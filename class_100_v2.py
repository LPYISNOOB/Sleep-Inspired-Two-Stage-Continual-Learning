import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import numpy as np
from tqdm import tqdm
import random
from copy import deepcopy

# ====================== 依赖 ======================
from spikingjelly.activation_based import neuron, surrogate, functional

# ====================== 全局随机种子 ======================
def _set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
_set_seed(42)

# ====================== 配置 ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = './data'

# 回放设置
SAMPLES_PER_CLASS = 2000
REPLAY_RATIO = 4.0

# 知识蒸馏
KD_TEMP = 2.0
KD_LAMBDA = 1.0                     # 增强 KD 强度，约束旧类别输出分布

# Sleep Phase 设置
SLEEP_EPOCHS = 10
SLEEP_BATCH_SIZE = 256
SLEEP_ITERS_PER_EPOCH = 100
SLEEP_LR = 2e-4

# 训练设置
EPOCHS_PER_TASK = 45
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-4

# Mixup 设置
MIXUP_ALPHA = 0.2

# 惊喜调制门控设置（创新点2）
GATE_BUDGET = 0.005                # 门控预算正则化强度（进一步降低，仅防坍塌）
SURPRISE_SCALE = 0.5               # 惊喜信号调制幅度
GATE_MOMENTUM = 0.9                # EMA 更新动量
WEIGHT_PRESERVE_LAMBDA = 0.05      # 参数级权重保持强度（门控调制）

# SNN 参数
SNN_NUM_STEPS = 12
PROJ_DIM = 1024
DROPOUT = 0.30

# 正则化
GRAD_CLIP_NORM = 1.0

# CIFAR-100 类增量标准设置
CLASSES_PER_TASK = 10
NUM_TASKS = 10

# CIFAR-100 归一化参数
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)


# ====================== 标签平滑损失 ======================
class LabelSmoothCE(nn.Module):
    """标签平滑交叉熵损失"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, logits, targets):
        n_classes = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_targets * log_probs).sum(dim=1).mean()


# ====================== Mixup 增强 ======================
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return x, y, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ====================== 创新点2: 惊喜调制门控生成器 ======================
class SurpriseModulatedGating(nn.Module):
    """
    惊喜调制门控生成器 (Surprise-Modulated Temporal-Synaptic Gating)

    相比原版 GatingGenerator 的核心改进：
    1. 多维输入：膜电位的 (均值, 标准差, 发放率) 三通道，而非单一全局标量
    2. 惊喜信号：维护旧任务膜电位统计的 EMA，计算当前输入与历史分布的差异
    3. 惊喜驱动调制：高惊喜（新知识）→ 门控趋近 1.0（允许可塑性）
                      低惊喜（旧知识）→ 门控趋近 0.1（保护稳定性）
    4. 门控预算正则化：对回放样本的门控平均值施加惩罚，防止收敛到 1.0

    抗坍塌机制：
    - EMA 锚定：熟悉的输入惊喜度自然降低，门控自动收缩
    - 逐样本计算：不同样本获得不同门控值，防止均匀坍塌
    - 惊喜归一化：batch 内惊喜度归一化，保持动态范围
    """
    def __init__(self, input_dim=3, hidden_dim=16, surprise_scale=SURPRISE_SCALE):
        super().__init__()
        self.surprise_scale = surprise_scale

        # 轻量 MLP: 3 → 16 → 8 → 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        # 初始化为高门控（~0.84），降低对前向学习的干扰，仅保留微量保护
        # sigmoid(2.0)≈0.88, final=0.1+0.8*0.88≈0.80
        with torch.no_grad():
            self.mlp[-1].bias.data.fill_(2.0)

        # EMA 统计缓冲区（旧任务膜电位参考分布）
        self.register_buffer('ema_v_mean', torch.tensor(0.0))
        self.register_buffer('ema_v_std', torch.tensor(0.0))
        self.register_buffer('ema_spike_rate', torch.tensor(0.0))
        self.register_buffer('ema_initialized', torch.tensor(0.0))  # 0=未初始化, 1=已初始化

    def forward(self, v_mean, v_std, spike_rate):
        """
        Args:
            v_mean: [B] 每个样本 T 步膜电位均值
            v_std:  [B] 每个样本 T 步膜电位标准差
            spike_rate: [B] 每个样本 T 步脉冲发放率
        Returns:
            gating: [B] 门控系数，范围 [0.1, 0.9]
        """
        B = v_mean.shape[0]
        stats = torch.stack([v_mean, v_std, spike_rate], dim=1)  # [B, 3]

        # 基础门控预测
        base = self.mlp(stats).squeeze(-1)  # [B]

        # 惊喜信号：当前统计与 EMA 的余弦距离
        if self.ema_initialized > 0.5:
            ema_stats = torch.tensor(
                [[self.ema_v_mean, self.ema_v_std, self.ema_spike_rate]],
                device=stats.device, dtype=stats.dtype
            ).expand(B, 3)
            surprise = 1.0 - F.cosine_similarity(stats, ema_stats, dim=1)  # [B], 0=熟悉, 1=新奇
            # 批内归一化，保持动态范围
            if B > 1:
                surprise = (surprise - surprise.min()) / (surprise.max() - surprise.min() + 1e-8)
        else:
            surprise = torch.ones(B, device=stats.device) * 0.5  # 首次任务：中性惊喜

        # 惊喜调制：高惊喜时推向 1.0（可塑性），低惊喜时保持基础值（稳定性）
        modulation = surprise * self.surprise_scale
        gate = torch.sigmoid(base + modulation)

        return (0.1 + 0.8 * gate).clamp(0.1, 0.9)

    @torch.no_grad()
    def update_ema(self, v_mean, v_std, spike_rate, momentum=GATE_MOMENTUM):
        """每个任务结束后，用回放样本更新 EMA 统计"""
        new_mean = v_mean.mean().item()
        new_std = v_std.mean().item()
        new_rate = spike_rate.mean().item()

        if self.ema_initialized < 0.5:
            self.ema_v_mean.fill_(new_mean)
            self.ema_v_std.fill_(new_std)
            self.ema_spike_rate.fill_(new_rate)
            self.ema_initialized.fill_(1.0)
        else:
            self.ema_v_mean.fill_(momentum * self.ema_v_mean.item() + (1 - momentum) * new_mean)
            self.ema_v_std.fill_(momentum * self.ema_v_std.item() + (1 - momentum) * new_std)
            self.ema_spike_rate.fill_(momentum * self.ema_spike_rate.item() + (1 - momentum) * new_rate)

    def get_ema_stats(self):
        return {
            'v_mean': self.ema_v_mean.item(),
            'v_std': self.ema_v_std.item(),
            'spike_rate': self.ema_spike_rate.item(),
            'initialized': self.ema_initialized.item()
        }


# ====================== 混合架构：ReLU ResNet 骨干 + SNN 分类头 ======================
class ReLUPreActBlock(nn.Module):
    """PreAct 风格残差块 + ReLU 激活"""
    def __init__(self, in_ch, out_ch, stride=1, drop2d=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(drop2d) if drop2d > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.shortcut = nn.Identity() if (stride == 1 and in_ch == out_ch) else \
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                          nn.BatchNorm2d(out_ch))
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = self.drop(F.relu(self.bn2(out), inplace=True))
        out = self.conv2(out)
        return out + residual


class CIFARResNetExtractor(nn.Module):
    """PreActResNet-18 特征提取器 — 全 ReLU，连续特征输出"""
    def __init__(self, proj_dim=1024, dropout=0.30):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.stage1 = nn.Sequential(ReLUPreActBlock(64, 64, 1, 0.00),
                                    ReLUPreActBlock(64, 64, 1, 0.00))
        self.stage2 = nn.Sequential(ReLUPreActBlock(64, 128, 2, 0.10),
                                    ReLUPreActBlock(128, 128, 1, 0.10))
        self.stage3 = nn.Sequential(ReLUPreActBlock(128, 256, 2, 0.15),
                                    ReLUPreActBlock(256, 256, 1, 0.15))
        self.stage4 = nn.Sequential(ReLUPreActBlock(256, 512, 2, 0.20),
                                    ReLUPreActBlock(512, 512, 1, 0.20))
        self.bn_f = nn.BatchNorm2d(512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(512, proj_dim),
                                  nn.LayerNorm(proj_dim), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout))
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x)
        return self.proj(self.gap(F.relu(self.bn_f(x), True)).flatten(1))


class IncrementalHybridSNN(nn.Module):
    """
    混合架构 — Class-IL 单头扩展：
    - 特征提取：标准 CNN（ReLU），一次前向得到连续特征 [B, proj_dim]
    - 分类决策：SNN LIF 速率编码，T 步累积发放率
    - 门控保护：SurpriseModulatedGating 逐样本调控输出幅度
    """
    def __init__(self, proj_dim=PROJ_DIM, dropout=DROPOUT, T=SNN_NUM_STEPS):
        super().__init__()
        self.T = T
        self.extractor = CIFARResNetExtractor(proj_dim, dropout)

        # SNN 分类头
        self.classifier = nn.Linear(proj_dim, 0)  # 动态扩展
        self.lif_cls = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                                      surrogate_function=surrogate.ATan(), decay_input=False)

        # 惊喜调制门控（创新点2）
        self.gating = SurpriseModulatedGating()

        self.current_num_classes = 0
        self.seen_classes = []

        # 存储最近一次前向的门控值（供训练损失使用）
        self._last_gating = None

    def update_num_classes(self, new_classes):
        num_new = len(new_classes)
        new_total = self.current_num_classes + num_new
        print(f"更新输出维度: {self.current_num_classes} -> {new_total}")

        old_cls = self.classifier
        old_w = old_cls.weight.data.clone() if old_cls.in_features > 0 else None
        old_b = old_cls.bias.data.clone() if (old_cls.in_features > 0 and old_cls.bias is not None) else None

        in_features = old_cls.in_features
        self.classifier = nn.Linear(in_features, new_total).to(device)

        if old_w is not None:
            self.classifier.weight.data[:self.current_num_classes] = old_w
            if old_b is not None:
                self.classifier.bias.data[:self.current_num_classes] = old_b

        if num_new > 0:
            with torch.no_grad():
                self.classifier.weight.data[self.current_num_classes:].normal_(0, 0.01)
                if self.classifier.bias is not None:
                    self.classifier.bias.data[self.current_num_classes:].zero_()

        self.seen_classes.extend(new_classes)
        self.current_num_classes = new_total

    def _get_membrane_stats(self):
        """从 LIF 神经元状态提取逐样本膜电位统计"""
        # lif_cls.v: [B, C] 膜电位矩阵
        v = self.lif_cls.v.detach()  # [B, C]
        v_mean = v.mean(dim=1)       # [B] 每个样本的平均膜电位
        v_std = v.std(dim=1)         # [B] 标准差
        return v_mean, v_std

    def forward(self, x):
        feat = self.extractor(x)                     # [B, proj_dim]

        spk_rec = []
        v_mean_sum = 0.0
        v_std_sum = 0.0
        spike_count = 0.0

        for t in range(self.T):
            out = self.classifier(feat)
            spk = self.lif_cls(out)
            spk_rec.append(spk)

            # 采集该步的膜电位统计（逐样本）
            vm, vs = self._get_membrane_stats()      # [B]
            v_mean_sum = v_mean_sum + vm
            v_std_sum = v_std_sum + vs
            spike_count = spike_count + (spk.detach() > 0).float().mean(dim=1)  # [B]

        # T 步平均统计
        v_mean = v_mean_sum / self.T                 # [B]
        v_std = v_std_sum / self.T                   # [B]
        spike_rate = spike_count / self.T            # [B]

        # 惊喜调制门控（创新点2）
        gating = self.gating(v_mean, v_std, spike_rate)  # [B]
        self._last_gating = gating

        # 速率编码输出（门控不再缩放前向输出，改为参数级正则）
        output = torch.stack(spk_rec, dim=0).sum(dim=0)  # [B, C]
        return output

    def get_class_mapping(self):
        class_to_idx = {cls: idx for idx, cls in enumerate(self.seen_classes)}
        idx_to_class = {idx: cls for idx, cls in enumerate(self.seen_classes)}
        return class_to_idx, idx_to_class

    def update_gating_ema(self, buffer):
        """任务结束后用回放样本更新门控 EMA"""
        if not buffer.buffer:
            return
        replay_x, _ = buffer.get_replay_batch(256)
        if replay_x is None:
            return
        replay_x = replay_x.to(device)
        self.eval()
        with torch.no_grad():
            functional.reset_net(self)
            _ = self.forward(replay_x)  # 触发 gating 前向，但不使用输出
            # forward 中已计算 gating，现在提取统计并更新 EMA
            # 重新计算统计（因为 forward 中的 _last_gating 已存储）
            feat = self.extractor(replay_x)
            v_mean_sum, v_std_sum, spike_sum = 0.0, 0.0, 0.0
            for t in range(self.T):
                out = self.classifier(feat)
                spk = self.lif_cls(out)
                vm, vs = self._get_membrane_stats()
                v_mean_sum += vm
                v_std_sum += vs
                spike_sum += (spk.detach() > 0).float().mean(dim=1)
            self.gating.update_ema(
                v_mean_sum / self.T,
                v_std_sum / self.T,
                spike_sum / self.T
            )
        self.train()


# ====================== 回放缓冲区 ======================
class ReplayBuffer:
    def __init__(self, samples_per_class=SAMPLES_PER_CLASS):
        self.samples_per_class = samples_per_class
        self.buffer = {}
        self.all_classes = set()

    def add_samples(self, images: torch.Tensor, labels: torch.Tensor):
        for i in range(len(images)):
            img = images[i].clone().cpu().unsqueeze(0)
            lab = int(labels[i].item())
            self.all_classes.add(lab)
            if lab not in self.buffer:
                self.buffer[lab] = []
            if len(self.buffer[lab]) < self.samples_per_class:
                self.buffer[lab].append(img)
            else:
                idx = np.random.randint(0, len(self.buffer[lab]))
                self.buffer[lab][idx] = img

    def get_replay_batch(self, desired_size: int):
        if not self.buffer:
            return None, None
        available_classes = list(self.buffer.keys())
        if not available_classes:
            return None, None

        num_classes = len(available_classes)
        samples_per_cls = max(1, desired_size // num_classes)

        replay_images = []
        replay_labels = []

        for cls in available_classes:
            cls_list = self.buffer[cls]
            n = min(samples_per_cls, len(cls_list))
            if n == 0:
                continue
            selected_idx = np.random.choice(len(cls_list), n, replace=False)
            for idx in selected_idx:
                replay_images.append(cls_list[idx])
                replay_labels.append(cls)

        if len(replay_images) == 0:
            return None, None
        return torch.cat(replay_images, dim=0), torch.tensor(replay_labels, dtype=torch.long)

    def get_class_distribution(self):
        return {cls: len(self.buffer.get(cls, [])) for cls in sorted(self.buffer.keys())}


# ====================== 数据集（CIFAR-100 在线增强 + Subset） ======================
def get_cifar100_task_datasets(task_id: int):
    classes = list(range(task_id * CLASSES_PER_TASK, (task_id + 1) * CLASSES_PER_TASK))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    full_train = CIFAR100(root=DATA_ROOT, train=True, download=True, transform=train_transform)
    train_idx = [i for i, (_, y) in enumerate(full_train) if y in classes]
    train_subset = Subset(full_train, train_idx)

    buffer_set = CIFAR100(root=DATA_ROOT, train=True, download=True, transform=test_transform)
    buffer_data = torch.stack([buffer_set[i][0] for i in train_idx])
    buffer_labels = torch.tensor([buffer_set[i][1] for i in train_idx])
    buffer_ds = TensorDataset(buffer_data, buffer_labels)

    test_set = CIFAR100(root=DATA_ROOT, train=False, download=True, transform=test_transform)
    test_idx = [i for i, (_, y) in enumerate(test_set) if y in classes]
    test_data = torch.stack([test_set[i][0] for i in test_idx])
    test_labels = torch.tensor([test_set[i][1] for i in test_idx])
    test_ds = TensorDataset(test_data, test_labels)

    return train_subset, test_ds, classes, buffer_ds


# ====================== 知识蒸馏损失 ======================
def distillation_loss(old_logits, new_logits, temperature=KD_TEMP):
    """iCaRL 风格蒸馏：保持旧类输出分布"""
    old_soft = F.softmax(old_logits / temperature, dim=1)
    new_log_soft = F.log_softmax(new_logits / temperature, dim=1)
    return -(old_soft * new_log_soft).sum(dim=1).mean() * (temperature ** 2)


# ====================== 创新点3: 日间训练 — 非对称回放策略 ======================
def train_on_task(model, train_subset, buffer, old_model=None):
    """
    日间训练（Day Learning）— 非对称回放策略：

    与原版一致的合并批次前向 + 三项创新增强：
    1. 非对称损失：新数据 LabelSmoothCE + Mixup / 回放数据标准 CE
    2. KD 蒸馏：合并输出上约束旧类别预测分布
    3. 惊喜调制门控：高初始门控（~0.84），门控预算防坍塌
    """
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_TASK)
    criterion_ls = LabelSmoothCE(smoothing=0.1)      # 新样本：标签平滑
    criterion_ce = nn.CrossEntropyLoss()              # 旧样本：标准 CE

    class_to_idx, _ = model.get_class_mapping()
    model.train()
    use_mixup = True

    for epoch in range(EPOCHS_PER_TASK):
        pbar = tqdm(train_loader, desc=f"Day Epoch {epoch+1}/{EPOCHS_PER_TASK}")
        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_gate = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            replay_x, replay_y = buffer.get_replay_batch(int(len(batch_y) * REPLAY_RATIO))

            # 构建合并批次
            if replay_x is not None:
                replay_x = replay_x.to(device)
                replay_y = replay_y.to(device)
                combined_x = torch.cat([batch_x, replay_x], dim=0)
                combined_y = torch.cat([batch_y, replay_y], dim=0)
            else:
                combined_x, combined_y = batch_x, batch_y

            combined_mapped = torch.tensor(
                [class_to_idx[y.item()] for y in combined_y], device=device)
            new_y_mapped = torch.tensor(
                [class_to_idx[y.item()] for y in batch_y], device=device)

            # ===== 非对称训练：新数据 Mixup + 回放数据标准 CE =====
            if use_mixup and epoch < EPOCHS_PER_TASK // 2 and replay_x is not None:
                # Mixup 仅对新数据
                mixed_x, y_a, y_b, lam = mixup_data(batch_x, new_y_mapped)
                functional.reset_net(model)
                mixed_out = model(mixed_x)
                mix_loss = mixup_criterion(criterion_ls, mixed_out, y_a, y_b, lam)

                # 回放数据标准 CE（无 Mixup）
                functional.reset_net(model)
                replay_mapped = torch.tensor(
                    [class_to_idx[y.item()] for y in replay_y], device=device)
                replay_out = model(replay_x)
                replay_loss = criterion_ce(replay_out, replay_mapped)

                ce_loss = (mix_loss + replay_loss) / 2

                # 合并前向用于 KD + 准确率（与原版一致）
                functional.reset_net(model)
                output = model(combined_x)
            elif use_mixup and epoch < EPOCHS_PER_TASK // 2:
                # 无回放数据时仅对新数据 Mixup
                mixed_x, y_a, y_b, lam = mixup_data(batch_x, new_y_mapped)
                functional.reset_net(model)
                output = model(mixed_x)
                ce_loss = mixup_criterion(criterion_ls, output, y_a, y_b, lam)
            else:
                # 无 Mixup：合并批次统一前向
                functional.reset_net(model)
                output = model(combined_x)
                ce_loss = criterion_ls(output, combined_mapped)

            # 准确率统计
            _, predicted = output.max(1)
            correct += (predicted == combined_mapped).sum().item()
            total += combined_mapped.size(0)

            # ===== KD 蒸馏（合并输出上约束旧类别分布，与原版一致） =====
            kd_loss = torch.tensor(0.0, device=device)
            if old_model is not None:
                old_model.eval()
                functional.reset_net(old_model)
                with torch.no_grad():
                    old_output = old_model(combined_x)
                old_num = old_model.current_num_classes
                kd_loss = distillation_loss(
                    old_output[:, :old_num], output[:, :old_num]
                ) / max(old_num, 1) * KD_LAMBDA

            # ===== 门控预算正则化（抗坍塌） =====
            gate_budget_loss = torch.tensor(0.0, device=device)
            gate_mean_val = model._last_gating.mean() if model._last_gating is not None else 1.0
            if model._last_gating is not None:
                gate_budget_loss = model._last_gating.mean() * GATE_BUDGET

            # ===== 参数级权重保持（门控调制，Adam 无法抵消） =====
            weight_preserve_loss = torch.tensor(0.0, device=device)
            if old_model is not None:
                old_num = old_model.current_num_classes
                # 仅约束旧类别对应的分类器权重
                old_w = old_model.classifier.weight.data[:old_num]  # [old_num, proj_dim]
                new_w = model.classifier.weight[:old_num]
                # 门控越低（越熟悉）→ (1-gate) 越大 → 惩罚越强
                preserve_strength = (1.0 - gate_mean_val) * WEIGHT_PRESERVE_LAMBDA
                weight_preserve_loss = ((new_w - old_w) ** 2).sum() * preserve_strength

            loss = ce_loss + kd_loss + gate_budget_loss + weight_preserve_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_kd += kd_loss.item()
            total_gate += gate_budget_loss.item() + weight_preserve_loss.item()

            gating_mean = model._last_gating.mean().item() if model._last_gating is not None else 0.0
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                ce=f"{ce_loss.item():.3f}",
                kd=f"{kd_loss.item():.3f}",
                wp=f"{weight_preserve_loss.item():.4f}",
                gate=f"{gating_mean:.3f}",
                acc=f"{100.*correct/total:.1f}%" if total > 0 else "N/A"
            )

        scheduler.step()
        n = len(train_loader)
        print(f"  Epoch {epoch+1}: loss={total_loss/n:.4f}, ce={total_ce/n:.4f}, "
              f"kd={total_kd/n:.4f}, gate+wp={total_gate/n:.4f}, "
              f"acc={100.*correct/total:.2f}%" if total > 0 else "N/A")


# ====================== 创新点1: 夜间巩固 — 增强 Sleep Phase ======================
def sleep_phase(model, buffer, old_model=None):
    """
    夜间巩固（Night Consolidation）：
    - 纯回放复习 + KD 蒸馏约束
    - 低学习率全局巩固
    - 完成后更新门控 EMA 统计
    """
    if not buffer.buffer:
        print("  [Sleep] 无旧样本，跳过")
        return

    optimizer = optim.Adam(model.parameters(), lr=SLEEP_LR)
    criterion_ce = nn.CrossEntropyLoss()
    class_to_idx, _ = model.get_class_mapping()

    print(f"  [Sleep Phase] 开始 {SLEEP_EPOCHS} 轮夜间巩固...")
    print(f"  [Sleep] 缓冲区类别分布: {buffer.get_class_distribution()}")

    for epoch in range(SLEEP_EPOCHS):
        model.train()
        total_loss = 0.0
        valid_iters = 0
        correct = 0
        total = 0

        for it in range(SLEEP_ITERS_PER_EPOCH):
            replay_x, replay_y = buffer.get_replay_batch(SLEEP_BATCH_SIZE)

            if replay_x is None or len(replay_x) < 2:
                break

            replay_x = replay_x.to(device)
            replay_y = replay_y.to(device)
            replay_mapped = torch.tensor([class_to_idx[y.item()] for y in replay_y], device=device)

            functional.reset_net(model)
            output = model(replay_x)
            ce_loss = criterion_ce(output, replay_mapped)

            # KD 蒸馏约束（夜间巩固中发挥更大作用）
            kd_loss = torch.tensor(0.0, device=device)
            if old_model is not None:
                old_model.eval()
                functional.reset_net(old_model)
                with torch.no_grad():
                    old_output = old_model(replay_x)
                old_num = old_model.current_num_classes
                kd_loss = distillation_loss(
                    old_output[:, :old_num],
                    output[:, :old_num]
                ) / max(old_num, 1) * KD_LAMBDA

            loss = ce_loss + kd_loss

            _, predicted = output.max(1)
            correct += (predicted == replay_mapped).sum().item()
            total += replay_mapped.size(0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            total_loss += loss.item()
            valid_iters += 1

            if (it + 1) % 20 == 0:
                gating_mean = model._last_gating.mean().item() if model._last_gating is not None else 0.0
                print(f"    Sleep Epoch {epoch+1}/{SLEEP_EPOCHS} | Iter {it+1}/{SLEEP_ITERS_PER_EPOCH} | "
                      f"Loss: {loss.item():.4f}, Gate: {gating_mean:.3f}, Acc: {100.*correct/total:.2f}%")

        if valid_iters > 0:
            avg_loss = total_loss / valid_iters
            avg_acc = 100. * correct / total
            print(f"  [Sleep] Epoch {epoch+1} 完成，avg loss: {avg_loss:.4f}, acc: {avg_acc:.2f}%")
        else:
            print(f"  [Sleep] Epoch {epoch+1} 无有效迭代")

    # 睡眠后更新门控 EMA（锚定旧知识统计）
    print(f"  [Sleep] 更新门控 EMA...")
    model.update_gating_ema(buffer)
    ema_stats = model.gating.get_ema_stats()
    print(f"  [Sleep] 门控 EMA: v_mean={ema_stats['v_mean']:.4f}, "
          f"v_std={ema_stats['v_std']:.4f}, spike_rate={ema_stats['spike_rate']:.4f}")


# ====================== 评估函数 ======================
def evaluate(model, test_datasets, task_id, class_to_idx, idx_to_class):
    model.eval()
    total_correct = 0
    total_samples = 0

    task_correct = {i: 0 for i in range(task_id + 1)}
    task_total = {i: 0 for i in range(task_id + 1)}

    idx_to_class_tensor = torch.tensor(list(idx_to_class.values()), device=device)

    with torch.no_grad():
        for t_id in range(task_id + 1):
            test_ds = test_datasets[t_id]
            loader = DataLoader(test_ds, batch_size=256, shuffle=False)

            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                functional.reset_net(model)
                output = model(x)
                pred_idx = output.argmax(dim=1)
                pred_class = idx_to_class_tensor[pred_idx]

                total_correct += (pred_class == y).sum().item()
                total_samples += y.size(0)

                mask = (y >= t_id * CLASSES_PER_TASK) & (y < (t_id + 1) * CLASSES_PER_TASK)
                task_total[t_id] += mask.sum().item()
                task_correct[t_id] += ((pred_class == y) & mask).sum().item()

    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
    task_accs = {t: task_correct[t] / task_total[t] * 100 if task_total[t] > 0 else 0
                 for t in range(task_id + 1)}
    return overall_acc, task_accs


# ====================== 主程序 ======================
def main():
    print("="*70)
    print("Split CIFAR-100 (10×10) | Class-IL | SMTS-Gating + Asymmetric Replay + Day-Night")
    print("="*70)
    print(f"设备: {device} | EPOCHS={EPOCHS_PER_TASK} | REPLAY={REPLAY_RATIO}")
    print(f"创新点1: 日间学习-夜间巩固双阶段框架")
    print(f"创新点2: 惊喜调制门控 (SurpriseModulatedGating) — 抗坍塌设计")
    print(f"创新点3: 非对称回放策略 (新样本Mixup+平滑 / 旧样本精确锚定+KD)")
    print(f"门控预算={GATE_BUDGET} | 惊喜尺度={SURPRISE_SCALE} | KD强度={KD_LAMBDA}")

    os.makedirs(DATA_ROOT, exist_ok=True)

    model = IncrementalHybridSNN(proj_dim=PROJ_DIM, dropout=DROPOUT, T=SNN_NUM_STEPS).to(device)
    buffer = ReplayBuffer()
    test_datasets = []
    old_model = None

    all_overall_accs = []
    all_task_accs = []

    for task_id in range(NUM_TASKS):
        print(f"\n{'='*70}")
        print(f"任务 {task_id} (类别 {task_id*CLASSES_PER_TASK}~{(task_id+1)*CLASSES_PER_TASK-1})")
        print(f"{'='*70}")

        train_subset, test_ds, task_classes, buffer_ds = get_cifar100_task_datasets(task_id)
        test_datasets.append(test_ds)

        # 保存旧模型供 KD 蒸馏
        if task_id > 0:
            old_model = deepcopy(model).eval()
            for p in old_model.parameters():
                p.requires_grad = False

        model.update_num_classes(task_classes)

        class_to_idx, idx_to_class = model.get_class_mapping()
        print(f"训练: {len(train_subset)} | 测试: {len(test_ds)} | 共 {len(class_to_idx)} 类")

        print("\n>>> 阶段1: 日间学习（非对称回放 + Mixup + KD + 门控预算）")
        train_on_task(model, train_subset, buffer, old_model)

        print("\n>>> 阶段2: 更新回放缓冲区")
        buffer.add_samples(buffer_ds.tensors[0], buffer_ds.tensors[1])
        print(f"缓冲区类别数: {len(buffer.buffer)}")

        print("\n>>> 阶段3: 夜间巩固（纯回放 + KD + EMA 更新）")
        sleep_phase(model, buffer, old_model)

        print("\n>>> 阶段4: 评估")
        overall_acc, task_accs = evaluate(model, test_datasets, task_id, class_to_idx, idx_to_class)
        all_overall_accs.append(overall_acc)
        all_task_accs.append(task_accs)

        print(f"\n任务 {task_id} 结束后:")
        print(f"  总体准确率: {overall_acc:.2f}%")
        print("  各任务准确率:")
        for t_id, acc in task_accs.items():
            print(f"    任务 {t_id}: {acc:.2f}%")

        if task_id > 0:
            avg_forget = np.mean([all_task_accs[task_id][t] - all_task_accs[task_id-1][t] for t in range(task_id)])
            print(f"  平均遗忘率: {avg_forget:.2f}%")

        # 打印门控 EMA 状态
        ema_stats = model.gating.get_ema_stats()
        print(f"  门控EMA: v_mean={ema_stats['v_mean']:.4f}, "
              f"v_std={ema_stats['v_std']:.4f}, spike={ema_stats['spike_rate']:.4f}")

    print("\n" + "="*70)
    print("训练完成！最终结果：")
    print("="*70)

    print("\n各任务后总体准确率:")
    for i, acc in enumerate(all_overall_accs):
        print(f"  任务 {i} 后: {acc:.2f}%")

    print("\n最终各任务准确率:")
    for task_id, acc in all_task_accs[-1].items():
        print(f"  任务 {task_id}: {acc:.2f}%")

    if len(all_task_accs) > 1:
        print("\n遗忘分析:")
        for task_id in range(NUM_TASKS - 1):
            first_acc = all_task_accs[task_id][task_id]
            last_acc = all_task_accs[-1][task_id]
            forgetting = first_acc - last_acc
            print(f"  任务 {task_id}: {first_acc:.2f}% -> {last_acc:.2f}% (遗忘 {forgetting:.2f}%)")

    print("\n=== SMTS-Gating + Asymmetric Replay + Day-Night CIFAR-100 训练完成 ===")


if __name__ == "__main__":
    main()
