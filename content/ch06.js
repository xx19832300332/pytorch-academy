(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch06',
  num: '06',
  part: 'p2',
  file: 'data-pipeline',
  zh: {
    title: '数据管道：Dataset 与 DataLoader',
    sub: '决定训练速度与正确性的，往往是数据这一侧',
    html: `
<p class="lead">模型再快，喂不饱也是白搭。PyTorch 把数据准备拆成两个角色：<strong>Dataset</strong> 回答「第 i 个样本长什么样」，<strong>DataLoader</strong> 负责「成批、打乱、并行地取」。这一章把这条流水线搭明白。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>用 <code>TensorDataset</code> 或自定义 Dataset 描述任意数据</li>
    <li>正确设置 batch_size / shuffle / num_workers / drop_last</li>
    <li>划分训练集与验证集，并解释为什么验证集绝不能 shuffle 进训练</li>
    <li>用 transforms 做数据增强</li>
  </ul>
</div>

<h2>1. 两个角色，一张图</h2>
<pre data-lang="python"><code>dataset  ──[ 单个样本 ]──►  DataLoader  ──[ 一个 batch ]──►  model
  │                          │
  └ __len__  __getitem__     └ sampler + collate + 多进程</code></pre>

<p>Dataset 只需要实现两个魔法方法：<code>__len__</code>（有多少条）和 <code>__getitem__</code>（第 i 条是什么）。剩下的事 DataLoader 全包。</p>

<h2>2. 最快的写法：TensorDataset</h2>
<pre data-lang="python"><code>import torch
from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000,))

ds = TensorDataset(X, y)          # 按第 0 维对齐打包
print(len(ds))                    # 1000
print(ds[0])                      # (x_0, y_0) 一个元组

dl = DataLoader(ds, batch_size=64, shuffle=True)
xb, yb = next(iter(dl))
print(xb.shape, yb.shape)         # (64, 20) (64,)</code></pre>

<h2>3. 自定义 Dataset：处理图片、文本、CSV</h2>
<pre data-lang="python"><code>from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class CatDogDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = sorted(Path(root).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        label = 1 if "dog" in path.name else 0
        if self.transform:
            img = self.transform(img)
        return img, label

ds = CatDogDataset("data/train")</code></pre>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p><code>__getitem__</code> 每次只处理<strong>一条</strong>数据，这是关键设计。它让 DataLoader 可以用多个进程同时读 8 条，也能在内存装不下 100 万张图时按需加载。你不需要写「怎么读一整个数据集」，只需要写「怎么读第 i 条」。</p>
</div>

<div class="callout warn">
  <span class="tag">Windows 上的经典报错</span>
  <p>在 Windows 下把 <code>num_workers</code> 设成大于 0，可能会遇到 <code>RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase</code>。解决办法：把训练代码放进 <code>if __name__ == "__main__":</code> 里，或者先用 <code>num_workers=0</code> 把逻辑跑通。</p>
</div>

<h2>4. DataLoader 的四个关键参数</h2>
<table>
  <tr><th>参数</th><th>作用</th><th>建议</th></tr>
  <tr><td><code>batch_size</code></td><td>每批多少样本</td><td>从 32 / 64 起，显存够就加大</td></tr>
  <tr><td><code>shuffle</code></td><td>每轮是否打乱顺序</td><td>训练集 <code>True</code>，验证/测试集 <code>False</code></td></tr>
  <tr><td><code>num_workers</code></td><td>几个子进程读数据</td><td>CPU 核心数的 1/2~1/4；Windows 先试 0</td></tr>
  <tr><td><code>drop_last</code></td><td>丢掉最后不足一批的样本</td><td>训练时若用 BatchNorm，建议 <code>True</code></td></tr>
</table>

<div class="callout math">
  <span class="tag">数学批注 · 一个 epoch 有多少步</span>
  <p>步数 = ⌈样本数 / batch_size⌉（<code>drop_last=True</code> 时向下取整）。</p>
  <p>例如 60000 张图、batch 64 → 938 步。如果训练 5 个 epoch，模型参数会被更新 <code>938 × 5 = 4690</code> 次。<strong>「epoch」和「step」是两个不同的计数单位</strong>，看日志时别混。</p>
</div>

<h2>5. 划分训练集与验证集</h2>
<pre data-lang="python"><code>from torch.utils.data import random_split

ds = TensorDataset(X, y)
n_train = int(len(ds) * 0.8)
train_ds, val_ds = random_split(
    ds, [n_train, len(ds) - n_train],
    generator=torch.Generator().manual_seed(42)   # 固定划分，保证可复现
)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=128, shuffle=False)

print(len(train_ds), len(val_ds))    # 800 200</code></pre>

<div class="callout warn">
  <span class="tag">验证集不能 shuffle 的深层原因</span>
  <p>技术上讲 shuffle 后算出的总 loss 是一样的。但如果你的评估代码依赖「顺序」（比如把预测和标签按顺序对齐画图），shuffle 会让结果错位。<strong>更重要的是</strong>：验证集绝对不能参与梯度更新，也不要在验证集上调超参数几十次——那等于在验证集上过拟合，你会得到一个「看起来很准，上线就崩」的模型。</p>
</div>

<h2>6. transforms：预处理与数据增强</h2>
<pre data-lang="python"><code>from torchvision import transforms

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),        # 随机裁剪缩放（增强）
    transforms.RandomHorizontalFlip(),        # 随机水平翻转（增强）
    transforms.ColorJitter(0.2, 0.2, 0.2),    # 颜色扰动（增强）
    transforms.ToTensor(),                    # PIL -> tensor, 归一到 [0,1]
    transforms.Normalize([0.485,0.456,0.406], # ImageNet 均值
                         [0.229,0.224,0.225]),# ImageNet 标准差
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 为什么除以标准差</span>
  <p><code>Normalize</code> 做的是 <code>(x - mean) / std</code>，把每个通道拉成均值 0、标准差 1。这对训练很重要：如果输入都是 0~255 的大数，第一层的梯度会非常大，学习率很难调；归一化后各维度尺度一致，梯度下降才好走。</p>
  <p>那三个奇怪的数字是 ImageNet 上统计出来的均值/标准差。如果你用预训练模型，<strong>必须用同样的数值</strong>，否则输入分布和模型训练时不一致，效果会明显下降。</p>
</div>

<div class="callout tip">
  <span class="tag">训练用增强，验证不用</span>
  <p>数据增强的目的是「让模型见到更多变化」，属于训练技巧。验证集要评估真实表现，只能用确定性的处理（resize + center crop）。搞混了会导致验证分数虚高或剧烈波动。</p>
</div>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li>Dataset 实现 <code>__len__</code> + <code>__getitem__</code>，一次只返回一条。</li>
    <li>DataLoader 负责批量化、打乱、多进程和 collate。</li>
    <li>训练集 <code>shuffle=True</code>，验证集 <code>False</code>。</li>
    <li>Windows 上 <code>num_workers&gt;0</code> 要配合 <code>__main__</code> 保护。</li>
    <li>step 与 epoch 不同：一个 epoch 包含 ⌈N/batch⌉ 个 step。</li>
    <li>transforms 训练要增强，验证只做确定性预处理。</li>
    <li>用预训练模型时，Normalize 的参数必须和它训练时一致。</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q">数据集有 1000 条，batch_size=64，drop_last=False，一个 epoch 有多少个 step？</p>
  <ol class="opts">
    <li>15</li>
    <li>16</li>
    <li>64</li>
    <li>1000</li>
  </ol>
  <p class="exp">1000/64 = 15.6，向上取整为 16，最后一批只有 40 条。若 drop_last=True 则是 15。</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">下面哪个<b>不是</b> Dataset 必须实现的方法？</p>
  <ol class="opts">
    <li><code>__len__</code></li>
    <li><code>__getitem__</code></li>
    <li><code>__collate__</code></li>
    <li>不需要其他方法</li>
  </ol>
  <p class="exp">collate 是 DataLoader 的职责（可用 collate_fn 定制）。Dataset 只需回答「多长」和「第 i 个是什么」。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">训练时输入像素是 0~255 的整数，最可能出现的现象是？</p>
  <ol class="opts">
    <li>梯度很大、loss 震荡或变 NaN，学习率必须调得极小</li>
    <li>训练速度变快</li>
    <li>完全没有影响</li>
    <li>自动归一化，无需处理</li>
  </ol>
  <p class="exp">输入尺度大 → 梯度大 → 同样的学习率会迈太大步子。归一化到 0 均值 1 方差能让优化稳定，这是标准操作。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>用 <code>TensorDataset</code> + <code>DataLoader</code> 造一个回归数据集，打印 3 个 batch 的形状。</li>
    <li>给自定义 Dataset 加一个「返回文件名」的功能，方便调试错样本。</li>
    <li>分别用 <code>shuffle=True/False</code> 取两轮数据，打印第一个 batch 的标签，观察差异。</li>
    <li>把 <code>num_workers</code> 从 0 改成 4，用 <code>time.time()</code> 对比读 100 个 batch 的耗时。</li>
  </ol>
</div>`
  },
  en: {
    title: 'Data pipeline: Dataset and DataLoader',
    sub: 'Training speed and correctness are often decided on the data side',
    html: `
<p class="lead">A fast model starved of data is useless. PyTorch splits data prep into two roles: <strong>Dataset</strong> answers "what is sample i", <strong>DataLoader</strong> handles batching, shuffling and parallelism.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Describe any dataset with <code>TensorDataset</code> or a custom Dataset</li>
    <li>Set batch_size / shuffle / num_workers / drop_last correctly</li>
    <li>Split train/val and explain why validation data never trains the model</li>
    <li>Apply transforms for preprocessing and augmentation</li>
  </ul>
</div>

<h2>1. Two roles</h2>
<pre data-lang="python"><code>dataset  ──[ one sample ]──►  DataLoader  ──[ one batch ]──►  model
   │                            │
   └ __len__  __getitem__       └ sampler + collate + workers</code></pre>

<h2>2. The fastest path: TensorDataset</h2>
<pre data-lang="python"><code>import torch
from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000,))

ds = TensorDataset(X, y)          # packs along dim 0
print(len(ds))                    # 1000
print(ds[0])                      # (x_0, y_0)

dl = DataLoader(ds, batch_size=64, shuffle=True)
xb, yb = next(iter(dl))
print(xb.shape, yb.shape)         # (64, 20) (64,)</code></pre>

<h2>3. A custom Dataset</h2>
<pre data-lang="python"><code>from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class CatDogDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = sorted(Path(root).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        label = 1 if "dog" in path.name else 0
        if self.transform:
            img = self.transform(img)
        return img, label

ds = CatDogDataset("data/train")</code></pre>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p><code>__getitem__</code> handles <strong>one</strong> sample. That design is what lets DataLoader read 8 at a time in parallel and stream a million images that do not fit in RAM. You never write "load the whole dataset", only "load item i".</p>
</div>

<div class="callout warn">
  <span class="tag">Classic Windows error</span>
  <p>With <code>num_workers&gt;0</code> on Windows you may hit <code>RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase</code>. Guard the entry point with <code>if __name__ == "__main__":</code>, or develop with <code>num_workers=0</code>.</p>
</div>

<h2>4. The four knobs that matter</h2>
<table>
  <tr><th>Arg</th><th>Effect</th><th>Suggestion</th></tr>
  <tr><td><code>batch_size</code></td><td>samples per step</td><td>start at 32/64, grow while memory allows</td></tr>
  <tr><td><code>shuffle</code></td><td>reshuffle each epoch</td><td>True for train, False for val/test</td></tr>
  <tr><td><code>num_workers</code></td><td>loader subprocesses</td><td>a quarter to half your cores; 0 first on Windows</td></tr>
  <tr><td><code>drop_last</code></td><td>drop the ragged last batch</td><td>True when using BatchNorm</td></tr>
</table>

<div class="callout math">
  <span class="tag">Math note · steps per epoch</span>
  <p>steps = ⌈N / batch_size⌉ (floor if <code>drop_last=True</code>).</p>
  <p>60000 images at batch 64 → 938 steps. Five epochs means 4690 parameter updates. <strong>epoch and step are different units</strong>; do not mix them up when reading logs.</p>
</div>

<h2>5. Train / validation split</h2>
<pre data-lang="python"><code>from torch.utils.data import random_split

ds = TensorDataset(X, y)
n_train = int(len(ds) * 0.8)
train_ds, val_ds = random_split(
    ds, [n_train, len(ds) - n_train],
    generator=torch.Generator().manual_seed(42)   # reproducible split
)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=128, shuffle=False)</code></pre>

<div class="callout warn">
  <span class="tag">Why validation must stay clean</span>
  <p>The bigger risk is not shuffling — it is leakage. Never update on validation data, and do not tune hyperparameters against it dozens of times: you are then fitting the validation set, and end up with a model that looks great offline and fails in production.</p>
</div>

<h2>6. transforms: preprocessing and augmentation</h2>
<pre data-lang="python"><code>from torchvision import transforms

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),                    # PIL -> tensor in [0,1]
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])</code></pre>

<div class="callout math">
  <span class="tag">Math note · why divide by std</span>
  <p><code>Normalize</code> computes <code>(x - mean) / std</code>, making each channel zero-mean unit-variance. Raw 0–255 inputs produce huge gradients in the first layer, which makes the learning rate nearly impossible to tune. Normalised features share a scale, so gradient descent behaves.</p>
  <p>Those odd constants are ImageNet statistics. With a pretrained model you <strong>must</strong> use the same values, or the input distribution will not match what it was trained on.</p>
</div>

<div class="callout tip">
  <span class="tag">Augment train, never val</span>
  <p>Augmentation is a training trick to show more variation. Validation should measure real performance with deterministic preprocessing.</p>
</div>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li>Dataset implements <code>__len__</code> + <code>__getitem__</code>, one sample at a time.</li>
    <li>DataLoader handles batching, shuffling, workers and collate.</li>
    <li>shuffle True for train, False for val.</li>
    <li>On Windows, <code>num_workers&gt;0</code> needs the <code>__main__</code> guard.</li>
    <li>One epoch = ⌈N/batch⌉ steps; step ≠ epoch.</li>
    <li>Augment the training set, keep validation deterministic.</li>
    <li>Match Normalize constants to the pretrained model.</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q">1000 samples, batch_size=64, drop_last=False: how many steps per epoch?</p>
  <ol class="opts">
    <li>15</li>
    <li>16</li>
    <li>64</li>
    <li>1000</li>
  </ol>
  <p class="exp">1000/64 = 15.6, round up to 16 with a ragged last batch of 40. With drop_last=True it is 15.</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">Which is NOT something a Dataset must implement?</p>
  <ol class="opts">
    <li><code>__len__</code></li>
    <li><code>__getitem__</code></li>
    <li><code>__collate__</code></li>
    <li>Nothing else is required</li>
  </ol>
  <p class="exp">Collation belongs to DataLoader (customisable via collate_fn).</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">Feeding raw 0–255 pixel values as input most likely causes:</p>
  <ol class="opts">
    <li>Huge gradients, oscillating loss or NaN unless the LR is tiny</li>
    <li>Faster training</li>
    <li>No effect at all</li>
    <li>Automatic normalisation</li>
  </ol>
  <p class="exp">Large input scale means large gradients. Normalising to zero mean, unit variance is standard practice.</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Build a regression dataset with TensorDataset and print three batch shapes.</li>
    <li>Extend the custom Dataset to also return the filename, for debugging bad samples.</li>
    <li>Compare the first batch's labels with shuffle on and off.</li>
    <li>Time 100 batches with num_workers=0 vs 4 using <code>time.time()</code>.</li>
  </ol>
</div>`
  }
});
