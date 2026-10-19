(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch15',
  num: '15',
  part: 'p6',
  file: 'capstone-vision',
  zh: {
    title: '综合实战：图像分类全流程',
    sub: '把前面 15 章拼成一条能跑通的流水线',
    html: `
<p class="lead">前面每一章都在拆零件：张量、autograd、nn.Module、DataLoader、损失、优化器、训练循环。这一章不再讲新概念，只做一件事——<b>把它们串成一条从原始数据到能用的模型的完整流水线</b>。走完这一遍，你就会知道真实项目里每一步该写在哪。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>独立写出「数据 → 模型 → 训练 → 评估 → 保存 → 推理」六段式代码</li>
    <li>解释每一步为什么必须存在，删掉会出什么问题</li>
    <li>把训练好的模型存成文件，在新进程里加载并预测一张新图片</li>
    <li>看懂训练日志，判断是欠拟合、过拟合还是学习率不对</li>
  </ul>
</div>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p>「流水线」（pipeline）是工程里的常用说法：指一串按固定顺序执行、前一步的输出就是后一步输入的环节。做菜也是流水线——洗、切、炒、装盘，顺序错了就做不成。深度学习项目同样是流水线，而且顺序错一步（比如验证集也 shuffle 了）结果就不可信。</p>
</div>

<h2>1. 先看全景：六段式</h2>
<table>
  <thead><tr><th>段</th><th>做什么</th><th>对应章节</th></tr></thead>
  <tbody>
    <tr><td>① 数据</td><td>Dataset + DataLoader，切成 batch</td><td>第 6 章</td></tr>
    <tr><td>② 模型</td><td>继承 nn.Module，写 forward</td><td>第 5 章</td></tr>
    <tr><td>③ 损失与优化器</td><td>选 loss，建 optimizer</td><td>第 7 章</td></tr>
    <tr><td>④ 训练循环</td><td>前向 → 损失 → 反传 → 更新</td><td>第 8 章</td></tr>
    <tr><td>⑤ 评估与保存</td><td>验证集算指标，存 state_dict</td><td>第 8、9 章</td></tr>
    <tr><td>⑥ 推理</td><td>加载权重，eval + no_grad 预测</td><td>第 14 章</td></tr>
  </tbody>
</table>

<div class="callout tip">
  <span class="tag">先想清楚再敲</span>
  <p>写代码之前，先在纸上回答三个问题：<b>输入长什么样？输出几个类别？用什么指标判断好坏？</b> 这三个问题答不上来，写多少行都是白写。本章的例子：输入 28×28 灰度图，输出 10 个类别，指标用准确率。</p>
</div>

<h2>2. 第①②段：数据与模型</h2>
<pre data-lang="python"><code>import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- ① 数据 ----------
# ToTensor: 把 PIL 图片变成 [0,1] 的 float 张量，并调整成 (C,H,W)
# Normalize: 用 MNIST 的全局均值/标准差做标准化，让输入分布更稳定
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_ds = datasets.MNIST("data", train=True,  download=True, transform=tf)
val_ds   = datasets.MNIST("data", train=False, download=True, transform=tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
# 注意：验证集绝对不能 shuffle，否则无法复现、也不影响指标但会让人误判
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

# ---------- ② 模型 ----------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.head(self.features(x))

model = Net().to(device)
print(model)
print("参数量：", sum(p.numel() for p in model.parameters()))</code></pre>

<div class="callout math">
  <span class="tag">数学批注</span>
  <p>为什么 <code>32 * 7 * 7</code>？卷积和池化都在改形状，必须自己算：输入 28×28，两次 <b>3×3 卷积 + padding=1</b> 尺寸不变，两次 <b>2×2 最大池化</b> 每次减半。所以 28 → 14 → 7。通道数最后是 32，展平后就是 32×7×7 = 1568 个数。</p>
  <p class="formula">H_out = ⌊(H + 2P − K) / S⌋ + 1</p>
  <p>拿不准就去第 10 章的交互计算器里填一遍，别硬背。写错这个数字是最常见的报错来源，典型提示是 <code>mat1 and mat2 shapes cannot be multiplied</code>。</p>
</div>

<h2>3. 第③④段：损失、优化器、训练循环</h2>
<pre data-lang="python"><code>criterion = nn.CrossEntropyLoss()
# 只把需要训练的参数交给优化器；weight_decay 就是 L2 正则
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()                     # 开启 dropout / BN 的训练行为
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)   # 数据和模型必须在同一设备

        optimizer.zero_grad()               # 清空上一轮残留的梯度
        logits = model(x)                   # ① 前向
        loss = criterion(logits, y)         # ② 算损失
        loss.backward()                     # ③ 反向传播求梯度
        optimizer.step()                    # ④ 按梯度更新参数

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        seen       += x.size(0)
    return total_loss / seen, correct / seen

@torch.no_grad()                     # 评估不需要建图，省显存也更快
def evaluate(model, loader):
    model.eval()                     # 关掉 dropout，BN 用滑动统计量
    correct, seen = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        seen    += x.size(0)
    return correct / seen

best = 0.0
for epoch in range(5):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    va_acc = evaluate(model, val_loader)
    print(f"epoch {epoch+1}  loss {tr_loss:.4f}  train {tr_acc:.3f}  val {va_acc:.3f}")
    if va_acc &gt; best:                       # 只保留最好的一版
        best = va_acc
        torch.save(model.state_dict(), "mnist_cnn.pt")
print("best val acc:", best)</code></pre>

<div class="callout warn">
  <span class="tag">四个必踩的坑</span>
  <ul>
    <li><b>忘记 zero_grad</b>：梯度会累加，等效学习率越来越大，loss 会突然炸。</li>
    <li><b>忘记 model.train() / eval()</b>：dropout 在验证时还在随机丢神经元，指标偏低且每次不同。</li>
    <li><b>验证时忘了 no_grad</b>：能跑，但显存暴涨，大模型直接 OOM。</li>
    <li><b>只存了 model 没存结构信息</b>：以后加载时必须先建出一模一样的类，再 load_state_dict。</li>
  </ul>
</div>

<h2>4. 第⑤⑥段：保存、加载、推理</h2>
<p>训练完成只是半程。真正的交付物是「一个能被别人加载、并对新数据给出预测的文件」。</p>
<pre data-lang="python"><code># ---------- 保存：只存参数（推荐） ----------
torch.save(model.state_dict(), "mnist_cnn.pt")

# ---------- 在新进程里加载 ----------
model2 = Net().to(device)                 # 先有结构
model2.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
model2.eval()                             # 一定要 eval，否则 dropout 会让结果不稳定

# ---------- 对一张图推理 ----------
img, label = val_ds[0]                    # img: (1,28,28)
with torch.no_grad():
    logits = model2(img.unsqueeze(0).to(device))   # 补一个 batch 维 -> (1,1,28,28)
    prob = logits.softmax(1)
    pred = prob.argmax(1).item()
print("真实:", label, "预测:", pred, "置信度:", round(prob.max().item(), 3))</code></pre>

<div class="callout basic">
  <span class="tag">为什么非要加 unsqueeze(0)</span>
  <p>模型永远按「一批数据」来处理，输入必须是四维 <code>(N, C, H, W)</code>。单张图是 <code>(1, 28, 28)</code> 三维，缺了最前面的「几张」这一维。<code>unsqueeze(0)</code> 就是在最前面插一个 1，变成 <code>(1, 1, 28, 28)</code>，意思是「这一批只有 1 张图」。</p>
</div>

<h2>5. 怎么读训练日志</h2>
<table>
  <thead><tr><th>现象</th><th>含义</th><th>怎么办</th></tr></thead>
  <tbody>
    <tr><td>train 和 val 都低，且不涨</td><td>欠拟合 / 学习率太小</td><td>加大模型、调大 lr、多训几轮</td></tr>
    <tr><td>train 高、val 低且差距变大</td><td>过拟合</td><td>加 dropout / weight_decay、做数据增强</td></tr>
    <tr><td>loss 变 NaN</td><td>学习率太大或数据没标准化</td><td>lr 除以 10，检查 Normalize</td></tr>
    <tr><td>val 每次都不一样</td><td>忘了 eval()</td><td>评估前调用 model.eval()</td></tr>
  </tbody>
</table>

<h2>6. 本章小测</h2>

<div class="quiz" data-answer="2">
  <p class="q">训练循环里，<code>optimizer.zero_grad()</code> 应该放在哪里？</p>
  <ol class="opts">
    <li>放在 loss.backward() 之后、step() 之前</li>
    <li>每个 epoch 开头调用一次就够了</li>
    <li>放在 backward() 之前（每次迭代都清）</li>
    <li>不需要，PyTorch 会自动清零</li>
  </ol>
  <p class="exp">必须在每次 backward 之前清。放在 backward 之后会把刚算好的梯度白清掉；每个 epoch 只清一次则批间梯度累加，等效学习率失控。</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q">验证集 DataLoader 设置 <code>shuffle=True</code> 的后果是？</p>
  <ol class="opts">
    <li>训练会更快，指标不受影响</li>
    <li>不影响准确率数值，但让结果不可复现、也不好做逐样本对比</li>
    <li>会让验证准确率虚高</li>
    <li>会直接报错</li>
  </ol>
  <p class="exp">验证不需要打乱：它只是遍历算指标，顺序不影响平均值，但打乱后每次顺序不同，排查问题时没法逐样本对照，也破坏了可复现性。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">加载已保存的模型做推理，漏掉 <code>model.eval()</code> 最可能的症状是？</p>
  <ol class="opts">
    <li>同一张图多次预测结果可能不一样</li>
    <li>直接报 shape 不匹配</li>
    <li>显存占用翻倍</li>
    <li>梯度变成 None</li>
  </ol>
  <p class="exp">eval() 关掉 dropout、让 BatchNorm 用滑动统计量。不调用就还在随机丢弃神经元，预测自然抖动。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <p>把上面的代码完整跑一遍（CPU 上 5 个 epoch 大约几分钟），然后依次做三件事：</p>
  <ol>
    <li>把 <code>Dropout(0.2)</code> 删掉，对比 val 准确率的变化——这就是正则化在起作用。</li>
    <li>把 <code>lr</code> 从 1e-3 改成 1.0，观察 loss 是否变成 NaN，理解「学习率过大」的物理含义。</li>
    <li>加载保存的权重，随便挑 10 张验证集图片做批量预测，把预测错的图找出来看看——人工看错例，是提升模型最快的直觉训练。</li>
  </ol>
</div>

<div class="keypoints">
  <h4>本章要点</h4>
  <ul>
    <li>六段式：数据 → 模型 → 损失优化器 → 训练 → 评估保存 → 推理</li>
    <li>训练循环四步：zero_grad → forward → backward → step</li>
    <li>train() / eval() 必须成对切换，验证要用 no_grad</li>
    <li>只存 state_dict，加载时先建结构再 load，最后 eval()</li>
    <li>单张图推理要 unsqueeze(0) 补 batch 维</li>
  </ul>
</div>
`
  },
  en: {
    title: 'Capstone: the full image-classification pipeline',
    sub: 'Wire the previous 15 chapters into one pipeline that actually runs',
    html: `
<p class="lead">Every earlier chapter took a part apart: tensors, autograd, <code>nn.Module</code>, <code>DataLoader</code>, losses, optimizers, the training loop. This chapter adds no new concepts — it <b>connects them into one pipeline from raw data to a model you can ship</b>. Walk it once and you will know where each piece belongs in a real project.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Write the six-stage flow yourself: data → model → train → evaluate → save → infer</li>
    <li>Explain why each stage must exist and what breaks without it</li>
    <li>Save a trained model and load it in a fresh process to predict a new image</li>
    <li>Read a training log and tell underfitting from overfitting from a bad learning rate</li>
  </ul>
</div>

<div class="callout basic">
  <span class="tag">Beginner note</span>
  <p>A <b>pipeline</b> is an ordered chain where each stage feeds the next. Cooking is a pipeline: wash, chop, fry, plate — get the order wrong and dinner fails. Deep learning is the same, and one wrong step (say, shuffling the validation set) quietly makes your numbers untrustworthy.</p>
</div>

<h2>1. The big picture: six stages</h2>
<table>
  <thead><tr><th>Stage</th><th>What happens</th><th>Chapter</th></tr></thead>
  <tbody>
    <tr><td>① Data</td><td>Dataset + DataLoader, split into batches</td><td>6</td></tr>
    <tr><td>② Model</td><td>Subclass nn.Module, write forward</td><td>5</td></tr>
    <tr><td>③ Loss + optimizer</td><td>Pick a loss, build an optimizer</td><td>7</td></tr>
    <tr><td>④ Training loop</td><td>forward → loss → backward → step</td><td>8</td></tr>
    <tr><td>⑤ Evaluate &amp; save</td><td>Validate, store state_dict</td><td>8, 9</td></tr>
    <tr><td>⑥ Inference</td><td>Load weights, eval + no_grad</td><td>14</td></tr>
  </tbody>
</table>

<div class="callout tip">
  <span class="tag">Think before typing</span>
  <p>Answer three questions on paper first: <b>what shape is the input, how many classes are there, and what metric decides success?</b> Our example: 28×28 grayscale in, 10 classes out, accuracy as the metric.</p>
</div>

<h2>2. Stages ① and ②: data and model</h2>
<pre data-lang="python"><code>import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- ① data ----------
tf = transforms.Compose([
    transforms.ToTensor(),                        # PIL image -> float tensor (C,H,W) in [0,1]
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean/std
])

train_ds = datasets.MNIST("data", train=True,  download=True, transform=tf)
val_ds   = datasets.MNIST("data", train=False, download=True, transform=tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

# ---------- ② model ----------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.head(self.features(x))

model = Net().to(device)
print("params:", sum(p.numel() for p in model.parameters()))</code></pre>

<div class="callout math">
  <span class="tag">Math note</span>
  <p>Where does <code>32 * 7 * 7</code> come from? Convolutions and pooling change shapes, so you compute it. Input 28×28; two <b>3×3 convs with padding=1</b> keep the size; two <b>2×2 max-pools</b> halve it each time: 28 → 14 → 7. With 32 channels, flattening gives 32×7×7 = 1568 numbers.</p>
  <p class="formula">H_out = ⌊(H + 2P − K) / S⌋ + 1</p>
  <p>When unsure, use the interactive calculator in chapter 10. Getting this number wrong is the single most common error, usually surfacing as <code>mat1 and mat2 shapes cannot be multiplied</code>.</p>
</div>

<h2>3. Stages ③ and ④: loss, optimizer, loop</h2>
<pre data-lang="python"><code>criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()                     # enable dropout / BN training behaviour
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)   # data and model must share a device

        optimizer.zero_grad()               # 1. clear stale gradients
        logits = model(x)                   # 2. forward
        loss = criterion(logits, y)         # 3. loss
        loss.backward()                     # 4. backprop
        optimizer.step()                    # 5. update

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        seen       += x.size(0)
    return total_loss / seen, correct / seen

@torch.no_grad()                     # no graph needed for evaluation
def evaluate(model, loader):
    model.eval()                     # dropout off, BN uses running stats
    correct, seen = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        seen    += x.size(0)
    return correct / seen

best = 0.0
for epoch in range(5):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    va_acc = evaluate(model, val_loader)
    print(f"epoch {epoch+1}  loss {tr_loss:.4f}  train {tr_acc:.3f}  val {va_acc:.3f}")
    if va_acc &gt; best:
        best = va_acc
        torch.save(model.state_dict(), "mnist_cnn.pt")
print("best val acc:", best)</code></pre>

<div class="callout warn">
  <span class="tag">Four mistakes everyone makes</span>
  <ul>
    <li><b>Forgetting zero_grad</b>: gradients accumulate, the effective learning rate grows, loss explodes.</li>
    <li><b>Forgetting train()/eval()</b>: dropout still fires during validation, so accuracy is lower and jitters.</li>
    <li><b>No no_grad in validation</b>: it runs, but memory balloons; big models OOM.</li>
    <li><b>Saving the whole model instead of state_dict</b>: you still need the class definition, so save state_dict.</li>
  </ul>
</div>

<h2>4. Stages ⑤ and ⑥: save, load, infer</h2>
<p>Training is only half the job. The deliverable is a file someone else can load and run on new data.</p>
<pre data-lang="python"><code>torch.save(model.state_dict(), "mnist_cnn.pt")

model2 = Net().to(device)                 # structure first
model2.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
model2.eval()                             # always eval before inference

img, label = val_ds[0]                    # img: (1,28,28)
with torch.no_grad():
    logits = model2(img.unsqueeze(0).to(device))   # add batch dim -> (1,1,28,28)
    prob = logits.softmax(1)
    pred = prob.argmax(1).item()
print("true:", label, "pred:", pred, "conf:", round(prob.max().item(), 3))</code></pre>

<div class="callout basic">
  <span class="tag">Why unsqueeze(0)</span>
  <p>Models always process a batch, so the input must be 4-D <code>(N, C, H, W)</code>. A single image is 3-D <code>(1, 28, 28)</code> — the leading "how many images" dimension is missing. <code>unsqueeze(0)</code> inserts a 1 there, meaning "a batch of exactly one".</p>
</div>

<h2>5. Reading the training log</h2>
<table>
  <thead><tr><th>Symptom</th><th>Meaning</th><th>Fix</th></tr></thead>
  <tbody>
    <tr><td>train and val both low and flat</td><td>Underfitting / lr too small</td><td>Bigger model, higher lr, more epochs</td></tr>
    <tr><td>train high, val low, gap widening</td><td>Overfitting</td><td>More dropout / weight decay, augment data</td></tr>
    <tr><td>loss becomes NaN</td><td>lr too high or unnormalised data</td><td>Divide lr by 10, check Normalize</td></tr>
    <tr><td>val changes every run</td><td>Missing eval()</td><td>Call model.eval() before evaluating</td></tr>
  </tbody>
</table>

<h2>6. Quiz</h2>

<div class="quiz" data-answer="2">
  <p class="q">Where should <code>optimizer.zero_grad()</code> go in the loop?</p>
  <ol class="opts">
    <li>After backward(), before step()</li>
    <li>Once at the top of each epoch</li>
    <li>Before backward(), every iteration</li>
    <li>Nowhere — PyTorch clears automatically</li>
  </ol>
  <p class="exp">It must run before each backward. Placing it after backward throws away the gradients you just computed; clearing once per epoch lets gradients accumulate across batches.</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q">What happens if the validation DataLoader uses <code>shuffle=True</code>?</p>
  <ol class="opts">
    <li>Training gets faster and metrics are unaffected</li>
    <li>The accuracy value is unaffected, but results stop being reproducible</li>
    <li>Validation accuracy becomes artificially high</li>
    <li>It raises an error</li>
  </ol>
  <p class="exp">Order does not change an average, so accuracy is fine — but a shuffled validation pass breaks reproducibility and makes per-sample debugging impossible.</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">You load saved weights for inference but forget <code>model.eval()</code>. Most likely symptom?</p>
  <ol class="opts">
    <li>The same image can get different predictions across runs</li>
    <li>An immediate shape-mismatch error</li>
    <li>Memory usage doubles</li>
    <li>Gradients become None</li>
  </ol>
  <p class="exp">eval() disables dropout and switches BatchNorm to running statistics. Without it, neurons are still randomly dropped and predictions jitter.</p>
</div>

<div class="exercise">
  <h4>Hands-on</h4>
  <p>Run the code end to end (a few minutes on CPU for 5 epochs), then do three things:</p>
  <ol>
    <li>Delete <code>Dropout(0.2)</code> and compare validation accuracy — that is regularisation at work.</li>
    <li>Set <code>lr</code> to 1.0 and watch loss turn NaN — that is what "learning rate too high" physically means.</li>
    <li>Load the saved weights, predict 10 validation images, and eyeball the mistakes — reading failure cases is the fastest way to build intuition.</li>
  </ol>
</div>

<div class="keypoints">
  <h4>Key points</h4>
  <ul>
    <li>Six stages: data → model → loss/optimizer → train → evaluate/save → infer</li>
    <li>Loop body: zero_grad → forward → backward → step</li>
    <li>train() / eval() always come in pairs; validate under no_grad</li>
    <li>Save state_dict, rebuild the class, load, then eval()</li>
    <li>Single-image inference needs unsqueeze(0)</li>
  </ul>
</div>
`
  }
});
