(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch13',
  num: '13',
  part: 'p4',
  file: 'gpu-performance',
  zh: {
    title: 'GPU、显存与性能调优',
    sub: '让训练从 10 小时变成 1 小时',
    html: `
<p class="lead">同一个模型，写法不同可以差 5~10 倍速度。这一章讲几件立刻能用的事：怎么正确放到 GPU、怎么测时间、怎么用混合精度、显存爆了怎么办。不需要懂 CUDA 底层。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>写出设备无关（CPU/GPU 通用）的训练代码</li>
    <li>正确测量 GPU 耗时，不被异步执行骗到</li>
    <li>用 AMP 混合精度把速度提升 1.5~2 倍、显存减半</li>
    <li>遇到 OOM 时按顺序尝试 5 种解法</li>
  </ul>
</div>

<h2>1. 设备无关的写法</h2>
<pre data-lang="python"><code>import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)                       # cuda 或 cpu

model = model.to(device)            # 参数搬到 GPU

for xb, yb in loader:
    xb = xb.to(device)
    yb = yb.to(device)
    logits = model(xb)              # 三者必须同设备
    ...

# 想指定第 1 块卡
# device = torch.device("cuda:1")</code></pre>

<div class="callout warn">
  <span class="tag">头号报错</span>
  <p><code>Expected all tensors to be on the same device, but found cuda:0 and cpu</code> —— 模型在 GPU、数据在 CPU（或中途新建的张量没搬）。排查方式：在出错行前打印 <code>model.device</code> 相关的参数 <code>next(model.parameters()).device</code> 和 <code>xb.device</code>。注意<strong>新建的张量默认在 CPU</strong>，比如 <code>torch.zeros(...)</code> 要和输入同设备时得手动 <code>.to(device)</code>。</p>
</div>

<h2>2. 为什么 GPU 快</h2>
<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p>CPU 像 8 个博士生，擅长复杂推理但人少；GPU 像 5000 个中学生，每人只会做简单乘法，但同时开工。</p>
  <p>神经网络的主要计算是<strong>大规模矩阵乘法</strong>——正好是「很多简单乘加」，于是 GPU 大杀四方。但如果你写了很多 Python 循环、每次只算一个数，GPU 反而更慢：光是「派活」的开销就超过了计算本身。</p>
  <p>所以原则是：<strong>把循环写成张量运算，一次派给整个 batch。</strong></p>
</div>

<h2>3. 正确测时间</h2>
<pre data-lang="python"><code>import time

# ❌ 错：GPU 是异步的，这里测到的是「派活时间」不是计算时间
start = time.time()
y = model(x)
print(time.time() - start)      # 可能只有 0.0001 秒，明显不真实

# ✅ 对：先同步，确保前面的计算真的做完了
if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.time()
y = model(x)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print(time.time() - start)</code></pre>

<div class="callout basic">
  <span class="tag">异步执行</span>
  <p>PyTorch 把计算任务丢进 GPU 队列就立刻返回，Python 继续往下跑。所以不加 <code>synchronize()</code>，你测到的只是「下达命令」的耗时。这个坑让无数人得出「GPU 没比 CPU 快」的错误结论。</p>
</div>

<h2>4. 混合精度（AMP）：最划算的加速</h2>
<pre data-lang="python"><code>from torch.amp import autocast, GradScaler    # PyTorch 2.x 推荐入口

scaler = GradScaler("cuda")                   # 指定设备，不再依赖隐式判断

for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad()

    with autocast("cuda", dtype=torch.float16):   # 前向用半精度算
        loss = criterion(model(xb), yb)

    scaler.scale(loss).backward()           # 梯度放大，防止下溢
    scaler.step(optimizer)
    scaler.update()</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 为什么要 GradScaler</span>
  <p>float16 能表示的<strong>最小正数</strong>大约是 6e-8，而梯度经常比这更小，直接算会变成 0（下溢），参数就更新不了。</p>
  <p>GradScaler 的做法：先把 loss 乘以一个大数（如 65536）再反向，梯度整体放大，不会下溢；更新参数前再除回去。如果检测到 inf/nan，会自动跳这一步并调小缩放系数。</p>
  <p>收益：显存约减半，速度提升 1.5~2 倍（Tensor Core 加成）。代价是极少量的数值精度损失，实践中几乎无影响。</p>
</div>

<div class="callout tip">
  <span class="tag">新版本写法</span>
  <p>上面代码用的就是 2.x 的新接口：<code>torch.amp.autocast("cuda", ...)</code> 与 <code>GradScaler("cuda")</code>，设备名作为第一个参数显式传入。旧写法 <code>torch.cuda.amp.autocast()</code> / <code>GradScaler()</code> 在 2.4+ 会给出 FutureWarning，功能相同，但新代码请直接用新接口。</p>
</div>

<h2>5. 显存爆了怎么办（按顺序试）</h2>
<ol class="steps">
  <li><strong>减小 batch_size</strong>。最直接，通常减半就够。配合梯度累积保持等效 batch。</li>
  <li><strong>开 AMP</strong>。显存约减半，几乎免费。</li>
  <li><strong>及时释放</strong>：确认没把 loss/tensor 存进列表；评估用 <code>no_grad</code>；必要时 <code>del x</code> + <code>torch.cuda.empty_cache()</code>。</li>
  <li><strong>梯度检查点</strong>：<code>torch.utils.checkpoint</code>，用计算换显存，适合超深模型。</li>
  <li><strong>减小模型或输入尺寸</strong>：降分辨率、减层数、用更小的 backbone。</li>
</ol>

<pre data-lang="python"><code># 梯度累积：小显存模拟大 batch
accum_steps = 4
for i, (xb, yb) in enumerate(loader):
    loss = criterion(model(xb.to(device)), yb.to(device)) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 为什么除以 accum_steps</span>
  <p>梯度是累加的，累积 4 个 batch 的梯度和 ≈ 4 倍 batch 的梯度<strong>和</strong>。但大 batch 的 loss 是「平均」，所以要在每个小 batch 上先除以 4，累加后正好等价。</p>
  <p>注意：这和「真的大 batch」不完全等价（BatchNorm 仍按小 batch 统计），但对大多数情况够用。</p>
</div>

<h2>6. 三个免费的提速技巧</h2>
<pre data-lang="python"><code># 1) 固定形状 + channels_last，让卷积走更快的算法
model = model.to(memory_format=torch.channels_last)
x = x.to(memory_format=torch.channels_last)

# 2) DataLoader 异步拷贝
loader = DataLoader(ds, batch_size=64, shuffle=True,
                    num_workers=8, pin_memory=True)
xb = xb.to(device, non_blocking=True)

# 3) 编译整个模型（PyTorch 2.0+），通常有 20~50% 提升
model = torch.compile(model)</code></pre>

<div class="callout warn">
  <span class="tag">num_workers 不是越大越好</span>
  <p>设成 CPU 核心数以上会导致进程调度开销反超收益。经验值是核心数的 1/4 到 1/2；在容器或 Windows 上从 2~4 开始试，并用实际耗时验证。</p>
</div>

<h2>7. 多卡训练：一句话说明白</h2>
<table>
  <tr><th>方式</th><th>原理</th><th>建议</th></tr>
  <tr><td><code>DataParallel</code></td><td>单进程多线程，主卡收集梯度</td><td>已不推荐，主卡会成为瓶颈</td></tr>
  <tr><td><code>DistributedDataParallel</code></td><td>多进程，各自算梯度后 all-reduce 同步</td><td>标准做法，配合 <code>torchrun</code></td></tr>
</table>
<p>单卡能跑就别碰多卡，调度的复杂度远大于收益。真需要时用 <code>torchrun --nproc_per_node=4 train.py</code>，代码里用 <code>DistributedSampler</code> 切分数据。</p>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li>所有张量必须在同一设备，新建张量默认在 CPU。</li>
    <li>测 GPU 时间必须先 <code>synchronize()</code>。</li>
    <li>AMP = 前向用 float16 + GradScaler 防下溢，提速又省显存。</li>
    <li>OOM 的顺序：降 batch → AMP → 释放引用 → checkpointing → 减模型。</li>
    <li>梯度累积时 loss 要除以累积步数。</li>
    <li>多卡优先 DDP，单卡能跑就别上多卡。</li>
  </ol>
</div>

<div class="quiz" data-answer="2">
  <p class="q">不加 <code>torch.cuda.synchronize()</code> 测 GPU 训练时间，会得到什么结果？</p>
  <ol class="opts">
    <li>偏慢</li>
    <li>准确</li>
    <li>偏快，因为只测到了任务下发时间</li>
    <li>报错</li>
  </ol>
  <p class="exp">CUDA 执行是异步的，Python 端立刻返回。必须同步后再计时。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">GradScaler 的作用是？</p>
  <ol class="opts">
    <li>放大 loss 以避免 float16 梯度下溢，更新前再缩回</li>
    <li>自动调整学习率</li>
    <li>压缩模型参数</li>
    <li>加速数据加载</li>
  </ol>
  <p class="exp">float16 的最小数太小，梯度容易变 0。缩放是标准解法。</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q">梯度累积 4 步，每个小 batch 的 loss 应该？</p>
  <ol class="opts">
    <li>乘以 4</li>
    <li>除以 4</li>
    <li>不变</li>
    <li>取平方</li>
  </ol>
  <p class="exp">梯度累加，所以要除以累积步数才能和大 batch 的平均梯度等价。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>用 <code>time.perf_counter()</code> + <code>synchronize()</code> 对比 CPU 与 GPU 上一个 1024×1024 矩阵乘的耗时。</li>
    <li>给第 8 章的 MNIST 训练加上 AMP，记录每轮耗时与显存（<code>torch.cuda.max_memory_allocated()</code>）。</li>
    <li>把 batch_size 设为 1024 制造 OOM，然后按本章顺序修复，记录哪一步解决问题。</li>
    <li>试试 <code>torch.compile(model)</code>，对比首次运行（含编译）与第二次运行的耗时。</li>
  </ol>
</div>`
  },
  en: {
    title: 'GPU, memory and performance',
    sub: 'Turn a 10-hour run into a 1-hour run',
    html: `
<p class="lead">The same model can be 5–10× slower depending on how you write it. This chapter covers what you can use immediately: device handling, honest timing, mixed precision, and fixing out-of-memory. No CUDA internals required.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Write device-agnostic training code</li>
    <li>Time GPU work correctly despite async execution</li>
    <li>Use AMP for 1.5–2× speedup and half the memory</li>
    <li>Fix OOM in a principled order</li>
  </ul>
</div>

<h2>1. Device-agnostic code</h2>
<pre data-lang="python"><code>device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

for xb, yb in loader:
    xb = xb.to(device)
    yb = yb.to(device)
    logits = model(xb)</code></pre>

<div class="callout warn">
  <span class="tag">Error #1</span>
  <p><code>Expected all tensors to be on the same device, but found cuda:0 and cpu</code> — the model or data did not move. Remember that <strong>newly created tensors default to CPU</strong>: <code>torch.zeros(...)</code> must be moved explicitly.</p>
</div>

<h2>2. Why GPUs are fast</h2>
<div class="callout basic">
  <span class="tag">For beginners</span>
  <p>A CPU is a few brilliant PhDs; a GPU is thousands of schoolkids who can only multiply, but all at once. Neural nets are dominated by big matrix multiplies — many simple multiply-adds — so GPUs win.</p>
  <p>Write Python loops computing one number at a time and the GPU becomes <em>slower</em>, because dispatch overhead exceeds the work. Rule: <strong>express loops as tensor ops over the whole batch.</strong></p>
</div>

<h2>3. Timing correctly</h2>
<pre data-lang="python"><code># WRONG: async, you measure dispatch only
start = time.time()
y = model(x)
print(time.time() - start)

# RIGHT
torch.cuda.synchronize()
start = time.time()
y = model(x)
torch.cuda.synchronize()
print(time.time() - start)</code></pre>

<div class="callout basic">
  <span class="tag">Async execution</span>
  <p>PyTorch queues work on the GPU and returns immediately. Without <code>synchronize()</code> you time the queueing, which is why people wrongly conclude GPUs are not faster.</p>
</div>

<h2>4. Mixed precision (AMP)</h2>
<pre data-lang="python"><code>from torch.amp import autocast, GradScaler   # recommended entry point in 2.x
scaler = GradScaler("cuda")                  # explicit device

for xb, yb in loader:
    optimizer.zero_grad()
    with autocast("cuda", dtype=torch.float16):
        loss = criterion(model(xb.to(device)), yb.to(device))

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()</code></pre>

<div class="callout math">
  <span class="tag">Math note · why GradScaler</span>
  <p>float16's smallest normal positive is ~6e-8, smaller than many gradients, which flush to zero. GradScaler multiplies the loss by a large factor before backward so gradients stay representable, then divides before the update. On inf/nan it skips the step and reduces the factor.</p>
  <p>Gain: about half the memory and 1.5–2× speed. Cost: negligible precision loss in practice.</p>
</div>

<h2>5. Fixing OOM, in order</h2>
<ol class="steps">
  <li><strong>Lower batch_size</strong>; use gradient accumulation to keep the effective batch.</li>
  <li><strong>Enable AMP</strong> — roughly halves memory for free.</li>
  <li><strong>Free references</strong>: do not store loss tensors, use <code>no_grad</code> in eval, and if needed <code>del</code> + <code>empty_cache()</code>.</li>
  <li><strong>Gradient checkpointing</strong> — trade compute for memory.</li>
  <li><strong>Shrink the model or input resolution.</strong></li>
</ol>

<pre data-lang="python"><code>accum_steps = 4
for i, (xb, yb) in enumerate(loader):
    loss = criterion(model(xb.to(device)), yb.to(device)) / accum_steps
    loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()</code></pre>

<div class="callout math">
  <span class="tag">Math note · why divide</span>
  <p>Gradients accumulate, so summing four micro-batch gradients equals four times the gradient. Since a large batch uses the mean loss, divide each micro-batch by four first.</p>
</div>

<h2>6. Three free speedups</h2>
<pre data-lang="python"><code>model = model.to(memory_format=torch.channels_last)
loader = DataLoader(ds, batch_size=64, shuffle=True,
                    num_workers=8, pin_memory=True)
xb = xb.to(device, non_blocking=True)
model = torch.compile(model)     # PyTorch 2.0+, often +20-50%</code></pre>

<div class="callout warn">
  <span class="tag">num_workers is not "more is better"</span>
  <p>Beyond your core count, scheduling overhead dominates. Start at a quarter to half your cores and measure.</p>
</div>

<h2>7. Multi-GPU in one paragraph</h2>
<table>
  <tr><th>Approach</th><th>How</th><th>Advice</th></tr>
  <tr><td><code>DataParallel</code></td><td>single process, gathers on GPU 0</td><td>legacy, GPU 0 becomes a bottleneck</td></tr>
  <tr><td><code>DistributedDataParallel</code></td><td>multi-process, all-reduce gradients</td><td>standard, use with <code>torchrun</code></td></tr>
</table>
<p>Do not touch multi-GPU if one card works.</p>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li>Everything on the same device; new tensors default to CPU.</li>
    <li><code>synchronize()</code> before timing.</li>
    <li>AMP = float16 forward + GradScaler.</li>
    <li>OOM order: batch → AMP → free refs → checkpointing → smaller model.</li>
    <li>Divide loss by accumulation steps.</li>
    <li>Prefer DDP, but stay single-GPU if you can.</li>
  </ol>
</div>

<div class="quiz" data-answer="2">
  <p class="q">Timing without <code>cuda.synchronize()</code> gives:</p>
  <ol class="opts">
    <li>Too slow</li>
    <li>Accurate</li>
    <li>Too fast — you only measured dispatch</li>
    <li>An error</li>
  </ol>
  <p class="exp">CUDA is asynchronous; sync first.</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">GradScaler:</p>
  <ol class="opts">
    <li>Scales the loss up so float16 gradients do not underflow, then scales back</li>
    <li>Tunes the learning rate</li>
    <li>Compresses weights</li>
    <li>Speeds up the dataloader</li>
  </ol>
  <p class="exp">Standard fix for fp16 underflow.</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q">With 4 accumulation steps, each micro-batch loss should be:</p>
  <ol class="opts">
    <li>Multiplied by 4</li>
    <li>Divided by 4</li>
    <li>Unchanged</li>
    <li>Squared</li>
  </ol>
  <p class="exp">Gradients add, so divide to match a true mean over the large batch.</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Time a 1024×1024 matmul on CPU vs GPU with proper synchronisation.</li>
    <li>Add AMP to the Chapter 8 MNIST loop; record epoch time and peak memory.</li>
    <li>Force an OOM with batch_size=1024, then fix it following the list.</li>
    <li>Compare <code>torch.compile</code> first run (compilation) vs second run.</li>
  </ol>
</div>`
  }
});
