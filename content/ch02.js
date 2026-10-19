(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch02',
  num: '02',
  part: 'p1',
  file: 'tensors',
  zh: {
    title: '张量基础：创建、形状、索引',
    sub: '所有 PyTorch 代码都建立在「会摆弄张量」之上',
    html: `
<p class="lead">张量就是多维数组，但它比 NumPy 多了两件事：<strong>能放到 GPU 上算</strong>、<strong>能自动求导</strong>。这一章把创建、属性、索引、变形四件事练熟，后面的模型代码读起来才不费劲。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>用 6 个常用函数创建想要的张量，并说清 dtype 与 device</li>
    <li>看懂任意 <code>torch.Size([...])</code>，知道每个轴代表什么</li>
    <li>熟练使用切片、布尔索引取出想要的数据</li>
    <li>分清 <code>reshape</code> 与 <code>view</code>，不再被 contiguous 报错卡住</li>
  </ul>
</div>

<h2>1. 创建张量：六个函数覆盖 90% 场景</h2>
<pre data-lang="python"><code>import torch

# 从 Python 数据构造
a = torch.tensor([[1, 2, 3], [4, 5, 6]])      # dtype 自动推断为 int64
b = torch.tensor([1.0, 2.0], dtype=torch.float32)

# 全 0 / 全 1 / 指定值
z = torch.zeros(2, 3)                # 2 行 3 列
o = torch.ones(2, 3)
f = torch.full((2, 3), 7.0)          # 全部填 7.0

# 等差数列 / 随机
r = torch.arange(0, 10, 2)           # tensor([0, 2, 4, 6, 8])
n = torch.randn(2, 3)                # 标准正态 N(0,1)
u = torch.rand(2, 3)                 # [0,1) 均匀分布

# 单位矩阵、与已有张量同形状
eye = torch.eye(3)
same = torch.zeros_like(a)           # 形状和 dtype 都跟 a 一样

print(a.shape, a.dtype, a.device)
# torch.Size([2, 3]) torch.int64 cpu</code></pre>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p><strong>dtype</strong> 是「每个数字用什么格式存」。<code>torch.int64</code> 是整数，<code>torch.float32</code> 是单精度小数（4 字节），<code>torch.float64</code> 是双精度（8 字节）。神经网络几乎只用 float32：够准、省一半显存、GPU 上快得多。</p>
  <p><strong>device</strong> 是「数据存在哪」。<code>cpu</code> 是内存，<code>cuda:0</code> 是第一块显卡。同一个运算的两个张量必须在同一设备上，否则报错。</p>
</div>

<div class="callout warn">
  <span class="tag">坑：默认整数类型</span>
  <p><code>torch.tensor([1, 2, 3])</code> 因为全是整数，dtype 是 <code>int64</code>。如果你把它丢进 <code>nn.Linear</code>，会报 <code>expected scalar type Float but found Long</code>。养成习惯：<strong>浮点数据显式写 <code>dtype=torch.float32</code>，或者末尾加 .float()</strong>。</p>
</div>

<h2>2. 形状：读 Size 的方法</h2>
<p>看到一个形状，从<strong>最右边</strong>开始读，那通常是最「细」的维度。</p>
<table>
  <tr><th>形状</th><th>怎么读</th><th>典型来源</th></tr>
  <tr><td><code>(784,)</code></td><td>一个 784 维向量</td><td>一张 28×28 的 MNIST 图片拉平</td></tr>
  <tr><td><code>(32, 784)</code></td><td>32 个样本，每个 784 维</td><td>一个 batch 的输入</td></tr>
  <tr><td><code>(32, 10)</code></td><td>32 个样本，每个 10 个分数</td><td>10 分类模型的输出 logits</td></tr>
  <tr><td><code>(32, 3, 32, 32)</code></td><td>32 张 3 通道 32×32 图</td><td>CIFAR-10 的一个 batch</td></tr>
  <tr><td><code>(64, 10, 32, 32)</code></td><td>64 个卷积核，输入 10 通道，核 32×32</td><td>卷积层权重</td></tr>
</table>

<pre data-lang="python"><code>x = torch.randn(32, 3, 32, 32)
print(x.shape)        # torch.Size([32, 3, 32, 32])
print(x.dim())        # 4  —— 轴数
print(x.numel())      # 98304 —— 元素总数 32*3*32*32
print(x.size(0))      # 32 —— 第 0 轴长度，等价于 x.shape[0]</code></pre>

<h2>3. 索引与切片</h2>
<pre data-lang="python"><code>x = torch.arange(24).reshape(2, 3, 4)   # 形状 (2,3,4)

x[0]              # 第 0 个「3x4 矩阵」，形状 (3,4)
x[0, 1]           # 第 0 个矩阵的第 1 行，形状 (4,)
x[0, 1, 2]        # 一个标量张量 tensor(6)
x[:, 0]           # 所有矩阵的第 0 行，形状 (2,4)
x[..., -1]        # ... 表示「前面所有轴都取全部」，取最后一列，形状 (2,3)
x[0:2, 1:3]       # 切片，含头不含尾

# 布尔索引：选出满足条件的元素
y = torch.tensor([1.0, -2.0, 3.0, -4.0])
print(y[y &gt; 0])                 # tensor([1., 3.])
print(y.abs())                  # 绝对值

# 取值：单个元素的 tensor 用 .item() 变成 Python 数字
loss = torch.tensor(0.523)
print(loss.item())              # 0.523，类型是 float
print(float(loss))              # 同样效果</code></pre>

<div class="callout tip">
  <span class="tag">好习惯</span>
  <p>要把 loss 打印成好看的数字、存进列表、或者做 <code>if loss &lt; best</code> 判断时，一定先 <code>.item()</code>。直接拿 tensor 比较会得到一个 tensor，很容易写错逻辑，而且会一直挂着计算图不放，导致显存缓慢泄漏。</p>
</div>

<h2>4. 变形：reshape / view / squeeze</h2>
<pre data-lang="python"><code>x = torch.arange(12)              # (12,)

x.reshape(3, 4)                   # (3,4)  —— 推荐，最安全
x.view(3, 4)                      # (3,4)  —— 要求内存连续，否则报错
x.reshape(2, -1)                  # (2,6)  —— -1 表示「这一维你自己算」

# 增删长度为 1 的轴
v = torch.randn(3)
v.unsqueeze(0).shape              # (1,3)  加一个轴
v.unsqueeze(1).shape              # (3,1)
t = torch.randn(1, 3, 1)
t.squeeze().shape                 # (3,)   删掉所有长度为 1 的轴
t.squeeze(0).shape                # (3,1)  只删指定轴

# 转置与换轴
m = torch.randn(2, 3)
m.t().shape                       # (3,2)  只能用于 2 维
m.transpose(0, 1).shape           # (3,2)  交换任意两个轴
img = torch.randn(3, 32, 32)
img.permute(1, 2, 0).shape        # (32,32,3)  重排所有轴（画图时常用）

# 展平
torch.randn(8, 3, 4).flatten(1).shape   # (8,12)  从第 1 轴开始压平</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 为什么 view 会报错</span>
  <p>张量在内存里是一维排布的。<code>(2,3)</code> 和 <code>(3,2)</code> 的元素顺序相同，只是「怎么切行」不同，所以 reshape 不需要搬数据，改个描述就行（这正是 view 快的原因）。</p>
  <p>但 <code>transpose</code> 之后，逻辑顺序和内存顺序不一致了（这叫 <strong>non-contiguous</strong>），此时 <code>view</code> 无法只靠改描述完成，就会报 <code>view size is not compatible with input tensor's size and stride</code>。解法二选一：</p>
  <p class="formula">x = x.contiguous().view(...)   # 先搬成连续内存
x = x.reshape(...)             # 或者直接用 reshape，它会自动处理</p>
  <p>结论：<strong>日常用 reshape，只有在明确要零拷贝时才用 view。</strong></p>
</div>

<h2>5. 和 NumPy 互转</h2>
<pre data-lang="python"><code>import numpy as np

arr = np.array([[1.0, 2.0], [3.0, 4.0]])
t = torch.from_numpy(arr)         # 共享内存！改一个另一个也变
t2 = torch.tensor(arr)            # 拷贝一份，互不影响

back = t.numpy()                  # tensor -> numpy（CPU 张量才行）

# GPU 上的张量要先搬回 CPU
# t.cuda().numpy()  -> 报错
# t.cuda().cpu().numpy()  -> 正确</code></pre>

<div class="callout warn">
  <span class="tag">坑：from_numpy 是共享内存</span>
  <p>用 <code>from_numpy</code> 得到的是同一块内存的「视图」。你改了张量，原来的 numpy 数组也变了。如果只想读数据无所谓，但如果后面要改，请用 <code>torch.tensor(arr)</code> 显式拷贝。</p>
</div>

<h2>6. 随机种子：让实验可复现</h2>
<pre data-lang="python"><code>torch.manual_seed(42)       # 固定 CPU 随机数
torch.randn(3)              # 每次运行结果都一样
# 有 GPU 时再加一句：
# torch.cuda.manual_seed_all(42)</code></pre>
<p>调试阶段一定要设种子，否则「这次效果好」可能只是随机初始化运气好。</p>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li>创建用 <code>tensor / zeros / ones / full / arange / randn</code>，够用了。</li>
    <li><code>dtype</code> 决定精度与显存，网络里默认 float32；整数张量不能进 Linear。</li>
    <li><code>dim()</code> 是轴数，<code>shape</code> 是每轴长度，<code>numel()</code> 是元素总数。</li>
    <li><code>...</code> 表示「其余轴全取」，切片含头不含尾。</li>
    <li>日常变形用 <code>reshape</code>；<code>view</code> 要求内存连续；<code>permute</code> 用于换轴。</li>
    <li>标量取 Python 数字用 <code>.item()</code>。</li>
  </ol>
</div>

<div class="quiz" data-answer="2">
  <p class="q"><code>torch.arange(24).reshape(2, 3, 4)</code> 的 <code>numel()</code> 和 <code>dim()</code> 分别是？</p>
  <ol class="opts">
    <li>24 和 4</li>
    <li>3 和 24</li>
    <li>24 和 3 —— 元素 24 个，轴 3 个</li>
    <li>6 和 4</li>
  </ol>
  <p class="exp">numel = 2×3×4 = 24 个元素；dim = 3 个轴。这两个数字经常被初学者搞反。</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q">执行 <code>x = torch.randn(4, 6)</code> 后，<code>x.t().view(4, 6)</code> 最可能发生什么？</p>
  <ol class="opts">
    <li>正常返回 (4,6)</li>
    <li>报 stride 相关错误，因为转置后内存不连续</li>
    <li>返回 (6,4)</li>
    <li>自动改成 (4,6) 并拷贝数据</li>
  </ol>
  <p class="exp">x.t() 是 (6,4) 且不连续，view(4,6) 无法只改描述完成，于是报错。改用 reshape 或先 .contiguous()。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">要把一个 batch 的图片 <code>(32, 3, 32, 32)</code> 喂给只能接收向量的全连接层，应该？</p>
  <ol class="opts">
    <li><code>x.reshape(32, -1)</code> 得到 (32, 3072)</li>
    <li><code>x.reshape(-1, 32)</code> 得到 (3072, 32)</li>
    <li><code>x.squeeze()</code></li>
    <li><code>x.t()</code></li>
  </ol>
  <p class="exp">要保留第 0 轴（样本数 32），把后面三维压成一维：3×32×32 = 3072。所以是 <code>reshape(32, -1)</code>，或者 <code>flatten(1)</code>。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>创建形状 <code>(3, 4)</code> 的随机张量，取出第 2 行、第 3 列的元素，并转成 Python 浮点数。</li>
    <li>把 <code>(2, 3, 4)</code> 变成 <code>(2, 12)</code>、<code>(6, 4)</code>、<code>(24,)</code>，三种写法各试一次。</li>
    <li>用 <code>torch.manual_seed(0)</code> 生成两次 <code>randn(2,2)</code>，验证结果相同；去掉种子再试。</li>
    <li>制造一次 view 报错，再用 reshape 修好，把两种报错/输出都记在笔记里。</li>
  </ol>
</div>`
  },
  en: {
    title: 'Tensor basics: create, shape, index',
    sub: 'Every PyTorch program is ultimately tensor juggling',
    html: `
<p class="lead">A tensor is a multi-dimensional array, plus two superpowers over NumPy: it <strong>runs on GPU</strong> and it <strong>tracks gradients</strong>. This chapter drills the four essential skills: creating, inspecting, indexing, reshaping.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Create any tensor with six common functions, and reason about dtype and device</li>
    <li>Read any <code>torch.Size([...])</code> and say what each axis means</li>
    <li>Slice, index and boolean-mask to get exactly the data you want</li>
    <li>Tell <code>reshape</code> from <code>view</code> and stop fighting contiguous errors</li>
  </ul>
</div>

<h2>1. Creating tensors: six functions cover 90%</h2>
<pre data-lang="python"><code>import torch

# from Python data
a = torch.tensor([[1, 2, 3], [4, 5, 6]])      # dtype inferred as int64
b = torch.tensor([1.0, 2.0], dtype=torch.float32)

# zeros / ones / constant
z = torch.zeros(2, 3)
o = torch.ones(2, 3)
f = torch.full((2, 3), 7.0)

# range / random
r = torch.arange(0, 10, 2)           # tensor([0, 2, 4, 6, 8])
n = torch.randn(2, 3)                # standard normal
u = torch.rand(2, 3)                 # uniform [0,1)

eye = torch.eye(3)
same = torch.zeros_like(a)           # same shape and dtype as a

print(a.shape, a.dtype, a.device)
# torch.Size([2, 3]) torch.int64 cpu</code></pre>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p><strong>dtype</strong> is how each number is stored: <code>int64</code> integers, <code>float32</code> single precision (4 bytes), <code>float64</code> double (8 bytes). Neural nets use float32 almost exclusively: accurate enough, half the memory, much faster on GPU.</p>
  <p><strong>device</strong> is where it lives: <code>cpu</code> or <code>cuda:0</code>. Both operands of an op must be on the same device.</p>
</div>

<div class="callout warn">
  <span class="tag">Trap: integer default</span>
  <p><code>torch.tensor([1, 2, 3])</code> is <code>int64</code>. Feed it to <code>nn.Linear</code> and you get <code>expected scalar type Float but found Long</code>. Habit: write <code>dtype=torch.float32</code> explicitly, or append <code>.float()</code>.</p>
</div>

<h2>2. Reading a shape</h2>
<p>Read a shape from the <strong>right</strong>: the rightmost axis is usually the finest-grained.</p>
<table>
  <tr><th>Shape</th><th>Read it as</th><th>Where it comes from</th></tr>
  <tr><td><code>(784,)</code></td><td>one 784-dim vector</td><td>a flattened 28×28 MNIST image</td></tr>
  <tr><td><code>(32, 784)</code></td><td>32 samples of 784 features</td><td>one input batch</td></tr>
  <tr><td><code>(32, 10)</code></td><td>32 samples, 10 scores each</td><td>logits of a 10-class model</td></tr>
  <tr><td><code>(32, 3, 32, 32)</code></td><td>32 RGB 32×32 images</td><td>a CIFAR-10 batch</td></tr>
  <tr><td><code>(64, 10, 32, 32)</code></td><td>64 filters, 10 input channels, 32×32 kernel</td><td>conv layer weights</td></tr>
</table>

<pre data-lang="python"><code>x = torch.randn(32, 3, 32, 32)
print(x.shape)        # torch.Size([32, 3, 32, 32])
print(x.dim())        # 4  -> number of axes
print(x.numel())      # 98304 -> total elements
print(x.size(0))      # 32 -> length of axis 0</code></pre>

<h2>3. Indexing and slicing</h2>
<pre data-lang="python"><code>x = torch.arange(24).reshape(2, 3, 4)   # shape (2,3,4)

x[0]              # first 3x4 matrix, shape (3,4)
x[0, 1]           # row 1 of that matrix, shape (4,)
x[0, 1, 2]        # scalar tensor(6)
x[:, 0]           # row 0 of every matrix, shape (2,4)
x[..., -1]        # ... means "all leading axes", last column, shape (2,3)
x[0:2, 1:3]       # slices, end exclusive

y = torch.tensor([1.0, -2.0, 3.0, -4.0])
print(y[y &gt; 0])                 # tensor([1., 3.])

loss = torch.tensor(0.523)
print(loss.item())              # 0.523 as a Python float</code></pre>

<div class="callout tip">
  <span class="tag">Good habit</span>
  <p>Call <code>.item()</code> before printing a scalar, storing it in a list, or comparing with <code>if loss &lt; best</code>. Comparing tensors returns tensors, which breaks your logic and keeps the graph alive (a slow memory leak).</p>
</div>

<h2>4. Reshaping</h2>
<pre data-lang="python"><code>x = torch.arange(12)              # (12,)

x.reshape(3, 4)                   # recommended, always safe
x.view(3, 4)                      # requires contiguous memory
x.reshape(2, -1)                  # -1 means "you figure this axis out"

v = torch.randn(3)
v.unsqueeze(0).shape              # (1,3)  add an axis
v.unsqueeze(1).shape              # (3,1)
t = torch.randn(1, 3, 1)
t.squeeze().shape                 # (3,)   drop all size-1 axes
t.squeeze(0).shape                # (3,1)  drop a specific one

m = torch.randn(2, 3)
m.t().shape                       # (3,2)  2-D only
m.transpose(0, 1).shape           # (3,2)  swap any two axes
img = torch.randn(3, 32, 32)
img.permute(1, 2, 0).shape        # (32,32,3)  reorder axes (for plotting)

torch.randn(8, 3, 4).flatten(1).shape   # (8,12)  flatten from axis 1</code></pre>

<div class="callout math">
  <span class="tag">Math note · why view fails</span>
  <p>Data lives in memory as a flat sequence. <code>(2,3)</code> and <code>(3,2)</code> hold the same elements in the same order, only the row split differs, so reshape needs no copy — that is why view is fast.</p>
  <p>After <code>transpose</code>, logical order no longer matches memory order (call it <strong>non-contiguous</strong>), so <code>view</code> cannot express the change and raises a stride error.</p>
  <p class="formula">x = x.contiguous().view(...)   # copy into contiguous order first
x = x.reshape(...)             # or just let reshape handle it</p>
  <p>Rule of thumb: use <code>reshape</code> daily; use <code>view</code> only when you specifically want zero-copy.</p>
</div>

<h2>5. NumPy interop</h2>
<pre data-lang="python"><code>import numpy as np

arr = np.array([[1.0, 2.0], [3.0, 4.0]])
t = torch.from_numpy(arr)         # SHARED memory
t2 = torch.tensor(arr)            # a copy

back = t.numpy()                  # CPU tensors only
# t.cuda().numpy()                # error
# t.cuda().cpu().numpy()          # correct</code></pre>

<div class="callout warn">
  <span class="tag">Trap: shared memory</span>
  <p><code>from_numpy</code> returns a view onto the same buffer. Mutating the tensor mutates the array. Use <code>torch.tensor(arr)</code> if you need independence.</p>
</div>

<h2>6. Seeds for reproducibility</h2>
<pre data-lang="python"><code>torch.manual_seed(42)
torch.randn(3)              # same numbers every run
# with a GPU also:
# torch.cuda.manual_seed_all(42)</code></pre>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li>Create with <code>tensor / zeros / ones / full / arange / randn</code>.</li>
    <li>dtype defaults matter; integer tensors cannot enter <code>nn.Linear</code>.</li>
    <li><code>dim()</code> = axes, <code>shape</code> = lengths, <code>numel()</code> = element count.</li>
    <li><code>...</code> takes all remaining axes; slices exclude the end index.</li>
    <li>Prefer <code>reshape</code>; <code>view</code> needs contiguity; <code>permute</code> reorders axes.</li>
    <li>Use <code>.item()</code> to turn a scalar tensor into a Python number.</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q">For <code>torch.arange(24).reshape(2, 3, 4)</code>, what are <code>numel()</code> and <code>dim()</code>?</p>
  <ol class="opts">
    <li>3 and 24</li>
    <li>24 and 3</li>
    <li>24 and 4</li>
    <li>6 and 4</li>
  </ol>
  <p class="exp">numel = 2×3×4 = 24 elements; dim = 3 axes. Beginners often swap these two.</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q">After <code>x = torch.randn(4, 6)</code>, what does <code>x.t().view(4, 6)</code> most likely do?</p>
  <ol class="opts">
    <li>Returns a (4,6) tensor</li>
    <li>Raises a stride error because the transpose is non-contiguous</li>
    <li>Returns (6,4)</li>
    <li>Silently copies into (4,6)</li>
  </ol>
  <p class="exp">x.t() is (6,4) and non-contiguous, so view cannot fulfil the request. Use reshape or .contiguous().view().</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">To feed images <code>(32, 3, 32, 32)</code> into a linear layer:</p>
  <ol class="opts">
    <li><code>x.reshape(32, -1)</code> → (32, 3072)</li>
    <li><code>x.reshape(-1, 32)</code> → (3072, 32)</li>
    <li><code>x.squeeze()</code></li>
    <li><code>x.t()</code></li>
  </ol>
  <p class="exp">Keep axis 0 (the batch) and flatten the rest: 3×32×32 = 3072. So <code>reshape(32, -1)</code> or <code>flatten(1)</code>.</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Create a (3,4) random tensor, fetch the element at row 2 / column 3 as a Python float.</li>
    <li>Turn (2,3,4) into (2,12), (6,4) and (24,) three different ways.</li>
    <li>With <code>manual_seed(0)</code>, generate <code>randn(2,2)</code> twice and compare; then remove the seed.</li>
    <li>Provoke a view error, fix it with reshape, and note both messages.</li>
  </ol>
</div>`
  }
});
