(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch03',
  num: '03',
  part: 'p1',
  file: 'ops-broadcasting',
  zh: {
    title: '运算、广播与归约',
    sub: '90% 的形状报错，都死在这一章',
    html: `
<p class="lead">会创建张量只是第一步。真正的日常是：加起来、乘起来、按维度求和、把几个张量拼起来。其中<strong>广播（broadcasting）</strong>是最容易出错、也最值得花时间弄懂的一节——它能让代码短很多，也能让 bug 藏得很深。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>分清 <code>*</code>、<code>@</code>、<code>torch.matmul</code> 的区别</li>
    <li>用广播规则预测任何两个张量运算后的形状（不运行代码）</li>
    <li>正确使用 <code>dim</code> 与 <code>keepdim</code> 做归约</li>
    <li>用 <code>cat / stack</code> 拼张量，不再搞混两者</li>
  </ul>
</div>

<h2>1. 逐元素运算</h2>
<pre data-lang="python"><code>import torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[10., 20.], [30., 40.]])

print(a + b)          # 对应位置相加
print(a * b)          # 对应位置相乘（不是矩阵乘！）
print(a / 2)          # 标量广播
print(a ** 2)
print(torch.exp(a))   # e^x，逐元素
print(torch.sqrt(a))

# 比较运算返回布尔张量
print(a &gt; 2)          # tensor([[False, False], [ True,  True]])
print(torch.clamp(a, max=2.5))    # 把超过 2.5 的截断
print(torch.where(a &gt; 2, a, torch.zeros_like(a)))  # 条件选择</code></pre>

<div class="callout warn">
  <span class="tag">第一大坑：* 不是矩阵乘</span>
  <p><code>a * b</code> 是逐元素相乘，要求两形状能广播；<code>a @ b</code> 才是矩阵乘法。在 NumPy 里 <code>*</code> 也是逐元素，所以从 NumPy 过来的同学容易顺手写错，而且形状恰好能广播时它<strong>不报错</strong>，结果却完全不对——这是最危险的一类 bug。</p>
</div>

<h2>2. 矩阵乘法的三种写法</h2>
<pre data-lang="python"><code>x = torch.randn(32, 784)     # 一个 batch
W = torch.randn(784, 10)     # 权重

y1 = x @ W                   # 推荐：最简洁
y2 = torch.matmul(x, W)      # 同义，函数式
y3 = torch.mm(x, W)          # 只支持 2 维

print(y1.shape)              # (32, 10)

# 批量矩阵乘：前几维是 batch，最后两维做矩阵乘
A = torch.randn(8, 4, 5)
B = torch.randn(8, 5, 6)
print(torch.bmm(A, B).shape)      # (8, 4, 6)
print(torch.matmul(A, B).shape)   # 同上，matmul 更通用

# 高维等价于「把最后两维当矩阵」
C = torch.randn(2, 3, 4, 5)
D = torch.randn(2, 3, 5, 6)
print(torch.matmul(C, D).shape)   # (2, 3, 4, 6)</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 什么时候用 @</span>
  <p>凡是「<strong>用一组权重对一批样本做线性组合</strong>」的地方，都是矩阵乘：全连接层、注意力里的 QKᵀ、把 one-hot 变成 embedding 等等。</p>
  <p>记法：<code>(N, in) @ (in, out) → (N, out)</code>。这里的 N 是「多少个样本」，out 是「每个样本输出多少个数」。中间维度 <code>in</code> 是「每个样本有多少特征」，它必须和权重的行数一致。</p>
</div>

<h2>3. 广播：一条三维右对齐规则</h2>
<p>当你对两个形状不同的张量做逐元素运算时，PyTorch 会尝试把它们「撑」成一样大。规则只有两条：</p>
<ol class="steps">
  <li>从<strong>最右边</strong>的轴开始，逐轴比较。</li>
  <li>每一对轴，要么<strong>相等</strong>，要么其中一个是 <strong>1</strong>（1 会被复制扩展），否则报错。</li>
</ol>

<div class="widget" data-widget="shape"></div>

<pre data-lang="python"><code># 例子 1：标量自动广播
x = torch.randn(3, 4)
x + 1.0                    # (3,4)，1.0 被当成 (3,4) 全 1

# 例子 2：(3,1) 和 (1,4)
a = torch.randn(3, 1)
b = torch.randn(1, 4)
print((a + b).shape)       # (3, 4)  —— 两个 1 都被扩展

# 例子 3：给每列乘一个系数（很常用）
w = torch.randn(4)         # (4,) 会被当成 (1,4)
print((x * w).shape)       # (3,4)

# 例子 4：给每行乘一个系数
w2 = torch.randn(3, 1)
print((x * w2).shape)      # (3,4)

# 例子 5：不能广播
# torch.randn(3, 4) + torch.randn(3, 3)   -> 报错：4 != 3
</code></pre>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p>把广播想成「<strong>复印</strong>」：形状是 1 的那一维太小了，只有一行/一列，PyTorch 就把它复印成需要的行数/列数。复印不占额外内存，是原地展开的视图，所以又快又省。</p>
  <p>这也解释了为什么 <code>(3,1) + (1,4)</code> 会得到 <code>(3,4)</code>：左边复印成 4 列，右边复印成 3 行，正好铺满。</p>
</div>

<div class="callout math">
  <span class="tag">数学批注 · 广播的应用</span>
  <p>机器学习里大量公式都是「逐样本减均值、除标准差」，写成代码就是广播：</p>
  <p class="formula">x_norm = (x - x.mean(dim=0)) / x.std(dim=0)</p>
  <p>这里 <code>x</code> 是 <code>(N, D)</code>，<code>mean(dim=0)</code> 得到 <code>(D,)</code>，广播后自动对每个样本都减同一个均值向量。<strong>理解广播 = 理解为什么这些代码不用写循环。</strong></p>
</div>

<h2>4. 归约：sum / mean / max 与 dim</h2>
<pre data-lang="python"><code>x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])      # (2,3)

x.sum()                  # tensor(21.)  全部相加
x.sum(dim=0)             # tensor([5., 7., 9.])   按列求和 -> 形状 (3,)
x.sum(dim=1)             # tensor([ 6., 15.])     按行求和 -> 形状 (2,)

x.mean(dim=1, keepdim=True)   # 形状 (2,1)，保持轴数
x.max(dim=1)             # 返回 namedtuple(values, indices)
print(x.max(dim=1).values)    # tensor([3., 6.])
print(x.max(dim=1).indices)   # tensor([2, 2])   —— 这就是 argmax
x.argmax(dim=1)          # tensor([2, 2])</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · dim 到底指哪一维</span>
  <p>初学者最容易记反。<strong><code>dim=k</code> 表示「把第 k 个轴消掉」</strong>。形状 <code>(2,3)</code> 沿 <code>dim=1</code> 归约，第 1 轴（长度 3）被压掉，剩下 <code>(2,)</code>，也就是「每一行变成一个数」。</p>
  <p>加 <code>keepdim=True</code> 则保留这个轴但长度变成 1，得到 <code>(2,1)</code>——这在后面做广播时非常关键，否则 <code>(2,)</code> 和 <code>(2,3)</code> 相减会变成右对齐比较，结果完全不是你想要的。</p>
</div>

<div class="callout warn">
  <span class="tag">高频错误：(N,) 和 (N,1) 不是一回事</span>
  <p>假设你有 <code>pred (32,)</code> 和 <code>target (32,)</code>，相减得到 <code>(32,)</code>，没问题。但如果 <code>pred</code> 是 <code>(32,1)</code>，那么 <code>pred - target</code> 会广播成 <code>(32,32)</code>，loss 直接算错，而且不报错。遇到诡异的大 loss，第一件事就是打印所有形状。</p>
</div>

<h2>5. 拼接与拆分</h2>
<pre data-lang="python"><code>a = torch.tensor([[1., 2.], [3., 4.]])   # (2,2)
b = torch.tensor([[5., 6.]])             # (1,2)

torch.cat([a, b], dim=0)      # (3,2)  沿已有轴拼接，除 dim 外形状必须一致
torch.cat([a, a], dim=1)      # (2,4)  沿列拼接

c = torch.tensor([[7., 8.]])
torch.stack([a, a])           # (2,2,2)  新建一个轴
torch.stack([c, c], dim=1)    # (1,2,2)

x = torch.randn(8, 4)
x.chunk(2, dim=0)             # 两个 (4,4)
torch.split(x, 3, dim=0)      # (3,4) (3,4) (2,4)</code></pre>

<div class="callout tip">
  <span class="tag">怎么选 cat 还是 stack</span>
  <p>问自己：「结果应该比原来多一个维度吗？」<br>把 100 张图片堆成一批——每张 <code>(3,32,32)</code>，目标 <code>(100,3,32,32)</code>，多了一维 → <strong>stack</strong>。<br>把两段长文本接起来——<code>(128,768)</code> 和 <code>(64,768)</code> 变 <code>(192,768)</code>，轴数不变 → <strong>cat</strong>。</p>
</div>

<h2>6. 原地操作（in-place）与它的风险</h2>
<pre data-lang="python"><code>x = torch.randn(3)
x.add_(1)          # 带下划线 = 原地修改，等价于 x = x + 1 但不新建内存
x.relu_()
x.clamp_(min=0)

# 风险：会破坏自动求导需要的原始值
# w = torch.randn(3, requires_grad=True)
# w.add_(1)      -> 可能报 "a leaf Variable that requires grad is being used in an in-place operation"</code></pre>
<p>结论：<strong>默认写非原地版本</strong>（<code>x = x + 1</code>），只在明确知道自己在省内存时用下划线版本。</p>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li><code>*</code> 逐元素，<code>@</code> 矩阵乘，两者绝不通用。</li>
    <li>矩阵乘形状：<code>(N,in)@(in,out)→(N,out)</code>，中间维度消失。</li>
    <li>广播：右对齐，逐轴比；相等或其中一个为 1，否则报错。</li>
    <li><code>dim=k</code> = 消掉第 k 轴；需要保留时加 <code>keepdim=True</code>。</li>
    <li><code>cat</code> 沿已有轴拼，<code>stack</code> 新建一个轴。</li>
    <li>原地操作带下划线，可能破坏 autograd，默认别用。</li>
  </ol>
</div>

<div class="quiz" data-answer="2">
  <p class="q">形状 <code>(32, 1, 8)</code> 和 <code>(4, 8)</code> 相加，结果形状是？</p>
  <ol class="opts">
    <li>报错</li>
    <li>(32, 1, 8)</li>
    <li>(32, 4, 8)</li>
    <li>(32, 4)</li>
  </ol>
  <p class="exp">右对齐：(8,8) 相等；(1,4) 中 1 被扩展成 4；左边缺的一轴按 1 处理再扩展成 32。得 (32,4,8)。用页面上的计算器验证一下。</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q"><code>x</code> 形状 <code>(64, 10)</code>，<code>x.mean(dim=1)</code> 的结果形状是？</p>
  <ol class="opts">
    <li>(64, 10)</li>
    <li>(64,)</li>
    <li>(10,)</li>
    <li>(64, 1)</li>
  </ol>
  <p class="exp">dim=1 把长度 10 的那一轴消掉，剩 (64,)。若要 (64,1) 需加 keepdim=True——这一步在广播时经常决定对错。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">已有 <code>a (2,3)</code>，想得到 <code>(2,3,5)</code> 并把 a 复制 5 份，应该？</p>
  <ol class="opts">
    <li><code>a.unsqueeze(-1).expand(2,3,5)</code></li>
    <li><code>torch.cat([a]*5)</code></li>
    <li><code>a.reshape(2,3,5)</code></li>
    <li><code>a.stack(5)</code></li>
  </ol>
  <p class="exp">先在新轴上加一维 (2,3,1)，再用 expand 沿该轴扩展成 5（不复制内存）。cat 只能拼接现有轴，reshape 不改变元素数量（6≠30）。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>用广播把 <code>(32,3,32,32)</code> 的每张图减去各自的均值（提示：先对 dim=(2,3) 求均值并 keepdim）。</li>
    <li>构造 <code>(5,4)</code> 和 <code>(4,)</code>，验证相乘是「每列乘一个系数」，再用 <code>(5,1)</code> 实现「每行乘一个系数」。</li>
    <li>用 <code>chunk</code> 把一个 batch 切成 4 份，再用 <code>cat</code> 拼回去，检查是否和原张量完全相等。</li>
    <li>在页面的广播计算器里试 3 组「能广播」和 3 组「不能广播」的形状，把规则说给自己听。</li>
  </ol>
</div>`
  },
  en: {
    title: 'Operations, broadcasting and reductions',
    sub: 'Where 90% of shape errors are born',
    html: `
<p class="lead">Creating tensors is step one. Daily work is adding, multiplying, reducing along axes and concatenating. <strong>Broadcasting</strong> is the concept most worth mastering: it shortens code a lot and hides bugs very well.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Tell <code>*</code>, <code>@</code> and <code>torch.matmul</code> apart</li>
    <li>Predict the output shape of any element-wise op without running code</li>
    <li>Use <code>dim</code> and <code>keepdim</code> correctly</li>
    <li>Choose between <code>cat</code> and <code>stack</code> without hesitation</li>
  </ul>
</div>

<h2>1. Element-wise operations</h2>
<pre data-lang="python"><code>import torch

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[10., 20.], [30., 40.]])

print(a + b)          # element-wise add
print(a * b)          # element-wise multiply (NOT matmul)
print(a / 2)          # scalar broadcast
print(torch.exp(a))
print(torch.sqrt(a))

print(a &gt; 2)          # boolean tensor
print(torch.clamp(a, max=2.5))
print(torch.where(a &gt; 2, a, torch.zeros_like(a)))</code></pre>

<div class="callout warn">
  <span class="tag">Trap #1: * is not matmul</span>
  <p><code>a * b</code> multiplies element-wise; <code>a @ b</code> is matrix multiplication. When shapes happen to be broadcastable, the wrong one <strong>does not error</strong> — it silently gives a wrong answer. That is the most dangerous bug class in this chapter.</p>
</div>

<h2>2. Three ways to multiply matrices</h2>
<pre data-lang="python"><code>x = torch.randn(32, 784)
W = torch.randn(784, 10)

y1 = x @ W                   # preferred
y2 = torch.matmul(x, W)      # same thing, functional
y3 = torch.mm(x, W)          # 2-D only

A = torch.randn(8, 4, 5)
B = torch.randn(8, 5, 6)
print(torch.bmm(A, B).shape)      # (8, 4, 6)
print(torch.matmul(A, B).shape)   # same; matmul is general

C = torch.randn(2, 3, 4, 5)
D = torch.randn(2, 3, 5, 6)
print(torch.matmul(C, D).shape)   # (2, 3, 4, 6)</code></pre>

<div class="callout math">
  <span class="tag">Math note · when to use @</span>
  <p>Whenever you "apply a weight matrix to a batch of samples": fully connected layers, attention's QKᵀ, one-hot to embedding, all of it.</p>
  <p>Mnemonic: <code>(N, in) @ (in, out) → (N, out)</code>. N is how many samples, in is how many features per sample, out is how many outputs per sample.</p>
</div>

<h2>3. Broadcasting: one right-aligned rule</h2>
<p>When two differently-shaped tensors meet in an element-wise op, PyTorch tries to stretch them. Two rules:</p>
<ol class="steps">
  <li>Compare axes starting from the <strong>rightmost</strong>.</li>
  <li>Each pair must be <strong>equal</strong>, or one of them must be <strong>1</strong> (stretched), otherwise it errors.</li>
</ol>

<div class="widget" data-widget="shape"></div>

<pre data-lang="python"><code>x = torch.randn(3, 4)
x + 1.0                    # scalar broadcast to (3,4)

a = torch.randn(3, 1)
b = torch.randn(1, 4)
print((a + b).shape)       # (3, 4)

w = torch.randn(4)
print((x * w).shape)       # (3,4) — one coefficient per column

w2 = torch.randn(3, 1)
print((x * w2).shape)      # (3,4) — one coefficient per row

# torch.randn(3, 4) + torch.randn(3, 3)   -> error: 4 != 3</code></pre>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p>Think of broadcasting as <strong>photocopying</strong>. A size-1 axis is too small — one row or one column — so PyTorch copies it to the needed length. No memory is allocated; it is a view, which is why it is fast.</p>
  <p>That is also why <code>(3,1) + (1,4)</code> is <code>(3,4)</code>: the left copies into 4 columns, the right copies into 3 rows.</p>
</div>

<div class="callout math">
  <span class="tag">Math note · why it matters</span>
  <p>Standardisation is written with broadcasting:</p>
  <p class="formula">x_norm = (x - x.mean(dim=0)) / x.std(dim=0)</p>
  <p><code>x</code> is <code>(N,D)</code>, the mean is <code>(D,)</code>, and broadcasting applies it to every row. Understanding broadcasting is why these lines have no loops.</p>
</div>

<h2>4. Reductions: sum / mean / max with dim</h2>
<pre data-lang="python"><code>x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])      # (2,3)

x.sum()                  # tensor(21.)
x.sum(dim=0)             # tensor([5., 7., 9.])  -> (3,)
x.sum(dim=1)             # tensor([ 6., 15.])    -> (2,)

x.mean(dim=1, keepdim=True)   # (2,1)
x.max(dim=1)             # namedtuple(values, indices)
print(x.argmax(dim=1))   # tensor([2, 2])</code></pre>

<div class="callout math">
  <span class="tag">Math note · what dim means</span>
  <p><strong><code>dim=k</code> means "remove axis k".</strong> For <code>(2,3)</code>, reducing over <code>dim=1</code> collapses the length-3 axis, leaving <code>(2,)</code> — one number per row.</p>
  <p><code>keepdim=True</code> keeps the axis with length 1, giving <code>(2,1)</code>. This matters enormously for broadcasting: <code>(2,)</code> minus <code>(2,3)</code> aligns from the right and produces nonsense, while <code>(2,1)</code> behaves as you intended.</p>
</div>

<div class="callout warn">
  <span class="tag">Frequent bug: (N,) vs (N,1)</span>
  <p>If <code>pred</code> is <code>(32,1)</code> and <code>target</code> is <code>(32,)</code>, then <code>pred - target</code> broadcasts to <code>(32,32)</code>. No error, wrong loss, invisible bug. When a loss looks absurd, print every shape first.</p>
</div>

<h2>5. Concatenate and split</h2>
<pre data-lang="python"><code>a = torch.tensor([[1., 2.], [3., 4.]])   # (2,2)
b = torch.tensor([[5., 6.]])             # (1,2)

torch.cat([a, b], dim=0)      # (3,2)
torch.cat([a, a], dim=1)      # (2,4)

torch.stack([a, a])           # (2,2,2) — new axis

x = torch.randn(8, 4)
x.chunk(2, dim=0)             # two (4,4)
torch.split(x, 3, dim=0)      # (3,4) (3,4) (2,4)</code></pre>

<div class="callout tip">
  <span class="tag">cat or stack?</span>
  <p>Ask: should the result have one more dimension? Stacking 100 images of <code>(3,32,32)</code> into <code>(100,3,32,32)</code> adds an axis → <strong>stack</strong>. Joining <code>(128,768)</code> and <code>(64,768)</code> into <code>(192,768)</code> keeps the rank → <strong>cat</strong>.</p>
</div>

<h2>6. In-place ops and their risk</h2>
<pre data-lang="python"><code>x = torch.randn(3)
x.add_(1)          # trailing underscore = in-place
x.relu_()
x.clamp_(min=0)

# danger:
# w = torch.randn(3, requires_grad=True)
# w.add_(1)   -> "a leaf Variable that requires grad is being used in an in-place operation"</code></pre>
<p>Default to the non-in-place form; use the underscore version only when you deliberately want to save memory.</p>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li><code>*</code> is element-wise, <code>@</code> is matmul. Never interchangeable.</li>
    <li>Matmul shape: <code>(N,in)@(in,out)→(N,out)</code>.</li>
    <li>Broadcast: right-aligned; equal or one is 1; otherwise error.</li>
    <li><code>dim=k</code> removes axis k; add <code>keepdim=True</code> to keep it as 1.</li>
    <li><code>cat</code> joins along an existing axis, <code>stack</code> creates a new one.</li>
    <li>In-place ops can break autograd — avoid by default.</li>
  </ol>
</div>

<div class="quiz" data-answer="2">
  <p class="q">Adding a <code>(32, 1, 8)</code> tensor to a <code>(4, 8)</code> tensor gives:</p>
  <ol class="opts">
    <li>An error</li>
    <li>(32, 1, 8)</li>
    <li>(32, 4, 8)</li>
    <li>(32, 4)</li>
  </ol>
  <p class="exp">Right-aligned: 8 vs 8 equal; 1 vs 4 stretches to 4; the missing axis counts as 1 and stretches to 32. Result (32,4,8). Try it in the widget above.</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q">For <code>x</code> of shape <code>(64, 10)</code>, <code>x.mean(dim=1)</code> has shape:</p>
  <ol class="opts">
    <li>(64, 10)</li>
    <li>(64,)</li>
    <li>(10,)</li>
    <li>(64, 1)</li>
  </ol>
  <p class="exp">dim=1 removes the length-10 axis, leaving (64,). Add keepdim=True for (64,1).</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">Given <code>a</code> of shape <code>(2,3)</code>, produce <code>(2,3,5)</code> by repeating it 5 times:</p>
  <ol class="opts">
    <li><code>a.unsqueeze(-1).expand(2,3,5)</code></li>
    <li><code>torch.cat([a]*5)</code></li>
    <li><code>a.reshape(2,3,5)</code></li>
    <li><code>a.stack(5)</code></li>
  </ol>
  <p class="exp">Add a new axis (2,3,1), then expand it to 5 without copying memory. cat only joins existing axes and reshape cannot change the element count.</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Broadcast-subtract each image's own mean from a <code>(32,3,32,32)</code> batch (reduce over dim=(2,3) with keepdim).</li>
    <li>With <code>(5,4)</code> and <code>(4,)</code>, multiply to scale columns; then use <code>(5,1)</code> to scale rows.</li>
    <li>Chunk a batch into 4, cat it back, and check equality with the original.</li>
    <li>Try three broadcastable and three non-broadcastable pairs in the widget.</li>
  </ol>
</div>`
  }
});
