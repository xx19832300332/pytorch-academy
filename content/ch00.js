(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch00',
  num: '00',
  part: 'p0',
  file: 'prerequisites',
  zh: {
    title: '预备知识：Python 与数学的最小集合',
    sub: '不必先学完高数和线代，但下面这些「零件」你得认识',
    html: `
<p class="lead">很多教程第一页就开始写 <code>nn.Module</code>，结果读者卡在「为什么这里要转置」「梯度到底是个啥」。这一章先把后面 14 章会用到的 Python 语法、张量直觉、矩阵乘法、导数和概率，用最小的篇幅过一遍。看不懂的地方不用背，混个脸熟就行，后面遇到会反复回来。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>读懂 PyTorch 示例里的 Python 语法（类、函数、列表推导、字典）</li>
    <li>说清楚「标量 / 向量 / 矩阵 / 张量」的区别，以及 shape 的含义</li>
    <li>手算一个 2×2 的矩阵乘法，理解它为什么能表达「一层神经网络」</li>
    <li>用一句话解释导数和梯度，知道它为什么能让模型变好</li>
  </ul>
</div>

<h2>1. Python 的最小可用子集</h2>
<p>PyTorch 的代码 95% 是普通 Python。你不需要成为 Python 高手，但下面五件事必须认识：变量与类型、列表/字典、函数、类、<code>import</code>。</p>

<pre data-lang="python"><code># 1) 变量：名字 = 值，不需要声明类型
lr = 0.001          # 浮点数 float
epochs = 10         # 整数 int
name = "resnet"     # 字符串 str
use_gpu = True      # 布尔 bool

# 2) 列表 list：有序，可以装任何东西
layers = [64, 128, 256]
layers.append(512)          # 末尾追加 -> [64, 128, 256, 512]
first = layers[0]           # 下标从 0 开始 -> 64
last = layers[-1]           # 负数从后往前 -> 512
part = layers[1:3]          # 切片：含头不含尾 -> [128, 256]

# 3) 字典 dict：键 -> 值，用来存配置最方便
config = {"batch_size": 32, "lr": 0.001}
config["lr"] = 0.0005       # 改值
print(config.get("momentum", 0.9))   # 取不到就给默认值 0.9

# 4) 函数 def：把一段逻辑打包
def accuracy(correct, total):
    return correct / total

print(accuracy(87, 100))    # 0.87

# 5) 类 class：把「数据 + 行为」打包，PyTorch 的模型全是类
class Counter:
    def __init__(self):     # 构造函数，创建对象时自动调用
        self.n = 0          # self.n 是每个对象自己的属性
    def add(self):          # 方法
        self.n += 1

c = Counter()
c.add(); c.add()
print(c.n)                  # 2</code></pre>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p><strong>列表推导式</strong>是 Python 里最常见的简写，读作「对于每个 x，产生 f(x)」：</p>
  <p><code>[x * 2 for x in [1, 2, 3]]</code> 等价于先建空表、循环、再 append，结果是 <code>[2, 4, 6]</code>。你在 PyTorch 代码里会看到 <code>[m for m in model.modules()]</code> 这种写法，意思就是「把模型里所有层收集成一个列表」。</p>
  <p><strong>self</strong> 就是「这个对象自己」。<code>self.n</code> 表示「我这个对象的 n」，不是全局变量。</p>
</div>

<h2>2. 从标量到张量：shape 到底在说什么</h2>
<p>神经网络里的所有数据——图片、句子、权重——最终都变成一堆数字排成的<strong>多维数组</strong>，PyTorch 管它叫 <strong>张量（tensor）</strong>。维度（dim）就是「这个数组有几层方括号」。</p>

<table>
  <tr><th>名字</th><th>英文</th><th>形状</th><th>PyTorch 例子</th><th>机器学习里的典型含义</th></tr>
  <tr><td>标量</td><td>scalar</td><td>()</td><td><code>torch.tensor(3.14)</code></td><td>一个损失值、一个学习率</td></tr>
  <tr><td>向量</td><td>vector</td><td>(n,)</td><td><code>torch.randn(784)</code></td><td>一张 28×28 图片拉平后的像素</td></tr>
  <tr><td>矩阵</td><td>matrix</td><td>(m, n)</td><td><code>torch.randn(32, 784)</code></td><td>32 张图片组成一个 batch</td></tr>
  <tr><td>三维张量</td><td>3-D tensor</td><td>(c, h, w)</td><td><code>torch.randn(3, 224, 224)</code></td><td>一张彩色图片（RGB 三通道）</td></tr>
  <tr><td>四维张量</td><td>4-D tensor</td><td>(b, c, h, w)</td><td><code>torch.randn(8, 3, 224, 224)</code></td><td>一批 8 张彩色图片</td></tr>
</table>

<div class="callout math">
  <span class="tag">数学批注 · 维度</span>
  <p>数学里「维度」有两个意思，初学最容易混：</p>
  <ul>
    <li><strong>空间的维数</strong>：一条直线的点是 1 维，平面是 2 维，我们生活的空间是 3 维。这里说的是「需要几个数字才能定位一个点」。</li>
    <li><strong>数组的轴数（rank / ndim）</strong>：PyTorch 说的 <code>tensor.dim() == 3</code> 是指「有 3 个轴」，不是「3 维空间」。形状 <code>(3, 224, 224)</code> 表示通道、高、宽三个轴。</li>
  </ul>
  <p>所以看到 <code>x.dim()</code> 时，请读作「x 有几个轴」，看到 <code>x.shape</code> 时，读作「每个轴各有多长」。这两个信息一起，才算把张量描述完整。</p>
</div>

<h2>3. 矩阵乘法到底在算什么</h2>
<p>别怕公式。矩阵乘法就是「<strong>把一行的每个数，和另一列的每个数配对相乘，再加起来</strong>」。这也是神经网络一层做的事：每个输出神经元，都在看所有输入的一个加权和。</p>

<pre data-lang="python"><code>import torch

x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])      # shape (2, 3)

W = torch.tensor([[1., 0.],
                  [0., 1.],
                  [1., 1.]])          # shape (3, 2)

y = x @ W                             # 矩阵乘法，(2,3) @ (3,2) -> (2,2)
print(y)
# tensor([[ 4.,  5.],
#         [10., 11.]])</code></pre>

<p>第一行怎么来的？<code>1×1 + 2×0 + 3×1 = 4</code>，<code>1×0 + 2×1 + 3×1 = 5</code>。就这样一对一对乘、加。</p>

<div class="callout math">
  <span class="tag">数学批注 · 形状规则</span>
  <p>矩阵乘法 <code>(a, b) @ (b, c)</code> 能成立的条件是：<strong>左边的列数 = 右边的行数</strong>，结果形状是 <code>(a, c)</code>。中间那个 <code>b</code> 必须相等，并且它会在结果里「消失」。</p>
  <p class="formula">(N, in_features) @ (in_features, out_features) → (N, out_features)</p>
  <p>这正是 <code>nn.Linear(in_features, out_features)</code> 的形状约定。记住这条，以后 90% 的形状报错都能自己定位：报错说 <code>mat1 and mat2 shapes cannot be multiplied (32x784 and 128x10)</code>，就是在说 784 ≠ 128。</p>
  <p>另外还有一个符号 <code>*</code>，它是「逐元素相乘」（对应位置各乘各的），和 <code>@</code> 完全不同。这是新手最常见的 bug 来源之一。</p>
</div>

<h2>4. 导数与梯度：模型是怎么「学会」的</h2>
<p>一句话：<strong>导数告诉你「把输入往右挪一点点，输出会变大还是变小、变多少」</strong>。模型训练就是把这句话用到极致：我们希望损失（loss）变小，那就沿着导数相反的方向调参数。</p>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p>把损失函数想成一片山地，你在山上某个位置，蒙着眼睛想走到谷底。你唯一能做的是用脚感受脚下哪个方向是下坡——这个「坡度」就是导数。梯度就是「把所有方向的坡度打包成一个向量」，它指向最陡的上坡方向，所以我们要<strong>减去</strong>它。</p>
  <p>学习率（learning rate）就是「每步走多远」。步子太大会跨过谷底来回震荡，太小会走到天荒地老。</p>
</div>

<p>一个初中水平的例子就够了：<code>y = x²</code> 的导数是 <code>2x</code>。当 <code>x = 3</code> 时导数为 6，说明 x 增加一点点，y 会以 6 倍的速度增加；想让 y 变小，就该让 x 减小。</p>

<pre data-lang="python"><code>x = 3.0
for step in range(5):
    grad = 2 * x          # y = x^2 的导数
    x = x - 0.3 * grad    # 沿负梯度方向走一步，0.3 是学习率
    print(f"x={x:.4f}, y={x**2:.4f}")
# x 会一路减小，y 一路逼近 0</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 偏导与梯度</span>
  <p>当函数有很多输入时（比如模型有 100 万个参数），我们一次只对一个参数求导，其他参数当常数——这叫<strong>偏导数</strong>，写作 ∂L/∂w。</p>
  <p>把所有偏导数排成一个向量，就是<strong>梯度</strong> ∇L。第 4 章你会看到，PyTorch 的 <code>loss.backward()</code> 做的就是「自动把所有偏导数算出来」，这件事叫反向传播（backpropagation）。</p>
  <p class="formula">w ← w − lr · ∂L/∂w &nbsp;&nbsp;（这就是所有优化器的祖先：梯度下降）</p>
</div>

<h2>5. 概率：为什么需要 softmax</h2>
<p>分类模型的输出通常是一组「分数」，比如猫 2.3、狗 0.7、鸟 0.1。这些分数叫 <strong>logits</strong>，可以任意大任意小。但我们想要的是「概率」：都在 0~1 之间，加起来等于 1。</p>
<p><strong>softmax</strong> 就是做这个转换的：先对每个分数取指数（把负数变正、放大差距），再除以总和做归一化。</p>
<div class="formula">softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)</div>
<p>代入上面的例子：exp(2.3)=9.97, exp(0.7)=2.01, exp(0.1)=1.11，总和 13.09，于是概率约为 (0.76, 0.15, 0.08)。模型说「76% 是猫」。</p>

<div class="callout tip">
  <span class="tag">先记住这个坑</span>
  <p>PyTorch 的 <code>nn.CrossEntropyLoss</code> <strong>内部已经包含 softmax</strong>。所以模型最后一层要输出原始 logits，<strong>不要</strong>先手动 softmax 再送进去，否则等于做了两遍，训练会变得很糟。这是第 7 章的重点。</p>
</div>

<h2>6. 术语表（后面会反复出现）</h2>
<div class="terms">
  <div class="term"><b>张量<span class="en">tensor</span></b>多维数组，PyTorch 的基本数据结构，能放到 GPU 上算。</div>
  <div class="term"><b>形状<span class="en">shape</span></b>每个轴的长度，如 <code>(32, 3, 224, 224)</code>。形状错了，代码基本就跑不动。</div>
  <div class="term"><b>批次<span class="en">batch</span></b>一次喂给模型的样本数。太大爆显存，太小训练抖。</div>
  <div class="term"><b>轮次<span class="en">epoch</span></b>把整个训练集完整看一遍叫一轮。</div>
  <div class="term"><b>参数<span class="en">parameter</span></b>模型里会被训练改变的数，主要是权重 W 和偏置 b。</div>
  <div class="term"><b>梯度<span class="en">gradient</span></b>损失对参数的偏导，指示「往哪调会让损失变大」。</div>
  <div class="term"><b>损失<span class="en">loss</span></b>一个数字，衡量预测和答案差多远。训练就是让它变小。</div>
  <div class="term"><b>推理<span class="en">inference</span></b>训练完只用模型做预测，不需要算梯度。</div>
</div>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li>PyTorch 代码的主体是普通 Python，读懂类、函数、列表、字典就够起步。</li>
    <li>张量的 <code>dim()</code> 是轴数，<code>shape</code> 是每轴长度，两者一起才能描述清楚。</li>
    <li>矩阵乘法 <code>(N,in)@(in,out)</code>，中间维度必须相等并消失——这是 <code>nn.Linear</code> 的形状约定。</li>
    <li><code>@</code> 是矩阵乘，<code>*</code> 是逐元素乘，别混。</li>
    <li>梯度 = 坡度，训练 = 沿负梯度方向小步走，学习率 = 步长。</li>
    <li><code>CrossEntropyLoss</code> 内含 softmax，模型输出 logits 就好。</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q">形状为 <code>(64, 20)</code> 的张量和形状为 <code>(20, 5)</code> 的张量做矩阵乘法，结果是？</p>
  <ol class="opts">
    <li>(64, 5)</li>
    <li>(64, 20, 5)</li>
    <li>(20, 20)</li>
    <li>报错，因为形状不同</li>
  </ol>
  <p class="exp">中间那个 20 相等并「消掉」，剩下 (64, 5)。这就是 <code>nn.Linear(20, 5)</code> 对一批 64 个样本做的事：每个样本从 20 维变成 5 维。</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">如果 loss 对某个参数的偏导是负数，说明什么？</p>
  <ol class="opts">
    <li>这个参数不需要更新</li>
    <li>增大该参数会让 loss 变大</li>
    <li>增大该参数会让 loss 变小</li>
    <li>该参数已经是最优值</li>
  </ol>
  <p class="exp">偏导为负说明「往正方向走，loss 在下降」。而更新公式是 w ← w − lr·grad，负负得正，所以参数会自动被增大，正好朝 loss 下降的方向。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>用纸笔算一遍 <code>[[1,2],[3,4]] @ [[1,0],[0,1]]</code>，再用 PyTorch 验证。想想为什么结果等于原矩阵（提示：单位矩阵）。</li>
    <li>把上面梯度下降的代码跑一遍，把学习率改成 0.6，观察 x 是否来回震荡甚至发散。这个现象叫什么？</li>
    <li>写出形状为 <code>(8, 3, 32, 32)</code> 的张量中，每个数字代表什么（提示：b, c, h, w）。</li>
  </ol>
</div>`
  },
  en: {
    title: 'Prerequisites: the minimum Python and math',
    sub: 'You do not need a full calculus course — just recognise these parts',
    html: `
<p class="lead">Most tutorials jump straight into <code>nn.Module</code> and leave readers stuck on "why transpose here" and "what even is a gradient". This chapter covers the Python syntax, tensor intuition, matrix multiplication, derivatives and probability that the next 14 chapters rely on. You do not need to memorise anything; just get familiar, and come back when something feels shaky.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Read the Python used in PyTorch examples (classes, functions, comprehensions, dicts)</li>
    <li>Explain scalar vs vector vs matrix vs tensor, and what a shape means</li>
    <li>Multiply two 2×2 matrices by hand and see why that expresses one network layer</li>
    <li>Explain derivative and gradient in one sentence, and why they make models learn</li>
  </ul>
</div>

<h2>1. The minimum useful subset of Python</h2>
<p>95% of PyTorch code is ordinary Python. You need five things: variables, lists/dicts, functions, classes, and <code>import</code>.</p>

<pre data-lang="python"><code># 1) Variables: name = value, no type declaration
lr = 0.001          # float
epochs = 10         # int
name = "resnet"     # str
use_gpu = True      # bool

# 2) list: ordered, holds anything
layers = [64, 128, 256]
layers.append(512)          # [64, 128, 256, 512]
first = layers[0]           # 64, zero-indexed
last = layers[-1]           # 512, negative counts from the end
part = layers[1:3]          # slice, end exclusive -> [128, 256]

# 3) dict: key -> value, perfect for configs
config = {"batch_size": 32, "lr": 0.001}
config["lr"] = 0.0005
print(config.get("momentum", 0.9))   # default when missing

# 4) def: package logic into a function
def accuracy(correct, total):
    return correct / total

# 5) class: data + behaviour. Every PyTorch model is a class.
class Counter:
    def __init__(self):     # constructor, called on creation
        self.n = 0          # per-object attribute
    def add(self):
        self.n += 1

c = Counter()
c.add(); c.add()
print(c.n)                  # 2</code></pre>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p>A <strong>list comprehension</strong> is shorthand for "build a list by transforming each item": <code>[x * 2 for x in [1, 2, 3]]</code> gives <code>[2, 4, 6]</code>. You will meet <code>[m for m in model.modules()]</code>, which just collects every layer into a list.</p>
  <p><strong>self</strong> means "this object". <code>self.n</code> is "my own n", not a global.</p>
</div>

<h2>2. From scalar to tensor: what a shape tells you</h2>
<p>Everything inside a network — images, sentences, weights — becomes a multi-dimensional array of numbers. PyTorch calls it a <strong>tensor</strong>. The number of dimensions is the number of nested bracket levels.</p>

<table>
  <tr><th>Kind</th><th>Shape</th><th>Example</th><th>Typical meaning in ML</th></tr>
  <tr><td>scalar</td><td>()</td><td><code>torch.tensor(3.14)</code></td><td>a loss value, a learning rate</td></tr>
  <tr><td>vector</td><td>(n,)</td><td><code>torch.randn(784)</code></td><td>one flattened 28×28 image</td></tr>
  <tr><td>matrix</td><td>(m, n)</td><td><code>torch.randn(32, 784)</code></td><td>a batch of 32 images</td></tr>
  <tr><td>3-D tensor</td><td>(c, h, w)</td><td><code>torch.randn(3, 224, 224)</code></td><td>one RGB image</td></tr>
  <tr><td>4-D tensor</td><td>(b, c, h, w)</td><td><code>torch.randn(8, 3, 224, 224)</code></td><td>a batch of 8 RGB images</td></tr>
</table>

<div class="callout math">
  <span class="tag">Math note · dimensions</span>
  <p>"Dimension" has two meanings, and beginners mix them up:</p>
  <ul>
    <li><strong>Dimension of a space</strong>: a line is 1-D, a plane 2-D, physical space 3-D — how many numbers locate a point.</li>
    <li><strong>Number of axes (rank)</strong>: <code>x.dim() == 3</code> means "three axes", not "three-dimensional space". Shape <code>(3, 224, 224)</code> is channel, height, width.</li>
  </ul>
  <p>So read <code>x.dim()</code> as "how many axes", and <code>x.shape</code> as "how long each axis is". Together they fully describe a tensor.</p>
</div>

<h2>3. What matrix multiplication actually computes</h2>
<p>Take each number in a row of the left matrix, multiply by the matching number in a column of the right matrix, sum them up. That is exactly what one network layer does: every output neuron looks at a weighted sum of all inputs.</p>

<pre data-lang="python"><code>import torch

x = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])      # shape (2, 3)

W = torch.tensor([[1., 0.],
                  [0., 1.],
                  [1., 1.]])          # shape (3, 2)

y = x @ W                             # (2,3) @ (3,2) -> (2,2)
print(y)
# tensor([[ 4.,  5.],
#         [10., 11.]])</code></pre>

<p>First row: <code>1*1 + 2*0 + 3*1 = 4</code> and <code>1*0 + 2*1 + 3*1 = 5</code>. Pairwise multiply, then add.</p>

<div class="callout math">
  <span class="tag">Math note · shape rule</span>
  <p><code>(a, b) @ (b, c)</code> works only when the left columns equal the right rows; the result is <code>(a, c)</code>. The inner <code>b</code> disappears.</p>
  <p class="formula">(N, in_features) @ (in_features, out_features) → (N, out_features)</p>
  <p>That is exactly the contract of <code>nn.Linear(in_features, out_features)</code>. Remember it and you can debug most shape errors yourself: <code>mat1 and mat2 shapes cannot be multiplied (32x784 and 128x10)</code> simply says 784 ≠ 128.</p>
  <p>Note the sibling operator <code>*</code>: element-wise multiply. Completely different from <code>@</code>, and a classic source of silent bugs.</p>
</div>

<h2>4. Derivatives and gradients: how a model learns</h2>
<p>In one sentence: a derivative tells you <strong>how much the output changes when you nudge the input</strong>. Training uses that: we want the loss to go down, so we nudge parameters in the opposite direction of the derivative.</p>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p>Picture the loss as a mountain range and yourself standing on it blindfolded. The only signal you have is the slope under your feet — that slope is the derivative. The gradient packs all the slopes into one vector pointing uphill, so we <strong>subtract</strong> it.</p>
  <p>The learning rate is your step size. Too big and you jump across the valley and oscillate; too small and you never arrive.</p>
</div>

<p>The derivative of <code>y = x²</code> is <code>2x</code>. At <code>x = 3</code> the slope is 6, meaning y grows six times as fast as x — to make y smaller, decrease x.</p>

<pre data-lang="python"><code>x = 3.0
for step in range(5):
    grad = 2 * x          # derivative of x^2
    x = x - 0.3 * grad    # step against the gradient; 0.3 is the LR
    print(f"x={x:.4f}, y={x**2:.4f}")
# x shrinks, y approaches 0</code></pre>

<div class="callout math">
  <span class="tag">Math note · partial derivatives</span>
  <p>With many inputs (a model may have millions of parameters) we differentiate with respect to one at a time, treating the rest as constants: that is a <strong>partial derivative</strong>, written ∂L/∂w.</p>
  <p>Stack all partials into a vector and you get the <strong>gradient</strong> ∇L. Chapter 4 shows that <code>loss.backward()</code> computes all of them automatically — that is backpropagation.</p>
  <p class="formula">w ← w − lr · ∂L/∂w &nbsp;&nbsp;(gradient descent, ancestor of every optimiser)</p>
</div>

<h2>5. Probability: why softmax exists</h2>
<p>A classifier usually outputs raw scores — cat 2.3, dog 0.7, bird 0.1. These are <strong>logits</strong> and can be any number. What we want are probabilities in [0, 1] that sum to 1.</p>
<p><strong>Softmax</strong> does that: exponentiate each score (turns negatives positive, magnifies gaps), then divide by the sum.</p>
<div class="formula">softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)</div>
<p>Above: exp(2.3)=9.97, exp(0.7)=2.01, exp(0.1)=1.11, sum 13.09 → (0.76, 0.15, 0.08). "76% cat".</p>

<div class="callout tip">
  <span class="tag">Remember this trap</span>
  <p><code>nn.CrossEntropyLoss</code> <strong>already contains softmax</strong>. Feed it raw logits; do not apply softmax yourself first, or you apply it twice and training degrades badly. This is a key point in Chapter 7.</p>
</div>

<h2>6. Glossary</h2>
<div class="terms">
  <div class="term"><b>tensor</b>A multi-dimensional array — PyTorch's core data structure, can live on GPU.</div>
  <div class="term"><b>shape</b>Length of each axis, e.g. <code>(32, 3, 224, 224)</code>. Wrong shapes are the #1 bug source.</div>
  <div class="term"><b>batch</b>Samples fed to the model at once. Too large: OOM. Too small: noisy.</div>
  <div class="term"><b>epoch</b>One full pass over the training set.</div>
  <div class="term"><b>parameter</b>Numbers the model learns: mainly weights W and biases b.</div>
  <div class="term"><b>gradient</b>Partial derivative of the loss w.r.t. a parameter; points uphill.</div>
  <div class="term"><b>loss</b>A single number measuring how wrong the predictions are.</div>
  <div class="term"><b>inference</b>Using a trained model to predict, with no gradient computation.</div>
</div>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li>PyTorch code is mostly plain Python. Classes, functions, lists and dicts are enough to start.</li>
    <li><code>dim()</code> is the number of axes; <code>shape</code> is their lengths. You need both.</li>
    <li>Matrix multiply is <code>(N,in)@(in,out)</code>; the inner dim must match and disappears.</li>
    <li><code>@</code> is matrix multiply, <code>*</code> is element-wise. Never mix them up.</li>
    <li>Gradient = slope; training = small steps downhill; learning rate = step size.</li>
    <li><code>CrossEntropyLoss</code> includes softmax — feed logits.</li>
  </ol>
</div>

<div class="quiz" data-answer="0">
  <p class="q">Multiplying a <code>(64, 20)</code> tensor by a <code>(20, 5)</code> tensor gives:</p>
  <ol class="opts">
    <li>(64, 5)</li>
    <li>(64, 20, 5)</li>
    <li>(20, 20)</li>
    <li>An error, shapes differ</li>
  </ol>
  <p class="exp">The inner 20 matches and collapses, leaving (64, 5). That is what <code>nn.Linear(20, 5)</code> does to a batch of 64 samples: 20 features to 5.</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">If the partial derivative of the loss w.r.t. a parameter is negative:</p>
  <ol class="opts">
    <li>The parameter should not be updated</li>
    <li>Increasing it makes the loss grow</li>
    <li>Increasing it makes the loss shrink</li>
    <li>It is already optimal</li>
  </ol>
  <p class="exp">A negative derivative means the loss decreases as the parameter grows. With w ← w − lr·grad, the minus sign flips it, so w increases — downhill.</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Compute <code>[[1,2],[3,4]] @ [[1,0],[0,1]]</code> by hand, then verify in PyTorch. Why is the result unchanged? (Identity matrix.)</li>
    <li>Run the gradient descent snippet with lr = 0.6. Does x oscillate or diverge? What is that called?</li>
    <li>For a tensor of shape <code>(8, 3, 32, 32)</code>, say what each number means.</li>
  </ol>
</div>`
  }
});
