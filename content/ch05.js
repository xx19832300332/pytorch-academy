(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch05',
  num: '05',
  part: 'p2',
  file: 'nn-module',
  zh: {
    title: '搭建模型：nn.Module',
    sub: '从「一堆张量」升级成「一个可以训练的对象」',
    html: `
<p class="lead">前面所有的参数都要手动创建、手动更新。一旦模型有 100 层，这件事就不可能了。<code>nn.Module</code> 帮你做三件事：<strong>自动收集参数</strong>、<strong>把前向过程写成函数</strong>、<strong>支持嵌套组合</strong>。这一章学完，你就能读懂并写出任意模型结构。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>写出一个带 <code>__init__</code> 和 <code>forward</code> 的自定义模型</li>
    <li>说清 <code>nn.Linear(in, out)</code> 里那两个数字和参数量</li>
    <li>用 <code>Sequential</code> 和 <code>ModuleList</code> 组织多层结构</li>
    <li>统计模型参数量，判断它大概需要多少显存</li>
  </ul>
</div>

<h2>1. 一个最小的自定义模型</h2>
<pre data-lang="python"><code>import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, in_dim=784, hidden=128, out_dim=10):
        super().__init__()                     # 必须调用父类构造函数
        self.fc1 = nn.Linear(in_dim, hidden)   # 会自己创建权重和偏置
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):                      # 定义「数据怎么流」
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = TwoLayerNet()
print(model)
# TwoLayerNet(
#   (fc1): Linear(in_features=784, out_features=128, bias=True)
#   (relu): ReLU()
#   (fc2): Linear(in_features=128, out_features=10, bias=True)
# )

x = torch.randn(32, 784)
out = model(x)          # 注意：不是 model.forward(x)
print(out.shape)        # (32, 10)</code></pre>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p>为什么写 <code>model(x)</code> 而不是 <code>model.forward(x)</code>？因为 <code>nn.Module</code> 重写了括号运算符 <code>__call__</code>，它会在调用 <code>forward</code> 前后做一些必要的事（比如执行 hook）。直接调 <code>forward</code> 会跳过这些，虽然多数时候也能跑，但某些高级功能就失效了。<strong>养成写 <code>model(x)</code> 的习惯。</strong></p>
  <p><code>super().__init__()</code> 是「先让父类把该准备的东西准备好」。不写它，<code>nn.Module</code> 内部的参数登记机制就没启动，后面加层会报错。</p>
</div>

<div class="callout warn">
  <span class="tag">坑：忘记 super().__init__()</span>
  <p>报错信息通常是 <code>AttributeError: cannot assign module before Module.__init__() call</code>。原因就是你在 <code>super().__init__()</code> 之前写了 <code>self.fc1 = ...</code>。把 <code>super().__init__()</code> 永远放在第一行。</p>
</div>

<h2>2. nn.Linear 到底做了什么</h2>
<p>一行公式：<code>y = xW + b</code>。其中 <code>W</code> 形状是 <code>(in_features, out_features)</code>，<code>b</code> 形状是 <code>(out_features,)</code>。</p>

<pre data-lang="python"><code>layer = nn.Linear(3, 1)
print(layer.weight.shape)    # torch.Size([1, 3])  —— 注意 PyTorch 存的是转置形式
print(layer.bias.shape)      # torch.Size([1])
print(layer.weight)          # 初始化时是均匀分布的小随机数

x = torch.randn(4, 3)        # 4 个样本，每个 3 维
print(layer(x).shape)        # (4, 1)

# 等价的纯张量写法，理解它为什么能省掉循环
y_manual = x @ layer.weight.T + layer.bias
print(torch.allclose(layer(x), y_manual))   # True</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 参数量怎么算</span>
  <p><code>nn.Linear(in, out)</code> 的参数量 = <code>in × out</code>（权重）+ <code>out</code>（偏置）。</p>
  <p>例如 <code>nn.Linear(784, 128)</code> → 784×128 + 128 = <strong>100,480</strong> 个参数。每个 float32 占 4 字节，约 <strong>0.4 MB</strong>。</p>
  <p>一个经验公式：模型参数量 <code>P</code>，推理大约占 <code>4P</code> 字节，训练时还要存梯度（<code>4P</code>）和优化器状态（Adam 是 <code>8P</code>），所以训练显存大概是推理的 <strong>3 倍以上</strong>。这解释了为什么「推理能跑的模型，训练可能爆显存」。</p>
</div>

<pre data-lang="python"><code>def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_params(TwoLayerNet()))   # 101882 = 100480 + 128 + 1290 + 10

# 看看每一层各占多少
for name, p in model.named_parameters():
    print(f"{name:10s} {str(tuple(p.shape)):12s} {p.numel()}")</code></pre>

<h2>3. Parameter 和普通张量的区别</h2>
<pre data-lang="python"><code>class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randn(3, 1)          # 普通张量，不会被训练！

class Good(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(3, 1))   # 注册为参数

b, g = Bad(), Good()
print(len(list(b.parameters())))   # 0 —— 优化器看不到它
print(len(list(g.parameters())))   # 1</code></pre>

<div class="callout warn">
  <span class="tag">最隐蔽的 bug</span>
  <p>用普通张量当权重，模型能跑、loss 也能算，但<strong>永远不会变好</strong>——因为 <code>model.parameters()</code> 里没有它，优化器根本不知道它的存在。如果你遇到「训练了 100 轮准确率还是 10%」，先检查这点。</p>
</div>

<h2>4. 组合：Sequential 与 ModuleList</h2>
<pre data-lang="python"><code># Sequential：一条直线，省事
net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10),
)
print(net[0])        # 可以用下标访问

# ModuleList：需要在 forward 里循环或做分支时用
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.act = nn.ReLU()

    def forward(self, x):
        for fc in self.fcs:          # 动态地遍历
            x = x + self.act(fc(x))  # 残差连接
        return x

print(ResBlock(16)(torch.randn(4, 16)).shape)   # (4, 16)</code></pre>

<div class="callout warn">
  <span class="tag">不要用 Python list 装层</span>
  <p><code>self.layers = [nn.Linear(4,4), nn.Linear(4,4)]</code> 是<strong>不会</strong>被注册的——参数照样不会出现在 <code>parameters()</code> 里。必须用 <code>nn.ModuleList</code> 或 <code>nn.Sequential</code>。</p>
</div>

<h2>5. 常用层速查</h2>
<table>
  <tr><th>层</th><th>作用</th><th>关键参数</th></tr>
  <tr><td><code>nn.Linear</code></td><td>全连接，y = xW + b</td><td><code>in_features, out_features</code></td></tr>
  <tr><td><code>nn.Conv2d</code></td><td>二维卷积（第 10 章）</td><td><code>in_ch, out_ch, kernel_size, stride, padding</code></td></tr>
  <tr><td><code>nn.BatchNorm2d</code></td><td>批归一化，稳定训练</td><td><code>num_features</code></td></tr>
  <tr><td><code>nn.Dropout</code></td><td>随机丢弃，抑制过拟合</td><td><code>p</code>（丢弃概率）</td></tr>
  <tr><td><code>nn.ReLU / GELU / Sigmoid</code></td><td>激活函数，引入非线性</td><td>—</td></tr>
  <tr><td><code>nn.Embedding</code></td><td>把整数 id 变成向量</td><td><code>num_embeddings, embedding_dim</code></td></tr>
  <tr><td><code>nn.Flatten</code></td><td>把多维压成二维</td><td><code>start_dim</code></td></tr>
</table>

<div class="callout basic">
  <span class="tag">为什么必须有激活函数</span>
  <p>如果只有线性层，两层叠加等价于一层：<code>(xW₁)W₂ = x(W₁W₂)</code>。无论堆多深，模型都只能画直线。ReLU 这类非线性把这条限制打破，这才让「深度」有意义。</p>
  <p><code>ReLU(x) = max(0, x)</code>，就这么简单，但效果极好，是默认选择。</p>
</div>

<h2>6. 初始化：为什么不能全 0</h2>
<pre data-lang="python"><code>import torch.nn.init as init

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        init.zeros_(m.bias)

model.apply(init_weights)     # 递归应用到每一层</code></pre>
<p>如果所有权重都是 0，每层输出的梯度完全相同，所有神经元会一直保持一样——这叫<strong>对称性不破</strong>，等于只有一个神经元在工作。所以需要随机初始化。</p>
<p>PyTorch 的默认初始化已经不错，<strong>大多数情况你不需要手动改</strong>，知道有这回事就行。</p>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li>自定义模型 = 继承 <code>nn.Module</code> + <code>__init__</code> 定义层 + <code>forward</code> 定义流向。</li>
    <li><code>super().__init__()</code> 必须是第一行。</li>
    <li>用 <code>model(x)</code>，不要 <code>model.forward(x)</code>。</li>
    <li><code>nn.Linear(in,out)</code> 参数量 = in×out + out。</li>
    <li>自定义权重必须用 <code>nn.Parameter</code> 或放进 Module 容器，否则不会被训练。</li>
    <li>层列表用 <code>nn.ModuleList</code>，不要用 Python list。</li>
    <li>没有非线性激活，堆多少层都等于一层。</li>
  </ol>
</div>

<div class="quiz" data-answer="2">
  <p class="q"><code>nn.Linear(256, 64)</code> 的参数量（含偏置）是？</p>
  <ol class="opts">
    <li>256 × 64 = 16384</li>
    <li>256 + 64 = 320</li>
    <li>256 × 64 + 64 = 16448</li>
    <li>256 × 256 = 65536</li>
  </ol>
  <p class="exp">权重 in×out，偏置每个输出一个。16448 个参数，float32 约 64 KB。</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q">在 <code>__init__</code> 里写 <code>self.w = torch.randn(3,1)</code>，会发生什么？</p>
  <ol class="opts">
    <li>正常工作，w 会被训练</li>
    <li>w 不会出现在 parameters() 里，永远不会被更新</li>
    <li>立即报错</li>
    <li>w 会自动转成 Parameter</li>
  </ol>
  <p class="exp">普通张量不会被注册。模型能跑但学不到东西，是最隐蔽的 bug 之一。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">两个连续的 <code>nn.Linear</code> 中间不加激活函数，等价于？</p>
  <ol class="opts">
    <li>一个线性层</li>
    <li>两个线性层，表达力翻倍</li>
    <li>一个非线性层</li>
    <li>无法计算</li>
  </ol>
  <p class="exp">矩阵乘满足结合律：(xW₁)W₂ = x(W₁W₂)，两个线性映射复合仍是线性映射。这就是必须有激活函数的原因。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>写一个三层 MLP（784→256→64→10），打印参数量，验证是否等于手算结果。</li>
    <li>把 <code>nn.Parameter</code> 改成普通张量，训练几轮看 loss 有没有变化，体会这个坑。</li>
    <li>用 <code>nn.Sequential</code> 重写上面的模型，比较两种写法的可读性。</li>
    <li>给模型加一个 <code>nn.Dropout(0.5)</code>，分别在 <code>model.train()</code> 和 <code>model.eval()</code> 下对同一输入前向两次，观察输出是否相同，并解释原因。</li>
  </ol>
</div>`
  },
  en: {
    title: 'Building models: nn.Module',
    sub: 'From loose tensors to a trainable object',
    html: `
<p class="lead">So far every weight was created and updated by hand — impossible at 100 layers. <code>nn.Module</code> gives you three things: <strong>automatic parameter collection</strong>, <strong>a forward function</strong>, and <strong>nesting</strong>. After this chapter you can read and write any architecture.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Write a custom model with <code>__init__</code> and <code>forward</code></li>
    <li>Explain <code>nn.Linear(in, out)</code> and its parameter count</li>
    <li>Organise layers with <code>Sequential</code> and <code>ModuleList</code></li>
    <li>Count parameters and estimate memory</li>
  </ul>
</div>

<h2>1. A minimal custom model</h2>
<pre data-lang="python"><code>import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, in_dim=784, hidden=128, out_dim=10):
        super().__init__()                     # required
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = TwoLayerNet()
print(model)

x = torch.randn(32, 784)
out = model(x)          # not model.forward(x)
print(out.shape)        # (32, 10)</code></pre>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p>Why <code>model(x)</code>? <code>nn.Module</code> overrides <code>__call__</code>, which wraps <code>forward</code> with hooks. Calling <code>forward</code> directly skips them. Just always use parentheses.</p>
  <p><code>super().__init__()</code> lets the parent set up its internal bookkeeping; without it, assigning layers raises <code>cannot assign module before Module.__init__() call</code>. Keep it first.</p>
</div>

<h2>2. What nn.Linear actually computes</h2>
<p><code>y = xW + b</code>, with <code>W</code> of shape <code>(in_features, out_features)</code> and <code>b</code> of shape <code>(out_features,)</code>.</p>

<pre data-lang="python"><code>layer = nn.Linear(3, 1)
print(layer.weight.shape)    # torch.Size([1, 3]) — stored transposed
print(layer.bias.shape)      # torch.Size([1])

x = torch.randn(4, 3)
print(layer(x).shape)        # (4, 1)

y_manual = x @ layer.weight.T + layer.bias
print(torch.allclose(layer(x), y_manual))   # True</code></pre>

<div class="callout math">
  <span class="tag">Math note · counting parameters</span>
  <p><code>nn.Linear(in, out)</code> has <code>in × out</code> weights plus <code>out</code> biases.</p>
  <p><code>nn.Linear(784, 128)</code> → 784×128 + 128 = <strong>100,480</strong> params ≈ 0.4 MB in float32.</p>
  <p>Roughly, inference needs <code>4P</code> bytes, training needs gradients (<code>4P</code>) plus optimiser state (Adam: <code>8P</code>) — about 3× inference. That is why a model that runs for inference may OOM during training.</p>
</div>

<pre data-lang="python"><code>def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_params(TwoLayerNet()))   # 101882

for name, p in model.named_parameters():
    print(f"{name:10s} {str(tuple(p.shape)):12s} {p.numel()}")</code></pre>

<h2>3. Parameter vs plain tensor</h2>
<pre data-lang="python"><code>class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randn(3, 1)          # not registered

class Good(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(3, 1))

print(len(list(Bad().parameters())))    # 0
print(len(list(Good().parameters())))   # 1</code></pre>

<div class="callout warn">
  <span class="tag">The sneakiest bug</span>
  <p>A plain tensor still runs and still produces a loss, but it <strong>never updates</strong> — the optimiser never sees it. If accuracy is stuck at 10% after 100 epochs, check this first.</p>
</div>

<h2>4. Composing: Sequential and ModuleList</h2>
<pre data-lang="python"><code>net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10),
)
print(net[0])

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        self.act = nn.ReLU()

    def forward(self, x):
        for fc in self.fcs:
            x = x + self.act(fc(x))
        return x</code></pre>

<div class="callout warn">
  <span class="tag">Never a plain Python list</span>
  <p><code>self.layers = [nn.Linear(...), nn.Linear(...)]</code> is not registered. Use <code>nn.ModuleList</code> or <code>nn.Sequential</code>.</p>
</div>

<h2>5. Layer cheat sheet</h2>
<table>
  <tr><th>Layer</th><th>What it does</th><th>Key args</th></tr>
  <tr><td><code>nn.Linear</code></td><td>fully connected, y = xW + b</td><td><code>in_features, out_features</code></td></tr>
  <tr><td><code>nn.Conv2d</code></td><td>2-D convolution (Ch. 10)</td><td><code>in_ch, out_ch, kernel_size, stride, padding</code></td></tr>
  <tr><td><code>nn.BatchNorm2d</code></td><td>batch normalisation</td><td><code>num_features</code></td></tr>
  <tr><td><code>nn.Dropout</code></td><td>random dropout</td><td><code>p</code></td></tr>
  <tr><td><code>nn.ReLU / GELU</code></td><td>non-linearity</td><td>—</td></tr>
  <tr><td><code>nn.Embedding</code></td><td>integer id to vector</td><td><code>num_embeddings, embedding_dim</code></td></tr>
  <tr><td><code>nn.Flatten</code></td><td>flatten dims</td><td><code>start_dim</code></td></tr>
</table>

<div class="callout basic">
  <span class="tag">Why activations are mandatory</span>
  <p>Stacking linear layers without activations is one linear layer: <code>(xW₁)W₂ = x(W₁W₂)</code>. ReLU (just <code>max(0,x)</code>) breaks that, which is what makes depth meaningful.</p>
</div>

<h2>6. Initialisation: why not zeros</h2>
<pre data-lang="python"><code>import torch.nn.init as init

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        init.zeros_(m.bias)

model.apply(init_weights)</code></pre>
<p>All-zero weights make every neuron receive identical gradients forever — the symmetry never breaks. PyTorch's defaults are already sensible; you rarely need this.</p>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li>A model = subclass <code>nn.Module</code> + layers in <code>__init__</code> + flow in <code>forward</code>.</li>
    <li><code>super().__init__()</code> must come first.</li>
    <li>Call <code>model(x)</code>, not <code>model.forward(x)</code>.</li>
    <li><code>nn.Linear(in,out)</code> has in×out + out params.</li>
    <li>Custom weights need <code>nn.Parameter</code> or a Module container.</li>
    <li>Use <code>nn.ModuleList</code>, never a plain list.</li>
    <li>No non-linearity means any depth collapses to one layer.</li>
  </ol>
</div>

<div class="quiz" data-answer="2">
  <p class="q">How many parameters does <code>nn.Linear(256, 64)</code> have, bias included?</p>
  <ol class="opts">
    <li>256 × 64 = 16384</li>
    <li>256 + 64 = 320</li>
    <li>256 × 64 + 64 = 16448</li>
    <li>256 × 256 = 65536</li>
  </ol>
  <p class="exp">Weights in×out, one bias per output.</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q"><code>self.w = torch.randn(3,1)</code> inside <code>__init__</code> means:</p>
  <ol class="opts">
    <li>It works and w trains normally</li>
    <li>w never appears in parameters() and never updates</li>
    <li>Immediate error</li>
    <li>It is auto-converted to a Parameter</li>
  </ol>
  <p class="exp">Plain tensors are not registered. The model runs, learns nothing.</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">Two consecutive <code>nn.Linear</code> with no activation in between are equivalent to:</p>
  <ol class="opts">
    <li>A single linear layer</li>
    <li>Two layers with double capacity</li>
    <li>A non-linear layer</li>
    <li>Nothing computable</li>
  </ol>
  <p class="exp">(xW₁)W₂ = x(W₁W₂). Linear composed with linear is still linear.</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Write a 3-layer MLP (784→256→64→10), print the parameter count, verify by hand.</li>
    <li>Swap <code>nn.Parameter</code> for a plain tensor, train a few epochs, watch the loss freeze.</li>
    <li>Rewrite the model with <code>nn.Sequential</code> and compare readability.</li>
    <li>Add <code>nn.Dropout(0.5)</code>, forward the same input twice under <code>model.train()</code> and <code>model.eval()</code>, explain the difference.</li>
  </ol>
</div>`
  }
});
