(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch04',
  num: '04',
  part: 'p1',
  file: 'autograd',
  zh: {
    title: '自动求导：autograd 是怎么工作的',
    sub: '不用手推公式，但你得知道它在替你推什么',
    html: `
<p class="lead">PyTorch 最值钱的功能不是张量，而是<strong>自动求导</strong>。你只需要写前向计算，它在背后记录每一步运算，然后一次 <code>backward()</code> 把所有权重的梯度算出来。这一章会拆开这个黑盒，让你在报错时知道该看哪里。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>说清 <code>requires_grad</code>、<code>.grad</code>、<code>backward()</code> 三者的关系</li>
    <li>手写一段梯度下降，用 autograd 求导并跟解析解对比</li>
    <li>解释为什么每个 step 都要 <code>zero_grad()</code></li>
    <li>知道什么时候必须用 <code>torch.no_grad()</code> 或 <code>.detach()</code></li>
  </ul>
</div>

<h2>1. 一个最小的例子</h2>
<pre data-lang="python"><code>import torch

x = torch.tensor(3.0, requires_grad=True)   # 声明：我要对它求导
y = x ** 2                                  # 记录运算：y = x^2

y.backward()                                # 反向传播，算 dy/dx
print(x.grad)                               # tensor(6.)  —— 因为 2x = 6
</code></pre>
<p>你从没写过「导数是 2x」，PyTorch 知道。<code>requires_grad=True</code> 相当于在这块张量上插了一面旗子：<strong>凡是用到它的计算，都要记进账本</strong>。</p>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p>把 autograd 想成银行流水：每次你用带旗子的张量做运算，它就在账本上记一笔「谁、对谁、做了什么」。<code>backward()</code> 就是拿这个账本从最后一笔往前翻，按链式法则把导数一层层乘回去。</p>
  <p>这也是「反向传播」名字的由来——从输出往输入方向回放。</p>
</div>

<h2>2. 计算图与链式法则</h2>
<p>复合函数的导数等于各层导数的乘积，这就是<strong>链式法则</strong>：</p>
<div class="formula">如果 y = f(g(x))，则 dy/dx = f′(g(x)) · g′(x)</div>

<pre data-lang="python"><code># y = (2x + 1)^2 ，在 x = 1 处
x = torch.tensor(1.0, requires_grad=True)
u = 2 * x + 1        # u = 3
y = u ** 2           # y = 9

y.backward()
print(x.grad)        # tensor(12.)

# 手算验证：dy/du = 2u = 6，du/dx = 2，相乘 = 12 ✓</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 为什么要「图」</span>
  <p>如果每次都重新推导整个表达式，代价极高。计算图把结果缓存下来：前向时每个节点记住自己的输出，反向时直接查表用。</p>
  <p>更重要的是，链式法则允许<strong>局部计算</strong>：每个节点只关心「我的输出怎么随我的输入变」，不用知道整个网络长什么样。这就是为什么你可以随便堆层，求导照样能算。</p>
</div>

<h2>3. 梯度是「累加」的，不是「覆盖」的</h2>
<p>这是新手第一大坑，必须亲手撞一次。</p>
<pre data-lang="python"><code>x = torch.tensor(2.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(i, x.grad)

# tensor(4.)
# tensor(8.)
# tensor(12.)   <- 一直在加！</code></pre>
<p>原因：<code>.grad</code> 默认是<strong>累加</strong>的。PyTorch 这么设计是因为有些模型（比如梯度累积训练大 batch）确实需要它。所以标准训练循环里必须有：</p>
<pre data-lang="python"><code>optimizer.zero_grad()     # 清空上一轮的梯度
loss.backward()           # 重新算
optimizer.step()          # 更新参数</code></pre>

<div class="callout warn">
  <span class="tag">顺序千万别搞反</span>
  <p>正确顺序是 <strong>zero_grad → backward → step</strong>。如果忘了 zero_grad，梯度会越积越大，训练几步就发散成 NaN；如果先 step 再 backward，你用的是上一批数据算出的旧梯度。</p>
</div>

<h2>4. 为什么 backward() 只能对标量调用</h2>
<pre data-lang="python"><code>x = torch.randn(3, requires_grad=True)
y = x * 2                 # 形状 (3,)

# y.backward()            # 报错：grad can be implicitly created only for scalar outputs

# 两种正确写法：
y.sum().backward()                                   # 先聚合成一个数（最常用）

# 或者传入「每个输出对应的权重」，即雅可比向量积
y.backward(torch.ones_like(y))</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 标量才叫「梯度」</span>
  <p>严格说，标量对向量求导得到的是梯度（一个同形状的向量）；向量对向量求导得到的是<strong>雅可比矩阵</strong>，维度会爆炸。深度学习只关心「损失这个标量怎么变」，所以约定先聚合。</p>
  <p>这也解释了为什么训练时永远写 <code>loss.backward()</code>，而 <code>loss</code> 必须是一个标量（<code>loss.item()</code> 能取出单个数字）。</p>
</div>

<h2>5. no_grad 与 detach：什么时候该切断</h2>
<pre data-lang="python"><code># 1) 评估/推理时，不需要图，能省大量显存和计算
model.eval()
with torch.no_grad():
    preds = model(x)

# 2) 只想要数值，不想让它参与求导
y = x.detach()             # 共享数据，但脱离计算图

# 3) 转成 numpy 前必须先脱离
arr = x.detach().cpu().numpy()</code></pre>

<div class="callout warn">
  <span class="tag">内存泄漏的常见原因</span>
  <p>在评估循环里把 <code>loss</code> 存进列表（比如 <code>losses.append(loss)</code>）却不 detach，会导致整张计算图被一直引用，显存越跑越大。正确写法：<code>losses.append(loss.item())</code>。</p>
</div>

<h2>6. 手写一次线性回归</h2>
<p>不用 <code>nn.Module</code>、不用优化器，只用 autograd，把「学习」这件事完整走一遍。</p>
<pre data-lang="python"><code>import torch

torch.manual_seed(0)

# 造数据：真实规律 y = 3x + 2，加一点噪声
N = 100
x = torch.randn(N, 1)
true_w, true_b = 3.0, 2.0
y = true_w * x + true_b + 0.1 * torch.randn(N, 1)

# 初始化参数（随机猜一个）
w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.1
for epoch in range(200):
    # 1) 前向
    pred = x @ w + b
    loss = ((pred - y) ** 2).mean()

    # 2) 反向 + 更新
    loss.backward()
    with torch.no_grad():          # 更新参数时不该被记录进图
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 50 == 0:
        print(epoch, loss.item())

print("学到的 w, b =", w.item(), b.item())   # 接近 3.0 和 2.0</code></pre>

<p>跑完你会看到 loss 一路下降，w、b 逼近真实值。这三行——<strong>前向算 loss、backward 求梯度、沿负梯度更新</strong>——就是所有深度学习训练的骨架，第 8 章只是把它变得更工程化。</p>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li><code>requires_grad=True</code> 才会被追踪；只有 <code>float</code> 张量能求导。</li>
    <li><code>backward()</code> 从输出倒推，用链式法则把局部导数相乘。</li>
    <li><code>.grad</code> 是累加的，所以每个 step 必须先 <code>zero_grad()</code>。</li>
    <li><code>backward()</code> 只能对标量调用，通常先 <code>loss.mean()/.sum()</code>。</li>
    <li>推理用 <code>torch.no_grad()</code>；取数值用 <code>.item()</code> / <code>.detach()</code>。</li>
    <li>训练三件套：前向算 loss → backward → 更新。</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q">连续两次 <code>y.backward()</code>（中间没有清零），第二次之后 <code>x.grad</code> 会？</p>
  <ol class="opts">
    <li>被覆盖成第二次的梯度</li>
    <li>等于两次梯度之和</li>
    <li>变成 0</li>
    <li>报错</li>
  </ol>
  <p class="exp">梯度默认累加。这正是必须 zero_grad 的原因，也是「梯度累积」这种技巧能成立的原因。</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">下面哪种情况最需要 <code>torch.no_grad()</code>？</p>
  <ol class="opts">
    <li>计算训练损失</li>
    <li>调用 loss.backward()</li>
    <li>在验证集上做预测</li>
    <li>初始化模型参数</li>
  </ol>
  <p class="exp">推理不需要梯度，关掉追踪能显著省显存、提速。训练时的前向必须保留图，否则 backward 会报错。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q"><code>x</code> 是整数张量且 <code>requires_grad=True</code>，会怎样？</p>
  <ol class="opts">
    <li>报错：只有浮点张量可以求导</li>
    <li>正常求导</li>
    <li>自动转成 float</li>
    <li>梯度恒为 0</li>
  </ol>
  <p class="exp">导数是连续概念，整数没有意义，PyTorch 直接拒绝。所以要把输入转成 float32/float64。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>对 <code>y = 3x³ + 2x</code> 在 x=2 处求导，先用纸笔算（答案 38），再用 autograd 验证。</li>
    <li>把上面线性回归的 lr 改成 1.0 和 0.001，各跑一次，画出 loss 曲线，解释现象。</li>
    <li>故意删掉 <code>w.grad.zero_()</code>，看看 loss 怎么变化，把现象记进笔记。</li>
    <li>把 <code>loss.item()</code> 改成 <code>loss</code> 存进列表，观察显存（可用 <code>torch.cuda.memory_allocated()</code>）。</li>
  </ol>
</div>`
  },
  en: {
    title: 'Autograd: how gradients appear',
    sub: 'You never derive by hand — but you should know what it derives for you',
    html: `
<p class="lead">PyTorch's real superpower is not tensors, it is <strong>automatic differentiation</strong>. You write the forward pass; it records every operation and computes all gradients in one <code>backward()</code> call. This chapter opens the black box so you know where to look when it errors.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Explain how <code>requires_grad</code>, <code>.grad</code> and <code>backward()</code> relate</li>
    <li>Write gradient descent by hand and compare autograd with the analytic answer</li>
    <li>Explain why every step needs <code>zero_grad()</code></li>
    <li>Know when <code>no_grad()</code> or <code>.detach()</code> is required</li>
  </ul>
</div>

<h2>1. The smallest example</h2>
<pre data-lang="python"><code>import torch

x = torch.tensor(3.0, requires_grad=True)   # please track me
y = x ** 2                                  # records y = x^2

y.backward()                                # backpropagate
print(x.grad)                               # tensor(6.)  since 2x = 6
</code></pre>
<p>You never wrote "the derivative is 2x". <code>requires_grad=True</code> plants a flag: <strong>every computation touching this tensor gets recorded</strong>.</p>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p>Think of autograd as a ledger. Every operation involving a flagged tensor is written down. <code>backward()</code> reads the ledger backwards and multiplies local derivatives by the chain rule.</p>
  <p>That is literally why it is called back-propagation: you replay from output to input.</p>
</div>

<h2>2. The graph and the chain rule</h2>
<div class="formula">If y = f(g(x)) then dy/dx = f′(g(x)) · g′(x)</div>

<pre data-lang="python"><code># y = (2x + 1)^2 at x = 1
x = torch.tensor(1.0, requires_grad=True)
u = 2 * x + 1        # u = 3
y = u ** 2           # y = 9

y.backward()
print(x.grad)        # tensor(12.)

# by hand: dy/du = 2u = 6, du/dx = 2, product = 12</code></pre>

<div class="callout math">
  <span class="tag">Math note · why a graph</span>
  <p>Re-deriving the whole expression each time would be prohibitive. The graph caches intermediate values: the forward pass stores each node's output, the backward pass looks them up.</p>
  <p>Better still, the chain rule allows <strong>local</strong> computation — each node only needs to know how its output depends on its own inputs. That is why you can stack arbitrary layers and gradients still work.</p>
</div>

<h2>3. Gradients accumulate, they do not overwrite</h2>
<pre data-lang="python"><code>x = torch.tensor(2.0, requires_grad=True)

for i in range(3):
    y = x ** 2
    y.backward()
    print(i, x.grad)

# tensor(4.)
# tensor(8.)
# tensor(12.)   <- it keeps adding</code></pre>
<p>This is intentional (it enables gradient accumulation for large effective batches), which is why the standard loop must clear first:</p>
<pre data-lang="python"><code>optimizer.zero_grad()     # clear previous step's gradients
loss.backward()
optimizer.step()</code></pre>

<div class="callout warn">
  <span class="tag">Order matters</span>
  <p>Always <strong>zero_grad → backward → step</strong>. Forget the first and gradients grow until the loss becomes NaN. Call step before backward and you apply stale gradients.</p>
</div>

<h2>4. Why backward() needs a scalar</h2>
<pre data-lang="python"><code>x = torch.randn(3, requires_grad=True)
y = x * 2                 # shape (3,)

# y.backward()            # error: grad can be implicitly created only for scalar outputs

y.sum().backward()        # usual fix
y.backward(torch.ones_like(y))   # or supply per-output weights (VJP)</code></pre>

<div class="callout math">
  <span class="tag">Math note · gradient vs Jacobian</span>
  <p>Scalar w.r.t. vector gives a gradient (same shape). Vector w.r.t. vector gives a <strong>Jacobian matrix</strong>, which explodes in size. Deep learning only cares how one scalar (the loss) changes, hence the convention.</p>
</div>

<h2>5. no_grad and detach</h2>
<pre data-lang="python"><code>model.eval()
with torch.no_grad():
    preds = model(x)

y = x.detach()             # shares data, leaves the graph

arr = x.detach().cpu().numpy()</code></pre>

<div class="callout warn">
  <span class="tag">A common leak</span>
  <p>Appending the loss tensor itself in an eval loop keeps the whole graph alive and memory grows. Append <code>loss.item()</code> instead.</p>
</div>

<h2>6. Linear regression by hand</h2>
<pre data-lang="python"><code>import torch

torch.manual_seed(0)

N = 100
x = torch.randn(N, 1)
true_w, true_b = 3.0, 2.0
y = true_w * x + true_b + 0.1 * torch.randn(N, 1)

w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.1
for epoch in range(200):
    pred = x @ w + b
    loss = ((pred - y) ** 2).mean()

    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 50 == 0:
        print(epoch, loss.item())

print("learned w, b =", w.item(), b.item())   # close to 3.0 and 2.0</code></pre>

<p>Those three lines — forward loss, backward gradients, step against the gradient — are the skeleton of every training loop. Chapter 8 just makes it production-shaped.</p>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li>Only <code>requires_grad=True</code> float tensors are tracked.</li>
    <li><code>backward()</code> multiplies local derivatives along the graph.</li>
    <li><code>.grad</code> accumulates; call <code>zero_grad()</code> every step.</li>
    <li><code>backward()</code> needs a scalar — reduce with <code>mean()/sum()</code>.</li>
    <li>Use <code>torch.no_grad()</code> for inference, <code>.detach()</code> for values.</li>
    <li>Training triples: forward, backward, update.</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q">Calling <code>y.backward()</code> twice without clearing: after the second call <code>x.grad</code> is:</p>
  <ol class="opts">
    <li>Overwritten with the second gradient</li>
    <li>The sum of both gradients</li>
    <li>Zero</li>
    <li>An error</li>
  </ol>
  <p class="exp">Gradients accumulate. That is why zero_grad exists, and why gradient accumulation works as a technique.</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">Where is <code>torch.no_grad()</code> most needed?</p>
  <ol class="opts">
    <li>Computing the training loss</li>
    <li>Calling loss.backward()</li>
    <li>Predicting on a validation set</li>
    <li>Initialising parameters</li>
  </ol>
  <p class="exp">Inference needs no graph, saving memory and time. The training forward pass must keep it.</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">An integer tensor with <code>requires_grad=True</code> will:</p>
  <ol class="opts">
    <li>Raise an error — only float tensors can require grad</li>
    <li>Work fine</li>
    <li>Be cast to float automatically</li>
    <li>Have zero gradient</li>
  </ol>
  <p class="exp">Derivatives are continuous; integers make no sense. Cast to float32/float64.</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Differentiate <code>y = 3x³ + 2x</code> at x=2 by hand (38), then verify with autograd.</li>
    <li>Run the regression with lr = 1.0 and lr = 0.001; plot the loss and explain both behaviours.</li>
    <li>Delete <code>w.grad.zero_()</code> and observe what happens to the loss.</li>
    <li>Store <code>loss</code> instead of <code>loss.item()</code> in a list and watch memory with <code>torch.cuda.memory_allocated()</code>.</li>
  </ol>
</div>`
  }
});
