(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch07',
  num: '07',
  part: 'p2',
  file: 'loss-optimizer',
  zh: {
    title: '损失函数与优化器',
    sub: '选错这两个，模型再好也学不会',
    html: `
<p class="lead">模型给出预测，<strong>损失函数</strong>负责打分「错得多离谱」，<strong>优化器</strong>负责根据这个分数调整参数。这两个组件加起来只有十几行代码，但选错的代价极大——尤其分类任务里，用错损失函数会让准确率永远卡住。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>根据任务类型（回归/二分类/多分类）选对损失函数</li>
    <li>说清 <code>CrossEntropyLoss</code> 与 logits 的关系，不再多做一次 softmax</li>
    <li>区分 SGD / Momentum / Adam 的差异并知道默认该用哪个</li>
    <li>用学习率调度器和参数组做更细的控制</li>
  </ul>
</div>

<h2>1. 任务决定损失函数</h2>
<table>
  <tr><th>任务</th><th>输出</th><th>损失函数</th><th>备注</th></tr>
  <tr><td>回归（预测数值）</td><td>任意实数</td><td><code>nn.MSELoss</code></td><td>对异常值敏感；<code>L1Loss</code> 更稳健</td></tr>
  <tr><td>二分类</td><td>1 个 logit</td><td><code>nn.BCEWithLogitsLoss</code></td><td>内部已含 sigmoid</td></tr>
  <tr><td>多分类（互斥）</td><td>C 个 logits</td><td><code>nn.CrossEntropyLoss</code></td><td>内部已含 softmax</td></tr>
  <tr><td>多标签（可同时为真）</td><td>C 个 logits</td><td><code>nn.BCEWithLogitsLoss</code></td><td>每个标签独立 sigmoid</td></tr>
  <tr><td>排序 / 对比学习</td><td>向量</td><td><code>TripletMarginLoss</code> 等</td><td>进阶内容</td></tr>
</table>

<h2>2. 回归：MSELoss</h2>
<pre data-lang="python"><code>import torch, torch.nn as nn

pred = torch.tensor([[1.0], [2.0], [3.0]])
target = torch.tensor([[1.5], [2.0], [2.5]])

mse = nn.MSELoss()
print(mse(pred, target))     # tensor(0.1667) = mean of (0.25, 0, 0.25)
</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 为什么平方而不是绝对值</span>
  <p>MSE = <code>mean((pred - target)²)</code>。平方让误差大的样本受到更重的惩罚，而且它的导数 <code>2(pred - target)</code> 是线性的，越远梯度越大，收敛快。</p>
  <p>缺点是<strong>对离群点极其敏感</strong>：一个错得离谱的样本能主导整个 loss。数据脏的时候用 <code>L1Loss</code>（绝对值）或 <code>SmoothL1Loss</code>（Huber）更稳。</p>
</div>

<h2>3. 多分类：CrossEntropyLoss 的正确用法</h2>
<pre data-lang="python"><code># 输入：32 个样本，10 个类别，输出是原始 logits
logits = torch.randn(32, 10)
target = torch.randint(0, 10, (32,))      # 形状 (32,)，值是类别下标

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)
print(loss)                                # 一个标量

# ❌ 错误：先 softmax 再送进去，等于做了两遍
probs = torch.softmax(logits, dim=1)
# loss = criterion(probs, target)   -> 损失偏小、梯度被压缩、学得很慢</code></pre>

<div class="callout warn">
  <span class="tag">本课最重要的一条约定</span>
  <p><code>nn.CrossEntropyLoss</code> = <code>log_softmax</code> + <code>NLLLoss</code>，<strong>内部自带 softmax</strong>。所以：</p>
  <ul>
    <li>模型最后一层<strong>不要</strong>加 softmax，直接输出 logits。</li>
    <li>推理时如果要概率，再用 <code>torch.softmax(logits, dim=1)</code>，或者 <code>F.softmax</code>。</li>
    <li>target 是<strong>类别下标</strong>（int64），不是 one-hot。若你手上是 one-hot，用 <code>target.argmax(dim=1)</code> 转一下。</li>
  </ul>
</div>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p><strong>logits</strong> 是模型直接输出的「分数」，可以是负数、可以很大，没有归一化。<strong>softmax</strong> 把分数变成概率。<strong>交叉熵</strong>衡量「模型给正确答案的概率有多低」，公式是 <code>-log(p_correct)</code>。</p>
  <p>直观理解：如果模型给正确答案 0.9 的概率，<code>-log(0.9)=0.105</code>，损失小；如果只给 0.1，<code>-log(0.1)=2.3</code>，损失大。这就是为什么「提高正确类别的概率」等价于「降低损失」。</p>
</div>

<h2>4. 二分类：BCEWithLogitsLoss</h2>
<pre data-lang="python"><code># 二分类：输出 1 个 logit（不是 2 个！）
logits = torch.randn(16, 1)
target = torch.randint(0, 2, (16, 1)).float()   # 必须是 float

criterion = nn.BCEWithLogitsLoss()
print(criterion(logits, target))

# 推理时转成概率
probs = torch.sigmoid(logits)
preds = (probs &gt; 0.5).long()</code></pre>

<div class="callout tip">
  <span class="tag">二分类的两种写法</span>
  <p>写法一（推荐）：输出 1 维 + <code>BCEWithLogitsLoss</code>。<br>写法二：输出 2 维 + <code>CrossEntropyLoss</code>。两者等价，任选其一，<strong>但不要混着用</strong>（比如 1 维输出却用 CrossEntropyLoss，会因维度不符报错）。</p>
</div>

<h2>5. 优化器：从 SGD 到 Adam</h2>
<pre data-lang="python"><code>import torch.optim as optim

# 最基础：随机梯度下降
opt = optim.SGD(model.parameters(), lr=0.01)

# 加动量：像下坡时带惯性，能冲过小坑
opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam：自适应学习率，通常是默认首选
opt = optim.Adam(model.parameters(), lr=1e-3)

# AdamW：Adam + 正确的权重衰减，Transformer 类模型的标准选择
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 三种优化器差在哪</span>
  <p><strong>SGD</strong>：<code>w ← w − lr·g</code>。简单，但各方向步长一样，遇到狭长山谷会来回震荡。</p>
  <p><strong>Momentum</strong>：累积历史梯度 <code>v ← βv + g</code>，然后 <code>w ← w − lr·v</code>。方向一致的维度越走越快，来回摇摆的维度互相抵消。</p>
  <p><strong>Adam</strong>：同时估计梯度的一阶矩（均值）和二阶矩（方差），并对每个参数<strong>单独</strong>缩放步长：<code>w ← w − lr · m̂ / (√v̂ + ε)</code>。梯度大的参数步子自动变小，梯度小的自动变大，所以对学习率不那么敏感。</p>
  <p>经验：CNN 从 SGD + momentum 起手常常泛化更好；Transformer/微调用 AdamW；不确定就先用 Adam(1e-3)。</p>
</div>

<h2>6. 学习率：最重要的超参数</h2>
<pre data-lang="python"><code># 手动调（最直接）
for g in opt.param_groups:
    g['lr'] = 1e-4

# 调度器：按计划自动降
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

for epoch in range(50):
    train_one_epoch()
    sched.step()          # 每个 epoch 后更新一次 lr
    print(opt.param_groups[0]['lr'])</code></pre>

<div class="callout warn">
  <span class="tag">调参第一优先级</span>
  <p>如果 loss 不动，先看学习率：太大 → loss 震荡/变 NaN；太小 → loss 下降极慢。常用的试法是从 <code>1e-3</code> 开始，按 3 倍或 10 倍上下扫（<code>3e-4, 1e-3, 3e-3, 1e-2</code>）。</p>
  <p>另外注意：换优化器、改 batch size 之后，最优学习率会变。batch 变大通常可以把 lr 按比例调大。</p>
</div>

<h2>7. 参数组：不同层用不同学习率</h2>
<pre data-lang="python"><code># 微调预训练模型时的标准写法
optimizer = optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},   # 预训练部分，小心动
    {"params": model.head.parameters(),     "lr": 1e-3},   # 新加的头，放心动
], weight_decay=0.01)</code></pre>
<p>这叫<strong>判别式微调</strong>（discriminative fine-tuning）：底层特征通用，改太快会毁掉；顶层任务相关，需要快速适应。</p>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li>回归用 MSE，多分类用 CrossEntropy，二分类用 BCEWithLogits。</li>
    <li><code>CrossEntropyLoss</code> 内含 softmax，模型输出 logits，别再套 softmax。</li>
    <li>CE 的 target 是类别下标（int64），不是 one-hot。</li>
    <li>优化器参数是 <code>model.parameters()</code>，别忘了传。</li>
    <li>Adam/AdamW 是默认首选，CNN 可试 SGD + momentum。</li>
    <li>学习率比优化器种类更重要，先调它。</li>
    <li>参数组能让不同层用不同 lr，微调时必备。</li>
  </ol>
</div>

<div class="quiz" data-answer="2">
  <p class="q">10 分类任务，模型最后输出 10 个 logits。应该？</p>
  <ol class="opts">
    <li>先 softmax，再送 CrossEntropyLoss</li>
    <li>先 sigmoid，再送 MSELoss</li>
    <li>直接把 logits 送 CrossEntropyLoss</li>
    <li>先 argmax，再送 CrossEntropyLoss</li>
  </ol>
  <p class="exp">CrossEntropyLoss 内部做 log_softmax，所以喂 logits。多套一次 softmax 会导致梯度被压缩，训练明显变慢。</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q">CrossEntropyLoss 的 target 应该是什么格式？</p>
  <ol class="opts">
    <li>(N, C) 的 one-hot 浮点张量</li>
    <li>(N,) 的 int64 类别下标</li>
    <li>(N, 1) 的浮点概率</li>
    <li>任意格式都行</li>
  </ol>
  <p class="exp">默认接受类别下标。若手上是 one-hot，用 argmax 转成下标，或改用带 label_smoothing 的方式。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">训练时 loss 在 3.0 附近剧烈跳动甚至出现 NaN，最该先检查？</p>
  <ol class="opts">
    <li>学习率是否太大</li>
    <li>模型层数是否太少</li>
    <li>是否忘了 import numpy</li>
    <li>数据集是否太小</li>
  </ol>
  <p class="exp">学习率过大是 loss 爆炸最常见的原因。降 10 倍再看，通常立刻稳定。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>手算 <code>MSELoss</code>：pred=[1,2,3]，target=[1.5,2,2.5]，验证输出 0.1667。</li>
    <li>构造一批 logits 和 label，分别用「logits 进 CE」和「softmax 后进 CE」，比较 loss 数值差异，解释为什么。</li>
    <li>同一个模型分别用 SGD(lr=0.1)、SGD+momentum、Adam(lr=1e-3) 训练 5 个 epoch，比较 loss 曲线。</li>
    <li>用 <code>CosineAnnealingLR</code> 打印 10 个 epoch 的学习率变化。</li>
  </ol>
</div>`
  },
  en: {
    title: 'Loss functions and optimisers',
    sub: 'Pick these wrong and no architecture will save you',
    html: `
<p class="lead">The model predicts, the <strong>loss</strong> scores how wrong it is, the <strong>optimiser</strong> adjusts parameters from that score. Together they are a dozen lines — and choosing wrongly caps your accuracy forever.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Pick the right loss for regression / binary / multi-class</li>
    <li>Explain <code>CrossEntropyLoss</code> and logits, and stop double-softmaxing</li>
    <li>Tell SGD, Momentum and Adam apart, and know the default</li>
    <li>Use schedulers and parameter groups</li>
  </ul>
</div>

<h2>1. Task decides the loss</h2>
<table>
  <tr><th>Task</th><th>Output</th><th>Loss</th><th>Note</th></tr>
  <tr><td>Regression</td><td>any real</td><td><code>nn.MSELoss</code></td><td>sensitive to outliers; <code>L1Loss</code> is robust</td></tr>
  <tr><td>Binary</td><td>1 logit</td><td><code>nn.BCEWithLogitsLoss</code></td><td>includes sigmoid</td></tr>
  <tr><td>Multi-class</td><td>C logits</td><td><code>nn.CrossEntropyLoss</code></td><td>includes softmax</td></tr>
  <tr><td>Multi-label</td><td>C logits</td><td><code>nn.BCEWithLogitsLoss</code></td><td>independent sigmoid per label</td></tr>
</table>

<h2>2. Regression: MSELoss</h2>
<pre data-lang="python"><code>import torch, torch.nn as nn

pred = torch.tensor([[1.0], [2.0], [3.0]])
target = torch.tensor([[1.5], [2.0], [2.5]])

print(nn.MSELoss()(pred, target))     # tensor(0.1667)</code></pre>

<div class="callout math">
  <span class="tag">Math note · why square</span>
  <p>MSE = <code>mean((pred - target)²)</code>. Squaring punishes large errors harder and its derivative <code>2(pred - target)</code> grows with distance, so learning speeds up when far off.</p>
  <p>Downside: a single wild outlier can dominate. For noisy labels use <code>L1Loss</code> or <code>SmoothL1Loss</code>.</p>
</div>

<h2>3. Multi-class: the right way</h2>
<pre data-lang="python"><code>logits = torch.randn(32, 10)
target = torch.randint(0, 10, (32,))      # (32,) class indices

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)

# WRONG: softmax first — that applies it twice
probs = torch.softmax(logits, dim=1)
# criterion(probs, target)  -> shrunken loss, compressed gradients</code></pre>

<div class="callout warn">
  <span class="tag">The most important convention here</span>
  <p><code>CrossEntropyLoss</code> = <code>log_softmax</code> + <code>NLLLoss</code>. Therefore:</p>
  <ul>
    <li>No softmax in the model — output raw logits.</li>
    <li>At inference, apply <code>softmax</code> only if you want probabilities.</li>
    <li>targets are <strong>class indices</strong> (int64), not one-hot. Convert with <code>argmax(dim=1)</code> if needed.</li>
  </ul>
</div>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p><strong>Logits</strong> are raw scores. <strong>Softmax</strong> turns them into probabilities. <strong>Cross-entropy</strong> is <code>-log(p_correct)</code>: giving the right class 0.9 costs 0.105, giving it 0.1 costs 2.3. Lower loss means higher probability on the truth.</p>
</div>

<h2>4. Binary: BCEWithLogitsLoss</h2>
<pre data-lang="python"><code>logits = torch.randn(16, 1)
target = torch.randint(0, 2, (16, 1)).float()   # must be float

criterion = nn.BCEWithLogitsLoss()
print(criterion(logits, target))

probs = torch.sigmoid(logits)
preds = (probs &gt; 0.5).long()</code></pre>

<div class="callout tip">
  <span class="tag">Two valid styles</span>
  <p>One logit + BCEWithLogits, or two logits + CrossEntropy. Both are fine; mixing them is not.</p>
</div>

<h2>5. Optimisers</h2>
<pre data-lang="python"><code>import torch.optim as optim

opt = optim.SGD(model.parameters(), lr=0.01)
opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
opt = optim.Adam(model.parameters(), lr=1e-3)
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)</code></pre>

<div class="callout math">
  <span class="tag">Math note · what differs</span>
  <p><strong>SGD</strong>: <code>w ← w − lr·g</code>. Simple, but one step size for all directions, so it zig-zags in narrow valleys.</p>
  <p><strong>Momentum</strong>: accumulate <code>v ← βv + g</code>, then <code>w ← w − lr·v</code>. Consistent directions accelerate, oscillating ones cancel.</p>
  <p><strong>Adam</strong>: estimate first and second moments and scale each parameter's step: <code>w ← w − lr · m̂ / (√v̂ + ε)</code>. Large gradients get smaller steps automatically, so it is less LR-sensitive.</p>
  <p>Rule of thumb: CNNs often generalise better with SGD+momentum; Transformers and fine-tuning use AdamW; when unsure start with Adam 1e-3.</p>
</div>

<h2>6. Learning rate: the hyperparameter that matters</h2>
<pre data-lang="python"><code>for g in opt.param_groups:
    g['lr'] = 1e-4

sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
for epoch in range(50):
    train_one_epoch()
    sched.step()</code></pre>

<div class="callout warn">
  <span class="tag">Debug this first</span>
  <p>Loss not moving? Check the learning rate. Too large → oscillation or NaN. Too small → glacial progress. Sweep <code>3e-4, 1e-3, 3e-3, 1e-2</code>. Also note the optimum shifts when you change optimiser or batch size (larger batch usually allows a larger LR).</p>
</div>

<h2>7. Parameter groups</h2>
<pre data-lang="python"><code>optimizer = optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},   # pretrained: gentle
    {"params": model.head.parameters(),     "lr": 1e-3},   # new head: fast
], weight_decay=0.01)</code></pre>
<p>Discriminative fine-tuning: low-level features are generic and easy to destroy; the new head must adapt quickly.</p>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li>MSE for regression, CrossEntropy for multi-class, BCEWithLogits for binary.</li>
    <li>CrossEntropy includes softmax — feed logits.</li>
    <li>CE targets are int64 class indices.</li>
    <li>Always pass <code>model.parameters()</code> to the optimiser.</li>
    <li>Adam/AdamW by default; SGD+momentum for many CNNs.</li>
    <li>The learning rate matters more than the optimiser choice.</li>
    <li>Parameter groups let layers use different LRs.</li>
  </ol>
</div>

<div class="quiz" data-answer="2">
  <p class="q">10-class model outputs 10 logits. You should:</p>
  <ol class="opts">
    <li>Softmax first, then CrossEntropyLoss</li>
    <li>Sigmoid first, then MSELoss</li>
    <li>Feed logits directly to CrossEntropyLoss</li>
    <li>Argmax first, then CrossEntropyLoss</li>
  </ol>
  <p class="exp">CE applies log_softmax internally. Softmaxing first compresses gradients and slows learning.</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q">What target format does CrossEntropyLoss expect?</p>
  <ol class="opts">
    <li>(N, C) one-hot floats</li>
    <li>(N,) int64 class indices</li>
    <li>(N, 1) probabilities</li>
    <li>Anything</li>
  </ol>
  <p class="exp">Class indices by default.</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">Loss jumps around 3.0 and turns into NaN. Check first:</p>
  <ol class="opts">
    <li>Whether the learning rate is too large</li>
    <li>Whether the model is too shallow</li>
    <li>Whether numpy is imported</li>
    <li>Whether the dataset is too small</li>
  </ol>
  <p class="exp">Exploding loss is almost always LR. Divide by 10 and retry.</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Verify MSELoss by hand: pred=[1,2,3], target=[1.5,2,2.5] → 0.1667.</li>
    <li>Compare CE on logits vs CE on softmax(logits); explain the difference.</li>
    <li>Train the same model with SGD 0.1, SGD+momentum and Adam 1e-3 for 5 epochs; compare curves.</li>
    <li>Print the LR across 10 epochs with CosineAnnealingLR.</li>
  </ol>
</div>`
  }
});
