(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch09',
  num: '09',
  part: 'p2',
  file: 'regularization',
  zh: {
    title: '过拟合与正则化',
    sub: '模型在训练集上 99%、测试集上 70%，怎么救',
    html: `
<p class="lead">训练集分数高不代表模型好，只代表它<strong>背下了答案</strong>。真本事是在没见过的数据上表现好。这一章讲清楚「为什么会背答案」，以及七种让它别背的办法。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>用偏差-方差的语言解释欠拟合与过拟合</li>
    <li>正确使用 Dropout、weight decay、BatchNorm、数据增强、早停</li>
    <li>知道 <code>model.train()</code> 与 <code>model.eval()</code> 为什么影响 Dropout</li>
    <li>按优先级顺序调参，而不是乱试</li>
  </ul>
</div>

<h2>1. 两张图看懂偏差与方差</h2>
<table>
  <tr><th></th><th>欠拟合（高偏差）</th><th>过拟合（高方差）</th></tr>
  <tr><td>训练 loss</td><td>高</td><td>很低</td></tr>
  <tr><td>验证 loss</td><td>高</td><td>先降后升</td></tr>
  <tr><td>直觉</td><td>没学会</td><td>背下来了</td></tr>
  <tr><td>解法</td><td>更大模型、训练更久、更好特征</td><td>更多数据、正则化、早停</td></tr>
</table>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p>想象准备考试。欠拟合是「书都没看」，什么题都不会；过拟合是「把练习册答案全背了」，练习题全对，换一道新题就崩。</p>
  <p>我们要的是「理解了原理」，这样新题也能做——这就是泛化（generalization）。</p>
</div>

<h2>2. Dropout：随机让神经元「请假」</h2>
<pre data-lang="python"><code>import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(128, 10),
)</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · Dropout 到底在做什么</span>
  <p>训练时，每个神经元以概率 <code>p</code> 被临时置零，且保留下来的值会除以 <code>(1-p)</code> 以保持期望不变（这叫 inverted dropout）。</p>
  <p>它有效的解释之一：每次前向都相当于在训练一个「随机删减版」的子网络，而所有子网络共享参数，相当于<strong>集成学习</strong>。另一个解释：它阻止神经元之间形成「固定搭档」，逼每个神经元独立有用。</p>
  <p><strong>推理时必须关掉</strong>（<code>model.eval()</code>），否则输出是随机的。</p>
</div>

<div class="callout warn">
  <span class="tag">常见错误</span>
  <p>在最后一层之前加 Dropout 通常有害（直接丢标签信息）；Dropout 一般放在隐藏层激活之后。CNN 里更常用 BatchNorm + 数据增强，Dropout 用在小模型的全连接层。Transformer 里 p=0.1 是标配。</p>
</div>

<h2>3. Weight decay：不让权重长得太大</h2>
<pre data-lang="python"><code># 方式一：交给优化器（推荐）
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 方式二：手动加 L2 项（理解用，实际少写）
l2 = sum((p ** 2).sum() for p in model.parameters())
loss = criterion(logits, y) + 1e-4 * l2</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · L2 正则</span>
  <p>原始目标是 <code>min L(w)</code>，加上正则后变成 <code>min L(w) + λ‖w‖²</code>。</p>
  <p>对 w 求导会多出一项 <code>2λw</code>，于是更新式变成 <code>w ← w(1 − 2λlr) − lr·g</code>：每步都先把 w 缩一点。所以 weight decay 的字面意思就是「权重衰减」。</p>
  <p>为什么权重小更不容易过拟合？因为输出对输入的微小变化不敏感，函数更平滑，不会为了迁就个别噪声点而剧烈扭曲。</p>
  <p>注意 <code>AdamW</code> 与 <code>Adam(weight_decay=...)</code> 的实现不同：AdamW 把衰减和梯度更新解耦，效果更符合预期，所以现在更推荐。</p>
</div>

<h2>4. BatchNorm：让每层输入分布稳定</h2>
<pre data-lang="python"><code>nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.BatchNorm2d(32),           # 放在卷积后、激活前（常见约定）
    nn.ReLU(),
)</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · BN 做了什么</span>
  <p>对一个 batch 内每个通道，计算均值 μ 和方差 σ²，做 <code>(x−μ)/√(σ²+ε)</code>，再用两个<strong>可学习</strong>参数 γ、β 缩放平移：<code>y = γ·x̂ + β</code>。</p>
  <p>好处：让每层输入保持在稳定范围，可以用更大学习率、收敛更快，还有轻微的正则效果。</p>
  <p>注意训练/推理的差异：训练用当前 batch 的 μ、σ²，同时用滑动平均记录全局统计量；推理时用滑动平均。<strong>这就是必须切 eval 模式的原因。</strong></p>
</div>

<div class="callout warn">
  <span class="tag">batch 太小时别用 BN</span>
  <p>batch=1 时方差为 0，BN 会失效。小 batch 场景（如大模型微调）改用 <code>LayerNorm</code> / <code>GroupNorm</code>。</p>
</div>

<h2>5. 数据增强：最划算的正则化</h2>
<pre data-lang="python"><code>from torchvision import transforms

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
])</code></pre>
<p>它等于「免费造更多数据」，而且造出来的样本符合真实世界的变化。通常比调正则化系数更有效——<strong>如果能加数据，优先加数据。</strong></p>

<div class="callout warn">
  <span class="tag">增强不能破坏语义</span>
  <p>数字识别里做水平翻转会把「2」变成奇怪的东西；医学影像里随意旋转可能不合理。增强方式必须符合任务的真实不变性。</p>
</div>

<h2>6. 早停：最省事的正则化</h2>
<p>过拟合的模型在验证集上有一条 U 形曲线。早停就是「在谷底附近停下」，配合保存最佳 checkpoint，几乎零成本。</p>
<p>它本质上是「限制训练时间」这种正则化，和 L2 有相似的理论解释。</p>

<h2>7. 调参优先级（重要）</h2>
<ol class="steps">
  <li><strong>先让训练能过拟合。</strong>用小子集训练，确认模型有能力学会。学不会说明架构或代码有问题。</li>
  <li><strong>再治过拟合。</strong>按性价比：更多数据 &gt; 数据增强 &gt; 早停 &gt; weight decay &gt; Dropout。</li>
  <li><strong>再调学习率与调度。</strong>它的收益往往比换模型大。</li>
  <li><strong>最后才换模型架构。</strong>换架构成本最高，收益最不确定。</li>
</ol>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li>过拟合 = 训练好、验证差；欠拟合 = 两边都差。</li>
    <li>Dropout 训练时随机置零并缩放，推理时必须关闭。</li>
    <li>weight decay 就是 L2 正则，AdamW 的实现更正确。</li>
    <li>BatchNorm 训练用 batch 统计、推理用滑动平均 → 必须切 eval。</li>
    <li>数据增强是最划算的正则化，但不能破坏语义。</li>
    <li>调参顺序：先过拟合，再治过拟合，再调 lr，最后换架构。</li>
  </ol>
</div>

<div class="quiz" data-answer="2">
  <p class="q">训练准确率 99.5%，验证准确率 72%。最不该做的是？</p>
  <ol class="opts">
    <li>增加数据增强</li>
    <li>加 Dropout 或 weight decay</li>
    <li>把模型改得更大更深</li>
    <li>使用早停</li>
  </ol>
  <p class="exp">这是典型过拟合，加大模型只会让它背得更牢。应该加约束或加数据。</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q">评估时忘记 <code>model.eval()</code>，Dropout 层会？</p>
  <ol class="opts">
    <li>自动关闭</li>
    <li>继续随机丢弃神经元，导致输出不稳定、指标偏低</li>
    <li>报错</li>
    <li>变成恒等映射</li>
  </ol>
  <p class="exp">Dropout 的行为由 module 的 training 标志决定，eval() 才会关闭它。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">weight decay 的作用机制是？</p>
  <ol class="opts">
    <li>每步把权重往 0 缩一点，限制模型复杂度</li>
    <li>随机删除神经元</li>
    <li>加快学习率</li>
    <li>增加训练数据</li>
  </ol>
  <p class="exp">L2 正则的导数项是 2λw，等价于每步做一次乘性收缩。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>用 MNIST 的一个小子集（比如每类 100 张）训练，观察明显过拟合的曲线。</li>
    <li>依次加 Dropout(0.5)、weight_decay=1e-4、数据增强，记录三种情况下验证准确率的变化。</li>
    <li>把 batch_size 改成 1 使用 BatchNorm2d，观察报错或异常（提示：训练模式下会报错）。</li>
    <li>用 <code>model.train()</code> 对同一输入连续前向 3 次（含 Dropout），看输出是否变化；再切 eval 试一次。</li>
  </ol>
</div>`
  },
  en: {
    title: 'Overfitting and regularisation',
    sub: '99% on train, 70% on test — how to fix it',
    html: `
<p class="lead">High training accuracy does not mean a good model; it often means the model <strong>memorised the answers</strong>. Real skill is performance on unseen data. This chapter explains why memorisation happens and seven ways to stop it.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Explain under/overfitting in bias-variance terms</li>
    <li>Apply dropout, weight decay, BatchNorm, augmentation and early stopping correctly</li>
    <li>Explain why <code>train()</code>/<code>eval()</code> changes Dropout and BN</li>
    <li>Tune in a sensible order instead of guessing</li>
  </ul>
</div>

<h2>1. Two failure modes</h2>
<table>
  <tr><th></th><th>Underfitting (high bias)</th><th>Overfitting (high variance)</th></tr>
  <tr><td>Train loss</td><td>high</td><td>very low</td></tr>
  <tr><td>Val loss</td><td>high</td><td>falls then rises</td></tr>
  <tr><td>Intuition</td><td>did not learn</td><td>memorised</td></tr>
  <tr><td>Fix</td><td>bigger model, train longer, better features</td><td>more data, regularise, early stop</td></tr>
</table>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p>Underfitting is not reading the book at all. Overfitting is memorising the answer key: perfect on practice questions, useless on a new one. We want understanding, which is what <em>generalisation</em> means.</p>
</div>

<h2>2. Dropout</h2>
<pre data-lang="python"><code>model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(128, 10),
)</code></pre>

<div class="callout math">
  <span class="tag">Math note · what dropout does</span>
  <p>During training each unit is zeroed with probability <code>p</code>, and survivors are divided by <code>(1-p)</code> so the expectation is unchanged (inverted dropout).</p>
  <p>Interpretation: each forward pass trains a random sub-network, and all sub-networks share weights — a form of ensembling. It also prevents neurons from co-adapting.</p>
  <p><strong>Must be off at inference</strong>, otherwise outputs are random.</p>
</div>

<div class="callout warn">
  <span class="tag">Common mistakes</span>
  <p>Dropout right before the output layer usually hurts. In CNNs, BatchNorm plus augmentation is more common; dropout shines in fully connected heads and Transformers (p=0.1).</p>
</div>

<h2>3. Weight decay</h2>
<pre data-lang="python"><code>optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# manual L2, for understanding
l2 = sum((p ** 2).sum() for p in model.parameters())
loss = criterion(logits, y) + 1e-4 * l2</code></pre>

<div class="callout math">
  <span class="tag">Math note · L2 regularisation</span>
  <p><code>min L(w) + λ‖w‖²</code>. Differentiating adds <code>2λw</code>, so the update becomes <code>w ← w(1 − 2λlr) − lr·g</code>: a shrink factor every step.</p>
  <p>Small weights make the function smoother — less able to contort itself around individual noisy points.</p>
  <p><code>AdamW</code> decouples the decay from the adaptive gradient, which is why it is preferred over <code>Adam(weight_decay=...)</code>.</p>
</div>

<h2>4. BatchNorm</h2>
<pre data-lang="python"><code>nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
)</code></pre>

<div class="callout math">
  <span class="tag">Math note · what BN does</span>
  <p>Per channel over the batch: <code>(x−μ)/√(σ²+ε)</code>, then a learnable affine <code>y = γ·x̂ + β</code>.</p>
  <p>It stabilises each layer's input distribution, allowing larger learning rates and faster convergence, with a mild regularising effect.</p>
  <p>Train uses batch statistics and updates running averages; inference uses the averages. <strong>Hence <code>eval()</code>.</strong></p>
</div>

<div class="callout warn">
  <span class="tag">Avoid BN with tiny batches</span>
  <p>With batch=1 the variance is zero and training raises an error. Use LayerNorm/GroupNorm instead.</p>
</div>

<h2>5. Data augmentation</h2>
<pre data-lang="python"><code>train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
])</code></pre>
<p>Free extra data that respects real-world variation. Usually more effective than tuning regularisation strength — <strong>if you can add data, add data.</strong></p>

<div class="callout warn">
  <span class="tag">Do not break semantics</span>
  <p>Flipping digits is wrong; arbitrary rotations may be wrong in medical imaging. Augmentation must match the invariances your task actually has.</p>
</div>

<h2>6. Early stopping</h2>
<p>Validation loss is U-shaped; stop near the bottom and keep the best checkpoint. Nearly free, and theoretically a form of restricting training time.</p>

<h2>7. Tuning order</h2>
<ol class="steps">
  <li><strong>Make it able to overfit.</strong> Train on a small subset; if it cannot memorise, the architecture or code is broken.</li>
  <li><strong>Then fight overfitting</strong>, in order of value: more data &gt; augmentation &gt; early stop &gt; weight decay &gt; dropout.</li>
  <li><strong>Then tune LR and schedule</strong> — usually a bigger win than swapping architectures.</li>
  <li><strong>Only then change architecture.</strong></li>
</ol>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li>Overfit: train good, val bad. Underfit: both bad.</li>
    <li>Dropout zeroes and rescales during training, must be off at inference.</li>
    <li>Weight decay is L2; AdamW implements it correctly.</li>
    <li>BN uses batch stats in training, running averages in eval.</li>
    <li>Augmentation is the best value regulariser, but must preserve semantics.</li>
    <li>Order: overfit first, then regularise, then tune LR, then change architecture.</li>
  </ol>
</div>

<div class="quiz" data-answer="2">
  <p class="q">Train 99.5%, validation 72%. What should you NOT do?</p>
  <ol class="opts">
    <li>Add augmentation</li>
    <li>Add dropout or weight decay</li>
    <li>Make the model bigger and deeper</li>
    <li>Use early stopping</li>
  </ol>
  <p class="exp">That is overfitting; a bigger model memorises harder.</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q">Forgetting <code>model.eval()</code> means Dropout will:</p>
  <ol class="opts">
    <li>Switch itself off</li>
    <li>Keep dropping units, making outputs unstable and metrics low</li>
    <li>Raise an error</li>
    <li>Become identity</li>
  </ol>
  <p class="exp">Behaviour follows the module's training flag.</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">Weight decay works by:</p>
  <ol class="opts">
    <li>Shrinking weights slightly each step, limiting complexity</li>
    <li>Randomly deleting units</li>
    <li>Increasing the learning rate</li>
    <li>Adding training data</li>
  </ol>
  <p class="exp">The L2 term contributes 2λw, a multiplicative shrink.</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Train on a small MNIST subset (100 per class) and observe clear overfitting.</li>
    <li>Add dropout 0.5, then weight_decay 1e-4, then augmentation; record validation accuracy each time.</li>
    <li>Try BatchNorm2d with batch_size=1 in train mode and read the error.</li>
    <li>Forward the same input three times with dropout in train mode, then in eval mode, and compare.</li>
  </ol>
</div>`
  }
});
