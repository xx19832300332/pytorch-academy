/* 常见问题（#/faq） */
window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] };

window.PYTORCH_COURSE.faq = {
  zh: {
    kicker: '常见问题 · 新手高频踩坑',
    title: 'FAQ：新手最常卡住的 14 个点',
    sub: '报错原文 → 为什么会这样 → 怎么改',
    html: `
<p class="lead">这些问题来自真实的新手求助，按「环境 → 张量形状 → 训练 → 效果 → 学习方法」分组。每条都给了<b>报错原文</b>，因为排查问题的第一步永远是看懂报错在说什么。</p>

<h2>1. 环境与安装</h2>

<h3>Q：<code>ModuleNotFoundError: No module named 'torch'</code></h3>
<p><b>原因</b>：装到了另一个 Python 解释器上。你机器上往往有系统 Python、conda 环境、venv 三套，<code>pip</code> 和 <code>python</code> 未必是同一个。</p>
<pre data-lang="python"><code>import sys
print(sys.executable)      # 当前 notebook 用的是哪个 python
print(sys.version)</code></pre>
<p>拿这个路径对应的 pip 重装：<code>python -m pip install torch</code>。用 <code>python -m pip</code> 而不是裸 <code>pip</code>，就不会装错地方。</p>

<h3>Q：<code>torch.cuda.is_available()</code> 一直是 False</h3>
<p><b>原因</b>：三种可能——没 N 卡、装了 CPU 版、驱动太旧。先用 <code>nvidia-smi</code> 看驱动是否正常；能显示显卡型号但 torch 仍为 False，多半是装了 CPU 版，去官网用 CUDA 版本命令重装。</p>
<p><b>不影响学习</b>：本课程第 1–12、14–16 章都能在 CPU 上跑完，只有第 13 章需要 GPU。</p>

<h3>Q：Jupyter 里改了代码，结果没变化</h3>
<p><b>原因</b>：你跑的不是你以为的那一格，或者变量是上一格留下的。永远先 <b>Kernel → Restart &amp; Run All</b>，这也是唯一可靠的排查起点。</p>

<h2>2. 张量与形状</h2>

<h3>Q：<code>RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x1568 and 3136x128)</code></h3>
<p><b>原因</b>：全连接层的输入维度写错了——这是最高频的报错。括号里告诉你：你喂进来的是 1568 维，但这层期望 3136 维。</p>
<p><b>怎么改</b>：倒推。卷积 + 池化之后的特征图尺寸要自己算 <code>⌊(H+2P−K)/S⌋+1</code>，再乘通道数。第 10 章有交互计算器，别硬记。</p>
<pre data-lang="python"><code># 最快的办法：先让模型只跑一次，把形状打出来
x = torch.randn(1, 1, 28, 28)
h = model.features(x)
print(h.shape)          # 照着这个数改 Linear 的输入维度</code></pre>

<h3>Q：<code>RuntimeError: Expected all tensors to be on the same device</code></h3>
<p><b>原因</b>：模型在 GPU 上，数据还在 CPU（或反过来）。<code>model.to(device)</code> 不会自动搬运后续输入。</p>
<p><b>怎么改</b>：每个 batch 都 <code>x, y = x.to(device), y.to(device)</code>；新建的张量（如 mask、位置编码）也要 <code>.to(device)</code>。</p>

<h3>Q：<code>view size is not compatible with input tensor's size and stride</code></h3>
<p><b>原因</b>：<code>view</code> 要求内存连续，<code>transpose</code>/<code>permute</code> 之后不连续。</p>
<p><b>怎么改</b>：改用 <code>reshape</code>（必要时自动复制），或先 <code>.contiguous()</code> 再 <code>view</code>。</p>

<h2>3. 训练过程</h2>

<h3>Q：loss 不下降，或一开始就是 NaN</h3>
<table>
  <thead><tr><th>检查项</th><th>典型错误</th></tr></thead>
  <tbody>
    <tr><td>学习率</td><td>用了 1.0 这种量级；先试 1e-3 / 1e-4</td></tr>
    <tr><td>是否 zero_grad</td><td>漏了，梯度累积导致爆炸</td></tr>
    <tr><td>输入是否标准化</td><td>像素 0–255 直接喂，梯度必然炸</td></tr>
    <tr><td>标签范围</td><td>CrossEntropyLoss 要求标签是 0..C-1 的 long</td></tr>
    <tr><td>损失函数选错</td><td>分类用了 MSELoss，或多分类用了 BCELoss</td></tr>
  </tbody>
</table>

<h3>Q：<code>RuntimeError: expected scalar type Long but found Float</code></h3>
<p><b>原因</b>：分类任务的标签必须是 <code>torch.long</code>，你的是 float。<code>y = y.long()</code> 即可。反过来，模型输入必须是 float：<code>x = x.float()</code>。</p>

<h3>Q：<code>CUDA out of memory</code></h3>
<p>按代价从低到高试：减小 <code>batch_size</code> → 用 <code>with torch.no_grad()</code> 包住验证 → <code>del</code> 掉不用的中间变量 → 梯度累积（小 batch 多步）→ 混合精度 AMP → 梯度检查点。</p>

<h3>Q：训练集准确率 99%，测试集只有 70%</h3>
<p><b>这是过拟合，不是 bug</b>。按顺序加：数据增强 → dropout → weight decay → 早停 → 减小模型。第 9 章有完整的判断表和手段清单。</p>

<h2>4. 效果与部署</h2>

<h3>Q：模型训练好了，怎么给别人用？</h3>
<pre data-lang="python"><code>torch.save(model.state_dict(), "model.pt")          # 存参数

model = MyNet()                                      # 对方要有同一个类
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()                                         # 必须</code></pre>
<p>把「模型类定义 + 词表/预处理配置 + 权重文件」三样一起交付，缺一样都跑不起来。</p>

<h3>Q：同一张图，每次预测结果都不一样</h3>
<p><b>原因</b>：忘了 <code>model.eval()</code>，dropout 还在随机丢神经元。加上就稳定了。</p>

<h2>5. 学习方法</h2>

<h3>Q：看得懂，但自己写不出来</h3>
<p><b>原因</b>：看懂是「识别」，写出来是「回忆」，两者的大脑活动完全不同。办法只有一个：<b>关掉教程，从空白文件重写一遍</b>。写不出来就说明哪一步没真懂，回去只看那一步。</p>

<h3>Q：数学基础差，能学吗</h3>
<p>能。本课程所有公式都配了直觉解释，你至少需要：会看矩阵乘法、知道导数是「变化率」、理解概率之和为 1。这三样在预备章和第 2 章都补过了。遇到推不动的公式，先记住它「在干什么」，比背推导有用得多。</p>

<h3>Q：学完这些能找到工作吗</h3>
<p>这些是<b>入场券</b>，不是全部。真实岗位还要求：能读懂论文、能清洗脏数据、能调优到指标达标、能把模型部署上线。建议路径：学完本课程 → 用真实数据集独立做一个项目（从找数据到部署）→ 复现一篇简单论文。</p>

<div class="keypoints">
  <h4>遇到报错的通用四步</h4>
  <ul>
    <li>读报错最后一行，它通常直接说了哪里不对</li>
    <li>看形状：几乎所有 PyTorch 报错都能用 print(x.shape) 定位</li>
    <li>看设备：CPU / CUDA 是否一致</li>
    <li>最小复现：把代码删到只剩出错那两句，问题往往自己就暴露了</li>
  </ul>
</div>
`
  },
  en: {
    kicker: 'FAQ · the traps beginners hit most',
    title: 'FAQ: 14 things that block beginners',
    sub: 'The actual error, why it happens, how to fix it',
    html: `
<p class="lead">Collected from real beginner questions, grouped as environment → shapes → training → results → how to study. Each entry quotes the <b>real error text</b>, because debugging always starts with reading what the error says.</p>

<h2>1. Environment</h2>

<h3>Q: <code>ModuleNotFoundError: No module named 'torch'</code></h3>
<p><b>Cause</b>: installed into a different interpreter. Machines usually carry a system Python, a conda env and a venv — <code>pip</code> and <code>python</code> need not be the same one.</p>
<pre data-lang="python"><code>import sys
print(sys.executable)      # which python this notebook runs
print(sys.version)</code></pre>
<p>Install with that interpreter: <code>python -m pip install torch</code>. Using <code>python -m pip</code> instead of bare <code>pip</code> prevents the mismatch.</p>

<h3>Q: <code>torch.cuda.is_available()</code> is always False</h3>
<p><b>Cause</b>: no NVIDIA GPU, a CPU-only build, or an old driver. Run <code>nvidia-smi</code> first; if the card shows up but torch says False, you installed the CPU build.</p>
<p><b>Not blocking</b>: chapters 1–12 and 14–16 run on CPU; only chapter 13 wants a GPU.</p>

<h3>Q: I changed the code but the result did not change</h3>
<p><b>Cause</b>: you ran a different cell, or stale variables. Always <b>Kernel → Restart &amp; Run All</b> first.</p>

<h2>2. Tensors and shapes</h2>

<h3>Q: <code>RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x1568 and 3136x128)</code></h3>
<p><b>Cause</b>: wrong input dimension on a linear layer — the most common error. The brackets tell you: you fed 1568 features, the layer expects 3136.</p>
<pre data-lang="python"><code>x = torch.randn(1, 1, 28, 28)
print(model.features(x).shape)     # read the real number, fix the Linear</code></pre>

<h3>Q: <code>Expected all tensors to be on the same device</code></h3>
<p><b>Cause</b>: model on GPU, data on CPU (or vice versa). <code>model.to(device)</code> does not move later inputs.</p>
<p><b>Fix</b>: move every batch — <code>x, y = x.to(device), y.to(device)</code> — and any tensor you create later (masks, position encodings).</p>

<h3>Q: <code>view size is not compatible with input tensor's size and stride</code></h3>
<p><b>Cause</b>: <code>view</code> needs contiguous memory; <code>transpose</code>/<code>permute</code> break that.</p>
<p><b>Fix</b>: use <code>reshape</code> (copies if needed) or call <code>.contiguous()</code> before <code>view</code>.</p>

<h2>3. Training</h2>

<h3>Q: Loss does not drop, or turns NaN immediately</h3>
<table>
  <thead><tr><th>Check</th><th>Typical mistake</th></tr></thead>
  <tbody>
    <tr><td>Learning rate</td><td>Using 1.0 scale; start at 1e-3 or 1e-4</td></tr>
    <tr><td>zero_grad</td><td>Missing, so gradients accumulate and explode</td></tr>
    <tr><td>Normalisation</td><td>Feeding raw 0–255 pixels</td></tr>
    <tr><td>Label range</td><td>CrossEntropyLoss wants long labels in 0..C-1</td></tr>
    <tr><td>Wrong loss</td><td>MSELoss for classification, BCELoss for multiclass</td></tr>
  </tbody>
</table>

<h3>Q: <code>expected scalar type Long but found Float</code></h3>
<p><b>Cause</b>: classification labels must be <code>torch.long</code>. Use <code>y = y.long()</code>. Conversely model inputs must be float: <code>x = x.float()</code>.</p>

<h3>Q: <code>CUDA out of memory</code></h3>
<p>In increasing cost: smaller <code>batch_size</code> → wrap validation in <code>torch.no_grad()</code> → <code>del</code> unused tensors → gradient accumulation → AMP → gradient checkpointing.</p>

<h3>Q: 99% train accuracy, 70% test</h3>
<p><b>That is overfitting, not a bug.</b> Add, in order: data augmentation → dropout → weight decay → early stopping → a smaller model. Chapter 9 has the full decision table.</p>

<h2>4. Results and shipping</h2>

<h3>Q: How do I hand a trained model to someone else?</h3>
<pre data-lang="python"><code>torch.save(model.state_dict(), "model.pt")          # weights

model = MyNet()                                      # they need the same class
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()                                         # mandatory</code></pre>
<p>Ship three things together: the model class, the preprocessing/vocabulary config, and the weights. Miss one and nothing runs.</p>

<h3>Q: The same image gives different predictions each time</h3>
<p><b>Cause</b>: missing <code>model.eval()</code> — dropout is still firing.</p>

<h2>5. How to study</h2>

<h3>Q: I understand it but cannot write it myself</h3>
<p><b>Cause</b>: understanding is recognition; writing is recall. Different mental work. The only fix is <b>closing the tutorial and rewriting it from an empty file</b>. Where you get stuck is exactly the part you did not really get.</p>

<h3>Q: Can I learn this with weak math?</h3>
<p>Yes. You need three things: read a matrix product, know a derivative is a rate of change, and know probabilities sum to one. The prep chapter covers all three. For a formula you cannot push through, remember what it <em>does</em> before memorising how it is derived.</p>

<h3>Q: Is this enough to get a job?</h3>
<p>It is a ticket in, not the whole job. Real roles also want: reading papers, cleaning messy data, tuning to a target, deploying. Suggested path: finish this course → build one independent project on real data end to end → reproduce one simple paper.</p>

<div class="keypoints">
  <h4>Four steps for any error</h4>
  <ul>
    <li>Read the last line of the traceback — it usually names the problem</li>
    <li>Print shapes: almost every PyTorch error is a shape error</li>
    <li>Check devices: CPU vs CUDA consistency</li>
    <li>Minimal repro: delete everything except the two failing lines</li>
  </ul>
</div>
`
  }
};
