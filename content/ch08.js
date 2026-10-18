(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch08',
  num: '08',
  part: 'p2',
  file: 'training-loop',
  zh: {
    title: '训练循环：把手上的零件组装起来',
    sub: '一个模板，覆盖你未来 80% 的项目',
    html: `
<p class="lead">到这里你已经有了模型、数据、损失、优化器。这一章把它们串成一条流水线，并加上<strong>验证、保存、早停、日志</strong>——这些才是「能跑」和「跑得好」的区别。学完这一章，你就能独立训练一个真实模型。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>默写标准训练循环的五步，顺序不出错</li>
    <li>写出规范的 <code>train_one_epoch</code> / <code>evaluate</code> 函数</li>
    <li>在 MNIST 上训练到 98% 以上准确率</li>
    <li>保存最佳模型、加载回来做推理</li>
    <li>读懂 loss 曲线判断过拟合还是欠拟合</li>
  </ul>
</div>

<h2>1. 五步骨架</h2>
<pre data-lang="python"><code>for epoch in range(epochs):
    model.train()                 # 1. 切到训练模式
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()     # 2. 清空旧梯度
        logits = model(xb)        # 3. 前向
        loss = criterion(logits, yb)
        loss.backward()           # 4. 反向求梯度
        optimizer.step()          # 5. 更新参数

    evaluate(model, val_loader)   # 每个 epoch 结束验证一次</code></pre>

<div class="callout warn">
  <span class="tag">顺序就是一切</span>
  <p>这五行必须按顺序：<code>zero_grad → forward → loss → backward → step</code>。<br>
  忘了 <code>zero_grad</code>：梯度累积，几步后爆炸。<br>
  忘了 <code>backward</code>：梯度是 None，<code>step()</code> 什么也没改。<br>
  忘了 <code>model.train()</code>：Dropout 和 BatchNorm 行为错误，结果不可靠。</p>
</div>

<h2>2. 完整可运行示例：MNIST 手写数字</h2>
<pre data-lang="python"><code>import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 数据 ----------
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST 的均值/标准差
])
train_ds = datasets.MNIST("data", train=True,  download=True, transform=tf)
val_ds   = datasets.MNIST("data", train=False, download=True, transform=tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

# ---------- 模型 ----------
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 256), nn.ReLU(),
    nn.Linear(256, 128),   nn.ReLU(),
    nn.Linear(128, 10),
).to(device)

# ---------- 损失与优化器 ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------- 训练 ----------
def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss_sum += criterion(out, yb).item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return loss_sum / total, correct / total

best_acc = 0.0
for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    val_loss, val_acc = evaluate(model, val_loader)
    print(f"epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if val_acc &gt; best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best.pt")
        print("  saved best:", best_acc)</code></pre>

<p>在 CPU 上大约 1~2 分钟一轮，5 轮后验证准确率通常能到 <strong>97.5%~98.3%</strong>。如果明显低于这个数，先看下面「常见失败模式」。</p>

<h2>3. 为什么要分 train 和 eval 两个函数</h2>
<table>
  <tr><th></th><th>train 模式</th><th>eval 模式</th></tr>
  <tr><td>Dropout</td><td>随机丢弃神经元</td><td>关闭</td></tr>
  <tr><td>BatchNorm</td><td>用当前 batch 统计量并更新滑动平均</td><td>用滑动平均的统计量</td></tr>
  <tr><td>梯度</td><td>需要</td><td>用 <code>no_grad</code> 关闭</td></tr>
  <tr><td>返回</td><td>loss</td><td>loss + 指标</td></tr>
</table>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p><code>model.train()</code> 和 <code>model.eval()</code> 不是「训练/不训练」的开关，而是<strong>切换某些层的行为模式</strong>。忘了在评估前调 <code>eval()</code>，Dropout 还在随机丢，BatchNorm 还在用当前 batch 的统计量，测出来的分数会忽高忽低。</p>
  <p>反过来，在训练时忘了 <code>train()</code>（比如从评估回来没切回去），Dropout 失效，模型容易过拟合。</p>
</div>

<h2>4. 指标计算：别用错平均方式</h2>
<pre data-lang="python"><code># ❌ 错：各 batch 的 loss 直接平均（最后一批可能更小）
avg = sum(batch_losses) / len(batch_losses)

# ✅ 对：按样本数加权
avg = sum(l * n for l, n in zip(batch_losses, batch_sizes)) / total_samples

# ✅ 更方便：直接累加 loss.item() * batch_size</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 加权平均</span>
  <p>假设两个 batch，第一个 128 个样本 loss=0.5，第二个 32 个样本 loss=1.0。简单平均是 0.75；但真实整体损失是 <code>(128×0.5 + 32×1.0) / 160 = 0.6</code>。</p>
  <p>当最后一个 batch 不满时（160 = 128 + 32），两种算法差异明显。别小看这 0.15，它会让你误判模型好坏。</p>
</div>

<h2>5. 早停与保存</h2>
<pre data-lang="python"><code>best_acc, patience, bad_epochs = 0.0, 3, 0

for epoch in range(50):
    train_one_epoch(...)
    val_loss, val_acc = evaluate(model, val_loader)

    if val_acc &gt; best_acc:
        best_acc, bad_epochs = val_acc, 0
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "acc": best_acc,
        }, "best.pt")
    else:
        bad_epochs += 1
        if bad_epochs &gt;= patience:
            print("early stop at epoch", epoch)
            break</code></pre>

<pre data-lang="python"><code># 加载回来做推理
ckpt = torch.load("best.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

with torch.no_grad():
    logits = model(x_test.to(device))
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)</code></pre>

<div class="callout tip">
  <span class="tag">保存 state_dict，不要保存整个模型</span>
  <p><code>torch.save(model)</code> 会把类的定义路径也存进去，换目录、改文件名就可能加载失败。<code>state_dict()</code> 只存参数字典，稳定且通用。加载时需要<strong>先重建同结构的模型</strong>，再 <code>load_state_dict</code>。</p>
</div>

<h2>6. 读懂 loss 曲线</h2>
<table>
  <tr><th>现象</th><th>诊断</th><th>对策</th></tr>
  <tr><td>训练 loss 不降</td><td>学习率不对 / 梯度没回传 / 数据标签错</td><td>先过拟合 10 个样本，能记住说明模型没问题</td></tr>
  <tr><td>训练降、验证升</td><td>过拟合</td><td>加数据、增强、Dropout、weight decay、早停</td></tr>
  <tr><td>两边都很高</td><td>欠拟合</td><td>加大模型、训练更久、提高学习率</td></tr>
  <tr><td>两边都剧烈震荡</td><td>学习率太大 / batch 太小</td><td>降 lr、加 batch、加梯度裁剪</td></tr>
  <tr><td>验证准确率远高于训练</td><td>train/eval 切换有问题，或数据泄漏</td><td>检查 <code>model.train()</code> 与数据划分</td></tr>
</table>

<div class="callout tip">
  <span class="tag">调试的黄金第一步</span>
  <p>拿 10 个样本训练 100 步，看能不能把 loss 压到接近 0。如果连 10 个样本都记不住，问题一定在代码里（形状、损失、优化器没接上），不在数据量或正则化。这个技巧能省你几个小时。</p>
</div>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li>五步顺序：zero_grad → forward → loss → backward → step。</li>
    <li>训练前 <code>model.train()</code>，评估前 <code>model.eval()</code> + <code>no_grad()</code>。</li>
    <li>loss 平均要按样本数加权，不能对 batch 简单平均。</li>
    <li>保存 <code>state_dict()</code>，加载前先重建模型结构。</li>
    <li>验证指标不再提升就存 checkpoint + 计数早停。</li>
    <li>调不通时先用 10 个样本过拟合，快速定位是代码问题还是数据问题。</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q">训练时忘了 <code>optimizer.zero_grad()</code>，会发生什么？</p>
  <ol class="opts">
    <li>梯度变成 0，模型不学习</li>
    <li>梯度不断累加，更新步长越来越大，可能发散</li>
    <li>报错</li>
    <li>没有影响，PyTorch 会自动清零</li>
  </ol>
  <p class="exp">.grad 是累加的。不清零，每一步用的都是「历史所有 batch 的梯度和」，等效学习率越来越大。</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">评估时忘了 <code>model.eval()</code>，最可能的后果是？</p>
  <ol class="opts">
    <li>显存不足</li>
    <li>直接报错</li>
    <li>Dropout 仍在随机丢弃、BatchNorm 用 batch 统计量，指标不稳定且偏低</li>
    <li>评估速度变快</li>
  </ol>
  <p class="exp">eval 模式会关闭 Dropout、让 BN 使用滑动平均。忘了切模式，验证分数会明显偏低且每次不同。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">训练 loss 一直在降，验证 loss 从第 3 轮开始上升。这说明？</p>
  <ol class="opts">
    <li>过拟合，应该早停或加正则</li>
    <li>欠拟合，应该加大模型</li>
    <li>学习率太小</li>
    <li>数据太多</li>
  </ol>
  <p class="exp">典型的过拟合曲线。第 3 轮的权重才是最好的，这也是要保存 best checkpoint 的原因。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>跑通本章 MNIST 代码，目标验证准确率 &gt; 98%。记录每轮耗时。</li>
    <li>把 <code>zero_grad()</code> 注释掉，看 loss 如何变化，把现象写进笔记。</li>
    <li>加上早停（patience=3），观察它在第几轮停。</li>
    <li>故意用 <code>sum(batch_losses)/len(batch_losses)</code> 算平均 loss，和加权版本比较，差多少？</li>
    <li>保存 best.pt，新开一个 notebook 只做加载 + 预测，输出前 5 张图的预测结果。</li>
  </ol>
</div>`
  },
  en: {
    title: 'The training loop: assembling the parts',
    sub: 'One template that covers 80% of your future projects',
    html: `
<p class="lead">You now have a model, data, a loss and an optimiser. This chapter wires them into a pipeline and adds <strong>validation, checkpointing, early stopping and logging</strong> — the difference between "it runs" and "it works".</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Recite the five steps in the right order</li>
    <li>Write clean <code>train_one_epoch</code> / <code>evaluate</code> functions</li>
    <li>Reach 98%+ on MNIST</li>
    <li>Save the best model and reload it for inference</li>
    <li>Read loss curves and diagnose over/under-fitting</li>
  </ul>
</div>

<h2>1. The five-step skeleton</h2>
<pre data-lang="python"><code>for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    evaluate(model, val_loader)</code></pre>

<div class="callout warn">
  <span class="tag">Order is everything</span>
  <p><code>zero_grad → forward → loss → backward → step</code>.<br>
  Skip zero_grad: gradients accumulate and explode.<br>
  Skip backward: gradients are None and step() does nothing.<br>
  Skip <code>model.train()</code>: Dropout and BatchNorm misbehave.</p>
</div>

<h2>2. Complete MNIST example</h2>
<pre data-lang="python"><code>import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
train_ds = datasets.MNIST("data", train=True,  download=True, transform=tf)
val_ds   = datasets.MNIST("data", train=False, download=True, transform=tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 256), nn.ReLU(),
    nn.Linear(256, 128),   nn.ReLU(),
    nn.Linear(128, 10),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss_sum += criterion(out, yb).item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return loss_sum / total, correct / total

best_acc = 0.0
for epoch in range(5):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    val_loss, val_acc = evaluate(model, val_loader)
    print(f"epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if val_acc &gt; best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best.pt")</code></pre>

<p>On CPU this is 1–2 minutes per epoch; after five epochs validation accuracy is typically <strong>97.5–98.3%</strong>.</p>

<h2>3. Why separate train and eval</h2>
<table>
  <tr><th></th><th>train mode</th><th>eval mode</th></tr>
  <tr><td>Dropout</td><td>randomly drops units</td><td>off</td></tr>
  <tr><td>BatchNorm</td><td>batch statistics, updates running averages</td><td>uses running averages</td></tr>
  <tr><td>Gradients</td><td>needed</td><td>disabled via <code>no_grad</code></td></tr>
</table>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p><code>train()</code> / <code>eval()</code> do not start or stop training; they <strong>switch layer behaviour</strong>. Forgetting <code>eval()</code> leaves Dropout active and BatchNorm using the current batch, so your metric jumps around.</p>
</div>

<h2>4. Averaging metrics correctly</h2>
<pre data-lang="python"><code># WRONG: plain mean over batches
avg = sum(batch_losses) / len(batch_losses)

# RIGHT: weighted by sample count
avg = sum(l * n for l, n in zip(batch_losses, batch_sizes)) / total_samples</code></pre>

<div class="callout math">
  <span class="tag">Math note · weighted mean</span>
  <p>Batch A: 128 samples, loss 0.5. Batch B: 32 samples, loss 1.0. Plain mean 0.75; true loss <code>(128×0.5 + 32×1.0)/160 = 0.6</code>. With a ragged last batch the difference is real.</p>
</div>

<h2>5. Early stopping and checkpoints</h2>
<pre data-lang="python"><code>best_acc, patience, bad_epochs = 0.0, 3, 0

for epoch in range(50):
    train_one_epoch(...)
    val_loss, val_acc = evaluate(model, val_loader)

    if val_acc &gt; best_acc:
        best_acc, bad_epochs = val_acc, 0
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "acc": best_acc,
        }, "best.pt")
    else:
        bad_epochs += 1
        if bad_epochs &gt;= patience:
            break</code></pre>

<pre data-lang="python"><code>ckpt = torch.load("best.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

with torch.no_grad():
    logits = model(x_test.to(device))
    preds = logits.argmax(dim=1)</code></pre>

<div class="callout tip">
  <span class="tag">Save state_dict, not the module</span>
  <p><code>torch.save(model)</code> pickles the class path and breaks when you move files. <code>state_dict()</code> is just weights — rebuild the same architecture, then load.</p>
</div>

<h2>6. Reading the curves</h2>
<table>
  <tr><th>Symptom</th><th>Diagnosis</th><th>Action</th></tr>
  <tr><td>Train loss flat</td><td>bad LR / no grad / bad labels</td><td>overfit 10 samples first</td></tr>
  <tr><td>Train down, val up</td><td>overfitting</td><td>more data, augmentation, dropout, weight decay, early stop</td></tr>
  <tr><td>Both high</td><td>underfitting</td><td>bigger model, longer training, higher LR</td></tr>
  <tr><td>Both oscillating</td><td>LR too high / batch too small</td><td>lower LR, larger batch, clip gradients</td></tr>
  <tr><td>Val &gt; train</td><td>train/eval misuse or leakage</td><td>check <code>model.train()</code> and the split</td></tr>
</table>

<div class="callout tip">
  <span class="tag">The golden debugging step</span>
  <p>Train on 10 samples for 100 steps. If it cannot drive the loss to ~0, the bug is in your code, not in the data or regularisation. This saves hours.</p>
</div>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li>Order: zero_grad → forward → loss → backward → step.</li>
    <li><code>train()</code> before training, <code>eval()</code> + <code>no_grad()</code> before validation.</li>
    <li>Weight losses by sample count, not by batch.</li>
    <li>Save <code>state_dict()</code>; rebuild the model before loading.</li>
    <li>Checkpoint on improvement and count epochs for early stopping.</li>
    <li>When stuck, overfit 10 samples to isolate code from data.</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q">Forgetting <code>optimizer.zero_grad()</code> causes:</p>
  <ol class="opts">
    <li>Zero gradients, no learning</li>
    <li>Gradients accumulating, effective step size growing, possibly divergence</li>
    <li>An error</li>
    <li>Nothing, PyTorch clears automatically</li>
  </ol>
  <p class="exp">.grad accumulates. Without clearing, each step uses the sum of all past batches.</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">Forgetting <code>model.eval()</code> before validation means:</p>
  <ol class="opts">
    <li>Out of memory</li>
    <li>An error</li>
    <li>Dropout stays active and BN uses batch stats, so metrics are noisy and low</li>
    <li>Faster validation</li>
  </ol>
  <p class="exp">eval() disables dropout and switches BN to running statistics.</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">Train loss keeps falling but validation loss rises from epoch 3. This is:</p>
  <ol class="opts">
    <li>Overfitting — early stop or regularise</li>
    <li>Underfitting — bigger model</li>
    <li>Learning rate too small</li>
    <li>Too much data</li>
  </ol>
  <p class="exp">Classic overfitting. The epoch-3 weights are the best ones — hence best checkpoints.</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Run the MNIST code to &gt;98% validation accuracy; record epoch times.</li>
    <li>Comment out <code>zero_grad()</code> and watch the loss; write down what happens.</li>
    <li>Add early stopping with patience=3 and see which epoch it stops at.</li>
    <li>Compare the naive batch-mean loss with the sample-weighted one.</li>
    <li>Reload best.pt in a fresh notebook and predict the first five test images.</li>
  </ol>
</div>`
  }
});
