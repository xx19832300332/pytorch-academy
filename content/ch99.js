(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch99',
  num: '99',
  part: 'p5',
  file: 'cheatsheet',
  zh: {
    title: '速查表：记住这些就够开工',
    sub: '一页纸的 API、形状约定与报错对照',
    html: `
<p class="lead">这不是教程，是工具页。忘了怎么写的时候来这里看一眼，比重看章节快。建议收藏、打印，或者抄到自己的笔记里。</p>

<h2>1. 张量创建与转换</h2>
<table>
  <tr><th>写法</th><th>结果</th></tr>
  <tr><td><code>torch.tensor(data, dtype=torch.float32)</code></td><td>从 Python 数据构造</td></tr>
  <tr><td><code>torch.zeros(2,3)</code> / <code>ones</code> / <code>full((2,3), 7.)</code></td><td>全 0 / 全 1 / 指定值</td></tr>
  <tr><td><code>torch.arange(0, 10, 2)</code></td><td>等差序列</td></tr>
  <tr><td><code>torch.randn(2,3)</code> / <code>torch.rand(2,3)</code></td><td>正态 / 均匀随机</td></tr>
  <tr><td><code>torch.zeros_like(x)</code></td><td>同形状同 dtype</td></tr>
  <tr><td><code>torch.eye(3)</code></td><td>单位矩阵</td></tr>
  <tr><td><code>x.float()</code> / <code>x.long()</code> / <code>x.to(torch.float16)</code></td><td>类型转换</td></tr>
  <tr><td><code>x.numpy()</code> / <code>torch.from_numpy(a)</code></td><td>与 NumPy 互转（后者共享内存）</td></tr>
  <tr><td><code>x.item()</code></td><td>单元素张量 → Python 数字</td></tr>
</table>

<h2>2. 形状操作</h2>
<table>
  <tr><th>写法</th><th>作用</th></tr>
  <tr><td><code>x.shape</code> / <code>x.dim()</code> / <code>x.numel()</code></td><td>各轴长度 / 轴数 / 元素数</td></tr>
  <tr><td><code>x.reshape(a, b, -1)</code></td><td>变形，安全，可能拷贝</td></tr>
  <tr><td><code>x.view(a, b)</code></td><td>变形，要求内存连续</td></tr>
  <tr><td><code>x.flatten(1)</code></td><td>从第 1 轴开始压平</td></tr>
  <tr><td><code>x.unsqueeze(0)</code> / <code>x.squeeze(1)</code></td><td>加一个轴 / 删长度为 1 的轴</td></tr>
  <tr><td><code>x.permute(2,0,1)</code></td><td>重排所有轴</td></tr>
  <tr><td><code>x.t()</code> / <code>x.transpose(0,1)</code></td><td>转置（2 维）/ 交换任意两轴</td></tr>
  <tr><td><code>x.contiguous()</code></td><td>变成连续内存</td></tr>
  <tr><td><code>x.expand(2,3,5)</code> / <code>x.repeat(2,1,1)</code></td><td>广播扩展（不复制）/ 真实复制</td></tr>
</table>

<h2>3. 运算与归约</h2>
<table>
  <tr><th>写法</th><th>作用</th></tr>
  <tr><td><code>a + b</code>, <code>a - b</code>, <code>a * b</code>, <code>a / b</code></td><td>逐元素</td></tr>
  <tr><td><code>a @ b</code> / <code>torch.matmul</code></td><td>矩阵乘</td></tr>
  <tr><td><code>torch.bmm</code></td><td>批量矩阵乘（3 维）</td></tr>
  <tr><td><code>x.sum(dim=1)</code> / <code>mean</code> / <code>std</code></td><td>沿轴归约</td></tr>
  <tr><td><code>x.max(dim=1)</code> / <code>x.argmax(dim=1)</code></td><td>最大值及其下标</td></tr>
  <tr><td><code>torch.cat([a,b], dim=0)</code></td><td>沿已有轴拼接</td></tr>
  <tr><td><code>torch.stack([a,b])</code></td><td>新建一个轴堆叠</td></tr>
  <tr><td><code>torch.clamp(x, min, max)</code></td><td>截断</td></tr>
  <tr><td><code>torch.where(cond, a, b)</code></td><td>条件选择</td></tr>
  <tr><td><code>torch.softmax(x, dim=-1)</code></td><td>softmax（分类输出转概率时用）</td></tr>
</table>

<h2>4. 自动求导</h2>
<table>
  <tr><th>写法</th><th>作用</th></tr>
  <tr><td><code>x.requires_grad_(True)</code></td><td>开启追踪</td></tr>
  <tr><td><code>loss.backward()</code></td><td>反向传播（loss 必须是标量）</td></tr>
  <tr><td><code>x.grad</code></td><td>梯度（累加）</td></tr>
  <tr><td><code>x.grad.zero_()</code> / <code>opt.zero_grad()</code></td><td>清零</td></tr>
  <tr><td><code>with torch.no_grad():</code></td><td>关闭追踪（推理）</td></tr>
  <tr><td><code>x.detach()</code></td><td>脱离计算图</td></tr>
</table>

<h2>5. 模型与层</h2>
<table>
  <tr><th>写法</th><th>形状约定</th></tr>
  <tr><td><code>nn.Linear(in, out)</code></td><td><code>(N, in) → (N, out)</code></td></tr>
  <tr><td><code>nn.Conv2d(cin, cout, k, s, p)</code></td><td><code>(N,cin,H,W) → (N,cout,H',W')</code></td></tr>
  <tr><td><code>nn.MaxPool2d(2)</code></td><td>宽高减半</td></tr>
  <tr><td><code>nn.BatchNorm2d(c)</code></td><td>通道数必须等于输入通道</td></tr>
  <tr><td><code>nn.Dropout(p)</code></td><td>训练时随机置零</td></tr>
  <tr><td><code>nn.Embedding(V, D)</code></td><td><code>(N,T) int → (N,T,D)</code></td></tr>
  <tr><td><code>nn.LSTM(in, hidden, batch_first=True)</code></td><td><code>(N,T,in) → out (N,T,hidden)</code></td></tr>
  <tr><td><code>nn.MultiheadAttention(d, heads, batch_first=True)</code></td><td><code>(N,T,d)</code>，d 要能被 heads 整除</td></tr>
  <tr><td><code>model.parameters()</code> / <code>named_parameters()</code></td><td>交给优化器 / 带名字排查</td></tr>
  <tr><td><code>model.train()</code> / <code>model.eval()</code></td><td>切模式（影响 Dropout/BN）</td></tr>
</table>

<h2>6. 训练循环模板</h2>
<pre data-lang="python"><code>model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    print(epoch, correct / total)</code></pre>

<h2>7. 报错对照表</h2>
<table>
  <tr><th>报错</th><th>原因与解法</th></tr>
  <tr><td><code>shapes cannot be multiplied</code></td><td>矩阵乘维度不匹配，检查 <code>(N,in)@(in,out)</code></td></tr>
  <tr><td><code>expected scalar type Float but found Long/Double</code></td><td>dtype 不一致，<code>.float()</code></td></tr>
  <tr><td><code>Expected all tensors to be on the same device</code></td><td>模型/数据/新建张量不同设备</td></tr>
  <tr><td><code>CUDA out of memory</code></td><td>降 batch → AMP → 释放引用 → checkpointing</td></tr>
  <tr><td><code>view size is not compatible with ... stride</code></td><td>非连续，改 <code>reshape</code> 或先 <code>contiguous()</code></td></tr>
  <tr><td><code>grad can be implicitly created only for scalar outputs</code></td><td>backward 前先 <code>.mean()/.sum()</code></td></tr>
  <tr><td><code>element 0 of tensors does not require grad</code></td><td>参数不是 Parameter，或整条链没开 requires_grad</td></tr>
  <tr><td><code>cannot assign module before Module.__init__() call</code></td><td>忘了 <code>super().__init__()</code></td></tr>
  <tr><td><code>Target 5 is out of bounds</code></td><td>标签超出类别数，检查 num_classes 与标签映射</td></tr>
  <tr><td><code>mat1 and mat2 must have the same dtype</code></td><td>AMP 或混合 dtype，检查输入转换</td></tr>
</table>

<h2>8. 常用小片段</h2>
<pre data-lang="python"><code># 参数量
sum(p.numel() for p in model.parameters())

# 只训练需要梯度的参数
optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

# 设备无关
device = "cuda" if torch.cuda.is_available() else "cpu"

# 固定随机种子
torch.manual_seed(42); torch.cuda.manual_seed_all(42)

# 保存 / 加载
torch.save(model.state_dict(), "m.pt")
model.load_state_dict(torch.load("m.pt", map_location=device))

# 推理
model.eval()
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)

# 统计各层输出形状（调试神器）
from torchinfo import summary
summary(model, input_size=(1, 3, 224, 224))</code></pre>

<div class="keypoints">
  <h4>三个必须刻进肌肉记忆的约定</h4>
  <ol>
    <li><code>(N, in) @ (in, out) = (N, out)</code>：一切形状问题的根。</li>
    <li>训练五步：<code>zero_grad → forward → loss → backward → step</code>。</li>
    <li>推理三步：<code>eval() → no_grad() → 模型输入补 batch 维</code>。</li>
  </ol>
</div>

<div class="exercise">
  <h4>怎么用这一页</h4>
  <ol>
    <li>把「报错对照表」抄到你自己的笔记开头，遇到报错先查表。</li>
    <li>用浏览器打印功能（Ctrl+P）导出成 PDF，放在手边。</li>
    <li>每学完一章，回来看看这一页里哪些条目你终于看懂了——这比读十遍都有效。</li>
  </ol>
</div>`
  },
  en: {
    title: 'Cheatsheet: enough to start building',
    sub: 'APIs, shape conventions and error messages on one page',
    html: `
<p class="lead">Not a tutorial — a tool. Come back when you forget the syntax; it is faster than re-reading a chapter.</p>

<h2>1. Creating and converting</h2>
<table>
  <tr><th>Call</th><th>Result</th></tr>
  <tr><td><code>torch.tensor(data, dtype=torch.float32)</code></td><td>from Python data</td></tr>
  <tr><td><code>torch.zeros(2,3)</code> / <code>ones</code> / <code>full((2,3), 7.)</code></td><td>zeros / ones / constant</td></tr>
  <tr><td><code>torch.arange(0, 10, 2)</code></td><td>range</td></tr>
  <tr><td><code>torch.randn(2,3)</code> / <code>torch.rand(2,3)</code></td><td>normal / uniform</td></tr>
  <tr><td><code>torch.zeros_like(x)</code></td><td>same shape and dtype</td></tr>
  <tr><td><code>x.float()</code> / <code>x.long()</code> / <code>x.to(torch.float16)</code></td><td>dtype casts</td></tr>
  <tr><td><code>x.numpy()</code> / <code>torch.from_numpy(a)</code></td><td>NumPy bridge (shares memory)</td></tr>
  <tr><td><code>x.item()</code></td><td>scalar tensor → Python number</td></tr>
</table>

<h2>2. Shape manipulation</h2>
<table>
  <tr><th>Call</th><th>Effect</th></tr>
  <tr><td><code>x.shape</code> / <code>x.dim()</code> / <code>x.numel()</code></td><td>lengths / rank / element count</td></tr>
  <tr><td><code>x.reshape(a, b, -1)</code></td><td>safe reshape, may copy</td></tr>
  <tr><td><code>x.view(a, b)</code></td><td>requires contiguous memory</td></tr>
  <tr><td><code>x.flatten(1)</code></td><td>flatten from axis 1</td></tr>
  <tr><td><code>x.unsqueeze(0)</code> / <code>x.squeeze(1)</code></td><td>add / remove size-1 axis</td></tr>
  <tr><td><code>x.permute(2,0,1)</code></td><td>reorder axes</td></tr>
  <tr><td><code>x.t()</code> / <code>x.transpose(0,1)</code></td><td>transpose 2-D / swap any two</td></tr>
  <tr><td><code>x.contiguous()</code></td><td>make contiguous</td></tr>
  <tr><td><code>x.expand(2,3,5)</code> / <code>x.repeat(2,1,1)</code></td><td>broadcast view / real copy</td></tr>
</table>

<h2>3. Math and reductions</h2>
<table>
  <tr><th>Call</th><th>Effect</th></tr>
  <tr><td><code>a + b</code>, <code>a * b</code>, <code>a / b</code></td><td>element-wise</td></tr>
  <tr><td><code>a @ b</code> / <code>torch.matmul</code></td><td>matrix multiply</td></tr>
  <tr><td><code>torch.bmm</code></td><td>batched matmul (3-D)</td></tr>
  <tr><td><code>x.sum(dim=1)</code> / <code>mean</code> / <code>std</code></td><td>reduce along an axis</td></tr>
  <tr><td><code>x.max(dim=1)</code> / <code>x.argmax(dim=1)</code></td><td>values and indices</td></tr>
  <tr><td><code>torch.cat([a,b], dim=0)</code></td><td>join along existing axis</td></tr>
  <tr><td><code>torch.stack([a,b])</code></td><td>stack into new axis</td></tr>
  <tr><td><code>torch.clamp(x, min, max)</code></td><td>clip</td></tr>
  <tr><td><code>torch.where(cond, a, b)</code></td><td>select</td></tr>
  <tr><td><code>torch.softmax(x, dim=-1)</code></td><td>probabilities at inference</td></tr>
</table>

<h2>4. Autograd</h2>
<table>
  <tr><th>Call</th><th>Effect</th></tr>
  <tr><td><code>x.requires_grad_(True)</code></td><td>start tracking</td></tr>
  <tr><td><code>loss.backward()</code></td><td>backprop (loss must be scalar)</td></tr>
  <tr><td><code>x.grad</code></td><td>gradient (accumulated)</td></tr>
  <tr><td><code>opt.zero_grad()</code></td><td>clear gradients</td></tr>
  <tr><td><code>with torch.no_grad():</code></td><td>disable tracking</td></tr>
  <tr><td><code>x.detach()</code></td><td>leave the graph</td></tr>
</table>

<h2>5. Models and layers</h2>
<table>
  <tr><th>Call</th><th>Shape contract</th></tr>
  <tr><td><code>nn.Linear(in, out)</code></td><td><code>(N, in) → (N, out)</code></td></tr>
  <tr><td><code>nn.Conv2d(cin, cout, k, s, p)</code></td><td><code>(N,cin,H,W) → (N,cout,H',W')</code></td></tr>
  <tr><td><code>nn.MaxPool2d(2)</code></td><td>halves H and W</td></tr>
  <tr><td><code>nn.BatchNorm2d(c)</code></td><td>c must match input channels</td></tr>
  <tr><td><code>nn.Dropout(p)</code></td><td>active only in train mode</td></tr>
  <tr><td><code>nn.Embedding(V, D)</code></td><td><code>(N,T) int → (N,T,D)</code></td></tr>
  <tr><td><code>nn.LSTM(in, hidden, batch_first=True)</code></td><td><code>(N,T,in) → (N,T,hidden)</code></td></tr>
  <tr><td><code>nn.MultiheadAttention(d, heads, batch_first=True)</code></td><td>d divisible by heads</td></tr>
  <tr><td><code>model.parameters()</code></td><td>pass to the optimiser</td></tr>
  <tr><td><code>model.train()</code> / <code>model.eval()</code></td><td>switch Dropout/BN behaviour</td></tr>
</table>

<h2>6. Training loop template</h2>
<pre data-lang="python"><code>model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    print(epoch, correct / total)</code></pre>

<h2>7. Error decoder</h2>
<table>
  <tr><th>Message</th><th>Cause and fix</th></tr>
  <tr><td><code>shapes cannot be multiplied</code></td><td>matmul dims mismatch</td></tr>
  <tr><td><code>expected scalar type Float but found Long</code></td><td>dtype mismatch, use <code>.float()</code></td></tr>
  <tr><td><code>Expected all tensors to be on the same device</code></td><td>move model/data/new tensors</td></tr>
  <tr><td><code>CUDA out of memory</code></td><td>batch → AMP → free refs → checkpointing</td></tr>
  <tr><td><code>view size is not compatible with ... stride</code></td><td>non-contiguous, use reshape</td></tr>
  <tr><td><code>grad can be implicitly created only for scalar outputs</code></td><td>reduce with mean/sum first</td></tr>
  <tr><td><code>element 0 of tensors does not require grad</code></td><td>not a Parameter, or no requires_grad in the chain</td></tr>
  <tr><td><code>cannot assign module before Module.__init__() call</code></td><td>missing <code>super().__init__()</code></td></tr>
  <tr><td><code>Target 5 is out of bounds</code></td><td>label exceeds num_classes</td></tr>
  <tr><td><code>mat1 and mat2 must have the same dtype</code></td><td>mixed dtype, often AMP related</td></tr>
</table>

<h2>8. Snippets</h2>
<pre data-lang="python"><code>sum(p.numel() for p in model.parameters())

optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42); torch.cuda.manual_seed_all(42)

torch.save(model.state_dict(), "m.pt")
model.load_state_dict(torch.load("m.pt", map_location=device))

model.eval()
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)

from torchinfo import summary
summary(model, input_size=(1, 3, 224, 224))</code></pre>

<div class="keypoints">
  <h4>Three things to memorise</h4>
  <ol>
    <li><code>(N, in) @ (in, out) = (N, out)</code> — the root of all shape bugs.</li>
    <li>Five steps: <code>zero_grad → forward → loss → backward → step</code>.</li>
    <li>Inference: <code>eval() → no_grad() → add batch dim</code>.</li>
  </ol>
</div>

<div class="exercise">
  <h4>How to use this page</h4>
  <ol>
    <li>Copy the error table to the top of your own notes.</li>
    <li>Print it to PDF (Ctrl+P) and keep it nearby.</li>
    <li>After each chapter, come back and see which lines finally make sense.</li>
  </ol>
</div>`
  }
});
