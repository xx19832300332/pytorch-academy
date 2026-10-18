(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch14',
  num: '14',
  part: 'p4',
  file: 'transfer-deploy',
  zh: {
    title: '迁移学习与模型部署',
    sub: '用别人的 100 万张图，解决你的 200 张图',
    html: `
<p class="lead">现实中你几乎不会从零训练模型。更常见的路径是：拿一个在海量数据上训好的模型，换成你的任务，用少量数据微调。这一章讲清楚怎么改、怎么训、怎么存、怎么上线。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>用 torchvision 加载预训练模型并替换分类头</li>
    <li>决定冻结哪些层、用多大学习率</li>
    <li>把模型存成 state_dict / TorchScript / ONNX 三种格式</li>
    <li>写出一个正确的推理函数（eval + no_grad + 预处理一致）</li>
  </ul>
</div>

<h2>1. 为什么迁移学习有效</h2>
<p>CNN 在 ImageNet 上学到的浅层特征（边缘、纹理、颜色）是<strong>通用</strong>的，猫狗识别和细胞识别都需要它们。真正任务相关的只是最后几层。</p>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p>像学开车：你已经会骑自行车（平衡感、看后视镜），学开车时这些能力直接迁移过去，只需要学新的部分（油门、离合器）。从零学起则要重新建立所有基础感觉，需要多得多的数据和时间。</p>
</div>

<h2>2. 换掉分类头</h2>
<pre data-lang="python"><code>import torch.nn as nn
from torchvision import models

# weights="DEFAULT" 会自动下载预训练权重
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 看看原来最后一层长什么样
print(model.fc)          # Linear(in_features=512, out_features=1000, bias=True)

# 换成你自己的类别数
model.fc = nn.Linear(512, 5)      # 5 分类

# 搬到 GPU
model = model.cuda()</code></pre>

<div class="callout warn">
  <span class="tag">必须匹配的预处理</span>
  <p>预训练模型是用特定的归一化参数训练的。用错参数，准确率会明显下降。用 <code>weights.transforms()</code> 最保险：</p>
  <p><code>preprocess = models.ResNet18_Weights.DEFAULT.transforms()</code></p>
</div>

<h2>3. 冻结与微调</h2>
<pre data-lang="python"><code># 方案 A：只训最后一层（数据很少时）
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(512, 5)      # 新建的层默认 requires_grad=True

optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)

# 方案 B：全部解冻，但不同层用不同学习率（数据中等时）
for p in model.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters()
                if not n.startswith("fc")], "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-3},
])</code></pre>

<table>
  <tr><th>数据量</th><th>策略</th><th>典型 lr</th></tr>
  <tr><td>每类 &lt; 100 张</td><td>只训分类头</td><td>1e-3</td></tr>
  <tr><td>每类 100~1000</td><td>解冻后几层 / 全解冻小 lr</td><td>1e-4 ~ 1e-5</td></tr>
  <tr><td>每类 &gt; 1000</td><td>全网络微调，甚至从头训练</td><td>1e-3 ~ 1e-4</td></tr>
</table>

<div class="callout tip">
  <span class="tag">BatchNorm 要小心</span>
  <p>微调时如果 batch 很小，冻结 BN 层（<code>p.requires_grad=False</code> 不够，还要 <code>model.eval()</code> 让 BN 用 running stats）通常更稳，否则 running mean/var 会被小 batch 带偏。</p>
</div>

<h2>4. 保存与加载：三种格式</h2>
<h3>a) state_dict（最常用）</h3>
<pre data-lang="python"><code>torch.save(model.state_dict(), "model.pt")

# 加载：必须先有相同结构的模型
model = models.resnet18()
model.fc = nn.Linear(512, 5)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()</code></pre>

<h3>b) TorchScript（不依赖源码）</h3>
<pre data-lang="python"><code>scripted = torch.jit.script(model)     # 或 torch.jit.trace(model, example)
scripted.save("model_scripted.pt")

# 加载不需要原模型类定义
loaded = torch.jit.load("model_scripted.pt")
loaded.eval()</code></pre>

<h3>c) ONNX（跨框架、可被 C++/Java/TensorRT 用）</h3>
<pre data-lang="python"><code>dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)</code></pre>

<div class="callout warn">
  <span class="tag">导出前必须 eval()</span>
  <p>忘记 <code>model.eval()</code> 就导出，Dropout 会被固化进图里，推理结果变成随机的。这是部署阶段最常见的 bug。</p>
</div>

<h2>5. 一个正确的推理函数</h2>
<pre data-lang="python"><code>from PIL import Image
import torch

@torch.no_grad()
def predict(image_path, model, class_names, device="cpu"):
    model.eval()

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)   # 加 batch 维

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    idx = probs.argmax().item()

    return class_names[idx], probs[idx].item()

label, conf = predict("cat.jpg", model, ["cat", "dog", "bird", "fish", "rabbit"])
print(f"{label} {conf:.2%}")</code></pre>

<div class="callout warn">
  <span class="tag">四个必须检查的点</span>
  <ol>
    <li><code>model.eval()</code>：关掉 Dropout / BN 的训练行为。</li>
    <li><code>@torch.no_grad()</code>：不建图，省显存、提速。</li>
    <li><code>unsqueeze(0)</code>：单张图也要有 batch 维。</li>
    <li><strong>预处理必须和训练时完全一致</strong>：尺寸、归一化、通道顺序。这是「训练 99%、上线 60%」的头号原因。</li>
  </ol>
</div>

<h2>6. 上线前的检查清单</h2>
<table>
  <tr><th>项目</th><th>检查什么</th></tr>
  <tr><td>一致性</td><td>推理预处理 == 训练预处理（尺寸、mean/std、BGR/RGB）</td></tr>
  <tr><td>类别顺序</td><td><code>class_names</code> 的顺序必须和训练时的 label 映射一致</td></tr>
  <tr><td>数值稳定性</td><td>同一张图，训练环境与部署环境的输出是否几乎相同</td></tr>
  <tr><td>性能</td><td>单张延迟、批量吞吐、显存峰值</td></tr>
  <tr><td>边界情况</td><td>全黑图、超大图、透明通道图会不会崩</td></tr>
  <tr><td>版本</td><td>记录 torch 版本、模型文件哈希</td></tr>
</table>

<div class="callout tip">
  <span class="tag">最简单可靠的上线方式</span>
  <p>先用 FastAPI/Flask 包一层 HTTP 接口，加载 TorchScript 模型，前面加个队列。等真的有性能压力，再考虑 ONNX Runtime、TensorRT、vLLM 这类专门的推理引擎。</p>
</div>

<h2>7. 你的下一步</h2>
<ul>
  <li><strong>想深入 NLP</strong>：学 HuggingFace <code>transformers</code>，跑通一个 BERT 微调。</li>
  <li><strong>想深入视觉</strong>：读 ResNet、YOLO 的代码，做一次目标检测。</li>
  <li><strong>想深入工程</strong>：学 Lightning / Accelerate 组织训练代码，学 DDP 与实验管理（wandb）。</li>
  <li><strong>想打比赛</strong>：Kaggle 上从「tabular + 小 CNN」开始，把本章的迁移学习流程走熟。</li>
</ul>
<p>最重要的一件事：<strong>做完一个完整项目</strong>。从数据到上线，哪怕很小，也比看十个教程有用。</p>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li>迁移学习 = 用预训练权重 + 换掉最后的分类头。</li>
    <li>数据少只训头，数据多再逐步解冻，用差异化学习率。</li>
    <li>预处理必须和预训练模型一致，优先用 <code>weights.transforms()</code>。</li>
    <li>保存用 <code>state_dict</code>；要脱离源码用 TorchScript；跨平台用 ONNX。</li>
    <li>导出和推理前一定要 <code>eval()</code>，推理用 <code>no_grad</code>。</li>
    <li>上线前逐项核对预处理、类别顺序、数值一致性。</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q">只有 500 张图做 5 分类，最合理的做法是？</p>
  <ol class="opts">
    <li>从零训练一个 ResNet</li>
    <li>加载预训练 ResNet，冻结主体只训分类头</li>
    <li>用更大的模型从头训练</li>
    <li>不用 CNN，用全连接层</li>
  </ol>
  <p class="exp">数据量小，从头训练必然过拟合。迁移学习正是为这种场景设计的。</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">导出 ONNX 前忘记 <code>model.eval()</code>，会怎样？</p>
  <ol class="opts">
    <li>导出失败</li>
    <li>文件变大</li>
    <li>Dropout 等训练行为被固化，推理结果随机</li>
    <li>没有影响</li>
  </ol>
  <p class="exp">导出的是当时的计算图行为，eval 必须在导出前调用。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">单张图片推理时，为什么还要 <code>unsqueeze(0)</code>？</p>
  <ol class="opts">
    <li>模型要求输入带 batch 维，即使只有 1 张</li>
    <li>为了加速</li>
    <li>为了转成 float</li>
    <li>不需要</li>
  </ol>
  <p class="exp">模型输入形状是 (B, C, H, W)，单张就是 B=1。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>用预训练 ResNet18 微调一个「猫狗」二分类，只用 200 张图，记录准确率。</li>
    <li>对比「冻结主体」和「全解冻」两种策略的收敛速度与最终准确率。</li>
    <li>把模型导出为 TorchScript 和 ONNX，比较文件大小与单张推理耗时。</li>
    <li>写一个 <code>predict()</code> 函数，用同一张图分别在训练脚本内和独立加载的模型中推理，比较输出是否一致。</li>
  </ol>
</div>`
  },
  en: {
    title: 'Transfer learning and deployment',
    sub: 'Use someone else’s million images for your 200',
    html: `
<p class="lead">In practice you almost never train from scratch. You take a model pretrained on huge data, swap the head for your task, and fine-tune on a small dataset. This chapter covers how to modify, train, save and ship it.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Load a pretrained torchvision model and replace the head</li>
    <li>Decide what to freeze and which learning rates to use</li>
    <li>Save as state_dict, TorchScript or ONNX</li>
    <li>Write a correct inference function</li>
  </ul>
</div>

<h2>1. Why it works</h2>
<p>Early CNN features (edges, textures, colours) are <strong>generic</strong> — useful for cats and for cells alike. Only the last layers are task-specific.</p>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p>You already balance on a bike, so learning to drive reuses that. Learning from zero would rebuild every basic skill, needing far more data and time.</p>
</div>

<h2>2. Replace the head</h2>
<pre data-lang="python"><code>import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
print(model.fc)          # Linear(in_features=512, out_features=1000)

model.fc = nn.Linear(512, 5)
model = model.cuda()</code></pre>

<div class="callout warn">
  <span class="tag">Preprocessing must match</span>
  <p>Pretrained models expect specific normalisation. Use <code>models.ResNet18_Weights.DEFAULT.transforms()</code> rather than hand-rolling it.</p>
</div>

<h2>3. Freeze or fine-tune</h2>
<pre data-lang="python"><code># A: train the head only (tiny data)
for p in model.parameters():
    p.requires_grad = False
model.fc = nn.Linear(512, 5)
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)

# B: unfreeze all with discriminative LRs (medium data)
optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters()
                if not n.startswith("fc")], "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-3},
])</code></pre>

<table>
  <tr><th>Data</th><th>Strategy</th><th>LR</th></tr>
  <tr><td>&lt; 100 per class</td><td>head only</td><td>1e-3</td></tr>
  <tr><td>100–1000</td><td>unfreeze last blocks / all with small LR</td><td>1e-4 – 1e-5</td></tr>
  <tr><td>&gt; 1000</td><td>full fine-tune, possibly from scratch</td><td>1e-3 – 1e-4</td></tr>
</table>

<div class="callout tip">
  <span class="tag">Careful with BatchNorm</span>
  <p>With small batches, freeze BN by switching to eval mode; otherwise its running statistics are dragged around by noisy small batches.</p>
</div>

<h2>4. Saving: three formats</h2>
<h3>a) state_dict (default)</h3>
<pre data-lang="python"><code>torch.save(model.state_dict(), "model.pt")

model = models.resnet18()
model.fc = nn.Linear(512, 5)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()</code></pre>

<h3>b) TorchScript (no source needed)</h3>
<pre data-lang="python"><code>scripted = torch.jit.script(model)
scripted.save("model_scripted.pt")

loaded = torch.jit.load("model_scripted.pt")
loaded.eval()</code></pre>

<h3>c) ONNX (cross-framework)</h3>
<pre data-lang="python"><code>dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)</code></pre>

<div class="callout warn">
  <span class="tag">eval() before export</span>
  <p>Exporting without <code>eval()</code> bakes Dropout into the graph and inference becomes random. The most common deployment bug.</p>
</div>

<h2>5. A correct inference function</h2>
<pre data-lang="python"><code>@torch.no_grad()
def predict(image_path, model, class_names, device="cpu"):
    model.eval()

    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)   # add batch dim

    probs = torch.softmax(model(x), dim=1)[0]
    idx = probs.argmax().item()
    return class_names[idx], probs[idx].item()</code></pre>

<div class="callout warn">
  <span class="tag">Four mandatory checks</span>
  <ol>
    <li><code>model.eval()</code> — disable dropout / BN training behaviour.</li>
    <li><code>@torch.no_grad()</code> — no graph, less memory.</li>
    <li><code>unsqueeze(0)</code> — even one image needs a batch dim.</li>
    <li><strong>Preprocessing must exactly match training</strong>: size, normalisation, channel order. This is the #1 cause of "99% offline, 60% online".</li>
  </ol>
</div>

<h2>6. Pre-launch checklist</h2>
<table>
  <tr><th>Item</th><th>Check</th></tr>
  <tr><td>Consistency</td><td>Inference preprocessing == training preprocessing</td></tr>
  <tr><td>Label order</td><td><code>class_names</code> matches the training label mapping</td></tr>
  <tr><td>Numerics</td><td>Same image gives nearly identical logits in both environments</td></tr>
  <tr><td>Performance</td><td>Single-image latency, batch throughput, peak memory</td></tr>
  <tr><td>Edge cases</td><td>All-black image, huge image, RGBA image</td></tr>
  <tr><td>Versioning</td><td>Record torch version and model file hash</td></tr>
</table>

<div class="callout tip">
  <span class="tag">Simplest reliable deployment</span>
  <p>Wrap a TorchScript model in a FastAPI endpoint, add a queue, and only move to ONNX Runtime / TensorRT / vLLM when you actually have load pressure.</p>
</div>

<h2>7. What next</h2>
<ul>
  <li><strong>NLP</strong>: learn HuggingFace <code>transformers</code>, fine-tune one BERT.</li>
  <li><strong>Vision</strong>: read ResNet and YOLO code, train a detector.</li>
  <li><strong>Engineering</strong>: Lightning / Accelerate, DDP, experiment tracking.</li>
  <li><strong>Competitions</strong>: start with tabular plus a small CNN on Kaggle.</li>
</ul>
<p>Most importantly: <strong>finish one complete project</strong>, data to deployment. Even a tiny one beats ten tutorials.</p>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li>Transfer learning = pretrained weights + new head.</li>
    <li>Little data: head only. More data: gradual unfreezing, discriminative LRs.</li>
    <li>Match preprocessing; prefer <code>weights.transforms()</code>.</li>
    <li>state_dict for normal use, TorchScript to drop source deps, ONNX for portability.</li>
    <li>Always <code>eval()</code> before export and inference; use <code>no_grad</code>.</li>
    <li>Verify preprocessing, label order and numeric parity before launch.</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q">500 images, 5 classes. Best approach?</p>
  <ol class="opts">
    <li>Train a ResNet from scratch</li>
    <li>Load a pretrained ResNet, freeze the body, train the head</li>
    <li>Use a bigger model from scratch</li>
    <li>Skip CNNs, use dense layers</li>
  </ol>
  <p class="exp">From scratch will overfit. Transfer learning exists for exactly this.</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">Exporting ONNX without <code>model.eval()</code>:</p>
  <ol class="opts">
    <li>Fails</li>
    <li>Makes a bigger file</li>
    <li>Bakes training behaviour (dropout) into the graph — random outputs</li>
    <li>No effect</li>
  </ol>
  <p class="exp">The exported graph captures current behaviour.</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">Why <code>unsqueeze(0)</code> for a single image?</p>
  <ol class="opts">
    <li>The model expects a batch dimension even of size 1</li>
    <li>For speed</li>
    <li>To convert to float</li>
    <li>Not needed</li>
  </ol>
  <p class="exp">Inputs are (B, C, H, W).</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Fine-tune pretrained ResNet18 on 200 cat/dog images; record accuracy.</li>
    <li>Compare freezing vs full fine-tuning: convergence speed and final accuracy.</li>
    <li>Export to TorchScript and ONNX; compare size and latency.</li>
    <li>Write predict() and verify identical output in-script vs freshly loaded model.</li>
  </ol>
</div>`
  }
});
