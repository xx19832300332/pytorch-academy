(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch10',
  num: '10',
  part: 'p3',
  file: 'cnn',
  zh: {
    title: '卷积神经网络 CNN',
    sub: '让模型自己学会「看边缘、看纹理、看形状」',
    html: `
<p class="lead">用全连接层处理图片有两个致命问题：参数爆炸、丢掉空间结构。一张 224×224 的彩色图拉平是 15 万个数字，接一个 1000 维隐藏层就要 1.5 亿参数。<strong>卷积</strong>用「局部连接 + 参数共享」把这个问题解决掉，也是过去十年计算机视觉的基础。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>手算一个 3×3 卷积核在原图上的输出</li>
    <li>用公式算出任意卷积层的输出尺寸与参数量</li>
    <li>说清 kernel / stride / padding / channel 各自的含义</li>
    <li>用 nn.Conv2d 搭一个 LeNet 并在 MNIST 上训练</li>
  </ul>
</div>

<h2>1. 卷积在做什么</h2>
<p>拿一个小窗口（卷积核，kernel）在图上滑动，每滑到一个位置，就把窗口内的数字和核的权重<strong>对应相乘再求和</strong>，得到输出图上的一个点。</p>

<pre data-lang="python"><code># 手算一次卷积：3x3 的图，2x2 的核
img = [[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]]
k   = [[1, 0],
       [0, -1]]

# 输出左上角 = 1*1 + 2*0 + 4*0 + 5*(-1) = -4
# 输出右上角 = 2*1 + 3*0 + 5*0 + 6*(-1) = -4
# 输出左上、右上、左下、右下 = [[-4,-4],[-4,-4]]</code></pre>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p>卷积核就是一个小「模板」。比如 <code>[[1,0],[0,-1]]</code> 这个核，会用左上角减右下角——如果一块区域亮度从左上到右下变化剧烈，输出就大。所以它实际上是个<strong>边缘检测器</strong>。</p>
  <p>关键在于：<strong>我们不需要手工设计这个模板</strong>。给模型一批图片，它会自己学出「什么模板有用」。浅层学到的核通常像边缘/颜色检测器，深层则像「眼睛」「轮子」这种复杂模式的组合——这正是 CNN 可解释性的来源之一。</p>
</div>

<h2>2. 三个超参数：kernel、stride、padding</h2>
<table>
  <tr><th>参数</th><th>含义</th><th>典型值</th></tr>
  <tr><td><code>kernel_size</code></td><td>窗口大小</td><td>3（最常用）、5、7（第一层）</td></tr>
  <tr><td><code>stride</code></td><td>每次滑几步</td><td>1（保持尺寸）、2（下采样）</td></tr>
  <tr><td><code>padding</code></td><td>四周补几圈 0</td><td><code>kernel//2</code>，即 3×3 核配 padding=1</td></tr>
</table>

<div class="callout math">
  <span class="tag">数学批注 · 输出尺寸公式</span>
  <p>输入边长 <code>H</code>，卷积核 <code>K</code>，步长 <code>S</code>，填充 <code>P</code>：</p>
  <p class="formula">H_out = ⌊(H + 2P − K) / S⌋ + 1</p>
  <p>例：<code>H=32, K=3, S=1, P=1</code> → (32+2−3)/1 + 1 = <strong>32</strong>，尺寸不变。<br>
  例：<code>H=32, K=3, S=2, P=1</code> → (32+2−3)/2 + 1 = 16，减半。</p>
  <p>记法：<strong>padding 取 kernel//2 且 stride=1 时尺寸不变</strong>，这是最常见的配置。</p>
</div>

<div class="widget" data-widget="conv"></div>

<h2>3. 通道：从「一张图」到「一组特征图」</h2>
<pre data-lang="python"><code>import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
print(conv.weight.shape)     # (32, 3, 3, 3)
print(conv.bias.shape)       # (32,)

# 输入 (batch, 通道, 高, 宽)
x = torch.randn(8, 3, 32, 32)
print(conv(x).shape)         # (8, 32, 32, 32)</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 参数量怎么算</span>
  <p><code>Conv2d(in_ch, out_ch, k)</code> 的参数量 = <code>in_ch × out_ch × k × k + out_ch</code>。</p>
  <p>上面的例子：3×32×3×3 + 32 = <strong>896</strong> 个参数。对比同尺寸的全连接层（3×32×32 输入，32 输出，需要 98 万参数），量级完全不同——这就是卷积高效的原因。</p>
  <p>为什么这么少还能强？因为<strong>参数共享</strong>：同一个 3×3 的核在全图所有位置复用，相当于假设「图像不同位置的特征规律相同」。这个假设（平移不变性）恰好对图像成立。</p>
</div>

<div class="callout basic">
  <span class="tag">out_channels 是什么</span>
  <p>一个卷积核提取一种特征，32 个输出通道 = 用 32 个不同的核，各自找一种模式（边缘、斑块、颜色渐变…）。所以 <code>out_channels</code> 越大，这一层能表达的「特征种类」越多，代价是参数和计算量增加。常见配置是逐层加倍：32 → 64 → 128 → 256。</p>
</div>

<h2>4. 池化：把图缩小，保留主要信息</h2>
<pre data-lang="python"><code>pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.randn(8, 32, 32, 32)
print(pool(x).shape)          # (8, 32, 16, 16)  —— 宽高减半

gap = nn.AdaptiveAvgPool2d(1) # 全局平均池化，压成 (8,32,1,1)
print(gap(x).shape)</code></pre>

<p>MaxPool 取窗口内最大值，作用是：减少计算量、扩大感受野、提供一点平移鲁棒性。<code>AdaptiveAvgPool2d(1)</code> 则常用来替代巨大的全连接层，让模型能接受任意输入尺寸。</p>

<div class="callout math">
  <span class="tag">数学批注 · 感受野</span>
  <p>感受野是「输出图上一个点，对应原图多大的区域」。</p>
  <p>3×3 卷积叠 2 层，感受野是 5×5；叠 3 层是 7×7。所以<strong>两个 3×3 卷积 ≈ 一个 5×5 卷积，但参数更少、非线性更多</strong>（中间多了一次 ReLU）。这就是 VGG 之后大家都用 3×3 堆叠的原因。</p>
  <p>加 stride=2 或池化会显著加速感受野增长，这也是深层网络能「看到整张图」的机制。</p>
</div>

<h2>5. LeNet：完整可跑的例子</h2>
<pre data-lang="python"><code>import torch, torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),      # (1,28,28) -> (6,24,24)
            nn.ReLU(),
            nn.MaxPool2d(2),         # -> (6,12,12)
            nn.Conv2d(6, 16, 5),     # -> (16,8,8)
            nn.ReLU(),
            nn.MaxPool2d(2),         # -> (16,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),            # -> (16*4*4=256)
            nn.Linear(256, 120), nn.ReLU(),
            nn.Linear(120, 84),  nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = LeNet()
print(model(torch.randn(1, 1, 28, 28)).shape)    # (1, 10)</code></pre>

<p>把第 8 章的训练循环换掉模型就能直接跑，MNIST 上能达到 <strong>98.5% 以上</strong>。CNN 比全连接强的地方在于：它不会把相邻像素拆散，而是利用局部结构。</p>

<h2>6. CIFAR-10：带增强的真实训练</h2>
<pre data-lang="python"><code>from torchvision import datasets, transforms

tf_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),
])
tf_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),
])

train_ds = datasets.CIFAR10("data", train=True,  download=True, transform=tf_train)
test_ds  = datasets.CIFAR10("data", train=False, download=True, transform=tf_test)

class SmallCNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        self.net = nn.Sequential(
            *block(3, 32),   nn.MaxPool2d(2),     # 32 -> 16
            *block(32, 64),  nn.MaxPool2d(2),     # 16 -> 8
            *block(64, 128), nn.MaxPool2d(2),     # 8  -> 4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, n),
        )
    def forward(self, x):
        return self.net(x)</code></pre>

<p>这个不到 10 万参数的小网络，配合增强训练 20 轮，CIFAR-10 能到 <strong>75%~80%</strong>。要知道这是从 32×32 的模糊小图上区分 10 类物体，比 MNIST 难得多。</p>

<div class="callout tip">
  <span class="tag">实际项目怎么做</span>
  <p>几乎没人从零训练 CNN 解决真实问题。标准做法是<strong>用预训练的 ResNet / EfficientNet</strong> 换掉最后分类层（第 14 章）。只有数据量很大、或图像领域很特殊（医学、遥感）时，才值得从头训练。</p>
</div>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li>卷积 = 小窗口滑动做加权求和，参数全图共享。</li>
    <li>输出尺寸 <code>⌊(H+2P−K)/S⌋+1</code>；P=K//2、S=1 时尺寸不变。</li>
    <li>Conv2d 参数量 = in×out×k×k + out，远小于全连接。</li>
    <li><code>out_channels</code> 是「学多少种特征」，通常逐层加倍。</li>
    <li>池化降尺寸、扩感受野；两个 3×3 优于一个 5×5。</li>
    <li>真实项目中优先用预训练模型微调。</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q">输入 <code>(8, 3, 32, 32)</code> 经过 <code>Conv2d(3, 16, 3, stride=1, padding=1)</code>，输出形状是？</p>
  <ol class="opts">
    <li>(8, 3, 32, 32)</li>
    <li>(8, 16, 32, 32)</li>
    <li>(8, 16, 30, 30)</li>
    <li>(8, 16, 16, 16)</li>
  </ol>
  <p class="exp">out_channels 变成 16；padding=1 + stride=1 + k=3 保持尺寸 32×32。</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q"><code>Conv2d(6, 16, 5)</code> 的参数量（含偏置）是？</p>
  <ol class="opts">
    <li>6 × 16 = 96</li>
    <li>6×16×5 + 16 = 496</li>
    <li>6×16×5×5 + 16 = 2416</li>
    <li>16×5×5 = 400</li>
  </ol>
  <p class="exp">每个输出通道有 6 个 in_ch × 5×5 的核，共 16 个，加 16 个偏置。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">为什么用两个 3×3 卷积替代一个 5×5？</p>
  <ol class="opts">
    <li>感受野相同，但参数更少、非线性更强</li>
    <li>计算量更大但更准</li>
    <li>可以避免使用激活函数</li>
    <li>两个没有本质区别</li>
  </ol>
  <p class="exp">两个 3×3 感受野是 5×5：参数 2×9=18 少于 25，且中间多一层 ReLU，表达力更强。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>用页面的卷积尺寸计算器验证：输入 28、k=5、s=1、p=0 时输出是多少？（答案 24）</li>
    <li>训练 LeNet 到 98.5%，记录参数量与耗时。</li>
    <li>把第一层卷积核可视化出来（<code>conv.weight[0,0]</code> 用 matplotlib 画灰度图），看看学到了什么。</li>
    <li>把 MaxPool2d 换成 AvgPool2d，比较准确率差异。</li>
    <li>在 CIFAR-10 上去掉数据增强，看准确率掉多少。</li>
  </ol>
</div>`
  },
  en: {
    title: 'Convolutional networks',
    sub: 'Let the model learn to see edges, textures and shapes',
    html: `
<p class="lead">Fully connected layers on images have two fatal problems: parameter explosion and loss of spatial structure. A flattened 224×224 RGB image has 150k numbers; one 1000-unit hidden layer needs 150M weights. <strong>Convolution</strong> solves this with local connectivity and weight sharing.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Compute a 3×3 convolution by hand</li>
    <li>Derive the output size and parameter count of any conv layer</li>
    <li>Explain kernel, stride, padding and channels</li>
    <li>Build LeNet with <code>nn.Conv2d</code> and train it on MNIST</li>
  </ul>
</div>

<h2>1. What convolution does</h2>
<p>Slide a small window (the kernel) over the image; at each position multiply the window contents by the kernel weights and sum. That produces one pixel of the output map.</p>

<pre data-lang="python"><code>img = [[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]]
k   = [[1, 0],
       [0, -1]]

# top-left  = 1*1 + 2*0 + 4*0 + 5*(-1) = -4
# top-right = 2*1 + 3*0 + 5*0 + 6*(-1) = -4
# output = [[-4,-4],[-4,-4]]</code></pre>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p>A kernel is a template. <code>[[1,0],[0,-1]]</code> subtracts the bottom-right from the top-left, so it responds to intensity changes — an <strong>edge detector</strong>.</p>
  <p>The point is that <strong>nobody designs these templates</strong>. Given images, the model learns them. Early layers learn edges and colour blobs; deeper layers compose them into "eyes" and "wheels".</p>
</div>

<h2>2. Three hyperparameters</h2>
<table>
  <tr><th>Arg</th><th>Meaning</th><th>Typical</th></tr>
  <tr><td><code>kernel_size</code></td><td>window size</td><td>3 (default), 5, 7 (first layer)</td></tr>
  <tr><td><code>stride</code></td><td>step between positions</td><td>1 (keep size), 2 (downsample)</td></tr>
  <tr><td><code>padding</code></td><td>zero border</td><td><code>kernel//2</code></td></tr>
</table>

<div class="callout math">
  <span class="tag">Math note · output size</span>
  <p class="formula">H_out = ⌊(H + 2P − K) / S⌋ + 1</p>
  <p><code>H=32, K=3, S=1, P=1</code> → 32 (unchanged). <code>H=32, K=3, S=2, P=1</code> → 16 (halved).</p>
  <p>Mnemonic: padding = kernel//2 with stride 1 preserves the size — the most common configuration.</p>
</div>

<div class="widget" data-widget="conv"></div>

<h2>3. Channels</h2>
<pre data-lang="python"><code>conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
print(conv.weight.shape)     # (32, 3, 3, 3)

x = torch.randn(8, 3, 32, 32)
print(conv(x).shape)         # (8, 32, 32, 32)</code></pre>

<div class="callout math">
  <span class="tag">Math note · parameter count</span>
  <p><code>in_ch × out_ch × k × k + out_ch</code>. Here 3×32×3×3 + 32 = <strong>896</strong>, versus ~980k for a comparable dense layer.</p>
  <p>The saving comes from <strong>weight sharing</strong>: one 3×3 kernel reused at every position, encoding the assumption that image statistics are translation-invariant.</p>
</div>

<div class="callout basic">
  <span class="tag">What out_channels means</span>
  <p>One kernel extracts one kind of pattern. 32 output channels means 32 kernels looking for 32 patterns. More channels means more expressive capacity at higher cost. Typical: 32 → 64 → 128 → 256.</p>
</div>

<h2>4. Pooling</h2>
<pre data-lang="python"><code>pool = nn.MaxPool2d(2, 2)
x = torch.randn(8, 32, 32, 32)
print(pool(x).shape)          # (8, 32, 16, 16)

print(nn.AdaptiveAvgPool2d(1)(x).shape)   # (8, 32, 1, 1)</code></pre>

<div class="callout math">
  <span class="tag">Math note · receptive field</span>
  <p>The receptive field is how much of the original image one output pixel sees. Two stacked 3×3 convs give 5×5; three give 7×7.</p>
  <p>Two 3×3 ≈ one 5×5 in receptive field, but with fewer parameters (18 vs 25) and an extra non-linearity. That is why VGG popularised stacking 3×3. Stride-2 or pooling grows the receptive field quickly.</p>
</div>

<h2>5. LeNet: a complete example</h2>
<pre data-lang="python"><code>class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),      # (1,28,28) -> (6,24,24)
            nn.ReLU(),
            nn.MaxPool2d(2),         # -> (6,12,12)
            nn.Conv2d(6, 16, 5),     # -> (16,8,8)
            nn.ReLU(),
            nn.MaxPool2d(2),         # -> (16,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*4*4, 120), nn.ReLU(),
            nn.Linear(120, 84),  nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))</code></pre>

<p>Swap the model into the Chapter 8 loop and you get <strong>98.5%+</strong> on MNIST.</p>

<h2>6. CIFAR-10 with augmentation</h2>
<pre data-lang="python"><code>tf_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),
])

class SmallCNN(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        self.net = nn.Sequential(
            *block(3, 32),   nn.MaxPool2d(2),
            *block(32, 64),  nn.MaxPool2d(2),
            *block(64, 128), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, n),
        )
    def forward(self, x):
        return self.net(x)</code></pre>

<p>Under 100k parameters plus augmentation: <strong>75–80%</strong> after ~20 epochs on CIFAR-10, which is much harder than MNIST.</p>

<div class="callout tip">
  <span class="tag">In real projects</span>
  <p>Almost nobody trains a CNN from scratch. Standard practice: take a pretrained ResNet/EfficientNet and replace the head (Chapter 14).</p>
</div>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li>Convolution = sliding window weighted sum with shared weights.</li>
    <li>Output size <code>⌊(H+2P−K)/S⌋+1</code>; P=K//2, S=1 keeps the size.</li>
    <li>Params = in×out×k×k + out.</li>
    <li><code>out_channels</code> = how many patterns; usually doubles with depth.</li>
    <li>Pooling shrinks size and grows receptive field; two 3×3 beat one 5×5.</li>
    <li>In practice fine-tune a pretrained model.</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q"><code>(8,3,32,32)</code> through <code>Conv2d(3,16,3,stride=1,padding=1)</code> gives:</p>
  <ol class="opts">
    <li>(8, 3, 32, 32)</li>
    <li>(8, 16, 32, 32)</li>
    <li>(8, 16, 30, 30)</li>
    <li>(8, 16, 16, 16)</li>
  </ol>
  <p class="exp">Channels become 16; padding 1 with stride 1 keeps 32×32.</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">Parameters in <code>Conv2d(6,16,5)</code> including bias:</p>
  <ol class="opts">
    <li>96</li>
    <li>496</li>
    <li>2416</li>
    <li>400</li>
  </ol>
  <p class="exp">6×16×5×5 + 16 = 2416.</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">Why replace one 5×5 with two 3×3?</p>
  <ol class="opts">
    <li>Same receptive field, fewer parameters, more non-linearity</li>
    <li>More compute but more accurate</li>
    <li>It avoids activations entirely</li>
    <li>No real difference</li>
  </ol>
  <p class="exp">18 vs 25 weights and an extra ReLU in between.</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Check with the widget: H=28, k=5, s=1, p=0 → 24.</li>
    <li>Train LeNet to 98.5%; record parameters and time.</li>
    <li>Visualise <code>conv.weight[0,0]</code> as a grayscale image. What did it learn?</li>
    <li>Swap MaxPool for AvgPool and compare accuracy.</li>
    <li>Remove augmentation on CIFAR-10 and measure the drop.</li>
  </ol>
</div>`
  }
});
