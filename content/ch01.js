(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch01',
  num: '01',
  part: 'p0',
  file: 'setup',
  zh: {
    title: '环境搭建与第一个张量',
    sub: '装好、跑通、能改——三步完成今天的学习',
    html: `
<p class="lead">这一章只做一件事：让 <code>import torch</code> 在你机器上成功执行，并且知道以后每段代码该写在哪里跑。配置环境是新手劝退率最高的环节，但只要你严格按顺序做，成功率很高。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>用 conda 或 venv 建一个独立环境，装好 PyTorch 和 Jupyter</li>
    <li>验证安装：打印版本号、检查 GPU 是否可用</li>
    <li>知道 .ipynb 是什么，会新建、运行、保存 notebook</li>
    <li>看懂最常见的 5 个报错并自己修</li>
  </ul>
</div>

<h2>1. 为什么一定要用「虚拟环境」</h2>
<p>如果你直接 <code>pip install torch</code> 装进系统 Python，迟早会遇到：项目 A 要 torch 1.x、项目 B 要 2.x，两个互相覆盖，最后系统 Python 一团糟。虚拟环境就是给每个项目一个独立的「工具箱」。</p>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p>可以用「厨房」来理解：系统 Python 是公共厨房，谁都能动；虚拟环境是你自己开的小灶，锅碗瓢盆互不干扰，弄坏了删掉重开一个就好。</p>
</div>

<h2>2. 安装：选一条路走</h2>
<h3>路线 A：conda（推荐新手，能顺便装 CUDA）</h3>
<pre data-lang="bash"><code># 1) 建一个叫 pt 的环境，指定 Python 3.11
conda create -n pt python=3.11 -y

# 2) 进入环境（每次开新终端都要执行）
conda activate pt

# 3) 安装 PyTorch（CPU 版，最省事）
conda install pytorch torchvision torchaudio -c pytorch -y

# 4) 再装 notebook 和常用库
conda install jupyterlab matplotlib numpy pandas -y</code></pre>

<h3>路线 B：venv + pip（系统里已有 Python 3.9+）</h3>
<pre data-lang="bash"><code>python -m venv .venv

# Windows
.venv\\Scripts\\activate
# macOS / Linux
source .venv/bin/activate

# 安装（CPU 版）
pip install torch torchvision torchaudio
pip install jupyterlab matplotlib</code></pre>

<div class="callout tip">
  <span class="tag">装 GPU 版</span>
  <p>不要自己猜命令。打开 pytorch.org 首页的安装向导，选择你的系统、包管理器、CUDA 版本，它会生成一行准确的命令。用 conda 装会连带 CUDA 运行库，用 pip 装则需要你驱动版本足够新。</p>
  <p>没有 NVIDIA 显卡就安心装 CPU 版，本课程所有示例都能在 CPU 上跑完（MNIST 训练一次大约几分钟）。</p>
</div>

<h2>3. 验证安装</h2>
<pre data-lang="python"><code>import torch

print(torch.__version__)          # 例如 2.3.1+cu121
print(torch.cuda.is_available())  # True 表示能用 GPU
print(torch.cuda.device_count())  # 显卡数量

# 第一段真正跑起来的 PyTorch 代码
x = torch.tensor([1.0, 2.0, 3.0])
print(x + 1)                      # tensor([2., 3., 4.])
print(x.device)                   # cpu</code></pre>

<p>如果这几行都正常输出，恭喜，环境就绪。如果 <code>torch.cuda.is_available()</code> 返回 <code>False</code> 但你有显卡，通常是驱动太旧或装成了 CPU 版，不必纠结，先用 CPU 学完前 12 章完全没问题。</p>

<h2>4. Jupyter：本课程的主要练习场</h2>
<p>Notebook 把代码切成一个个「单元格（cell）」，可以一格一格运行，结果（包括图）直接显示在下面。学习阶段它比写 .py 文件高效得多，因为你能立刻看到每一步的中间结果。</p>

<ol class="steps">
  <li>在项目目录下打开终端，激活环境，运行 <code>jupyter lab</code>（或 <code>jupyter notebook</code>）。</li>
  <li>浏览器自动打开后，点 Python 3 新建一个 notebook。</li>
  <li>在第一个 cell 里输入 <code>import torch</code>，按 <strong>Shift + Enter</strong> 运行，左侧方括号会变成数字，说明跑过了。</li>
  <li>本课程每章右上角都有「下载 Notebook」，下载后拖进 Jupyter 的文件列表即可打开，所有代码块已经按顺序排好。</li>
</ol>

<div class="callout warn">
  <span class="tag">最常见的坑：内核不对</span>
  <p>Notebook 报 <code>ModuleNotFoundError: No module named 'torch'</code>，但你明明装过？几乎一定是「notebook 用的 Python 不是你装包的那个环境」。解决办法：在目标环境里执行 <code>pip install ipykernel</code>，然后 <code>python -m ipykernel install --user --name pt --display-name "Python (pt)"</code>，再在 notebook 右上角内核菜单里选它。</p>
</div>

<h2>5. 五个高频报错对照表</h2>
<table>
  <tr><th>报错关键词</th><th>真实原因</th><th>怎么修</th></tr>
  <tr><td><code>No module named 'torch'</code></td><td>环境没激活 / 内核选错</td><td>激活环境；或给该环境装 ipykernel 后重选内核</td></tr>
  <tr><td><code>CUDA out of memory</code></td><td>显存不够</td><td>减小 batch_size；<code>del</code> 无用变量 + <code>torch.cuda.empty_cache()</code></td></tr>
  <tr><td><code>shapes cannot be multiplied</code></td><td>矩阵乘法维度不匹配</td><td>打印 shape，检查 <code>(N,in)@(in,out)</code></td></tr>
  <tr><td><code>expected scalar type Float but found Double</code></td><td>dtype 不一致</td><td><code>x.float()</code> 或统一用 <code>torch.float32</code></td></tr>
  <tr><td><code>RuntimeError: expected ... device cuda but got cpu</code></td><td>模型在 GPU、数据在 CPU（或相反）</td><td>两边都 <code>.to(device)</code></td></tr>
</table>

<h2>6. 建议的目录习惯</h2>
<pre data-lang="bash"><code>pytorch-study/
├── notebooks/          # 每个练习一个 .ipynb
├── data/               # 数据集下载到这里
├── checkpoints/        # 训练好的模型权重
└── src/                # 写成熟后抽出来的 .py 模块</code></pre>
<p>不要把所有东西堆在桌面。三个月后你会感谢现在分类的自己。</p>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li>永远在虚拟环境里安装，不要污染系统 Python。</li>
    <li>CPU 版足够学完基础；要 GPU 版请用官网向导生成的命令。</li>
    <li>验证三件事：版本号、<code>cuda.is_available()</code>、能跑 <code>x + 1</code>。</li>
    <li>Notebook 的「内核」必须指向你装包的那个环境，否则一定 ModuleNotFoundError。</li>
    <li>Shift + Enter 运行当前 cell；改代码后要重新运行，变量才会更新。</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q">在 notebook 里 <code>import torch</code> 报 ModuleNotFoundError，但终端里 <code>pip list</code> 能看到 torch。最可能的原因是？</p>
  <ol class="opts">
    <li>torch 安装损坏，需要重装系统</li>
    <li>notebook 的内核指向了另一个 Python 环境</li>
    <li>没有联网</li>
    <li>必须用管理员权限运行</li>
  </ol>
  <p class="exp">这是最典型的环境错配。装包的环境和跑代码的解释器不是同一个，包自然「看不见」。装 ipykernel 并切换内核即可。</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">关于虚拟环境，下面哪句是对的？</p>
  <ol class="opts">
    <li>一个虚拟环境可以同时服务所有项目，越多包越好</li>
    <li>虚拟环境会自动把包同步到系统 Python</li>
    <li>每个项目一个环境，互相隔离，删掉不心疼</li>
    <li>只有 Linux 才需要虚拟环境</li>
  </ol>
  <p class="exp">隔离正是它的目的。项目之间依赖版本冲突时，各自独立的环境能让你随时「推倒重来」而不用重装系统。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>建好环境后，打印 <code>torch.__version__</code> 和 <code>torch.cuda.is_available()</code>，截图或记下来。</li>
    <li>下载本章 Notebook，在 Jupyter 里从头到尾跑一遍（菜单里有 Run All）。</li>
    <li>故意把 <code>torch.tensor([1,2,3])</code> 和 <code>torch.tensor([1.0,2.0])</code> 相加，看看报错长什么样，读懂它再改对。</li>
  </ol>
</div>`
  },
  en: {
    title: 'Setup and your first tensor',
    sub: 'Install it, run it, change it — that is day one',
    html: `
<p class="lead">This chapter has one goal: make <code>import torch</code> succeed on your machine, and know where to write code from now on. Environment setup is where most beginners give up, but follow the order and it usually just works.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Create an isolated environment with conda or venv, install PyTorch and Jupyter</li>
    <li>Verify the install: print the version, check GPU availability</li>
    <li>Know what a .ipynb is, and create / run / save one</li>
    <li>Recognise and fix the five most common errors</li>
  </ul>
</div>

<h2>1. Why a virtual environment is mandatory</h2>
<p>Install straight into the system Python and sooner or later project A needs torch 1.x while project B needs 2.x. They overwrite each other and the whole install rots. A virtual environment gives each project its own toolbox.</p>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p>Think of the system Python as a shared kitchen. A virtual environment is your own small stove: your pots and pans, no interference, and if you break it you delete the folder and start again.</p>
</div>

<h2>2. Install: pick one path</h2>
<h3>Path A — conda (easiest for beginners)</h3>
<pre data-lang="bash"><code># 1) create an env named pt with Python 3.11
conda create -n pt python=3.11 -y

# 2) activate it (every new terminal)
conda activate pt

# 3) install PyTorch (CPU build)
conda install pytorch torchvision torchaudio -c pytorch -y

# 4) notebooks and friends
conda install jupyterlab matplotlib numpy pandas -y</code></pre>

<h3>Path B — venv + pip</h3>
<pre data-lang="bash"><code>python -m venv .venv

# Windows
.venv\\Scripts\\activate
# macOS / Linux
source .venv/bin/activate

pip install torch torchvision torchaudio
pip install jupyterlab matplotlib</code></pre>

<div class="callout tip">
  <span class="tag">GPU build</span>
  <p>Do not guess the command. Open the installer on pytorch.org, pick your OS, package manager and CUDA version, and copy the exact line it generates. conda bundles the CUDA runtime; pip needs a recent enough driver.</p>
  <p>No NVIDIA GPU? Install the CPU build and move on. Every example in this course runs on CPU (one MNIST run takes a few minutes).</p>
</div>

<h2>3. Verify</h2>
<pre data-lang="python"><code>import torch

print(torch.__version__)          # e.g. 2.3.1+cu121
print(torch.cuda.is_available())  # True means GPU is usable
print(torch.cuda.device_count())

x = torch.tensor([1.0, 2.0, 3.0])
print(x + 1)                      # tensor([2., 3., 4.])
print(x.device)                   # cpu</code></pre>

<p>If those print, you are ready. If <code>cuda.is_available()</code> is False but you do have a GPU, it is usually an old driver or a CPU-only build — not worth fighting now; Chapters 0–12 are fine on CPU.</p>

<h2>4. Jupyter: your practice field</h2>
<p>A notebook splits code into cells you run one at a time, with outputs (including plots) shown right below. While learning, that beats a .py file because you see every intermediate step.</p>

<ol class="steps">
  <li>Open a terminal in your project folder, activate the env, run <code>jupyter lab</code>.</li>
  <li>Create a new Python 3 notebook.</li>
  <li>Type <code>import torch</code> and press <strong>Shift + Enter</strong>. The bracket on the left becomes a number when it has run.</li>
  <li>Every chapter here has a "Download notebook" button — drag the .ipynb into Jupyter and all code cells are already ordered.</li>
</ol>

<div class="callout warn">
  <span class="tag">Classic trap: wrong kernel</span>
  <p><code>ModuleNotFoundError: No module named 'torch'</code> right after installing means the notebook is using a different Python. Fix: in the target env run <code>pip install ipykernel</code>, then <code>python -m ipykernel install --user --name pt --display-name "Python (pt)"</code>, then pick that kernel.</p>
</div>

<h2>5. Five errors you will hit</h2>
<table>
  <tr><th>Message</th><th>Real cause</th><th>Fix</th></tr>
  <tr><td><code>No module named 'torch'</code></td><td>Env not activated / wrong kernel</td><td>Activate, or install ipykernel and re-select</td></tr>
  <tr><td><code>CUDA out of memory</code></td><td>Not enough VRAM</td><td>Smaller batch; <code>del</code> + <code>torch.cuda.empty_cache()</code></td></tr>
  <tr><td><code>shapes cannot be multiplied</code></td><td>Matmul dims mismatch</td><td>Print shapes, check <code>(N,in)@(in,out)</code></td></tr>
  <tr><td><code>expected scalar type Float but found Double</code></td><td>dtype mismatch</td><td><code>x.float()</code></td></tr>
  <tr><td><code>expected device cuda but got cpu</code></td><td>Model and data on different devices</td><td><code>.to(device)</code> both</td></tr>
</table>

<h2>6. A sane folder layout</h2>
<pre data-lang="bash"><code>pytorch-study/
├── notebooks/          # one .ipynb per practice session
├── data/               # downloaded datasets
├── checkpoints/        # saved weights
└── src/                # .py modules once code matures</code></pre>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li>Always install inside a virtual environment.</li>
    <li>CPU build is enough for the basics; get the GPU command from the official installer page.</li>
    <li>Verify three things: version, <code>cuda.is_available()</code>, and <code>x + 1</code>.</li>
    <li>The notebook kernel must point at the env where you installed packages.</li>
    <li>Shift + Enter runs a cell; re-run after editing or old variables stay in memory.</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q"><code>import torch</code> fails in the notebook, but <code>pip list</code> in the terminal shows torch. Most likely cause?</p>
  <ol class="opts">
    <li>The install is corrupt and needs a new OS</li>
    <li>The notebook kernel points at a different Python</li>
    <li>No internet connection</li>
    <li>You need administrator rights</li>
  </ol>
  <p class="exp">Classic environment mismatch: the package lives in one interpreter, the code runs in another. Install ipykernel and switch kernels.</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">Which statement about virtual environments is correct?</p>
  <ol class="opts">
    <li>One env should serve every project, more packages is better</li>
    <li>They sync packages back into the system Python</li>
    <li>One env per project, isolated, safe to delete and recreate</li>
    <li>They are only needed on Linux</li>
  </ol>
  <p class="exp">Isolation is the point: conflicts between projects stop being your problem, and you can always nuke and rebuild.</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Print <code>torch.__version__</code> and <code>torch.cuda.is_available()</code> and note them down.</li>
    <li>Download this chapter's notebook and use Run All.</li>
    <li>Deliberately add <code>torch.tensor([1,2,3])</code> and <code>torch.tensor([1.0,2.0])</code>, read the error, then fix it.</li>
  </ol>
</div>`
  }
});
