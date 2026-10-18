/* 练习工具使用指南（独立页面，#/practice） */
window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] };

window.PYTORCH_COURSE.practice = {
  zh: {
    kicker: '练习工具 · 使用指南',
    title: '练习工具怎么用',
    sub: '下载 notebook、跑起来、改数字——把「看懂」变成「会用」',
    html: `
<p class="lead">这个网站有三类练习工具，管的事不一样，别混着用：<b>Notebook</b> 让你真的敲代码、看输出；<b>章末小测</b>检查你有没有读懂；<b>交互计算器</b>帮你把抽象的规则变成手感。下面是一份从零到跑通的操作说明，照着做就行。</p>

<table>
  <thead><tr><th>工具</th><th>在哪里</th><th>解决什么问题</th><th>花多久</th></tr></thead>
  <tbody>
    <tr><td>下载 Notebook</td><td>每章标题右侧的紫色按钮</td><td>动手写代码、看真实输出</td><td>20–30 分钟 / 章</td></tr>
    <tr><td>章末小测</td><td>每章结尾的「小测」区块</td><td>确认概念没搞反</td><td>3–5 分钟</td></tr>
    <tr><td>交互计算器</td><td>第 3 章（广播）、第 10 章（卷积）</td><td>建立形状变化的直觉</td><td>5 分钟，可反复玩</td></tr>
  </tbody>
</table>

<h2>1. Notebook：五步跑起来</h2>

<h3>第 1 步：下载</h3>
<p>打开任意一章（比如第 4 章「自动求导」），点标题栏右侧的 <b>「下载 Notebook」</b>。浏览器会把这个章节的正文、代码、练习一起打包成一个 <code>.ipynb</code> 文件，默认落在下载文件夹里，文件名形如：</p>
<pre data-lang="text"><code>C:\\Users\\LENOVO\\Downloads\\pytorch-04-autograd.ipynb</code></pre>
<div class="callout warn">
  <span class="tag">点了没反应？</span>
  <p>先按 <kbd>Ctrl</kbd>+<kbd>J</kbd> 打开下载列表，看文件是不是已经悄悄下好了（有些浏览器不弹提示）。如果显示「已拦截」，点列表里的「保留 / 允许」即可。这个按钮是用浏览器原生的 Blob 下载实现的，不需要联网，也不会上传任何东西。</p>
</div>

<h3>第 2 步：先搞清 .ipynb 是什么</h3>
<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p><code>.ipynb</code> 不是 Python 脚本，它是一个 <b>JSON 文本文件</b>，里面按顺序存着「一格一格的单元格」：有的格是说明文字（Markdown），有的格是代码。所以：<b>不能双击打开</b>（默认会用记事本打开，看到一堆花括号，别慌，文件没坏），也<b>不能</b>用 <code>python xxx.ipynb</code> 运行。它必须由 Jupyter 来解释。</p>
</div>

<h3>第 3 步：放进 Jupyter 能看到的目录</h3>
<p>三条路线，按你机器上已有的东西选一条，不要三条都装。</p>

<p><b>路线 A：你已经有 WSL 里的 Jupyter（你现在就是这套，最省事）</b></p>
<p>WSL 能直接读到 Windows 的下载文件夹，路径是 <code>/mnt/c/Users/你的用户名/Downloads</code>。打开 WSL 终端，依次执行：</p>
<pre data-lang="bash"><code># 1) 建一个专门放练习本的文件夹
mkdir -p ~/pytorch-practice

# 2) 把刚下载的 notebook 复制过去（* 表示所有章节）
cp /mnt/c/Users/LENOVO/Downloads/pytorch-*.ipynb ~/pytorch-practice/

# 3) 进目录、启动 Jupyter（装的是 lab 就用 lab）
cd ~/pytorch-practice
jupyter lab</code></pre>
<p>启动后终端会打印一行带 token 的地址，形如 <code>http://127.0.0.1:8888/lab?token=xxxx</code>。把它复制到 Windows 的浏览器地址栏打开即可（WSL 的 localhost 和 Windows 是通的）。左侧文件列表里点开 <code>pytorch-04-autograd.ipynb</code>。</p>

<p><b>路线 B：Windows 原生 Python</b></p>
<pre data-lang="bash"><code># 在下载目录里直接启动 Jupyter，最不容易找不到文件
cd C:\\Users\\LENOVO\\Downloads
python -m pip install jupyterlab
python -m jupyter lab</code></pre>

<p><b>路线 C：一点环境都不想装</b></p>
<p>打开 <code>colab.research.google.com</code> → 文件 → 上传笔记本 → 选这个 .ipynb。缺点是文件会传到云端，且需要联网；优点是自带 GPU，第 13 章（显存与混合精度）用它更舒服。</p>

<h3>第 4 步：逐格运行</h3>
<table>
  <thead><tr><th>快捷键</th><th>作用</th><th>什么时候用</th></tr></thead>
  <tbody>
    <tr><td><kbd>Shift</kbd>+<kbd>Enter</kbd></td><td>运行本格，跳到下一格</td><td>从头顺着学，最常用</td></tr>
    <tr><td><kbd>Ctrl</kbd>+<kbd>Enter</kbd></td><td>运行本格，停在原地</td><td>反复改同一格试参数</td></tr>
    <tr><td><kbd>Alt</kbd>+<kbd>Enter</kbd></td><td>运行并在下面插一格新的</td><td>自己想加实验</td></tr>
    <tr><td><kbd>Esc</kbd> 然后 <kbd>B</kbd></td><td>在下方插入单元格</td><td>写自己的代码</td></tr>
  </tbody>
</table>
<div class="callout tip">
  <span class="tag">顺序很重要</span>
  <p>左边的 <code>[1] [2] [3]</code> 是执行序号，不是行号。Jupyter 的变量是「共享的」：跳着跑会用到上一次留下的变量，结果可能莫名其妙。遇到诡异结果，先点菜单 <b>Kernel → Restart &amp; Run All</b>，从头跑一遍。</p>
</div>

<h3>第 5 步：改数字，看变化</h3>
<p>每个 notebook 的末尾都留了「改一改」的格子。以第 4 章为例，原代码算的是 <code>y = x**2</code> 在 <code>x=3</code> 处的梯度（答案是 6）。你把它改成 <code>x = 5</code>，先在心里算 <code>2x = 10</code>，再运行验证。这种「先预测、再验证」的动作，比读十遍文字都管用。</p>

<h2>2. 开跑之前：三行环境自检</h2>
<p>新建一格，把下面这段粘进去运行。这是排查一切问题的第一步——先确认你用哪个 Python、装没装 torch、有没有显卡。</p>
<pre data-lang="python"><code>import sys, platform
import torch

print('Python :', sys.version.split()[0], '|', platform.system())
print('解释器 :', sys.executable)
print('torch  :', torch.__version__)
print('CUDA   :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('显卡   :', torch.cuda.get_device_name(0))
x = torch.tensor([1.0, 2.0, 3.0])
print('张量   :', x, x.dtype, x.shape)</code></pre>
<p>输出怎么读：</p>
<ul>
  <li><code>torch 2.x.x</code> 正常打出 → 环境通了，可以开始。</li>
  <li><code>CUDA False</code> → 你在用 CPU。本课程第 1–12、14 章在 CPU 上完全跑得动，只有第 13 章需要 GPU，届时去 Colab 或租卡即可，不必现在折腾驱动。</li>
  <li><code>ModuleNotFoundError: No module named 'torch'</code> → 装到了另一个 Python 上。看 <code>sys.executable</code> 打印的路径，用它对应的 pip 重装：<code>python -m pip install torch</code>。</li>
</ul>

<h2>3. 章末小测怎么用</h2>
<p>每章末尾有 3 道单选题，<b>点一下选项就判卷</b>：正确项变绿，你点错的那项变红，并且立刻锁住——不能重答，这是故意的，避免「挨个试出来」的假懂。</p>
<p>答错了怎么办：不要急着往下翻，回到上面的正文找那一段重读一次，然后<b>用自己的话在笔记里写一句「我刚才错在……」</b>。笔记（右上角「笔记」按钮）会按章自动保存，也能一键导出成 <code>.md</code> 归档。</p>
<div class="callout basic">
  <span class="tag">小测的答案会记住</span>
  <p>你的作答状态会跟着「重点标记」一起存进浏览器本地存储，下次打开这一章还是判过的样子。想重做？清空这一章的高亮标记（选中已标记的文字 → 工具栏「取消」），或按首页底部的「清空全部学习数据」重来。</p>
</div>

<h2>4. 两个交互计算器怎么玩</h2>
<h3>广播形状计算器（第 3 章）</h3>
<p>输入两个形状，比如 <code>3,1</code> 和 <code>1,4</code>，点「计算」。它会把两个形状<b>右对齐</b>排好，逐维比较：相等就留下，有一边是 1 就扩展成另一边，都不是就报错。所以 <code>3,1 × 1,4 → 3,4</code>；而 <code>3,2 × 3,4</code> 在最后一维 2 和 4 上冲突，直接失败。</p>
<p>玩法建议：先自己用纸笔猜结果，再点计算。连续猜对 5 个，广播就再也不会错了。</p>

<h3>卷积输出尺寸计算器（第 10 章）</h3>
<p>填输入高宽 <code>H, W</code>、卷积核 <code>K</code>、填充 <code>P</code>、步长 <code>S</code>，它同时给出<b>输出尺寸</b>和<b>参数量</b>。公式是：</p>
<p class="formula">H_out = ⌊(H + 2P − K) / S⌋ + 1</p>
<p>拿它验证一个经典事实：<code>K=3, P=1, S=1</code> 时输出尺寸不变（这叫 same padding）；把 <code>S</code> 改成 2，尺寸就腰斩。参数量只和 <code>K, C_in, C_out</code> 有关，和输入多大完全无关——这一点是很多人第一次听会惊讶的地方。</p>

<h2>5. 常见故障对照表</h2>
<table>
  <thead><tr><th>现象</th><th>原因</th><th>解决</th></tr></thead>
  <tbody>
    <tr><td>点「下载 Notebook」没反应</td><td>浏览器静默下载或拦截</td><td><kbd>Ctrl</kbd>+<kbd>J</kbd> 看下载列表，允许保留</td></tr>
    <tr><td>双击 .ipynb 打开是花括号</td><td>被记事本接管了</td><td>文件没坏，改用 <code>jupyter lab</code> 打开</td></tr>
    <tr><td>Jupyter 文件列表里看不到文件</td><td>启动目录不对</td><td>先 <code>cd</code> 到 notebook 所在目录再启动</td></tr>
    <tr><td><code>No module named 'torch'</code></td><td>装到了别的 Python</td><td>跑自检代码看 <code>sys.executable</code>，用它装</td></tr>
    <tr><td><code>CUDA out of memory</code></td><td>显存不够</td><td>把 <code>batch_size</code> 减半，或 <code>del</code> 掉不用的张量</td></tr>
    <tr><td>改了代码但结果没变</td><td>跑的不是你以为的那一格</td><td>Kernel → Restart &amp; Run All</td></tr>
  </tbody>
</table>

<h2>6. 一天一章的节奏</h2>
<table>
  <thead><tr><th>环节</th><th>时长</th><th>做什么</th></tr></thead>
  <tbody>
    <tr><td>读</td><td>20 分钟</td><td>读正文，遇到「零基础批注 / 数学批注」停下来看懂</td></tr>
    <tr><td>跑</td><td>20 分钟</td><td>下载 notebook，从头 Shift+Enter 跑到底，看每格输出</td></tr>
    <tr><td>改</td><td>10 分钟</td><td>改形状、改数字、改学习率，先预测再运行</td></tr>
    <tr><td>测</td><td>5 分钟</td><td>做小测；错了就回读，写一句「错在哪」</td></tr>
  </tbody>
</table>
<p>16 章按这个节奏大约三周。别贪快——<b>能跑通、能改对</b>比看完更重要。</p>

<div class="keypoints">
  <h4>记住这五点</h4>
  <ul>
    <li>Notebook 要用 Jupyter 打开，双击和 <code>python xxx.ipynb</code> 都不行</li>
    <li>WSL 里读 Windows 下载目录：<code>/mnt/c/Users/LENOVO/Downloads</code></li>
    <li>遇到诡异结果，第一反应是 Restart &amp; Run All</li>
    <li>小测答错不能重答，所以先复习再点</li>
    <li>先预测、再运行——这是练习和看视频的区别</li>
  </ul>
</div>
`
  },
  en: {
    kicker: 'Practice tools · How-to',
    title: 'How to use the practice tools',
    sub: 'Download a notebook, run it, change the numbers — turn reading into doing',
    html: `
<p class="lead">Three kinds of practice tools, each with a different job. <b>Notebooks</b> make you write code and see real output. <b>Quizzes</b> check whether you actually understood. <b>Interactive calculators</b> turn abstract rules into intuition. Here is the full walkthrough.</p>

<table>
  <thead><tr><th>Tool</th><th>Where</th><th>Solves</th><th>Time</th></tr></thead>
  <tbody>
    <tr><td>Download notebook</td><td>Purple button, top right of every chapter</td><td>Hands-on coding, real output</td><td>20–30 min / chapter</td></tr>
    <tr><td>Chapter quiz</td><td>End of every chapter</td><td>Confirm you did not flip a concept</td><td>3–5 min</td></tr>
    <tr><td>Interactive widgets</td><td>Chapter 3 (broadcasting), chapter 10 (conv)</td><td>Build shape intuition</td><td>5 min, replayable</td></tr>
  </tbody>
</table>

<h2>1. Notebooks: five steps to running</h2>

<h3>Step 1: Download</h3>
<p>Open any chapter and click <b>Download notebook</b>. The whole chapter — prose, code, exercises — is packaged into a single <code>.ipynb</code> file in your downloads folder:</p>
<pre data-lang="text"><code>C:\\Users\\LENOVO\\Downloads\\pytorch-04-autograd.ipynb</code></pre>
<div class="callout warn">
  <span class="tag">Nothing happened?</span>
  <p>Press <kbd>Ctrl</kbd>+<kbd>J</kbd> to open the download list — it may have saved silently. If it says "blocked", choose Keep/Allow. The button uses the browser's native Blob download: no network, no upload.</p>
</div>

<h3>Step 2: Know what a .ipynb is</h3>
<div class="callout basic">
  <span class="tag">Beginner note</span>
  <p>A <code>.ipynb</code> is <b>not</b> a Python script. It is a JSON document holding an ordered list of cells: some Markdown (prose), some code. So <b>do not double-click it</b> (Notepad will show curly braces — the file is fine), and you cannot run it with <code>python xxx.ipynb</code>. Only Jupyter can execute it.</p>
</div>

<h3>Step 3: Put it where Jupyter can see it</h3>
<p>Pick one of three routes — do not install all three.</p>

<p><b>Route A: you already run Jupyter in WSL (recommended if that is your setup)</b></p>
<p>WSL reads the Windows downloads folder directly at <code>/mnt/c/Users/&lt;you&gt;/Downloads</code>:</p>
<pre data-lang="bash"><code># 1) a dedicated folder for practice notebooks
mkdir -p ~/pytorch-practice

# 2) copy everything you downloaded
cp /mnt/c/Users/LENOVO/Downloads/pytorch-*.ipynb ~/pytorch-practice/

# 3) launch Jupyter from that folder
cd ~/pytorch-practice
jupyter lab</code></pre>
<p>The terminal prints a URL with a token, e.g. <code>http://127.0.0.1:8888/lab?token=xxxx</code>. Paste it into your Windows browser (WSL localhost is shared with Windows), then open the notebook from the file list.</p>

<p><b>Route B: native Windows Python</b></p>
<pre data-lang="bash"><code># launch Jupyter inside the download folder so the file is right there
cd C:\\Users\\LENOVO\\Downloads
python -m pip install jupyterlab
python -m jupyter lab</code></pre>

<p><b>Route C: install nothing</b></p>
<p>Go to <code>colab.research.google.com</code> → File → Upload notebook. Trade-off: your file goes to the cloud and you need a connection, but you get a free GPU — handy for chapter 13.</p>

<h3>Step 4: Run cell by cell</h3>
<table>
  <thead><tr><th>Shortcut</th><th>Effect</th><th>When</th></tr></thead>
  <tbody>
    <tr><td><kbd>Shift</kbd>+<kbd>Enter</kbd></td><td>Run, move to next cell</td><td>Your default while studying</td></tr>
    <tr><td><kbd>Ctrl</kbd>+<kbd>Enter</kbd></td><td>Run, stay here</td><td>Tweaking one cell repeatedly</td></tr>
    <tr><td><kbd>Alt</kbd>+<kbd>Enter</kbd></td><td>Run and insert a cell below</td><td>Adding your own experiment</td></tr>
    <tr><td><kbd>Esc</kbd> then <kbd>B</kbd></td><td>Insert cell below</td><td>Writing your own code</td></tr>
  </tbody>
</table>
<div class="callout tip">
  <span class="tag">Order matters</span>
  <p><code>[1] [2] [3]</code> on the left is execution order, not line numbers. Variables are shared across cells, so running out of order gives confusing results. When in doubt: <b>Kernel → Restart &amp; Run All</b>.</p>
</div>

<h3>Step 5: Change the numbers</h3>
<p>Every notebook ends with "try it" cells. Chapter 4 differentiates <code>y = x**2</code> at <code>x=3</code> (answer: 6). Change it to <code>x = 5</code>, predict <code>2x = 10</code> in your head, then run. Predict-then-verify beats re-reading by a wide margin.</p>

<h2>2. Before you start: a three-line health check</h2>
<p>Make a new cell, paste this, run it. Always the first move when something feels wrong.</p>
<pre data-lang="python"><code>import sys, platform
import torch

print('Python :', sys.version.split()[0], '|', platform.system())
print('exe    :', sys.executable)
print('torch  :', torch.__version__)
print('CUDA   :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU    :', torch.cuda.get_device_name(0))
x = torch.tensor([1.0, 2.0, 3.0])
print('tensor :', x, x.dtype, x.shape)</code></pre>
<ul>
  <li><code>torch 2.x.x</code> prints → you are good to go.</li>
  <li><code>CUDA False</code> → you are on CPU. Chapters 1–12 and 14 run fine on CPU; only chapter 13 wants a GPU, and Colab covers that. Do not fight drivers today.</li>
  <li><code>No module named 'torch'</code> → it went into a different interpreter. Look at <code>sys.executable</code> and install with that one: <code>python -m pip install torch</code>.</li>
</ul>

<h2>3. How to use the quizzes</h2>
<p>Three multiple-choice questions per chapter. <b>One click grades you</b>: the correct option turns green, a wrong pick turns red, and the question locks. That is deliberate — it stops you from brute-forcing your way to a false sense of understanding.</p>
<p>If you miss one, do not scroll on. Re-read that section, then write one sentence in your notes: "I got it wrong because…". Notes autosave per chapter and export to <code>.md</code>.</p>
<div class="callout basic">
  <span class="tag">Answers are remembered</span>
  <p>Quiz state is stored with your highlights in browser localStorage, so a chapter stays graded next time. To retry, remove the mark on the answered block (select the text → "Remove"), or clear all study data from the home page.</p>
</div>

<h2>4. The two interactive calculators</h2>
<h3>Broadcast shape calculator (chapter 3)</h3>
<p>Enter two shapes, e.g. <code>3,1</code> and <code>1,4</code>. It right-aligns them and compares dimension by dimension: equal survives, a 1 stretches, anything else fails. So <code>3,1 × 1,4 → 3,4</code>, while <code>3,2 × 3,4</code> breaks on the last dim (2 vs 4).</p>
<p>Best way to use it: guess on paper first, then click. Five correct guesses in a row and broadcasting is yours for good.</p>

<h3>Conv output calculator (chapter 10)</h3>
<p>Enter <code>H, W</code>, kernel <code>K</code>, padding <code>P</code>, stride <code>S</code>. It returns the output size and the parameter count. The formula:</p>
<p class="formula">H_out = ⌊(H + 2P − K) / S⌋ + 1</p>
<p>Use it to verify two facts: with <code>K=3, P=1, S=1</code> the size is unchanged (same padding); set <code>S=2</code> and it halves. And the parameter count depends only on <code>K, C_in, C_out</code> — not on the input size at all, which surprises most people the first time.</p>

<h2>5. Troubleshooting</h2>
<table>
  <thead><tr><th>Symptom</th><th>Cause</th><th>Fix</th></tr></thead>
  <tbody>
    <tr><td>Download button seems dead</td><td>Silent or blocked download</td><td><kbd>Ctrl</kbd>+<kbd>J</kbd>, allow/keep the file</td></tr>
    <tr><td>.ipynb opens as curly braces</td><td>Notepad hijacked the extension</td><td>File is intact — open it with <code>jupyter lab</code></td></tr>
    <tr><td>Notebook not in Jupyter's list</td><td>Wrong start directory</td><td><code>cd</code> into the folder before launching</td></tr>
    <tr><td><code>No module named 'torch'</code></td><td>Installed into another Python</td><td>Check <code>sys.executable</code>, install with that</td></tr>
    <tr><td><code>CUDA out of memory</code></td><td>Not enough VRAM</td><td>Halve <code>batch_size</code>, <code>del</code> unused tensors</td></tr>
    <tr><td>Code changed but output did not</td><td>You ran a different cell</td><td>Kernel → Restart &amp; Run All</td></tr>
  </tbody>
</table>

<h2>6. A one-chapter-a-day rhythm</h2>
<table>
  <thead><tr><th>Phase</th><th>Time</th><th>What you do</th></tr></thead>
  <tbody>
    <tr><td>Read</td><td>20 min</td><td>Read the prose; stop at every "beginner note" and "math note"</td></tr>
    <tr><td>Run</td><td>20 min</td><td>Download the notebook, Shift+Enter all the way down</td></tr>
    <tr><td>Change</td><td>10 min</td><td>Change shapes, numbers, learning rates — predict first</td></tr>
    <tr><td>Test</td><td>5 min</td><td>Take the quiz; if wrong, re-read and note why</td></tr>
  </tbody>
</table>
<p>Sixteen chapters at this pace is about three weeks. Do not rush: <b>able to run and able to change</b> beats "finished reading".</p>

<div class="keypoints">
  <h4>Five things to remember</h4>
  <ul>
    <li>Notebooks open in Jupyter — not by double-clicking, not with <code>python x.ipynb</code></li>
    <li>WSL sees Windows downloads at <code>/mnt/c/Users/LENOVO/Downloads</code></li>
    <li>Weird results → Restart &amp; Run All first</li>
    <li>Quizzes lock after one click, so review before you answer</li>
    <li>Predict, then run — that is the difference between practicing and watching</li>
  </ul>
</div>
`
  }
};
