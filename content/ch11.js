(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch11',
  num: '11',
  part: 'p3',
  file: 'rnn',
  zh: {
    title: '序列模型：RNN、LSTM 与 GRU',
    sub: '处理「有先后顺序」的数据：文本、语音、股价、传感器',
    html: `
<p class="lead">图片的像素可以打乱顺序，句子的词不行——「狗咬人」和「人咬狗」是两件事。这类数据叫<strong>序列</strong>。循环神经网络（RNN）的思路是：<strong>带着一个「记忆」逐个读入元素</strong>。虽然 Transformer 现在更主流，但理解 RNN 是理解注意力机制的前提。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>说清隐藏状态（hidden state）的含义与 RNN 的展开过程</li>
    <li>正确使用 <code>nn.LSTM</code> 的输入输出形状与 <code>batch_first</code></li>
    <li>搭一个文本分类模型（Embedding + LSTM）</li>
    <li>解释梯度消失，以及 LSTM 的三个门怎么缓解它</li>
  </ul>
</div>

<h2>1. 为什么不能用全连接</h2>
<p>把一句话的 20 个词各自变成 100 维向量，然后拼成 2000 维送进全连接层——理论上可行，但有两个问题：</p>
<ul>
  <li><strong>长度不固定</strong>：这句话 20 个词，那句话 35 个词，全连接层的输入维度是固定的。</li>
  <li><strong>丢了顺序信息</strong>：拼接对位置敏感，但模型无法理解「同一个词在不同位置含义不同」，也无法共享「词与词之间的规律」。</li>
</ul>

<h2>2. RNN：带记忆地逐个读</h2>
<pre data-lang="python"><code># 手工实现一个最小 RNN，理解它在做什么
import torch

h = torch.zeros(4)             # 初始记忆（隐藏状态）
W_x = torch.randn(3, 4)        # 输入 -> 记忆
W_h = torch.randn(4, 4)        # 记忆 -> 记忆

for x_t in torch.randn(5, 3):  # 依次读入 5 个时间步，每步 3 维
    h = torch.tanh(x_t @ W_x + h @ W_h)   # 新记忆 = 当前输入 + 旧记忆

print(h.shape)                 # (4,)  最终记忆浓缩了整个序列</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 同一组权重反复用</span>
  <p>关键点：<code>W_x</code> 和 <code>W_h</code> 在<strong>每个时间步都相同</strong>。所以无论序列多长，参数量都一样——这是 RNN 能处理变长输入的根本原因，叫「参数共享 / 时间上的权重复用」。</p>
  <p>公式化：<code>h_t = tanh(x_t W_x + h_{t-1} W_h + b)</code>。<code>h_t</code> 既作为输出，也作为下一步的输入，形成一个循环。</p>
</div>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p>把 RNN 想成「边读边做笔记」：每读一个词，就在笔记本上更新一下理解。<code>h</code> 就是那本笔记。读完整句话，笔记里就是模型对这句话的「摘要」，再拿它去做分类或翻译。</p>
</div>

<h2>3. 用 PyTorch 的 LSTM</h2>
<pre data-lang="python"><code>import torch.nn as nn

lstm = nn.LSTM(
    input_size=128,      # 每个时间步的向量维度
    hidden_size=256,     # 记忆的维度
    num_layers=2,        # 叠两层
    batch_first=True,    # 关键！输入形状为 (batch, seq, feature)
    dropout=0.2,         # 层间 dropout
    bidirectional=False,
)

x = torch.randn(8, 20, 128)          # (batch=8, seq=20, feature=128)
out, (h_n, c_n) = lstm(x)

print(out.shape)      # (8, 20, 256) —— 每个时间步的输出
print(h_n.shape)      # (2, 8, 256)  —— 每层最后一步的隐藏状态
print(c_n.shape)      # (2, 8, 256)  —— 细胞状态（LSTM 特有）</code></pre>

<div class="callout warn">
  <span class="tag">batch_first 是最大的坑</span>
  <p>PyTorch 的 RNN 系列默认 <code>batch_first=False</code>，输入形状是 <code>(seq, batch, feature)</code>。这和 DataLoader 给的 <code>(batch, seq, feature)</code> 正好相反。忘写 <code>batch_first=True</code>，模型能跑但batch 被当成时间步，结果完全错误且不报错。<strong>永远显式写这个参数。</strong></p>
</div>

<h2>4. LSTM 的三个门</h2>
<p>普通 RNN 有个致命问题：<strong>梯度消失</strong>。反向传播时梯度要连乘很多次，如果每次乘的数都小于 1，传到前面就接近 0，模型学不到长距离依赖（比如「我昨天说的那本书……它」里的「它」指什么）。</p>

<div class="callout math">
  <span class="tag">数学批注 · 门控机制</span>
  <p>LSTM 引入一条「细胞状态」<code>c</code> 作为信息高速公路，并用三个门控制读写：</p>
  <p class="formula">f_t = σ(W_f·[h_{t-1}, x_t])  遗忘门：旧记忆留多少
i_t = σ(W_i·[h_{t-1}, x_t])  输入门：新信息写多少
c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(...)  更新记忆
o_t = σ(W_o·[h_{t-1}, x_t])  输出门：暴露多少给 h</p>
  <p><code>σ</code> 是 sigmoid，输出 0~1，所以「门」可以理解为「开多大」的比例。<code>⊙</code> 是逐元素乘。</p>
  <p>为什么能缓解梯度消失？因为 <code>c_t = f_t ⊙ c_{t-1} + ...</code> 这条路径上，梯度是<strong>相加</strong>而非连乘，只要遗忘门接近 1，梯度就能几乎无损地传很远。</p>
</div>

<p>GRU 是 LSTM 的简化版，只有两个门（更新门、重置门），参数更少、训练更快，效果常常相当。选哪个？<strong>先试 GRU</strong>，不够再上 LSTM。</p>

<h2>5. 完整例子：文本情感分类</h2>
<pre data-lang="python"><code>import torch, torch.nn as nn

class SentimentNet(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=128, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden * 2, num_classes)   # 双向要乘 2

    def forward(self, ids):
        # ids: (batch, seq) 的整数张量
        x = self.embed(ids)                # (batch, seq, emb)
        out, (h_n, c_n) = self.lstm(x)     # out: (batch, seq, 2*hidden)
        # 取最后一步（双向时 h_n 已包含两个方向）
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)   # (batch, 2*hidden)
        return self.fc(self.dropout(h))

model = SentimentNet(vocab_size=10000)
ids = torch.randint(0, 10000, (16, 30))    # 16 条评论，每条 30 个词
print(model(ids).shape)                    # (16, 2)</code></pre>

<div class="callout basic">
  <span class="tag">Embedding 是什么</span>
  <p>词不能直接进网络，要先变成数字。<code>nn.Embedding(V, D)</code> 内部就是一张 <code>(V, D)</code> 的表：输入词的整数 id，输出对应的 D 维向量。这张表<strong>也是可学习的参数</strong>——训练过程中，语义相近的词会慢慢靠在一起。</p>
  <p><code>padding_idx=0</code> 告诉模型「id 0 是补齐用的，别管它」，它的向量永远是 0，也不参与梯度更新。</p>
</div>

<h2>6. 变长序列：padding 与 pack</h2>
<pre data-lang="python"><code>from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

seqs = [torch.randn(5, 128), torch.randn(3, 128), torch.randn(7, 128)]
padded = pad_sequence(seqs, batch_first=True)     # (3, 7, 128)，短句补 0
lengths = torch.tensor([5, 3, 7])

packed = pack_padded_sequence(padded, lengths, batch_first=True,
                              enforce_sorted=False)
out, (h, c) = lstm(packed)                        # RNN 会跳过补齐部分
out, _ = pad_packed_sequence(out, batch_first=True)</code></pre>
<p>不做 pack 的话，补齐的 0 会被当成真实数据算进记忆里，短句的表示会被污染。数据量小时影响不大，做正经任务一定要 pack。</p>

<div class="callout tip">
  <span class="tag">现实提醒</span>
  <p>做 NLP 现在基本不直接用 LSTM 了：预训练 Transformer（BERT、GPT）效果碾压，而且有 HuggingFace 直接调用。学 RNN 的价值在于<strong>理解序列建模的思想</strong>，以及处理传感器、时序信号等场景（那里 LSTM 仍然很常用）。</p>
</div>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li>RNN 用同一组权重逐个读入，隐藏状态 h 是「记忆」。</li>
    <li>PyTorch RNN 必须显式写 <code>batch_first=True</code>。</li>
    <li>LSTM 返回 <code>out, (h_n, c_n)</code>，形状分别是 (B,T,H)、(L,B,H)。</li>
    <li>梯度消失靠门控 + 细胞状态的加法路径缓解。</li>
    <li>GRU 更轻量，通常先试它。</li>
    <li>变长序列要用 <code>pack_padded_sequence</code>。</li>
    <li>文本任务的现代做法是预训练 Transformer。</li>
  </ol>
</div>

<div class="quiz" data-answer="2">
  <p class="q"><code>nn.LSTM(128, 256, batch_first=True)</code> 接收 <code>(8, 20, 128)</code> 的输入，输出 <code>out</code> 的形状是？</p>
  <ol class="opts">
    <li>(20, 8, 128)</li>
    <li>(8, 128, 256)</li>
    <li>(8, 20, 256)</li>
    <li>(8, 256)</li>
  </ol>
  <p class="exp">batch_first 保持 batch 在第 0 维，时间步保留在中间，特征维变成 hidden_size=256。</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q">RNN 无法学到长距离依赖的主要原因是？</p>
  <ol class="opts">
    <li>参数量太大</li>
    <li>反向传播时梯度连乘导致消失</li>
    <li>隐藏状态维度太高</li>
    <li>不能用 GPU 加速</li>
  </ol>
  <p class="exp">长序列上梯度要穿过很多步连乘，数值指数衰减。LSTM 的门控和加法路径缓解了这一点。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q"><code>nn.Embedding(10000, 128)</code> 的参数量是？</p>
  <ol class="opts">
    <li>10000 × 128 = 1,280,000</li>
    <li>10000 + 128</li>
    <li>128</li>
    <li>10000</li>
  </ol>
  <p class="exp">它就是一张查表矩阵，每个词一个 128 维向量。这也是为什么词表大时 embedding 占显存很多。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>用最小 RNN 手写实现跑一遍，把 <code>tanh</code> 换成 <code>relu</code>，观察长序列下 h 是否饱和。</li>
    <li>把 <code>batch_first</code> 去掉，用同一批数据前向，比较输出形状，理解这个坑。</li>
    <li>把 <code>num_layers=2</code> 改成 1，参数量变化多少？用代码验证。</li>
    <li>用合成数据（正弦波序列）训练一个 LSTM 做下一步预测，画出预测曲线。</li>
  </ol>
</div>`
  },
  en: {
    title: 'Sequence models: RNN, LSTM, GRU',
    sub: 'For data where order matters: text, audio, prices, sensors',
    html: `
<p class="lead">Image pixels can be shuffled; words cannot — "dog bites man" is not "man bites dog". Such data is a <strong>sequence</strong>. RNNs read elements one at a time while carrying a <strong>memory</strong>. Transformers dominate now, but understanding RNNs is the prerequisite for understanding attention.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Explain the hidden state and the unrolled RNN</li>
    <li>Use <code>nn.LSTM</code> shapes and <code>batch_first</code> correctly</li>
    <li>Build a text classifier (Embedding + LSTM)</li>
    <li>Explain vanishing gradients and how gates mitigate them</li>
  </ul>
</div>

<h2>1. Why a dense layer fails</h2>
<ul>
  <li><strong>Variable length</strong>: 20 words here, 35 there. A dense layer needs a fixed input size.</li>
  <li><strong>Lost order</strong>: concatenation encodes position implicitly but shares no statistics across positions.</li>
</ul>

<h2>2. RNN: reading with memory</h2>
<pre data-lang="python"><code>import torch

h = torch.zeros(4)             # initial memory
W_x = torch.randn(3, 4)        # input -> memory
W_h = torch.randn(4, 4)        # memory -> memory

for x_t in torch.randn(5, 3):  # 5 time steps of 3 features
    h = torch.tanh(x_t @ W_x + h @ W_h)

print(h.shape)                 # (4,) — the whole sequence compressed</code></pre>

<div class="callout math">
  <span class="tag">Math note · shared weights over time</span>
  <p>The same <code>W_x</code> and <code>W_h</code> are used at every step, so the parameter count is independent of length — that is what enables variable-length input.</p>
  <p><code>h_t = tanh(x_t W_x + h_{t-1} W_h + b)</code>. h_t is both the output and the next step's input.</p>
</div>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p>An RNN takes notes while reading: each word updates the notebook. <code>h</code> is the notebook; when the sentence ends, the notebook summarises it and you classify from there.</p>
</div>

<h2>3. PyTorch LSTM</h2>
<pre data-lang="python"><code>lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,    # crucial
    dropout=0.2,
    bidirectional=False,
)

x = torch.randn(8, 20, 128)          # (batch, seq, feature)
out, (h_n, c_n) = lstm(x)

print(out.shape)      # (8, 20, 256)
print(h_n.shape)      # (2, 8, 256)
print(c_n.shape)      # (2, 8, 256)</code></pre>

<div class="callout warn">
  <span class="tag">The batch_first trap</span>
  <p>PyTorch RNNs default to <code>(seq, batch, feature)</code>. DataLoader gives <code>(batch, seq, feature)</code>. Forget <code>batch_first=True</code> and it still runs — with the batch treated as time steps, producing silently wrong results. <strong>Always write it explicitly.</strong></p>
</div>

<h2>4. The three gates of LSTM</h2>
<div class="callout math">
  <span class="tag">Math note · gating</span>
  <p>A cell state <code>c</code> acts as a highway, controlled by three gates:</p>
  <p class="formula">f_t = σ(W_f·[h_{t-1}, x_t])   forget: keep how much
i_t = σ(W_i·[h_{t-1}, x_t])   input: write how much
c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(...)
o_t = σ(W_o·[h_{t-1}, x_t])   output: expose how much</p>
  <p>σ squeezes to 0–1, so a gate is a dial. Why does this help gradients? Because <code>c_t = f_t ⊙ c_{t-1} + ...</code> gives an <strong>additive</strong> path; with f≈1 gradients flow almost unchanged over long spans.</p>
</div>

<p>GRU simplifies to two gates, fewer parameters, usually comparable quality. Try GRU first.</p>

<h2>5. Full example: sentiment classification</h2>
<pre data-lang="python"><code>class SentimentNet(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=128, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, ids):
        x = self.embed(ids)                # (B, T, E)
        out, (h_n, c_n) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)   # both directions
        return self.fc(self.dropout(h))</code></pre>

<div class="callout basic">
  <span class="tag">What is Embedding</span>
  <p><code>nn.Embedding(V, D)</code> is a lookup table of shape (V, D): integer id in, D-dim vector out. The table is <strong>learned</strong>, so similar words drift together. <code>padding_idx=0</code> freezes the pad vector at zero.</p>
</div>

<h2>6. Variable lengths: packing</h2>
<pre data-lang="python"><code>from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

seqs = [torch.randn(5, 128), torch.randn(3, 128), torch.randn(7, 128)]
padded = pad_sequence(seqs, batch_first=True)
lengths = torch.tensor([5, 3, 7])

packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
out, (h, c) = lstm(packed)
out, _ = pad_packed_sequence(out, batch_first=True)</code></pre>

<div class="callout tip">
  <span class="tag">Reality check</span>
  <p>Modern NLP uses pretrained Transformers. RNNs remain valuable for sensor and time-series data, and for understanding the ideas attention replaced.</p>
</div>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li>RNNs reuse weights across time; h is the memory.</li>
    <li>Always set <code>batch_first=True</code>.</li>
    <li>LSTM returns <code>out, (h_n, c_n)</code>.</li>
    <li>Gates plus an additive cell path mitigate vanishing gradients.</li>
    <li>Try GRU first; it is lighter.</li>
    <li>Pack padded sequences for variable lengths.</li>
    <li>Text work today means pretrained Transformers.</li>
  </ol>
</div>

<div class="quiz" data-answer="2">
  <p class="q"><code>nn.LSTM(128, 256, batch_first=True)</code> on <code>(8,20,128)</code> returns <code>out</code> of shape:</p>
  <ol class="opts">
    <li>(20, 8, 128)</li>
    <li>(8, 128, 256)</li>
    <li>(8, 20, 256)</li>
    <li>(8, 256)</li>
  </ol>
  <p class="exp">Batch first, time second, hidden last.</p>
</div>

<div class="quiz" data-answer="1">
  <p class="q">RNNs struggle with long-range dependencies mainly because:</p>
  <ol class="opts">
    <li>Too many parameters</li>
    <li>Gradients are multiplied repeatedly and vanish</li>
    <li>Hidden size is too large</li>
    <li>They cannot use GPUs</li>
  </ol>
  <p class="exp">Exponential decay across steps; gates and additive paths help.</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">Parameters in <code>nn.Embedding(10000, 128)</code>:</p>
  <ol class="opts">
    <li>10000 × 128 = 1,280,000</li>
    <li>10000 + 128</li>
    <li>128</li>
    <li>10000</li>
  </ol>
  <p class="exp">A lookup matrix; large vocabularies dominate memory.</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Run the minimal RNN; swap tanh for relu and see whether h saturates.</li>
    <li>Remove <code>batch_first</code>, forward the same batch, compare shapes.</li>
    <li>Change num_layers 2 → 1 and count parameters with code.</li>
    <li>Train an LSTM on a synthetic sine wave to predict the next value; plot it.</li>
  </ol>
</div>`
  }
});
