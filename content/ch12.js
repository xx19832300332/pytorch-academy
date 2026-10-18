(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch12',
  num: '12',
  part: 'p3',
  file: 'attention',
  zh: {
    title: '注意力与 Transformer',
    sub: '今天所有大模型的骨架，其实只有四个矩阵乘',
    html: `
<p class="lead">2017 年的《Attention Is All You Need》之后，注意力机制逐渐统一了 NLP、视觉、语音，成了 GPT、BERT、ViT 的共同基础。这章不推导论文，只把<strong>机制</strong>讲清楚，并让你能用 PyTorch 搭出一个能跑的小 Transformer。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>用「查字典」的比喻解释 Q、K、V</li>
    <li>手写缩放点积注意力，并解释为什么除以 √d</li>
    <li>说清多头注意力与位置编码的必要性</li>
    <li>用 <code>nn.TransformerEncoder</code> 搭一个文本分类器</li>
  </ul>
</div>

<h2>1. 注意力的直觉</h2>
<p>RNN 必须把信息一步步传递，长距离依赖容易丢。注意力换了个思路：<strong>每个位置直接看所有位置，自己决定该关注谁</strong>。</p>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p>用查字典打比方：你手上有个问题（<strong>Query</strong>），图书馆每个书脊上有个标签（<strong>Key</strong>），书里是内容（<strong>Value</strong>）。</p>
  <p>你先拿问题去和每个标签比对，算出「哪本书最相关」（相似度分数）；再把分数变成权重（softmax）；最后按权重把各本书的内容<strong>加权混合</strong>，得到答案。</p>
  <p>这个过程里没有循环、没有逐步传递，所有位置一步到位——这就是它能大规模并行的原因。</p>
</div>

<pre data-lang="python"><code>import torch, torch.nn.functional as F

# 3 个位置，每个位置 4 维
Q = torch.randn(3, 4)   # 我要找什么
K = torch.randn(3, 4)   # 每个位置能提供什么（标签）
V = torch.randn(3, 4)   # 每个位置的实际内容

scores = Q @ K.T                    # (3,3) 两两相似度
weights = F.softmax(scores, dim=-1) # 每行归一化成权重
out = weights @ V                   # (3,4) 加权求和

print(weights.sum(dim=-1))          # tensor([1., 1., 1.])</code></pre>

<h2>2. 公式与那个 √d</h2>
<div class="formula">Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V</div>

<div class="callout math">
  <span class="tag">数学批注 · 为什么要除以 √d</span>
  <p><code>Q·Kᵀ</code> 里每个元素是 d 个随机数相乘再相加。如果 Q、K 的各分量方差为 1，那么点积的方差约为 <code>d</code>，标准差是 <code>√d</code>。</p>
  <p>d=64 时，分数可能在 ±8 附近波动。softmax 遇到这种大差距会变得<strong>极度尖锐</strong>（几乎 one-hot），梯度接近 0，训练不动。除以 √d 把方差拉回 1，softmax 保持「温和」。</p>
  <p>这就是「scaled」的含义：不是为了数值稳定这么简单，而是为了<strong>让梯度能流动</strong>。</p>
</div>

<h2>3. 多头：不同角度同时看</h2>
<pre data-lang="python"><code>import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)

x = torch.randn(4, 10, 128)      # (batch, seq, dim)
attn_out, attn_weights = mha(x, x, x)   # 自注意力：Q=K=V=x

print(attn_out.shape)            # (4, 10, 128)
print(attn_weights.shape)        # (4, 10, 10) 每个头平均后的权重</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 多头的本质</span>
  <p>把 128 维拆成 8 个头，每个头 16 维，各自独立做一次注意力，最后拼接再线性变换。</p>
  <p>为什么有用？单个注意力只能表达一种「关注模式」，多头允许模型同时关注<strong>不同关系</strong>：一个头看语法依赖，一个头看指代，一个头看相邻词。参数量几乎不变（拆开算反而更省），表达力却成倍提升。</p>
  <p>注意 <code>embed_dim</code> 必须能被 <code>num_heads</code> 整除。</p>
</div>

<h2>4. 位置编码：注意力的盲点</h2>
<p>注意力是「集合运算」——它不知道谁在前谁在后。把一句话打乱，输出只是顺序变了，内容一样。所以必须额外注入位置信息。</p>

<pre data-lang="python"><code>class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, L, D)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])</code></pre>

<div class="callout math">
  <span class="tag">数学批注 · 正弦位置编码</span>
  <p>用不同频率的正弦、余弦函数给每个位置生成一个独特的「指纹」：<code>PE(pos, 2i) = sin(pos / 10000^(2i/d))</code>。</p>
  <p>好处是不用学习、能外推到比训练更长的序列，而且任意两个位置的相对距离可以通过线性变换得到。现代模型（如 LLaMA）多用 RoPE，思路类似但把相对位置编码进点积里。</p>
  <p><code>register_buffer</code> 的作用：把它当作模型的一部分保存/移动到 GPU，但<strong>不参与梯度更新</strong>（它不是参数）。</p>
</div>

<h2>5. 一个 Transformer 块</h2>
<pre data-lang="python"><code>class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=8, ff=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN 结构（现代实现的主流，比原论文更稳）
        h = self.norm1(x)
        a, _ = self.attn(h, h, h)
        x = x + self.drop(a)              # 残差连接
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

print(TransformerBlock()(torch.randn(4, 10, 128)).shape)   # (4,10,128)</code></pre>

<div class="callout basic">
  <span class="tag">残差连接为什么必须有</span>
  <p>Transformer 可以堆几十上百层，靠的就是 <code>x = x + f(x)</code> 这种残差结构：梯度有一条「直通车道」，可以无损地传回浅层。没有它，深层网络根本训不动。</p>
  <p>LayerNorm 放在注意力之前（Pre-LN）还是之后（Post-LN）？原论文是 Post-LN，但实践中 Pre-LN 更稳定，不需要小心翼翼地预热学习率，所以现在几乎都用 Pre-LN。</p>
</div>

<h2>6. 搭一个文本分类 Transformer</h2>
<pre data-lang="python"><code>class TinyTransformer(nn.Module):
    def __init__(self, vocab=10000, d_model=128, nhead=8,
                 num_layers=2, num_classes=2, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, ids, mask=None):
        x = self.pos(self.embed(ids))
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.norm(x)
        return self.head(x[:, 0])       # 用第一个位置做分类（类似 [CLS]）

model = TinyTransformer()
ids = torch.randint(1, 10000, (8, 32))
print(model(ids).shape)                 # (8, 2)</code></pre>

<div class="callout warn">
  <span class="tag">padding mask 别忘</span>
  <p><code>src_key_padding_mask</code> 的形状是 <code>(batch, seq)</code>，True 表示「这个位置是补齐的，别看它」。不传的话，模型会把 padding 的 0 也当成真实 token 计算注意力，短句结果会被稀释。</p>
</div>

<h2>7. 和 GPT / BERT 的关系</h2>
<table>
  <tr><th>模型</th><th>结构</th><th>训练目标</th><th>擅长</th></tr>
  <tr><td>BERT</td><td>Encoder（双向）</td><td>掩码语言模型</td><td>理解类任务：分类、抽取、检索</td></tr>
  <tr><td>GPT</td><td>Decoder（因果掩码）</td><td>预测下一个词</td><td>生成类任务：写作、对话、代码</td></tr>
  <tr><td>ViT</td><td>Encoder，输入是图像块</td><td>分类 / 对比学习</td><td>视觉</td></tr>
</table>
<p>所谓「大模型」，就是把同样的块堆到几十层、几千维，用海量数据训练。结构和你在上面写的没有本质区别，差别在规模与工程。</p>

<div class="keypoints">
  <h4>本章重点</h4>
  <ol>
    <li>注意力：Q 与 K 算相似度 → softmax 成权重 → 加权求和 V。</li>
    <li>除以 √d 是为了控制方差、避免 softmax 饱和导致梯度消失。</li>
    <li>多头 = 在不同子空间并行关注不同关系，embed_dim 要被 num_heads 整除。</li>
    <li>注意力本身无序，必须加位置编码。</li>
    <li>Transformer 块 = 注意力 + 前馈 + 残差 + LayerNorm，Pre-LN 更稳。</li>
    <li>BERT 是 Encoder，GPT 是 Decoder，ViT 把图切成块。</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q">缩放点积注意力里除以 √d_k 的主要目的是？</p>
  <ol class="opts">
    <li>减少参数量</li>
    <li>控制点积方差，避免 softmax 过于尖锐、梯度消失</li>
    <li>让输出变成概率</li>
    <li>加速矩阵乘法</li>
  </ol>
  <p class="exp">点积的方差随 d 线性增长，不缩放会让 softmax 近似 one-hot，梯度几乎为 0。</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">为什么 Transformer 需要位置编码？</p>
  <ol class="opts">
    <li>为了让模型更大</li>
    <li>为了处理变长输入</li>
    <li>因为注意力对输入顺序不敏感，本身没有位置概念</li>
    <li>为了减少显存</li>
  </ol>
  <p class="exp">自注意力是集合运算，打乱输入顺序输出只是重排。位置信息必须显式注入。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">一个 128 维、8 个头的多头注意力，每个头的维度是？</p>
  <ol class="opts">
    <li>16</li>
    <li>8</li>
    <li>128</li>
    <li>1024</li>
  </ol>
  <p class="exp">128 / 8 = 16。embed_dim 必须能被 num_heads 整除，否则报错。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>打印 <code>mha</code> 的注意力权重矩阵，观察每一行的和为 1，并解释为什么。</li>
    <li>把 <code>num_heads</code> 改成 3（128 不能被 3 整除），看报错信息。</li>
    <li>去掉位置编码，用同一句话的两个不同词序输入，比较模型输出差异。</li>
    <li>把 TransformerBlock 堆 4 层，统计参数量，估算推理显存。</li>
  </ol>
</div>`
  },
  en: {
    title: 'Attention and Transformers',
    sub: 'The skeleton of every modern large model is four matrix multiplies',
    html: `
<p class="lead">Since "Attention Is All You Need" (2017), attention has unified NLP, vision and speech, and underpins GPT, BERT and ViT. This chapter skips the paper and explains the <strong>mechanism</strong>, then gets you to build a small working Transformer.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Explain Q, K, V with the dictionary analogy</li>
    <li>Implement scaled dot-product attention and justify √d</li>
    <li>Explain multi-head attention and positional encoding</li>
    <li>Build a text classifier with <code>nn.TransformerEncoder</code></li>
  </ul>
</div>

<h2>1. The intuition</h2>
<p>RNNs pass information step by step, so long dependencies fade. Attention instead lets <strong>every position look at every position</strong> and decide what to attend to.</p>

<div class="callout basic">
  <span class="tag">For beginners</span>
  <p>You have a question (<strong>Query</strong>). Every book spine has a label (<strong>Key</strong>) and the book has content (<strong>Value</strong>). Compare your question with every label to get relevance scores, softmax them into weights, then blend the contents by weight.</p>
  <p>No loop, no sequential passing — which is exactly why it parallelises so well.</p>
</div>

<pre data-lang="python"><code>import torch, torch.nn.functional as F

Q = torch.randn(3, 4)
K = torch.randn(3, 4)
V = torch.randn(3, 4)

scores = Q @ K.T                    # (3,3) pairwise similarity
weights = F.softmax(scores, dim=-1)
out = weights @ V                   # (3,4)

print(weights.sum(dim=-1))          # tensor([1., 1., 1.])</code></pre>

<h2>2. The formula and that √d</h2>
<div class="formula">Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V</div>

<div class="callout math">
  <span class="tag">Math note · why divide by √d</span>
  <p>Each entry of <code>Q·Kᵀ</code> sums d products. For unit-variance components the dot product has variance ≈ d, i.e. standard deviation √d.</p>
  <p>At d=64, scores swing around ±8. Softmax over such gaps becomes nearly one-hot and gradients vanish. Dividing by √d restores unit variance and keeps softmax well-behaved — <strong>it is about gradient flow, not just numerics</strong>.</p>
</div>

<h2>3. Multi-head</h2>
<pre data-lang="python"><code>mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)

x = torch.randn(4, 10, 128)
attn_out, attn_weights = mha(x, x, x)   # self-attention: Q=K=V

print(attn_out.shape)            # (4, 10, 128)
print(attn_weights.shape)        # (4, 10, 10)</code></pre>

<div class="callout math">
  <span class="tag">Math note · what heads do</span>
  <p>Split 128 dims into 8 heads of 16, run attention in each, concatenate and project.</p>
  <p>One attention map expresses one kind of relation. Heads let the model track syntax, coreference and adjacency simultaneously at almost no extra cost. <code>embed_dim</code> must divide by <code>num_heads</code>.</p>
</div>

<h2>4. Positional encoding</h2>
<pre data-lang="python"><code>class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])</code></pre>

<div class="callout math">
  <span class="tag">Math note · sinusoidal PE</span>
  <p>Different frequencies give each position a unique fingerprint: <code>sin(pos / 10000^(2i/d))</code>. No parameters, extrapolates to longer sequences, and relative offsets become linear transforms. Modern models often use RoPE instead.</p>
  <p><code>register_buffer</code> keeps it with the model (moves to GPU, saved in state_dict) but excludes it from gradients.</p>
</div>

<h2>5. A Transformer block</h2>
<pre data-lang="python"><code>class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=8, ff=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)                 # Pre-LN
        a, _ = self.attn(h, h, h)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x</code></pre>

<div class="callout basic">
  <span class="tag">Why residuals are non-negotiable</span>
  <p><code>x = x + f(x)</code> gives gradients a highway straight back to shallow layers. Without it, dozens of stacked blocks will not train. Pre-LN (norm before attention) is more stable than the paper's Post-LN and needs no LR warmup gymnastics.</p>
</div>

<h2>6. A text classification Transformer</h2>
<pre data-lang="python"><code>class TinyTransformer(nn.Module):
    def __init__(self, vocab=10000, d_model=128, nhead=8,
                 num_layers=2, num_classes=2, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, ids, mask=None):
        x = self.pos(self.embed(ids))
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.head(self.norm(x)[:, 0])</code></pre>

<div class="callout warn">
  <span class="tag">Do not forget the padding mask</span>
  <p><code>src_key_padding_mask</code> is <code>(batch, seq)</code> with True on padded positions. Without it, padding zeros participate in attention and dilute short sequences.</p>
</div>

<h2>7. Relation to GPT and BERT</h2>
<table>
  <tr><th>Model</th><th>Stack</th><th>Objective</th><th>Good at</th></tr>
  <tr><td>BERT</td><td>Encoder (bidirectional)</td><td>masked LM</td><td>understanding: classification, extraction, retrieval</td></tr>
  <tr><td>GPT</td><td>Decoder (causal mask)</td><td>next token</td><td>generation: writing, chat, code</td></tr>
  <tr><td>ViT</td><td>Encoder over image patches</td><td>classification / contrastive</td><td>vision</td></tr>
</table>
<p>A "large model" is the same block stacked deeper and wider, trained on enormous data. The difference is scale and engineering, not structure.</p>

<div class="keypoints">
  <h4>Key points</h4>
  <ol>
    <li>Attention: score Q against K, softmax, blend V.</li>
    <li>√d controls variance and keeps gradients alive.</li>
    <li>Heads attend to different relations in parallel.</li>
    <li>Attention is order-agnostic; positional encoding is mandatory.</li>
    <li>Block = attention + FFN + residual + LayerNorm; Pre-LN is standard.</li>
    <li>BERT = encoder, GPT = decoder, ViT = patches.</li>
  </ol>
</div>

<div class="quiz" data-answer="1">
  <p class="q">Dividing by √d_k in scaled dot-product attention mainly:</p>
  <ol class="opts">
    <li>Reduces parameters</li>
    <li>Controls dot-product variance so softmax does not saturate</li>
    <li>Makes outputs probabilities</li>
    <li>Speeds up matmul</li>
  </ol>
  <p class="exp">Variance grows with d; without scaling softmax is near one-hot and gradients vanish.</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">Why do Transformers need positional encoding?</p>
  <ol class="opts">
    <li>To make the model bigger</li>
    <li>To handle variable length</li>
    <li>Because attention is permutation-invariant</li>
    <li>To save memory</li>
  </ol>
  <p class="exp">Self-attention treats input as a set; order must be injected.</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">Embed dim 128 with 8 heads means each head has dimension:</p>
  <ol class="opts">
    <li>16</li>
    <li>8</li>
    <li>128</li>
    <li>1024</li>
  </ol>
  <p class="exp">128/8=16. Otherwise PyTorch raises a divisibility error.</p>
</div>

<div class="exercise">
  <h4>Exercises</h4>
  <ol>
    <li>Print the attention weight matrix and verify each row sums to 1.</li>
    <li>Set num_heads=3 with dim 128 and read the error.</li>
    <li>Remove positional encoding and compare outputs for two permutations of the same tokens.</li>
    <li>Stack four blocks, count parameters, estimate inference memory.</li>
  </ol>
</div>`
  }
});
