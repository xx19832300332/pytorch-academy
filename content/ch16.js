(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch16',
  num: '16',
  part: 'p6',
  file: 'capstone-text',
  zh: {
    title: '综合实战：文本情感分类',
    sub: '不等长序列、词表、padding——文本任务的三个新问题',
    html: `
<p class="lead">图像是规整的网格，文本是不等长的序列。这一章用一个能在一分钟内训练完的情感分类器，把「变长输入怎么变成张量」「词表是什么」「padding 到多少」这三个文本任务特有的问题讲清楚。数据内置在代码里，不下载任何东西。</p>

<div class="objectives">
  <h4>学完你能做到</h4>
  <ul>
    <li>自己构建词表，把句子变成 id 序列</li>
    <li>用 <code>collate_fn</code> 把不等长的句子 pad 成同形状的批次</li>
    <li>写出 Embedding → 池化 → 全连接的文本分类模型</li>
    <li>理解 padding 位置为什么必须屏蔽（mask）</li>
  </ul>
</div>

<div class="callout basic">
  <span class="tag">零基础批注</span>
  <p><b>词表（vocabulary）</b>就是一本「字 → 编号」的字典。模型只认数字，不认汉字，所以第一步永远是把句子翻译成一串整数。「我 喜欢 这个」→ <code>[5, 21, 8]</code>。这本字典是你在训练前统计语料得到的，训练后必须原样保存——否则同一个字在推理时对应另一个编号，模型就废了。</p>
  <p><b>padding</b> 是「补位」。一批里有 3 个字的句子，也有 10 个字的，张量要求每行等长，所以短的用 0 补齐到最长的那句。补出来的 0 是假的，不能让模型把它当真实内容，这就是后面要 mask 的原因。</p>
</div>

<h2>1. 数据与词表</h2>
<pre data-lang="python"><code>import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 内置小语料：1 = 正面，0 = 负面。真实项目里这里换成几千上万条
RAW = [
    ("这个电影太好看了，我很喜欢", 1),
    ("剧情精彩，演员也演得很好", 1),
    ("非常推荐，值得二刷", 1),
    ("画面很美，音乐也很动人", 1),
    ("节奏舒服，看完心情很好", 1),
    ("故事一般，但结局还算惊喜", 1),
    ("太难看了，浪费时间", 0),
    ("剧情拖沓，看得我睡着了", 0),
    ("演员尴尬，台词也很假", 0),
    ("完全不推荐，后悔买票", 0),
    ("特效粗糙，逻辑混乱", 0),
    ("前面还行，后面彻底崩了", 0),
]

def tokenize(text):
    """按字切分：中文不需要分词库也能跑通，够用来理解流程"""
    return list(text)

# ---------- 构建词表 ----------
PAD, UNK = 0, 1                       # 0 专门留给 padding，1 留给没见过的字
vocab = {"<pad>": PAD, "<unk>": UNK}
for text, _ in RAW:
    for ch in tokenize(text):
        if ch not in vocab:
            vocab[ch] = len(vocab)

print("词表大小:", len(vocab))
print("示例编码:", [vocab.get(c, UNK) for c in tokenize(RAW[0][0])])</code></pre>

<div class="callout warn">
  <span class="tag">两个约定不能省</span>
  <p>第一，<code>PAD</code> 必须是 0，因为 <code>nn.Embedding</code> 的 <code>padding_idx=0</code> 依赖这个约定，它会把第 0 行向量固定为全零且不参与梯度。第二，<code>UNK</code> 必须存在，否则推理时遇到训练集里没出现过的字就会直接 <code>KeyError</code> —— 这是新手最常见的线上事故。</p>
</div>

<h2>2. Dataset 与 collate_fn：解决不等长</h2>
<pre data-lang="python"><code>class SentDataset(Dataset):
    def __init__(self, data, vocab):
        self.items = [(torch.tensor([vocab.get(c, UNK) for c in tokenize(t)]), y)
                      for t, y in data]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

def collate(batch):
    """把一批不等长的 id 序列 pad 成 (B, Lmax)，并生成 mask"""
    seqs, labels = zip(*batch)
    L = max(len(s) for s in seqs)
    out = torch.zeros(len(seqs), L, dtype=torch.long)      # 默认填 0 = PAD
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    mask = (out != PAD)                                     # True = 真实文字
    return out, mask, torch.tensor(labels)

ld = DataLoader(SentDataset(RAW, vocab), batch_size=4, shuffle=True, collate_fn=collate)
x, mask, y = next(iter(ld))
print("x     :", x.shape)      # (4, Lmax)
print("mask  :", mask.shape)   # 同形状，True 表示真字
print("labels:", y)</code></pre>

<div class="callout basic">
  <span class="tag">collate_fn 是什么</span>
  <p>DataLoader 默认会尝试用 <code>torch.stack</code> 把一批样本摞起来，但长度不同的序列摞不了，会直接报错。传入 <code>collate_fn</code> 就是告诉它：「这批样本我自己拼」。所以凡是变长数据（文本、语音、点云），都要自己写 collate。</p>
</div>

<h2>3. 模型：Embedding + mask 池化</h2>
<pre data-lang="python"><code>class TextClf(nn.Module):
    def __init__(self, vocab_size, dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.fc  = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 2))

    def forward(self, x, mask):
        v = self.emb(x)                                  # (B, L, dim)
        m = mask.unsqueeze(-1).float()                   # (B, L, 1)
        pooled = (v * m).sum(1) / m.sum(1).clamp(min=1)  # 只对真字求平均
        return self.fc(pooled)

model = TextClf(len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

for epoch in range(60):
    model.train()
    for x, mask, y in ld:
        optimizer.zero_grad()
        loss = criterion(model(x, mask), y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"epoch {epoch+1}  loss {loss.item():.4f}")</code></pre>

<div class="callout math">
  <span class="tag">数学批注</span>
  <p>那行 <code>(v * m).sum(1) / m.sum(1)</code> 是「掩码平均池化」。如果直接 <code>v.mean(1)</code>，被 pad 的 0 向量也会进分母，句子越短被稀释得越厉害，短句的表示就会系统性偏小——这是个隐蔽但严重的问题。</p>
  <p class="formula">pooled = Σ(vᵢ · mᵢ) / max(Σmᵢ, 1)</p>
  <p>分母用 <code>clamp(min=1)</code> 防止全 pad 的极端情况除以 0。这类「先乘 mask 再求和、最后除以真实长度」的写法，在注意力、序列池化里会反复出现，值得背下来。</p>
</div>

<h2>4. 推理：新句子怎么预测</h2>
<pre data-lang="python"><code>@torch.no_grad()
def predict(text):
    model.eval()
    ids = torch.tensor([[vocab.get(c, UNK) for c in tokenize(text)]])
    mask = (ids != PAD)
    prob = model(ids, mask).softmax(1)[0]
    return ("正面" if prob[1] &gt; prob[0] else "负面"), round(prob.max().item(), 3)

for s in ["这部片子真的很棒", "太糟糕了，不想再看", "还可以吧"]:
    print(s, "->", predict(s))</code></pre>

<div class="callout tip">
  <span class="tag">模型不是「懂」了，是「记住」了</span>
  <p>用 12 条语料训出来的模型只是背下了这几个句子，换个说法很可能翻车。这很正常——重点是流程，不是效果。想看真实效果，把 <code>RAW</code> 换成几千条真实评论（比如公开的电商评论数据集），把按字切分换成 jieba 分词，再跑一遍。</p>
</div>

<h2>5. 本章小测</h2>

<div class="quiz" data-answer="1">
  <p class="q">为什么 <code>nn.Embedding</code> 要传 <code>padding_idx=0</code>？</p>
  <ol class="opts">
    <li>让 pad 的向量等于 1，方便计算</li>
    <li>把第 0 行固定成全零且不更新梯度，避免 pad 参与学习</li>
    <li>只是语法要求，没有实际作用</li>
    <li>它会自动把序列补齐</li>
  </ol>
  <p class="exp">padding_idx 会把对应行置零并跳过梯度更新，否则模型会把「补位用的 0」也当成一个有含义的词去学。</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">不写 <code>collate_fn</code> 直接喂不等长序列会发生什么？</p>
  <ol class="opts">
    <li>DataLoader 会自动 pad</li>
    <li>会悄悄截断到最短长度</li>
    <li>默认的拼接逻辑失败并报错，因为形状不一致</li>
    <li>训练变慢但能跑完</li>
  </ol>
  <p class="exp">默认的 stack/cat 要求形状一致，长度不同就会报错。变长数据必须自己写 collate。</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">池化时不做 mask、直接对整行平均，主要后果是？</p>
  <ol class="opts">
    <li>短句的表示被 pad 稀释，系统性偏小</li>
    <li>会报维度错误</li>
    <li>只是浪费一点算力</li>
    <li>会让 loss 变负</li>
  </ol>
  <p class="exp">pad 的零向量会把分母撑大，句子越短被稀释越严重，模型对短句的判断就会系统性偏移。</p>
</div>

<div class="exercise">
  <h4>动手练习</h4>
  <ol>
    <li>把 <code>RAW</code> 扩到 40 条，看看预测那句「还可以吧」是否还准。</li>
    <li>把 <code>padding_idx=PAD</code> 改成不传，重复训练，对比 loss 曲线——观察 pad 参与学习后的差异。</li>
    <li>把池化从「掩码平均」改成「掩码最大池化」，比较收敛速度。</li>
    <li>进阶：把 Embedding 换成 <code>nn.EmbeddingBag(vocab, dim, mode="mean", padding_idx=0)</code>，看代码能简化多少行。</li>
  </ol>
</div>

<div class="keypoints">
  <h4>本章要点</h4>
  <ul>
    <li>文本第一步永远是「建词表 → 转 id」，词表必须随模型一起保存</li>
    <li>PAD 用 0、UNK 必须留，是两条不能省的约定</li>
    <li>变长序列必须自己写 collate_fn 做 padding</li>
    <li>池化要 mask，否则短句被系统性稀释</li>
    <li>推理流程和训练一致，只多一个 eval() + no_grad</li>
  </ul>
</div>
`
  },
  en: {
    title: 'Capstone: sentiment classification on text',
    sub: 'Variable-length sequences, vocabularies, padding — the three new problems in NLP',
    html: `
<p class="lead">Images are regular grids; text is a variable-length sequence. This chapter builds a sentiment classifier that trains in about a minute, to make three text-specific ideas concrete: how variable-length input becomes a tensor, what a vocabulary is, and how long padding should be. The data is embedded in the code — nothing to download.</p>

<div class="objectives">
  <h4>After this chapter you can</h4>
  <ul>
    <li>Build a vocabulary and turn sentences into id sequences</li>
    <li>Use <code>collate_fn</code> to pad unequal sentences into one batch</li>
    <li>Write an Embedding → pooling → linear text classifier</li>
    <li>Explain why padded positions must be masked out</li>
  </ul>
</div>

<div class="callout basic">
  <span class="tag">Beginner note</span>
  <p>A <b>vocabulary</b> is a character-to-id dictionary. Models only understand numbers, so step one is always translating a sentence into integers: "I like this" → <code>[5, 21, 8]</code>. You build this from your training corpus, and you must save it with the model — the same character must map to the same id at inference, or the model is worthless.</p>
  <p><b>Padding</b> is filling in. A batch may mix 3-character and 10-character sentences, but a tensor needs equal row lengths, so short ones get zeros. Those zeros are fake and must not be treated as content — hence masking.</p>
</div>

<h2>1. Data and vocabulary</h2>
<pre data-lang="python"><code>import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# tiny built-in corpus: 1 = positive, 0 = negative
RAW = [
    ("这个电影太好看了，我很喜欢", 1),
    ("剧情精彩，演员也演得很好", 1),
    ("非常推荐，值得二刷", 1),
    ("画面很美，音乐也很动人", 1),
    ("节奏舒服，看完心情很好", 1),
    ("故事一般，但结局还算惊喜", 1),
    ("太难看了，浪费时间", 0),
    ("剧情拖沓，看得我睡着了", 0),
    ("演员尴尬，台词也很假", 0),
    ("完全不推荐，后悔买票", 0),
    ("特效粗糙，逻辑混乱", 0),
    ("前面还行，后面彻底崩了", 0),
]

def tokenize(text):
    """character-level split: no tokenizer library needed, good enough for the pipeline"""
    return list(text)

PAD, UNK = 0, 1                       # 0 reserved for padding, 1 for unseen characters
vocab = {"<pad>": PAD, "<unk>": UNK}
for text, _ in RAW:
    for ch in tokenize(text):
        if ch not in vocab:
            vocab[ch] = len(vocab)

print("vocab size:", len(vocab))
print("encoded:", [vocab.get(c, UNK) for c in tokenize(RAW[0][0])])</code></pre>

<div class="callout warn">
  <span class="tag">Two conventions you cannot skip</span>
  <p>First, <code>PAD</code> must be 0, because <code>nn.Embedding</code>'s <code>padding_idx=0</code> relies on it: row 0 is pinned to zeros and receives no gradient. Second, <code>UNK</code> must exist, otherwise inference on a character never seen in training raises <code>KeyError</code> — the most common production accident for beginners.</p>
</div>

<h2>2. Dataset and collate_fn: handling variable length</h2>
<pre data-lang="python"><code>class SentDataset(Dataset):
    def __init__(self, data, vocab):
        self.items = [(torch.tensor([vocab.get(c, UNK) for c in tokenize(t)]), y)
                      for t, y in data]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

def collate(batch):
    """pad a batch of unequal id sequences to (B, Lmax) and build the mask"""
    seqs, labels = zip(*batch)
    L = max(len(s) for s in seqs)
    out = torch.zeros(len(seqs), L, dtype=torch.long)      # zeros == PAD
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    mask = (out != PAD)                                     # True = real characters
    return out, mask, torch.tensor(labels)

ld = DataLoader(SentDataset(RAW, vocab), batch_size=4, shuffle=True, collate_fn=collate)
x, mask, y = next(iter(ld))
print("x     :", x.shape)      # (4, Lmax)
print("mask  :", mask.shape)
print("labels:", y)</code></pre>

<div class="callout basic">
  <span class="tag">What collate_fn is for</span>
  <p>By default DataLoader tries to <code>torch.stack</code> a batch, which fails on unequal lengths. Passing <code>collate_fn</code> says "I will assemble this batch myself". Any variable-length data — text, audio, point clouds — needs one.</p>
</div>

<h2>3. Model: Embedding plus masked pooling</h2>
<pre data-lang="python"><code>class TextClf(nn.Module):
    def __init__(self, vocab_size, dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.fc  = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 2))

    def forward(self, x, mask):
        v = self.emb(x)                                  # (B, L, dim)
        m = mask.unsqueeze(-1).float()                   # (B, L, 1)
        pooled = (v * m).sum(1) / m.sum(1).clamp(min=1)  # average over real chars only
        return self.fc(pooled)

model = TextClf(len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

for epoch in range(60):
    model.train()
    for x, mask, y in ld:
        optimizer.zero_grad()
        loss = criterion(model(x, mask), y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"epoch {epoch+1}  loss {loss.item():.4f}")</code></pre>

<div class="callout math">
  <span class="tag">Math note</span>
  <p><code>(v * m).sum(1) / m.sum(1)</code> is masked average pooling. With plain <code>v.mean(1)</code>, padded zero vectors still enter the denominator, so shorter sentences get diluted more — a silent, systematic bias against short inputs.</p>
  <p class="formula">pooled = Σ(vᵢ · mᵢ) / max(Σmᵢ, 1)</p>
  <p>The <code>clamp(min=1)</code> guards against a division by zero on all-pad rows. This "multiply by mask, sum, divide by true length" pattern recurs in attention and sequence pooling — worth memorising.</p>
</div>

<h2>4. Inference on a new sentence</h2>
<pre data-lang="python"><code>@torch.no_grad()
def predict(text):
    model.eval()
    ids = torch.tensor([[vocab.get(c, UNK) for c in tokenize(text)]])
    mask = (ids != PAD)
    prob = model(ids, mask).softmax(1)[0]
    return ("positive" if prob[1] &gt; prob[0] else "negative"), round(prob.max().item(), 3)

for s in ["这部片子真的很棒", "太糟糕了，不想再看", "还可以吧"]:
    print(s, "->", predict(s))</code></pre>

<div class="callout tip">
  <span class="tag">It memorised, it did not understand</span>
  <p>Twelve sentences only teach the model those sentences; rephrase and it will likely fail. That is fine — the pipeline is the lesson, not the accuracy. For real results, replace <code>RAW</code> with a few thousand labelled reviews and swap character splitting for a proper tokenizer.</p>
</div>

<h2>5. Quiz</h2>

<div class="quiz" data-answer="1">
  <p class="q">Why pass <code>padding_idx=0</code> to <code>nn.Embedding</code>?</p>
  <ol class="opts">
    <li>So padded vectors equal 1, which is easier to compute</li>
    <li>To pin row 0 to zeros and exclude it from gradient updates</li>
    <li>It is only a syntax requirement</li>
    <li>It pads sequences automatically</li>
  </ol>
  <p class="exp">padding_idx zeroes that row and skips its gradient, otherwise the model learns "0" as a meaningful token.</p>
</div>

<div class="quiz" data-answer="2">
  <p class="q">What happens without a <code>collate_fn</code> on unequal-length sequences?</p>
  <ol class="opts">
    <li>DataLoader pads them automatically</li>
    <li>They are silently truncated to the shortest</li>
    <li>Default batching fails and raises a shape error</li>
    <li>It just runs slower</li>
  </ol>
  <p class="exp">The default stack requires identical shapes. Variable-length data always needs a custom collate.</p>
</div>

<div class="quiz" data-answer="0">
  <p class="q">If you average over the whole row without masking, the main consequence is:</p>
  <ol class="opts">
    <li>Short sentences get diluted by padding and skew small</li>
    <li>A dimension error is raised</li>
    <li>You only waste some compute</li>
    <li>Loss turns negative</li>
  </ol>
  <p class="exp">Zero vectors enlarge the denominator, so shorter inputs are diluted more, biasing the model against them.</p>
</div>

<div class="exercise">
  <h4>Hands-on</h4>
  <ol>
    <li>Grow <code>RAW</code> to 40 sentences and check whether "还可以吧" is still predicted well.</li>
    <li>Drop <code>padding_idx=PAD</code> and retrain — compare loss curves to see padding leak into learning.</li>
    <li>Switch masked mean pooling to masked max pooling and compare convergence.</li>
    <li>Advanced: replace the embedding + pooling with <code>nn.EmbeddingBag(vocab, dim, mode="mean", padding_idx=0)</code> and see how many lines disappear.</li>
  </ol>
</div>

<div class="keypoints">
  <h4>Key points</h4>
  <ul>
    <li>Text always starts with vocabulary → ids, and the vocab ships with the model</li>
    <li>PAD is 0 and UNK must exist — two conventions you cannot skip</li>
    <li>Variable-length data needs a custom collate_fn for padding</li>
    <li>Pool with a mask, or short sentences get systematically diluted</li>
    <li>Inference mirrors training, plus eval() and no_grad</li>
  </ul>
</div>
`
  }
});
