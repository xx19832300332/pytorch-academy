# PyTorch Academy · 从零开始的 PyTorch 教程站

一个**零依赖、无需构建、可离线运行**的 PyTorch 入门教程网站。18 章中英双语课程，自带全文搜索、三态学习进度、笔记、重点标记、随堂小测、错题本、成就徽章、深色模式、交互式计算器和 Jupyter Notebook 导出。

> 面向**大一新生 / 转行初学者**：每个陌生概念都有「零基础批注」，每处数学都有「数学批注」，不需要先补完微积分和线性代数才能开始。

🔗 在线预览：<https://pytorch-academy.app.workbuddy.host/>

---

## 特性

| 功能 | 说明 |
|---|---|
| **18 章系统课程** | 从 Python 最小集、环境搭建，到张量、autograd、nn.Module、数据加载、训练循环、过拟合、CNN、RNN、Transformer、显存优化、部署，再加**两个综合实战**（图像分类 / 文本分类）|
| **中英双语** | 每个章节、每个 UI 文案都有中英两版，右上角一键切换 |
| **全文搜索** | `Ctrl` / `⌘` + `K` 唤起，同时检索中英正文，结果带关键词高亮，支持 ↑↓ / Enter 键盘导航 |
| **三态学习进度** | 每章区分「未学 / 学习中 / 已完成」，侧栏圆点+颜色，阶段可折叠，顶栏 `n / 18` 徽标 |
| **章节笔记** | 按章自动保存，划词可直接「摘录 / 记疑问」进笔记，可一键导出为 `.md` 归档 |
| **错题本** | 小测答错自动收录，集中复习，答对后手动「我掌握了」移除 |
| **学习数据** | 累计学习时长、连续学习天数、完成度、待复习错题四项，外加 6 枚阶段成就徽章 |
| **阅读设置** | 深色模式、三档字号、新手/进阶模式（进阶隐藏「零基础批注」）|
| **断点续读** | 记住每章滚动位置，回来直接回到上次离开的地方 |
| **悬浮速查表** | 任意页面按 `A` 或点右下角按钮呼出常用 API，不用离开正文 |
| **重点标记** | 选中任意文字即可高亮（黄 / 绿 / 红 / 下划线四种样式），随章节持久化 |
| **随堂小测** | 每章 3 道单选题，点选即判卷并锁定，防止「挨个试出来」的假懂 |
| **交互计算器** | 广播形状计算器（第 3 章）、卷积输出尺寸 / 参数量计算器（第 10 章） |
| **Jupyter 导出** | 任意章节一键导出 `.ipynb`，正文转 Markdown 单元格、代码块转 code 单元格，本地 `jupyter lab` 直接跑 |
| **练习指南** | 内置「练习工具怎么用」页，含环境自检代码、三条环境路线、故障对照表 |
| **术语悬停** | 正文里的关键 API 悬停即显示一句话解释 |
| **完整离线** | 双击 `index.html` 就能用，不联网、不加载任何 CDN |

所有学习数据（进度、笔记、标记）都存在浏览器 `localStorage`，不上传、无账号、无后端。

---

## 快速开始

### 方式一：直接打开（最简单）

下载本仓库，双击 `index.html` 即可。没有构建步骤，没有依赖安装。

### 方式二：本地起一个静态服务器（推荐，避免部分浏览器的 `file://` 限制）

```bash
# Python
python -m http.server 8000

# 或 Node
npx serve .
```

然后访问 <http://localhost:8000>。

### 方式三：部署到任意静态托管

本项目是纯静态文件，可直接部署到 GitHub Pages / Netlify / Vercel / 对象存储：

```bash
# GitHub Pages：把仓库 Settings → Pages → Source 设为 main 分支根目录即可
```

---

## 项目结构

```
pytorch-academy/
├── index.html            # 应用外壳（顶栏、侧栏、正文区、目录、搜索层、笔记抽屉）
├── css/style.css         # 全部样式（3 栏栅格 + 1280/900/680px 三级响应式断点）
├── js/app.js             # 全部逻辑：路由、渲染、搜索、导出、标记、小测、计算器
├── content/
│   ├── ch00.js ~ ch14.js # 15 章正文，每章一个文件，含 zh / en 两份内容
│   ├── ch15.js, ch16.js  # 两个综合实战（图像分类 / 文本分类）
│   ├── ch99.js           # API 速查表
│   ├── practice.js       # 「练习工具怎么用」独立页
│   └── faq.js            # 常见问题（14 个高频报错与解法）
├── assets/               # 站点图标（ico + png）
└── tools/make_icon.py    # 纯标准库生成多尺寸 .ico 的小工具
```

**没有 `package.json`、没有 `node_modules`、没有打包产物。** 浏览器直接吃源码。

### 数据是怎么组织的

每个内容文件向全局注册表 `window.PYTORCH_COURSE.chapters` 里 push 一个对象：

```js
(window.PYTORCH_COURSE = window.PYTORCH_COURSE || { chapters: [] }).chapters.push({
  id: 'ch03',            // 唯一 id，同时是 URL hash：#/ch03
  num: '03',             // 显示用编号
  part: 'p1',            // 归属模块（p0~p5）
  file: 'ops',           // 导出 notebook 的文件名片段
  zh: { title: '…', sub: '…', html: `…` },
  en: { title: '…', sub: '…', html: `…` }
});
```

`js/app.js` 启动时把 `chapters` 按 `part` 分组渲染成侧栏，正文的 HTML 字符串直接注入 `.prose` 容器。

---

## 新增一章

1. 复制一份 `content/ch14.js` 改名为 `content/ch15.js`，修改 `id` / `num` / `part` / `title`。
2. 在 `index.html` 底部按顺序加一行：`<script src="content/ch15.js"></script>`。
3. 刷新页面，侧栏和搜索索引会自动带上新章节。

### 每章的推荐结构

正文 HTML 里建议包含这些类名，`app.js` 和样式表对它们有专门处理：

| 区块 | 类名 / 属性 | 作用 |
|---|---|---|
| 学完你能做到 | `.objectives` | 灰色目标卡片 |
| 零基础批注 | `.callout.basic` | 生活化解释陌生概念 |
| 数学批注 | `.callout.math` + `.formula` | 公式与推导 |
| 坑与提醒 | `.callout.warn` / `.callout.tip` | 黄色 / 蓝色提示 |
| 代码块 | `<pre data-lang="python"><code>…</code></pre>` | 自动语法高亮 + 一键复制 |
| 小测 | `<div class="quiz" data-answer="1">` | `data-answer` 是正确选项下标（从 0 起） |
| 交互工具 | `<div class="widget" data-widget="shape">` | 目前支持 `shape` / `conv` |
| 本章要点 | `.keypoints` | 结尾要点卡片 |

---

## 技术说明

- **路由**：hash 路由（`#/ch04`），无需服务端 rewrite，`file://` 下也能用。
- **语法高亮**：手写的单正则 Python 高亮器，无第三方库。
- **索引**：启动时把每章 HTML 按块级标签切词、去标签后建内存索引，搜索为子串匹配。
- **导出 `.ipynb`**：遍历正文顶层 DOM，`pre` → code cell，`h2/h3/h4/p/ul/ol/table/callout` → markdown cell，生成 nbformat 4.5 JSON。
- **持久化 key**：`pta.lang` / `pta.progress` / `pta.notes` / `pta.annotations` / `pta.stars` / `pta.last` / `pta.ui` / `pta.fold` / `pta.visited` / `pta.scroll` / `pta.wrong` / `pta.stats`。

---

## Roadmap

- [x] 深色主题 / 字号调节 / 新手-进阶模式
- [x] 代码块「一键复制代码并打开 Colab」（Pyodide 没有 PyTorch 轮子，无法在纯前端真正跑 torch）
- [ ] 笔记支持 Markdown 预览
- [ ] 错题本按知识点聚类
- [ ] 进度导出 / 导入（跨设备同步，需要后端）

欢迎开 Issue 认领。

---

## 贡献

欢迎提 Issue 或 PR：修正错别字、补充批注、新增章节都可以。新增内容请保持中英双语同步，并按上面的推荐结构组织。

代码风格：`js/app.js` 保持 ES5 写法（无 `let` / 箭头函数 / 模板字符串），以便在旧浏览器和 `file://` 环境下稳定运行；`content/*.js` 里用模板字符串书写长 HTML 是允许的。

---

## License

[MIT](./LICENSE)

---

## English

A dependency-free, build-free, offline-capable PyTorch course site. 18 bilingual (Chinese / English) chapters covering everything from Python basics and tensors to autograd, `nn.Module`, data pipelines, training loops, regularization, CNNs, RNNs, Transformers, memory optimization, deployment, plus two capstone projects.

Highlights: full-text search (`Ctrl`/`⌘`+`K`), tri-state progress with collapsible stages, per-chapter notes with quote-to-note and `.md` export, four highlight styles, per-chapter quizzes with a mistake book, study-time/streak analytics and achievement badges, dark mode with font scaling, beginner/advanced reading modes, scroll-position restore, a floating cheat sheet, two interactive calculators, and one-click **Jupyter Notebook export** for every chapter.

To run: open `index.html`, or serve the folder with any static server. No build step, no dependencies, no backend — all study data stays in `localStorage`.

Contributions welcome. New content should stay bilingual and follow the class-name conventions listed above.
