# 贡献指南

感谢你愿意改进这个项目。这里没有构建步骤、没有依赖，改完刷新浏览器就能看到效果。

## 本地跑起来

```bash
# 方式一：直接双击 index.html（file:// 也能用，但 service worker 不生效）
# 方式二：起一个静态服务器（推荐，能测 PWA 离线）
python -m http.server 8000
```

## 提交前必须做的事

```bash
node tools/validate.mjs
```

必须输出「全部通过 ✓」。CI 会用同样的命令检查 PR，不通过不会合并。

## 新增一章

1. 复制 `content/ch16.js` 改名为 `content/ch17.js`，修改 `id` / `num` / `part` / `title`
2. 在 `index.html` 底部按顺序加一行 `<script src="content/ch17.js"></script>`
3. 刷新页面，侧栏、搜索索引、路线图会自动带上新章节
4. 运行 `node tools/validate.mjs` 确认通过

### 每章的必备区块

校验脚本会检查这些区块是否存在、标签是否闭合、小测答案是否越界：

| 区块 | 标记 |
|---|---|
| 学完你能做到 | `.objectives` |
| 零基础批注（至少一条） | `.callout.basic` |
| 数学批注（建议） | `.callout.math` + `.formula` |
| 代码块 | `<pre data-lang="python"><code>…</code></pre>` |
| 小测（3 道） | `<div class="quiz" data-answer="下标">` + `<p class="exp">` 解析 |
| 动手练习 | `.exercise` |
| 本章要点 | `.keypoints` |

**中英双语必须同步**：`zh.html` 和 `en.html` 结构要对应，`data-answer` 不能只改一边。

## 代码风格

- `js/app.js` 保持 **ES5**（不用 `let` / 箭头函数 / 模板字符串），保证旧浏览器和 `file://` 下稳定
- `content/*.js` 里用模板字符串写长 HTML 是允许的
- 不要引入任何第三方库、CDN 或构建工具 —— 这是本项目的核心约束（可离线、单文件可控）

## 改了资源要发版

如果新增了会被缓存的静态文件（新的 css/js/图片），把 `sw.js` 顶部的 `VERSION` 改一下，
否则老用户会一直拿到旧缓存：

```js
var VERSION = 'v3-2026-10-18';  // 改成新日期
```

## 提交信息

用简短的中文说明「改了什么、为什么」，例如：

```
fix: 修正 ch09 小测第 3 题答案下标写反
feat: 章节内搜索支持跳到具体小节
```

## 不要做的事

- 不要粘贴他人书籍、课程、付费资料的原文（版权风险）
- 不要把 `node_modules`、构建产物、`.DS_Store` 之类带进仓库
- 不要为了「功能更多」引入后端依赖 —— 这个项目刻意保持纯静态
