/**
 * 内容与语法校验（无需联网、无依赖，只用 Node 内置模块）
 *
 * 用法：
 *   node tools/validate.mjs            # 校验，出错时 exit 1
 *   node tools/validate.mjs --verbose  # 额外打印每章规模
 *
 * 检查项：
 *   1. 所有 js/*.js、sw.js、content/*.js 能通过 vm 解析（等价于 node --check 的脚本层）
 *   2. 章节 id 唯一、必须有 zh / en 两份内容、必须有 html
 *   3. 每章 3 道小测，data-answer 下标不越界
 *   4. 关键标签成对闭合（div/p/ul/ol/li/pre/table/h2/h3）
 *   5. 必备区块存在（objectives / callout basic / keypoints / exercise）
 *   6. index.html 引用的内容脚本都存在，且 content/ 里没有被漏引用的章节
 */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const VERBOSE = process.argv.includes('--verbose');

const errors = [];
const warnings = [];
const fail = (m) => errors.push(m);
const warn = (m) => warnings.push(m);

function read(p) {
  return fs.readFileSync(path.join(ROOT, p), 'utf8');
}

/* ---------- 1. 语法：把每个脚本编译一遍 ---------- */
const scriptFiles = [];
for (const dir of ['js', 'content', 'tools']) {
  const abs = path.join(ROOT, dir);
  if (!fs.existsSync(abs)) continue;
  for (const f of fs.readdirSync(abs)) {
    if (f.endsWith('.js') || f.endsWith('.mjs')) scriptFiles.push(`${dir}/${f}`);
  }
}
if (fs.existsSync(path.join(ROOT, 'sw.js'))) scriptFiles.push('sw.js');

for (const f of scriptFiles) {
  if (f.startsWith('tools/')) continue; // 工具脚本自己就是被跑的对象，跳过
  const code = read(f);
  try {
    new vm.Script(code, { filename: f });
  } catch (e) {
    fail(`${f}: 语法错误 — ${e.message}`);
  }
}

/* ---------- 2. 加载内容 ---------- */
const contentDir = path.join(ROOT, 'content');
const contentFiles = fs.existsSync(contentDir)
  ? fs.readdirSync(contentDir).filter((f) => f.endsWith('.js')).sort()
  : [];

const sandbox = { window: {} };
vm.createContext(sandbox);
for (const f of contentFiles) {
  try {
    vm.runInContext(read(`content/${f}`), sandbox, { filename: f });
  } catch (e) {
    fail(`content/${f}: 运行失败 — ${e.message}`);
  }
}

const STORE = sandbox.window.PYTORCH_COURSE || {};
const chapters = STORE.chapters || [];

if (!chapters.length) fail('没有加载到任何章节（window.PYTORCH_COURSE.chapters 为空）');

/* ---------- 3. 逐章检查 ---------- */
const seenIds = new Set();
const VALID_PARTS = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6'];

for (const c of chapters) {
  const tag = c.id || '(no id)';
  if (!c.id) fail(`${tag}: 缺少 id`);
  else if (seenIds.has(c.id)) fail(`${tag}: id 重复`);
  else seenIds.add(c.id);

  if (!c.num) fail(`${tag}: 缺少 num`);
  if (!VALID_PARTS.includes(c.part)) fail(`${tag}: part="${c.part}" 不在 ${VALID_PARTS.join('/')} 中`);

  for (const lang of ['zh', 'en']) {
    const d = c[lang];
    if (!d) { fail(`${tag}: 缺少 ${lang} 内容`); continue; }
    if (!d.title) fail(`${tag}.${lang}: 缺少 title`);
    if (!d.html || d.html.length < 500) fail(`${tag}.${lang}: html 缺失或过短（${(d.html || '').length}）`);
    if (!d.html) continue;

    const h = d.html;

    /* 小测 */
    const quizRe = /<div class="quiz" data-answer="(\d+)">([\s\S]*?)<\/div>/g;
    let m;
    let quizCount = 0;
    while ((m = quizRe.exec(h)) !== null) {
      quizCount += 1;
      const ans = Number(m[1]);
      const options = (m[2].match(/<li>/g) || []).length;
      if (ans >= options) fail(`${tag}.${lang}: 小测第 ${quizCount} 题 data-answer=${ans}，但只有 ${options} 个选项`);
      if (!/class="exp"/.test(m[2])) warn(`${tag}.${lang}: 小测第 ${quizCount} 题没有 exp 解析`);

      /* 选项不能重复：出现过「两个选项其实是同一个答案」的硬伤 */
      const optBlock = m[2].match(/<ol class="opts">([\s\S]*?)<\/ol>/);
      if (optBlock) {
        const texts = (optBlock[1].match(/<li>([\s\S]*?)<\/li>/g) || [])
          .map((li) => li.replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim());
        const dup = texts.find((t, i) => t && texts.indexOf(t) !== i);
        if (dup) fail(`${tag}.${lang}: 小测第 ${quizCount} 题有重复选项「${dup}」`);

      }
    }
    if (quizCount !== 3) warn(`${tag}.${lang}: 小测数量为 ${quizCount}（建议 3）`);

    /* 标签平衡 */
    for (const t of ['div', 'p', 'ul', 'ol', 'li', 'pre', 'table', 'h2', 'h3']) {
      const open = (h.match(new RegExp(`<${t}[\\s>]`, 'g')) || []).length;
      const close = (h.match(new RegExp(`</${t}>`, 'g')) || []).length;
      if (open !== close) fail(`${tag}.${lang}: <${t}> 未闭合（${open} 开 / ${close} 闭）`);
    }

    /* 必备区块（速查表 ch99 是纯参考页，不要求） */
    if (c.id !== 'ch99') {
      for (const need of ['objectives', 'callout basic', 'keypoints', 'exercise']) {
        if (!h.includes(need)) fail(`${tag}.${lang}: 缺少必备区块 "${need}"`);
      }
    }

    /* 代码块语言标记 */
    if (!/pre data-lang=/.test(h) && c.id !== 'ch99') warn(`${tag}.${lang}: 没有带 data-lang 的代码块`);

    if (VERBOSE) {
      console.log(
        `  ${tag}.${lang}`.padEnd(14),
        String(h.length).padStart(6), 'chars |',
        String(quizCount).padStart(2), 'quiz |',
        String((h.match(/<pre /g) || []).length).padStart(2), 'code'
      );
    }
  }
}

/* ---------- 4. 独立页面 ---------- */
for (const key of ['practice', 'faq']) {
  const pg = STORE[key];
  if (!pg) { warn(`缺少独立页 ${key}（window.PYTORCH_COURSE.${key}）`); continue; }
  for (const lang of ['zh', 'en']) {
    if (!pg[lang] || !pg[lang].html) fail(`${key}.${lang}: 缺少内容`);
  }
}

/* ---------- 5. index.html 引用一致性 ---------- */
const html = read('index.html');
const referenced = new Set();
const re = /<script\s+src="(content\/[^"]+)"/g;
let mm;
while ((mm = re.exec(html)) !== null) referenced.add(mm[1]);

for (const f of referenced) {
  if (!fs.existsSync(path.join(ROOT, f))) fail(`index.html 引用了不存在的 ${f}`);
}
for (const f of contentFiles) {
  const rel = `content/${f}`;
  if (!referenced.has(rel)) fail(`${rel} 没有被 index.html 引用（内容不会加载）`);
}

/* 关键 meta */
for (const must of ['name="viewport"', 'rel="manifest"', 'name="theme-color"']) {
  if (!html.includes(must)) warn(`index.html 缺少 ${must}`);
}

/* ---------- 输出 ---------- */
console.log(`\n检查了 ${chapters.length} 个章节、${contentFiles.length} 个内容文件、${scriptFiles.length - 1} 个脚本`);

if (warnings.length) {
  console.log(`\n警告 ${warnings.length} 条：`);
  for (const w of warnings) console.log('  ! ' + w);
}

if (errors.length) {
  console.log(`\n错误 ${errors.length} 条：`);
  for (const e of errors) console.log('  x ' + e);
  process.exit(1);
}

console.log('\n全部通过 ✓');
