/* ============================================================
   PyTorch Academy — app logic
   纯前端：hash 路由 / 搜索 / 学习进度 / 笔记 / 划词标记 / ipynb 导出
   ============================================================ */
(function () {
  'use strict';

  var COURSE = window.PYTORCH_COURSE || { chapters: [] };
  var CH = COURSE.chapters;

  var PARTS = [
    { id: 'p0', zh: '阶段一 · 入门', en: 'Stage 1 · Getting started' },
    { id: 'p1', zh: '阶段二 · 张量与自动微分', en: 'Stage 2 · Tensors & autograd' },
    { id: 'p2', zh: '阶段三 · 模型与训练', en: 'Stage 3 · Models & training' },
    { id: 'p3', zh: '阶段四 · 网络结构', en: 'Stage 4 · Architectures' },
    { id: 'p4', zh: '阶段五 · 工程与部署', en: 'Stage 5 · Engineering' },
    { id: 'p5', zh: '附录', en: 'Appendix' }
  ];

  /* ---------------- i18n ---------------- */
  var I18N = {
    zh: {
      brandSub: '学习站',
      searchLabel: '搜索章节 / 概念 / 代码',
      searchPlaceholder: '搜索：张量、广播、autograd、CrossEntropy…',
      notes: '笔记',
      notesTitle: '本章笔记',
      notesSub: '自动保存在本机浏览器，不会上传',
      notesPh: '记录你的理解、疑问、代码片段…\n\n例如：\n- 没看懂的地方：\n- 我发现的一个坑：\n- 想试试的实验：',
      notesSaved: '已保存',
      notesIdle: '自动保存',
      exportNotes: '导出 .md',
      clearNote: '清空本章',
      tocTitle: '本页目录',
      completed: '已完成',
      markDone: '标记本章完成',
      done: '已掌握 ✓',
      important: '标为重要',
      notebook: '下载 Notebook',
      prev: '上一章',
      next: '下一章',
      copy: '复制',
      copied: '已复制',
      home: '首页',
      practice: '练习工具怎么用',
      searchEmpty: '没有找到相关内容，换个关键词试试。',
      searchHint: '支持中文 / 英文关键词，例如「广播」「broadcast」「optimizer」',
      searchCount: function (n) { return '共 ' + n + ' 条结果'; },
      searchNav: '↑↓ 选择 · Enter 打开 · Esc 关闭',
      heroTitle1: '从零开始，',
      heroTitle2: '真正学会 PyTorch',
      heroSub: '16 个章节（含速查表），从矩阵乘法到 Transformer。每章配可运行的 Jupyter Notebook、随堂小测与「零基础批注」，遇到陌生概念和数理知识就在原文旁边解释清楚。',
      startLearn: '开始学习',
      continueLearn: '继续上次学习',
      openNotebookTip: '下载本章 Notebook',
      statChapters: '章节总数',
      statDone: '已掌握',
      statNotes: '笔记字数',
      statMarks: '重点标记',
      roadmap: '课程路线图',
      tools: '实践工具',
      tool1t: 'Jupyter Notebook',
      tool1p: '每章右上角可下载 .ipynb，本地 Jupyter 或 Colab 打开即可逐格运行、修改、做实验。',
      tool2t: '随堂小测',
      tool2p: '每章末尾有选择题，点选项立即判分并给出解析，检验你是否真的读懂了。',
      tool3t: '交互小工具',
      tool3p: '广播形状计算器等可视化工具，把抽象的规则变成可以动手试的东西。',
      tool4t: '速查表',
      tool4p: '常用 API 一页纸，忘了怎么写的时候翻一翻，比重看教程快。',
      tips: '学习建议',
      tip1: '先跑通代码，再回头看解释。手感比记忆重要。',
      tip2: '每章的「动手练习」至少做一题，卡住再看答案。',
      tip3: '善用划词标记：选中文字即可高亮或标为重点，下次复习直接看红黄块。',
      resetAll: '重置全部数据',
      resetConfirm: '将清除所有进度、笔记和标记，确定吗？',
      chipTitle: '已掌握章节',
      foot: 'PyTorch 学习站 · 本地运行，数据只保存在你的浏览器里',
      langTag: '中文',
      notFound: '这一章不存在'
    },
    en: {
      brandSub: 'Academy',
      searchLabel: 'Search chapters, concepts, code',
      searchPlaceholder: 'Try: tensor, broadcast, autograd, CrossEntropy…',
      notes: 'Notes',
      notesTitle: 'Chapter notes',
      notesSub: 'Saved locally in your browser',
      notesPh: 'Your understanding, questions, snippets…\n\n- Still confused about:\n- A gotcha I found:\n- An experiment to try:',
      notesSaved: 'Saved',
      notesIdle: 'Autosave on',
      exportNotes: 'Export .md',
      clearNote: 'Clear chapter',
      tocTitle: 'On this page',
      completed: 'completed',
      markDone: 'Mark as complete',
      done: 'Mastered ✓',
      important: 'Mark important',
      notebook: 'Download notebook',
      prev: 'Previous',
      next: 'Next',
      copy: 'Copy',
      copied: 'Copied',
      home: 'Home',
      practice: 'How to practice',
      searchEmpty: 'Nothing found. Try another keyword.',
      searchHint: 'Chinese and English keywords both work, e.g. 广播 / broadcast / optimizer',
      searchCount: function (n) { return n + ' result(s)'; },
      searchNav: '↑↓ to move · Enter to open · Esc to close',
      heroTitle1: 'Learn PyTorch ',
      heroTitle2: 'properly, from zero',
      heroSub: '16 chapters including a cheatsheet, from matrix multiplication to Transformers. Every chapter ships a runnable Jupyter notebook, quick quizzes, and beginner notes that explain the unfamiliar math right where you meet it.',
      startLearn: 'Start learning',
      continueLearn: 'Resume where I left off',
      openNotebookTip: 'Download this chapter',
      statChapters: 'Chapters',
      statDone: 'Mastered',
      statNotes: 'Note chars',
      statMarks: 'Marks',
      roadmap: 'Roadmap',
      tools: 'Practice tools',
      tool1t: 'Jupyter notebooks',
      tool1p: 'Download a .ipynb per chapter and run it cell by cell in local Jupyter or Colab.',
      tool2t: 'Quizzes',
      tool2p: 'Instant feedback with explanations, so you know whether you really got it.',
      tool3t: 'Interactive widgets',
      tool3p: 'Shape/broadcast calculators turn abstract rules into something you can poke at.',
      tool4t: 'Cheatsheet',
      tool4p: 'One page of the APIs you actually use, for when you forget the syntax.',
      tips: 'How to study',
      tip1: 'Run the code first, read the explanation second. Muscle memory beats memorising.',
      tip2: 'Do at least one exercise per chapter before peeking at hints.',
      tip3: 'Select text to highlight or flag as key — review your red/yellow blocks before exams.',
      resetAll: 'Reset all data',
      resetConfirm: 'This clears progress, notes and marks. Continue?',
      chipTitle: 'Chapters mastered',
      foot: 'PyTorch Academy · Runs locally; your data stays in this browser',
      langTag: 'EN',
      notFound: 'Chapter not found'
    }
  };

  /* ---------------- storage ---------------- */
  var S = {
    get: function (k, d) {
      try { var v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); }
      catch (e) { return d; }
    },
    set: function (k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} },
    del: function (k) { try { localStorage.removeItem(k); } catch (e) {} }
  };
  var K = { lang: 'pta.lang', prog: 'pta.progress', note: 'pta.notes', ann: 'pta.annotations', star: 'pta.stars', last: 'pta.last' };

  var state = {
    lang: S.get(K.lang, 'zh'),
    chapter: null,
    undo: [],
    searchSel: 0,
    searchHits: []
  };
  if (state.lang !== 'en') state.lang = 'zh';

  function t(key) {
    var v = I18N[state.lang][key];
    return v === undefined ? key : v;
  }
  function L(obj) { return obj[state.lang] || obj.zh; }

  /* ---------------- helpers ---------------- */
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function stripHtml(html) {
    var d = document.createElement('div');
    d.innerHTML = html;
    return d.textContent.replace(/\s+/g, ' ').trim();
  }
  /* 搜索索引用：在块级边界补空格，避免表格单元格文字粘连 */
  function plainText(html) {
    var d = document.createElement('div');
    d.innerHTML = String(html).replace(/<\/(td|th|li|p|h2|h3|h4|div|tr|pre)>/g, ' ');
    return d.textContent.replace(/\s+/g, ' ').trim();
  }
  function txt(node) { return node.textContent.replace(/\s+/g, ' ').trim(); }
  function debounce(fn, ms) {
    var h;
    return function () {
      var a = arguments, self = this;
      clearTimeout(h);
      h = setTimeout(function () { fn.apply(self, a); }, ms);
    };
  }

  var els = {
    sidebar: $('#sidebar'), content: $('#content'), toc: $('#toc'),
    overlay: $('#searchOverlay'), sInput: $('#searchInput'), sResults: $('#searchResults'), sFoot: $('#searchFoot'),
    drawer: $('#noteDrawer'), noteArea: $('#noteArea'), noteStatus: $('#noteStatus'),
    noteTitle: $('#noteTitle'), noteSub: $('#noteSub'),
    toolbar: $('#selToolbar'), scrim: $('#scrim')
  };

  /* ---------------- python highlighter ---------------- */
  var KW = ['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue',
    'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'];
  var BUILTIN = ['print', 'range', 'len', 'int', 'float', 'str', 'list', 'dict', 'tuple', 'set', 'bool',
    'sum', 'min', 'max', 'abs', 'round', 'enumerate', 'zip', 'open', 'type', 'isinstance', 'super',
    'sorted', 'map', 'filter', 'any', 'all', 'input', 'format', 'repr', 'next', 'iter'];

  var RE_CODE = new RegExp([
    '(#[^\\n]*)',
    '("""[\\s\\S]*?"""|\'\'\'[\\s\\S]*?\'\'\'|"(?:[^"\\\\\\n]|\\\\.)*"|\'(?:[^\'\\\\\\n]|\\\\.)*\')',
    '(@[A-Za-z_][\\w.]*)',
    '\\b(\\d+\\.?\\d*(?:[eE][-+]?\\d+)?)\\b',
    '\\b(' + KW.join('|') + ')\\b',
    '\\b(' + BUILTIN.join('|') + ')\\b'
  ].join('|'), 'g');

  function highlightPython(src) {
    var out = '', last = 0, m;
    RE_CODE.lastIndex = 0;
    while ((m = RE_CODE.exec(src)) !== null) {
      out += esc(src.slice(last, m.index));
      var cls = m[1] ? 'tok-com' : m[2] ? 'tok-str' : m[3] ? 'tok-dec' : m[4] ? 'tok-num' : m[5] ? 'tok-kw' : 'tok-bi';
      out += '<span class="' + cls + '">' + esc(m[0]) + '</span>';
      last = m.index + m[0].length;
    }
    out += esc(src.slice(last));
    return out;
  }

  /* ---------------- sidebar ---------------- */
  function progressMap() { return S.get(K.prog, {}); }
  function starMap() { return S.get(K.star, {}); }

  function renderSidebar() {
    var done = progressMap(), stars = starMap();
    var total = CH.length;
    var doneCount = CH.filter(function (c) { return done[c.id]; }).length;
    var pct = total ? Math.round(doneCount / total * 100) : 0;

    var html = '<div class="sb-progress">' +
      '<div class="bar"><i style="width:' + pct + '%"></i></div>' +
      '<div class="meta"><span>' + doneCount + '/' + total + ' ' + t('completed') + '</span><span>' + pct + '%</span></div>' +
      '</div>';

    html += '<a class="sb-item' + (state.chapter === null ? ' active' : '') + '" href="#/home">' +
      '<span class="sb-num">★</span><span class="sb-title">' + t('home') + '</span></a>';

    html += '<a class="sb-item' + (state.chapter === 'practice' ? ' active' : '') + '" href="#/practice">' +
      '<span class="sb-num">?</span><span class="sb-title">' + t('practice') + '</span></a>';

    PARTS.forEach(function (p) {
      var items = CH.filter(function (c) { return c.part === p.id; });
      if (!items.length) return;
      html += '<div class="sb-part">' + esc(p[state.lang]) + '</div>';
      items.forEach(function (c) {
        html += '<a class="sb-item' + (state.chapter === c.id ? ' active' : '') + '" href="#/' + c.id + '">' +
          '<span class="sb-num">' + esc(c.num) + '</span>' +
          '<span class="sb-title">' + esc(L(c).title) + '</span>' +
          (stars[c.id] ? '<span class="sb-star">★</span>' : '') +
          (done[c.id] ? '<span class="sb-check">✓</span>' : '') +
          '</a>';
      });
    });
    els.sidebar.innerHTML = html;

    var active = $('.sb-item.active', els.sidebar);
    if (active && active.scrollIntoView) active.scrollIntoView({ block: 'nearest' });

    var chip = $('#progressChip');
    chip.textContent = doneCount + ' / ' + total;
    chip.title = t('chipTitle');
  }

  /* ---------------- home ---------------- */
  function renderHome() {
    state.chapter = null;
    var done = progressMap();
    var total = CH.length;
    var doneCount = CH.filter(function (c) { return done[c.id]; }).length;
    var pct = total ? Math.round(doneCount / total * 100) : 0;

    var notes = S.get(K.note, {});
    var noteChars = Object.keys(notes).reduce(function (s, k) { return s + (notes[k] || '').length; }, 0);
    var ann = S.get(K.ann, {});
    var markCount = Object.keys(ann).reduce(function (s, k) {
      var d = document.createElement('div'); d.innerHTML = ann[k];
      return s + d.querySelectorAll('.hl-yellow,.hl-green,.hl-red,.hl-underline').length;
    }, 0);

    var last = S.get(K.last, null);
    var nextCh = CH.filter(function (c) { return !done[c.id]; })[0] || CH[0];

    var html = '<section class="hero">' +
      '<div class="ch-kicker">' + (state.lang === 'zh' ? '零基础 · 系统课程 · 中英双语' : 'FROM ZERO · STRUCTURED · BILINGUAL') + '</div>' +
      '<h1>' + t('heroTitle1') + '<br><em>' + t('heroTitle2') + '</em></h1>' +
      '<p>' + t('heroSub') + '</p>' +
      '<div class="hero-actions">' +
      '<a class="ghost-btn primary" href="#/' + nextCh.id + '">' + t('startLearn') + '</a>' +
      (last ? '<a class="ghost-btn" href="#/' + last + '">' + t('continueLearn') + ' ←</a>' : '') +
      '<a class="ghost-btn" href="#/ch99">' + (state.lang === 'zh' ? 'API 速查表' : 'API cheatsheet') + '</a>' +
      '</div>' +
      '<div class="stat-row">' +
      '<div class="stat"><b>' + total + '</b><span>' + t('statChapters') + '</span></div>' +
      '<div class="stat"><b>' + doneCount + '<i style="font-size:14px;color:#98a2b3;font-style:normal"> / ' + pct + '%</i></b><span>' + t('statDone') + '</span></div>' +
      '<div class="stat"><b>' + noteChars + '</b><span>' + t('statNotes') + '</span></div>' +
      '<div class="stat"><b>' + markCount + '</b><span>' + t('statMarks') + '</span></div>' +
      '</div></section>';

    html += '<div class="sec-h"><h2>' + t('roadmap') + '</h2><span>' + doneCount + '/' + total + '</span></div>';
    html += '<div class="roadmap">';
    PARTS.forEach(function (p) {
      var items = CH.filter(function (c) { return c.part === p.id; });
      if (!items.length) return;
      html += '<div class="rm-part"><div class="rm-head"><b>' + esc(p[state.lang]) + '</b>' +
        '<span class="cnt">' + items.length + '</span></div><div class="rm-items">';
      items.forEach(function (c) {
        var d = L(c);
        html += '<a class="rm-item' + (done[c.id] ? ' done' : '') + '" href="#/' + c.id + '">' +
          '<span class="n">' + esc(c.num) + '</span>' +
          '<span class="t"><b>' + esc(d.title) + '</b><span>' + esc(d.sub || '') + '</span></span></a>';
      });
      html += '</div></div>';
    });
    html += '</div>';

    html += '<div class="sec-h"><h2>' + t('tools') + '</h2><a class="sec-more" href="#/practice">' +
      (state.lang === 'zh' ? '使用说明 →' : 'How to use →') + '</a></div>' +
      '<div class="tool-cards">' +
      '<a class="tool-card" href="#/practice"><h4>' + t('tool1t') + '</h4><p>' + t('tool1p') + '</p></a>' +
      '<a class="tool-card" href="#/practice"><h4>' + t('tool2t') + '</h4><p>' + t('tool2p') + '</p></a>' +
      '<a class="tool-card" href="#/practice"><h4>' + t('tool3t') + '</h4><p>' + t('tool3p') + '</p></a>' +
      '<a class="tool-card" href="#/ch99"><h4>' + t('tool4t') + '</h4><p>' + t('tool4p') + '</p></a>' +
      '</div>';

    html += '<div class="sec-h"><h2>' + t('tips') + '</h2></div>' +
      '<ol class="steps">' +
      '<li>' + t('tip1') + '</li><li>' + t('tip2') + '</li><li>' + t('tip3') + '</li>' +
      '</ol>';

    html += '<div style="margin:34px 0 10px"><button class="ghost-btn sm danger" id="btnResetAll">' + t('resetAll') + '</button></div>';

    els.content.innerHTML = html;
    els.toc.innerHTML = '';
    els.content.parentNode.parentNode.scrollTop = 0;
    window.scrollTo({ top: 0 });

    $('#btnResetAll').addEventListener('click', function () {
      if (!window.confirm(t('resetConfirm'))) return;
      [K.prog, K.note, K.ann, K.star, K.last].forEach(S.del);
      renderSidebar(); renderRoute();
    });
    renderSidebar();
  }

  /* ---------------- chapter ---------------- */
  function partLabel(pid) {
    var p = PARTS.filter(function (x) { return x.id === pid; })[0];
    return p ? p[state.lang] : '';
  }

  function renderChapter(id) {
    var idx = -1;
    for (var i = 0; i < CH.length; i++) { if (CH[i].id === id) { idx = i; break; } }
    if (idx < 0) { location.hash = '#/home'; return; }

    var ch = CH[idx];
    state.chapter = id;
    state.undo = [];
    S.set(K.last, id);

    var d = L(ch);
    var done = progressMap()[id];
    var starred = starMap()[id];
    var ann = S.get(K.ann, {});
    var key = id + '|' + state.lang;

    var prev = idx > 0 ? CH[idx - 1] : null;
    var next = idx < CH.length - 1 ? CH[idx + 1] : null;

    var html = '<div class="ch-head">' +
      '<div class="ch-kicker">' + esc(partLabel(ch.part)) + ' &nbsp;·&nbsp; ' + (state.lang === 'zh' ? '第 ' : 'Chapter ') + esc(ch.num) + '</div>' +
      '<h1>' + esc(d.title) + '</h1>' +
      '<p class="ch-sub">' + esc(d.sub || '') + '</p>' +
      '<div class="ch-actions">' +
      '<button class="ghost-btn" data-act="complete">' + (done ? t('done') : t('markDone')) + '</button>' +
      '<button class="ghost-btn" data-act="star">' + (starred ? '★ ' : '☆ ') + t('important') + '</button>' +
      '<button class="ghost-btn" data-act="notes">' + t('notes') + '</button>' +
      '<div class="right">' +
      '<button class="ghost-btn primary" data-act="notebook">' + t('notebook') + '</button>' +
      '</div></div></div>' +
      '<div id="chapterBody" class="prose">' + (ann[key] || d.html) + '</div>' +
      '<div class="ch-footer">' +
      (prev ? '<a class="nav-card" href="#/' + prev.id + '"><span>← ' + t('prev') + '</span><b>' + esc(L(prev).title) + '</b></a>' : '<span style="flex:1"></span>') +
      (next ? '<a class="nav-card next" href="#/' + next.id + '"><span>' + t('next') + ' →</span><b>' + esc(L(next).title) + '</b></a>' : '') +
      '</div>';

    els.content.innerHTML = html;
    document.title = d.title + ' · PyTorch Academy';

    enhanceChapter(ch);
    buildToc();
    renderSidebar();
    updateNoteUi();

    var visited = progressMap();
    if (!visited[id]) { /* 只记录已访问，不算完成 */ }
    window.scrollTo({ top: 0 });
    closeDrawer();
  }

  function enhanceChapter(ch) {
    var body = $('#chapterBody');

    /* 代码块包装 + 高亮 + 复制 */
    $$('pre', body).forEach(function (pre) {
      var lang = pre.getAttribute('data-lang') || 'python';
      var codeEl = pre.querySelector('code');
      var raw = codeEl ? codeEl.textContent : pre.textContent;
      if (codeEl && lang === 'python') codeEl.innerHTML = highlightPython(raw);
      var wrap = document.createElement('div');
      wrap.className = 'codeblock';
      var head = document.createElement('div');
      head.className = 'cb-head';
      head.innerHTML = '<span class="dots"><i></i><i></i><i></i></span>' +
        '<span class="cb-lang">' + esc(lang) + '</span>' +
        '<button class="cb-copy">' + t('copy') + '</button>';
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(head);
      wrap.appendChild(pre);
      pre.style.margin = '0';
      head.querySelector('.cb-copy').addEventListener('click', function () {
        var btn = this;
        copyText(raw, function (ok) { btn.textContent = ok ? t('copied') : t('copy'); });
      });
    });

    /* 测验 */
    $$('.quiz', body).forEach(function (quiz) {
      var ans = parseInt(quiz.getAttribute('data-answer'), 10) || 0;
      quiz.setAttribute('data-ans', ans);
      $$('ol.opts li', quiz).forEach(function (li, i) {
        li.setAttribute('data-i', i);
      });
    });

    /* 交互工具 */
    $$('[data-widget]', body).forEach(function (w) {
      var kind = w.getAttribute('data-widget');
      if (kind === 'shape') buildShapeWidget(w);
      if (kind === 'conv') buildConvWidget(w);
    });

    /* 头部动作 */
    $$('.ch-actions [data-act]', els.content).forEach(function (btn) {
      btn.addEventListener('click', function () {
        var act = btn.getAttribute('data-act');
        if (act === 'complete') toggleComplete(ch.id, btn);
        if (act === 'star') toggleStar(ch.id, btn);
        if (act === 'notes') openDrawer();
        if (act === 'notebook') exportNotebook(ch);
      });
    });
  }

  function toggleComplete(id, btn) {
    var p = progressMap();
    if (p[id]) delete p[id]; else p[id] = Date.now();
    S.set(K.prog, p);
    renderSidebar();
    if (btn) {
      btn.textContent = p[id] ? t('done') : t('markDone');
      btn.style.color = p[id] ? 'var(--green)' : '';
      btn.style.borderColor = p[id] ? '#86efac' : '';
    }
  }

  function toggleStar(id, btn) {
    var s = starMap();
    if (s[id]) delete s[id]; else s[id] = 1;
    S.set(K.star, s);
    renderSidebar();
    if (btn) btn.textContent = (s[id] ? '★ ' : '☆ ') + t('important');
    if (s[id] && window.__refreshTocStars) window.__refreshTocStars();
  }

  /* ---------------- TOC ---------------- */
  function buildToc() {
    var body = $('#chapterBody');
    if (!body) { els.toc.innerHTML = ''; return; }
    var heads = $$('h2, h3', body);
    if (!heads.length) { els.toc.innerHTML = ''; return; }

    var html = '<div class="toc-title">' + t('tocTitle') + '</div>';
    heads.forEach(function (h, i) {
      if (!h.id) h.id = 'sec-' + i;
      html += '<a href="#' + h.id + '" class="' + (h.tagName === 'H3' ? 'h3' : '') + '" data-toc="' + h.id + '">' + esc(h.textContent) + '</a>';
    });
    els.toc.innerHTML = html;

    $$('a[data-toc]', els.toc).forEach(function (a) {
      a.addEventListener('click', function (e) {
        e.preventDefault();
        var el = document.getElementById(a.getAttribute('data-toc'));
        if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 72, behavior: 'smooth' });
      });
    });

    var spy = function () {
      var cur = null;
      heads.forEach(function (h) {
        if (h.getBoundingClientRect().top < 120) cur = h.id;
      });
      $$('a[data-toc]', els.toc).forEach(function (a) {
        a.classList.toggle('on', a.getAttribute('data-toc') === cur);
      });
    };
    window.removeEventListener('scroll', window.__tocSpy || function () {});
    window.__tocSpy = debounce(spy, 90);
    window.addEventListener('scroll', window.__tocSpy, { passive: true });
    spy();
  }

  /* ---------------- quizzes ---------------- */
  function onQuizClick(e) {
    var li = e.target.closest ? e.target.closest('.quiz ol.opts li') : null;
    if (!li) return;
    var quiz = li.closest('.quiz');
    if (!quiz || quiz.classList.contains('answered')) return;
    var ans = parseInt(quiz.getAttribute('data-ans'), 10);
    var i = parseInt(li.getAttribute('data-i'), 10);
    quiz.classList.add('answered');
    $$('li', quiz.querySelector('ol.opts')).forEach(function (x, j) {
      if (j === ans) x.classList.add('correct');
    });
    if (i !== ans) li.classList.add('wrong');
    saveAnn();
  }

  /* ---------------- shape widget ---------------- */
  function buildShapeWidget(host) {
    var zh = state.lang === 'zh';
    host.innerHTML =
      '<h4>' + (zh ? '广播形状计算器' : 'Broadcast shape calculator') + '</h4>' +
      '<p class="hint">' + (zh ? '输入两个张量的形状，例如 3,1 与 1,4（从最后一维开始右对齐比较）' : 'Enter two shapes, e.g. 3,1 and 1,4 (compare right-aligned from the last dim)') + '</p>' +
      '<div class="inputs">' +
      '<input id="wa" value="3,1" placeholder="3,1">' +
      '<span style="color:#98a2b3">×</span>' +
      '<input id="wb" value="1,4" placeholder="1,4">' +
      '<button class="go">' + (zh ? '计算' : 'Compute') + '</button>' +
      '</div>' +
      '<div class="out"></div>';

    var run = function () {
      var a = parseShape(host.querySelector('#wa').value);
      var b = parseShape(host.querySelector('#wb').value);
      var out = host.querySelector('.out');
      if (!a || !b) {
        out.innerHTML = '<span style="color:#dc2626">' + (zh ? '请输入形如 3,1 的数字' : 'Use numbers like 3,1') + '</span>';
        return;
      }
      var res = broadcast(a, b);
      if (!res.ok) {
        out.innerHTML = '<div class="res">' + (zh ? '报错：形状不兼容' : 'Error: shapes not broadcastable') + '</div>' +
          '<div class="why">' + res.why + '</div>';
        return;
      }
      out.innerHTML = '<div class="res">(' + a.join(', ') + ') ⊕ (' + b.join(', ') + ') → torch.Size([' + res.shape.join(', ') + '])</div>' +
        '<div class="why">' + res.why + '</div>';
    };
    host.querySelector('button.go').addEventListener('click', run);
    host.querySelectorAll('input').forEach(function (i) {
      i.addEventListener('keydown', function (e) { if (e.key === 'Enter') run(); });
    });
    run();
  }

  function buildConvWidget(host) {
    var zh = state.lang === 'zh';
    host.innerHTML =
      '<h4>' + (zh ? '卷积输出尺寸与参数量计算器' : 'Conv output size & parameter calculator') + '</h4>' +
      '<p class="hint">' + (zh ? '六个数依次是：输入边长 H · kernel K · stride S · padding P · 输入通道 · 输出通道（鼠标悬停有提示）' : 'Six values in order: input H · kernel K · stride S · padding P · in_ch · out_ch (hover for tooltips)') + '</p>' +
      '<div class="inputs">' +
      '<input id="cH" value="28" title="H">' +
      '<input id="cK" value="5" title="K">' +
      '<input id="cS" value="1" title="S">' +
      '<input id="cP" value="0" title="P">' +
      '<input id="cI" value="1" title="in_ch">' +
      '<input id="cO" value="6" title="out_ch">' +
      '<button class="go">' + (zh ? '计算' : 'Compute') + '</button>' +
      '</div>' +
      '<div class="out"></div>';

    var run = function () {
      var H = parseInt(host.querySelector('#cH').value, 10);
      var K = parseInt(host.querySelector('#cK').value, 10);
      var S = parseInt(host.querySelector('#cS').value, 10);
      var P = parseInt(host.querySelector('#cP').value, 10);
      var CI = parseInt(host.querySelector('#cI').value, 10);
      var CO = parseInt(host.querySelector('#cO').value, 10);
      var out = host.querySelector('.out');
      var bad = [H, K, S, P, CI, CO].some(function (v) { return isNaN(v) || v < 0; }) || S < 1 || K < 1;
      if (bad) {
        out.innerHTML = '<span style="color:#dc2626">' + (zh ? '请填入非负整数，stride 与 kernel 至少为 1' : 'Use non-negative integers; stride and kernel must be ≥ 1') + '</span>';
        return;
      }
      var size = Math.floor((H + 2 * P - K) / S) + 1;
      if (size <= 0) {
        out.innerHTML = '<div class="res">' + (zh ? '非法配置：输出尺寸 ≤ 0' : 'Invalid: output size ≤ 0') + '</div>' +
          '<div class="why">' + (zh ? 'padding 不足或 kernel 大于输入。' : 'Padding too small or kernel larger than input.') + '</div>';
        return;
      }
      var params = CI * CO * K * K + CO;
      var mem = (size * size * CO * 4 / 1024 / 1024);
      out.innerHTML = '<div class="res">H_out = ⌊(' + H + ' + 2×' + P + ' − ' + K + ') / ' + S + '⌋ + 1 = ' + size + '</div>' +
        '<div class="why">' +
        (zh ? '输出形状：(batch, ' + CO + ', ' + size + ', ' + size + ')<br>' : 'Output shape: (batch, ' + CO + ', ' + size + ', ' + size + ')<br>') +
        (zh ? '参数量 = ' + CI + '×' + CO + '×' + K + '×' + K + ' + ' + CO + ' = <b>' + params.toLocaleString() + '</b>' : 'Params = ' + CI + '×' + CO + '×' + K + '×' + K + ' + ' + CO + ' = <b>' + params.toLocaleString() + '</b>') +
        ' &nbsp;·&nbsp; ' + (zh ? '单个样本输出约占 ' : 'per-sample output ≈ ') + mem.toFixed(2) + ' MB (float32)' +
        '</div>';
    };
    host.querySelector('button.go').addEventListener('click', run);
    host.querySelectorAll('input').forEach(function (i) {
      i.addEventListener('keydown', function (e) { if (e.key === 'Enter') run(); });
    });
    run();
  }

  function parseShape(str) {    if (!str) return null;
    var parts = String(str).replace(/[()\[\]\s]/g, '').split(',').filter(Boolean);
    var nums = [];
    for (var i = 0; i < parts.length; i++) {
      var n = parseInt(parts[i], 10);
      if (isNaN(n) || n < 0) return null;
      nums.push(n);
    }
    return nums.length ? nums : null;
  }

  function broadcast(a, b) {
    var zh = state.lang === 'zh';
    var n = Math.max(a.length, b.length);
    var ra = [], rb = [], shape = [], steps = [];
    for (var i = 0; i < n; i++) {
      ra.unshift(a[a.length - 1 - i] === undefined ? 1 : a[a.length - 1 - i]);
      rb.unshift(b[b.length - 1 - i] === undefined ? 1 : b[b.length - 1 - i]);
    }
    for (var j = n - 1; j >= 0; j--) {
      var x = ra[j], y = rb[j];
      if (x === y) { shape.unshift(x); steps.push('<code>' + x + '</code> vs <code>' + y + '</code> → ' + x + (zh ? '（相同）' : ' (equal)')); }
      else if (x === 1) { shape.unshift(y); steps.push('<code>1</code> vs <code>' + y + '</code> → ' + y + (zh ? '（1 被复制扩展）' : ' (1 stretched)')); }
      else if (y === 1) { shape.unshift(x); steps.push('<code>' + x + '</code> vs <code>1</code> → ' + x + (zh ? '（1 被复制扩展）' : ' (1 stretched)')); }
      else {
        return {
          ok: false,
          why: (zh ? '在第 ' + (n - j) + ' 维（从右数）上 ' : 'At dim ' + (n - j) + ' from the right, ') +
            x + ' ≠ ' + y + (zh ? '，且都不是 1，无法广播。需要手动 unsqueeze / expand。' : ', neither is 1, so no broadcast. Use unsqueeze/expand manually.')
        };
      }
    }
    return {
      ok: true, shape: shape,
      why: (zh ? '逐维结果：' : 'Per-dim result: ') + steps.reverse().join('；')
    };
  }

  /* ---------------- annotations ---------------- */
  function saveAnn() {
    if (!state.chapter) return;
    var body = $('#chapterBody');
    if (!body) return;
    var all = S.get(K.ann, {});
    all[state.chapter + '|' + state.lang] = body.innerHTML;
    S.set(K.ann, all);
  }

  function onSelectionEnd() {
    var body = $('#chapterBody');
    if (!body) return;
    var sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) { hideToolbar(); return; }
    var range = sel.getRangeAt(0);
    if (!body.contains(range.commonAncestorContainer)) { hideToolbar(); return; }
    var rect = range.getBoundingClientRect();
    if (!rect || (!rect.width && !rect.height)) { hideToolbar(); return; }

    var tb = els.toolbar;
    tb.hidden = false;
    var w = tb.offsetWidth || 190;
    var left = Math.min(Math.max(8, rect.left + rect.width / 2 - w / 2), window.innerWidth - w - 8);
    var top = rect.top - 42;
    if (top < 60) top = rect.bottom + 8;
    tb.style.left = left + 'px';
    tb.style.top = top + 'px';
  }

  function hideToolbar() { els.toolbar.hidden = true; }

  function applyMark(cls) {
    var sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return;
    var range = sel.getRangeAt(0);
    if (range.collapsed) return;
    var body = $('#chapterBody');
    if (!body || !body.contains(range.commonAncestorContainer)) return;

    if (cls === 'clear') {
      var node = range.startContainer;
      var el = node.nodeType === 1 ? node : node.parentNode;
      var marked = el.closest ? el.closest('[class*="hl-"]') : null;
      if (marked) {
        var parent = marked.parentNode;
        while (marked.firstChild) parent.insertBefore(marked.firstChild, marked);
        parent.removeChild(marked);
        saveAnn();
      }
      sel.removeAllRanges();
      hideToolbar();
      return;
    }

    var span = document.createElement('span');
    span.className = cls;
    try {
      range.surroundContents(span);
    } catch (err) {
      var frag = range.extractContents();
      span.appendChild(frag);
      range.insertNode(span);
    }
    sel.removeAllRanges();
    hideToolbar();
    saveAnn();
  }

  /* ---------------- drawer / notes ---------------- */
  function openDrawer() {
    els.drawer.classList.add('open');
    els.scrim.hidden = false;
    updateNoteUi();
    setTimeout(function () { els.noteArea.focus(); }, 240);
  }
  function closeDrawer() {
    els.drawer.classList.remove('open');
    if (els.overlay.hidden) els.scrim.hidden = true;
  }
  function updateNoteUi() {
    var id = state.chapter;
    var ch = CH.filter(function (c) { return c.id === id; })[0];
    els.noteTitle.textContent = ch ? (L(ch).title) : t('notesTitle');
    els.noteSub.textContent = t('notesSub');
    var notes = S.get(K.note, {});
    els.noteArea.value = (id && notes[id]) || '';
    els.noteArea.placeholder = t('notesPh');
    els.noteStatus.textContent = t('notesIdle');
    els.noteArea.disabled = !id;
  }

  var saveNote = debounce(function () {
    if (!state.chapter) return;
    var notes = S.get(K.note, {});
    var v = els.noteArea.value;
    if (v.trim()) notes[state.chapter] = v; else delete notes[state.chapter];
    S.set(K.note, notes);
    var d = new Date();
    els.noteStatus.textContent = t('notesSaved') + ' ' + ('0' + d.getHours()).slice(-2) + ':' + ('0' + d.getMinutes()).slice(-2);
  }, 420);

  /* ---------------- search ---------------- */
  var INDEX = null;
  function buildIndex() {
    if (INDEX) return INDEX;
    INDEX = (function () {
      var list = CH.slice();
      var pg = window.PYTORCH_COURSE && window.PYTORCH_COURSE.practice;
      if (pg) list = list.concat([{ id: 'practice', num: '?', zh: pg.zh, en: pg.en }]);
      return list;
    })().map(function (c) {
      return {
        id: c.id, num: c.num,
        zhTitle: c.zh.title, enTitle: c.en.title,
        zhText: plainText(c.zh.html), enText: plainText(c.en.html)
      };
    });
    return INDEX;
  }

  function runSearch(q) {
    var query = q.trim().toLowerCase();
    var box = els.sResults;
    if (!query) {
      box.innerHTML = '<div class="empty-state">' + t('searchHint') + '</div>';
      els.sFoot.textContent = t('searchNav');
      state.searchHits = [];
      return;
    }
    var terms = query.split(/\s+/).filter(Boolean);
    var hits = [];
    buildIndex().forEach(function (rec) {
      ['zh', 'en'].forEach(function (lang) {
        var text = lang === 'zh' ? rec.zhText : rec.enText;
        var title = lang === 'zh' ? rec.zhTitle : rec.enTitle;
        var lower = text.toLowerCase();
        var all = terms.every(function (t2) { return lower.indexOf(t2) >= 0 || title.toLowerCase().indexOf(t2) >= 0; });
        if (!all) return;
        var pos = lower.indexOf(terms[0]);
        if (pos < 0) pos = 0;
        var start = Math.max(0, pos - 55);
        var snip = text.slice(start, start + 165);
        if (start > 0) snip = '…' + snip;
        if (start + 165 < text.length) snip += '…';
        var score = (title.toLowerCase().indexOf(terms[0]) >= 0 ? -100 : 0) + pos;
        hits.push({ id: rec.id, num: rec.num, title: title, lang: lang, snip: snip, score: score });
      });
    });
    hits.sort(function (a, b) { return a.score - b.score; });
    var seen = {};
    hits = hits.filter(function (h) {
      if (seen[h.id]) return false;
      seen[h.id] = 1; return true;
    }).slice(0, 24);

    state.searchHits = hits;
    state.searchSel = 0;

    if (!hits.length) {
      box.innerHTML = '<div class="empty-state">' + t('searchEmpty') + '</div>';
      els.sFoot.textContent = '';
      return;
    }

    box.innerHTML = hits.map(function (h, i) {
      var s = esc(h.snip);
      terms.forEach(function (term) {
        var re = new RegExp('(' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
        s = s.replace(re, '<mark>$1</mark>');
      });
      return '<a class="sr-item' + (i === 0 ? ' on' : '') + '" href="#/' + h.id + '" data-h="' + i + '">' +
        '<div class="top"><b>' + esc(h.num) + '. ' + esc(h.title) + '</b>' +
        '<span class="tag">' + (h.lang === 'zh' ? '中文' : 'EN') + '</span></div>' +
        '<div class="snip">' + s + '</div></a>';
    }).join('');
    els.sFoot.innerHTML = '<span>' + t('searchCount')(hits.length) + '</span><span>' + t('searchNav') + '</span>';
  }

  function openSearch() {
    els.overlay.hidden = false;
    els.sInput.value = '';
    runSearch('');
    setTimeout(function () { els.sInput.focus(); }, 30);
  }
  function closeSearch() { els.overlay.hidden = true; }

  function searchMove(delta) {
    if (!state.searchHits.length) return;
    state.searchSel = (state.searchSel + delta + state.searchHits.length) % state.searchHits.length;
    $$('.sr-item', els.sResults).forEach(function (a, i) {
      a.classList.toggle('on', i === state.searchSel);
      if (i === state.searchSel && a.scrollIntoView) a.scrollIntoView({ block: 'nearest' });
    });
  }

  /* ---------------- notebook export ---------------- */
  function mdCell(src) { return { cell_type: 'markdown', metadata: {}, source: splitLines(src) }; }
  function codeCell(src) { return { cell_type: 'code', execution_count: null, metadata: {}, outputs: [], source: splitLines(src) }; }
  function splitLines(s) {
    var lines = String(s).replace(/\r/g, '').split('\n');
    var out = [];
    for (var i = 0; i < lines.length; i++) out.push(i === lines.length - 1 ? lines[i] : lines[i] + '\n');
    if (out.length > 1 && out[out.length - 1] === '') out.pop();
    return out;
  }

  function exportNotebook(ch) {
    var d = L(ch);
    var zh = state.lang === 'zh';
    var tmp = document.createElement('div');
    tmp.innerHTML = d.html;

    var cells = [];
    cells.push(mdCell('# ' + d.title + '\n\n' + (d.sub || '') + '\n\n' +
      (zh ? '> 配套练习 · PyTorch 学习站 · 第 ' + ch.num + ' 章\n> 逐格运行（Shift+Enter），改数字、改形状，看输出怎么变。' :
        '> Practice notebook · PyTorch Academy · Chapter ' + ch.num + '\n> Run cells with Shift+Enter, then change numbers and shapes to see what happens.')));

    cells.push(codeCell(zh ? '# 首次使用请先安装（已装可跳过）\n# !pip install torch torchvision --quiet\n\nimport torch\nprint(torch.__version__)' :
      '# Install once if needed (otherwise skip)\n# !pip install torch torchvision --quiet\n\nimport torch\nprint(torch.__version__)'));

    var md = [];
    function flush() {
      var s = md.join('\n\n').trim();
      if (s) cells.push(mdCell(s));
      md = [];
    }

    Array.prototype.forEach.call(tmp.children, function (node) {
      var tag = node.tagName.toLowerCase();
      if (tag === 'pre') {
        var code = node.querySelector('code');
        if (code) { flush(); cells.push(codeCell(code.textContent.replace(/\s+$/, ''))); }
      } else if (tag === 'h2' || tag === 'h3' || tag === 'h4') {
        md.push(new Array(tag === 'h2' ? 3 : tag === 'h3' ? 4 : 5).join('#') + ' ' + txt(node));
      } else if (tag === 'p') {
        var p = txt(node); if (p) md.push(p);
      } else if (tag === 'ul' || tag === 'ol') {
        var lines = Array.prototype.map.call(node.children, function (li, i) {
          return (tag === 'ul' ? '- ' : (i + 1) + '. ') + txt(li);
        });
        if (lines.length) md.push(lines.join('\n'));
      } else if (tag === 'div' && node.classList.contains('callout')) {
        var tagEl = node.querySelector('.tag');
        var label = tagEl ? txt(tagEl) : 'Note';
        var bodytxt = Array.prototype.filter.call(node.children, function (c) { return c !== tagEl; })
          .map(function (c) { return txt(c); }).filter(Boolean).join('\n\n');
        if (bodytxt) md.push('> **' + label + '** ' + bodytxt);
      } else if (tag === 'div' && node.classList.contains('keypoints')) {
        var ks = Array.prototype.map.call(node.querySelectorAll('li'), function (li) { return '- ' + txt(li); });
        if (ks.length) md.push('**' + (zh ? '本章重点' : 'Key points') + '**\n' + ks.join('\n'));
      } else if (tag === 'div' && node.classList.contains('exercise')) {
        var es = Array.prototype.map.call(node.querySelectorAll('li'), function (li) { return '- ' + txt(li); });
        if (es.length) md.push('**' + (zh ? '动手练习' : 'Exercises') + '**\n' + es.join('\n'));
      } else if (tag === 'table') {
        var rows = Array.prototype.map.call(node.querySelectorAll('tr'), function (tr) {
          return '| ' + Array.prototype.map.call(tr.children, function (c) { return txt(c); }).join(' | ') + ' |';
        });
        if (rows.length) {
          var head = rows[0];
          var sep = '| ' + Array.prototype.map.call(node.querySelectorAll('tr')[0].children, function () { return '---'; }).join(' | ') + ' |';
          md.push([head, sep].concat(rows.slice(1)).join('\n'));
        }
      }
    });
    flush();

    var nb = {
      cells: cells,
      metadata: {
        kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
        language_info: { name: 'python', version: '3.10' },
        title: d.title
      },
      nbformat: 4,
      nbformat_minor: 5
    };

    var name = 'pytorch-' + ch.num + '-' + (ch.file || ch.id) + '.ipynb';
    download(name, JSON.stringify(nb, null, 1), 'application/x-ipynb+json');
  }

  function download(name, text, mime) {
    var blob = new Blob([text], { type: (mime || 'text/plain') + ';charset=utf-8' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 60000);
  }

  function copyText(text, cb) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { cb(true); }, function () { cb(fallback()); });
    } else { cb(fallback()); }
    function fallback() {
      try {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select();
        var ok = document.execCommand('copy');
        ta.remove(); return ok;
      } catch (e) { return false; }
    }
  }

  /* ---------------- routing ---------------- */
  function renderRoute() {
    var hash = location.hash.replace(/^#\/?/, '');
    if (!hash || hash === 'home') renderHome();
    else if (hash === 'practice') renderPractice();
    else renderChapter(hash);
  }

  /* ---------------- practice guide ---------------- */
  function renderPractice() {
    var pg = window.PYTORCH_COURSE && window.PYTORCH_COURSE.practice;
    if (!pg) { location.hash = '#/home'; return; }
    var d = L(pg);

    state.chapter = 'practice';
    state.undo = [];

    var html = '<div class="ch-head">' +
      '<div class="ch-kicker">' + esc(d.kicker || '') + '</div>' +
      '<h1>' + esc(d.title) + '</h1>' +
      '<p class="ch-sub">' + esc(d.sub || '') + '</p>' +
      '<div class="ch-actions">' +
      '<button class="ghost-btn" data-act="notes">' + t('notes') + '</button>' +
      '<div class="right">' +
      '<a class="ghost-btn primary" href="#/home">' + (state.lang === 'zh' ? '返回目录' : 'Back to syllabus') + '</a>' +
      '</div></div></div>' +
      '<div id="chapterBody" class="prose">' + d.html + '</div>' +
      '<div class="ch-footer"><span style="flex:1"></span>' +
      '<a class="nav-card next" href="#/' + CH[0].id + '"><span>' + t('startLearn') + ' →</span><b>' + esc(L(CH[0]).title) + '</b></a>' +
      '</div>';

    els.content.innerHTML = html;
    document.title = d.title + ' · PyTorch Academy';

    enhanceChapter({ id: 'practice', num: '', file: 'practice' });
    buildToc();
    renderSidebar();
    updateNoteUi();
    window.scrollTo({ top: 0 });
    closeDrawer();
  }

  /* ---------------- UI text refresh ---------------- */
  function applyLangUi() {
    document.documentElement.lang = state.lang === 'zh' ? 'zh-CN' : 'en';
    $('#brandSub').textContent = t('brandSub');
    $('#searchLabel').textContent = t('searchLabel');
    els.sInput.placeholder = t('searchPlaceholder');
    $('#notesLabel').textContent = t('notes');
    $('#footNote').textContent = t('foot');
    $('#btnExportNotes').textContent = t('exportNotes');
    $('#btnClearNote').textContent = t('clearNote');
    els.noteTitle.textContent = t('notesTitle');
    els.noteSub.textContent = t('notesSub');
    $$('#langSeg button').forEach(function (b) {
      b.classList.toggle('on', b.getAttribute('data-lang') === state.lang);
    });
    $$('#selToolbar [data-mark]').forEach(function (b) {
      var k = b.getAttribute('data-mark');
      var map = {
        'hl-yellow': state.lang === 'zh' ? '高亮' : 'Highlight',
        'hl-green': state.lang === 'zh' ? '绿色标记' : 'Green mark',
        'hl-red': state.lang === 'zh' ? '重点标记' : 'Key point',
        'hl-underline': state.lang === 'zh' ? '下划线' : 'Underline',
        'clear': state.lang === 'zh' ? '取消' : 'Remove'
      };
      b.title = map[k] || '';
    });
  }

  /* ---------------- events ---------------- */
  function bind() {
    window.addEventListener('hashchange', function () {
      renderRoute();
      closeSearch();
      closeDrawer();
      if (window.innerWidth <= 900) els.sidebar.classList.remove('open');
    });

    /* language */
    $('#langSeg').addEventListener('click', function (e) {
      var b = e.target.closest('button');
      if (!b) return;
      var lang = b.getAttribute('data-lang');
      if (lang === state.lang) return;
      state.lang = lang;
      S.set(K.lang, lang);
      applyLangUi();
      renderRoute();
      renderSidebar();
    });

    /* search */
    $('#btnSearch').addEventListener('click', openSearch);
    els.sInput.addEventListener('input', debounce(function () { runSearch(els.sInput.value); }, 90));
    els.sInput.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') { closeSearch(); }
      if (e.key === 'ArrowDown') { e.preventDefault(); searchMove(1); }
      if (e.key === 'ArrowUp') { e.preventDefault(); searchMove(-1); }
      if (e.key === 'Enter') {
        var hit = state.searchHits[state.searchSel];
        if (hit) { location.hash = '#/' + hit.id; closeSearch(); }
      }
    });
    els.overlay.addEventListener('click', function (e) { if (e.target === els.overlay) closeSearch(); });
    els.sResults.addEventListener('click', function (e) {
      var a = e.target.closest('.sr-item');
      if (a) closeSearch();
    });

    /* notes */
    $('#btnNotes').addEventListener('click', function () {
      if (!state.chapter) { location.hash = '#/' + CH[0].id; return; }
      if (els.drawer.classList.contains('open')) closeDrawer(); else openDrawer();
    });
    $('#btnCloseNote').addEventListener('click', closeDrawer);
    els.scrim.addEventListener('click', function () {
      closeDrawer();
      els.sidebar.classList.remove('open');
      els.scrim.hidden = true;
    });
    els.noteArea.addEventListener('input', saveNote);
    $('#btnClearNote').addEventListener('click', function () {
      if (!state.chapter || !window.confirm(state.lang === 'zh' ? '清空本章笔记？' : 'Clear this chapter note?')) return;
      var notes = S.get(K.note, {});
      delete notes[state.chapter];
      S.set(K.note, notes);
      els.noteArea.value = '';
    });
    $('#btnExportNotes').addEventListener('click', function () {
      var notes = S.get(K.note, {});
      var keys = Object.keys(notes).filter(function (k) { return notes[k] && notes[k].trim(); });
      if (!keys.length) {
        window.alert(state.lang === 'zh' ? '还没有笔记。' : 'No notes yet.');
        return;
      }
      var md = '# PyTorch 学习笔记 / Study notes\n\n';
      keys.forEach(function (id) {
        var c = CH.filter(function (x) { return x.id === id; })[0];
        md += '## ' + (c ? L(c).title : id) + '\n\n' + notes[id] + '\n\n---\n\n';
      });
      download('pytorch-notes.md', md, 'text/markdown');
    });

    /* selection toolbar */
    document.addEventListener('mouseup', function (e) {
      if (els.toolbar.contains(e.target)) return;
      setTimeout(onSelectionEnd, 10);
    });
    document.addEventListener('scroll', hideToolbar, { passive: true });
    els.toolbar.addEventListener('mousedown', function (e) { e.preventDefault(); });
    els.toolbar.addEventListener('click', function (e) {
      var b = e.target.closest('button');
      if (b) applyMark(b.getAttribute('data-mark'));
    });

    /* quiz delegation */
    els.content.addEventListener('click', onQuizClick);

    /* keyboard */
    document.addEventListener('keydown', function (e) {
      var meta = e.ctrlKey || e.metaKey;
      if (meta && (e.key === 'k' || e.key === 'K')) { e.preventDefault(); openSearch(); }
      if (e.key === 'Escape') { closeSearch(); hideToolbar(); if (els.drawer.classList.contains('open')) closeDrawer(); }
      if (meta && (e.key === 'e' || e.key === 'E')) { e.preventDefault(); if (state.chapter) openDrawer(); }
    });

    /* mobile menu */
    $('#btnMenu').addEventListener('click', function () {
      var open = els.sidebar.classList.toggle('open');
      els.scrim.hidden = !open;
    });

    /* 离开页面前保存笔记 */
    window.addEventListener('beforeunload', function () {
      if (!state.chapter) return;
      var notes = S.get(K.note, {});
      var v = els.noteArea.value;
      if (v.trim()) notes[state.chapter] = v;
      S.set(K.note, notes);
    });
  }

  /* ---------------- boot ---------------- */
  function boot() {
    if (!CH.length) {
      els.content.innerHTML = '<div class="empty-state">content failed to load / 内容加载失败。请确认 content/ 目录与 index.html 在同一层级。</div>';
      return;
    }
    applyLangUi();
    bind();
    renderRoute();
    renderSidebar();
    setInterval(function () { renderSidebar(); }, 60000 * 5);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
