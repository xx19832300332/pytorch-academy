/* PyTorch Academy — service worker
   目标：整个站（含 18 章内容脚本）在首次访问后可离线使用。

   策略：
   - 预缓存：安装时抓 index.html，把里面引用到的本站资源全部缓存（新增章节后自动带上，不用改这里）
   - 导航请求（打开页面）：网络优先，失败回退缓存，再失败回退缓存的 index.html
   - 其他本站资源：缓存优先，命中即返回；未命中则网络请求并顺手写入缓存

   注意：service worker 只在 http/https 下生效，file:// 打开时不会注册。
   发版后旧的 VERSION 会让新 worker 重新预缓存，activate 时清掉旧缓存。 */

var VERSION = 'v3-2026-10-18';
var CACHE = 'pta-' + VERSION;
var EXTRA = [
  './',
  './index.html',
  './manifest.webmanifest',
  './assets/icon-192.png',
  './assets/icon-512.png',
  './assets/icon.ico'
];

self.addEventListener('install', function (event) {
  event.waitUntil((async function () {
    var cache = await caches.open(CACHE);

    /* 1) 基础资源，失败不阻塞 */
    await Promise.all(EXTRA.map(function (u) {
      return cache.add(new Request(u, { cache: 'reload' })).catch(function () { });
    }));

    /* 2) 从 index.html 里提取本站的脚本 / 样式 / 图标，一并缓存 */
    try {
      var res = await fetch('./index.html', { cache: 'reload' });
      var html = await res.text();
      var urls = {};
      var re = /(?:src|href)\s*=\s*["']([^"']+)["']/g;
      var m;
      while ((m = re.exec(html)) !== null) {
        var u = m[1];
        if (/^(https?:|data:|mailto:|#|\/\/)/.test(u)) continue;
        urls[u.replace(/^\.\//, './')] = 1;
      }
      await Promise.all(Object.keys(urls).map(function (u) {
        return cache.add(new Request(u, { cache: 'reload' })).catch(function () { });
      }));
    } catch (e) { /* 网络异常时退化为部分缓存，不影响安装 */ }

    self.skipWaiting();
  })());
});

self.addEventListener('activate', function (event) {
  event.waitUntil((async function () {
    var keys = await caches.keys();
    await Promise.all(keys.map(function (k) {
      return k === CACHE ? null : caches.delete(k);
    }));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', function (event) {
  var req = event.request;
  if (req.method !== 'GET') return;

  var url = new URL(req.url);
  if (url.origin !== self.location.origin) return;   /* 跨域一律不插手 */

  /* 页面导航：网络优先，断网走缓存 */
  if (req.mode === 'navigate' || (req.headers.get('accept') || '').indexOf('text/html') >= 0) {
    event.respondWith((async function () {
      try {
        var fresh = await fetch(req);
        var cache = await caches.open(CACHE);
        cache.put(req, fresh.clone());
        return fresh;
      } catch (e) {
        var cached = await caches.match(req);
        if (cached) return cached;
        var shell = await caches.match('./index.html');
        if (shell) return shell;
        return new Response('离线且没有缓存，请先联网打开一次本站。', {
          status: 503,
          headers: { 'Content-Type': 'text/plain; charset=utf-8' }
        });
      }
    })());
    return;
  }

  /* 静态资源：缓存优先 */
  event.respondWith((async function () {
    var hit = await caches.match(req);
    if (hit) return hit;
    try {
      var res = await fetch(req);
      if (res && res.status === 200 && res.type === 'basic') {
        var cache = await caches.open(CACHE);
        cache.put(req, res.clone());
      }
      return res;
    } catch (e) {
      return new Response('', { status: 504 });
    }
  })());
});
