// Wingbeat Lab: works offline, because fieldwork happens where there is no signal.
//
// v2 adds the two trained models. They are 227 KB in total and they MUST be
// cached: a model that only works where there is a mobile signal is useless for
// exactly the sites that matter. report.json is deliberately not in this list,
// because it changes when the models are re-evaluated and should not be pinned.
const CACHE = 'wingbeat-v2';
const FILES = ['./', './index.html', './dsp.js', './features.js', './model.js',
               './classify.js', './app.js', './manifest.webmanifest',
               './models/melfb.bin',
               './models/detector.json', './models/detector.bin',
               './models/culex.json', './models/culex.bin',
               './icons/icon-192.png', './icons/icon-512.png'];
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE)
    .then(c => Promise.all(FILES.map(f => c.add(f).catch(() => null))))
    .then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys()
    .then(k => Promise.all(k.filter(x => x !== CACHE).map(x => caches.delete(x))))
    .then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  const url = new URL(e.request.url);
  // report.json: try the network first so updated scores appear, fall back to
  // the cache, and treat its absence as a normal outcome rather than an error.
  if (url.pathname.endsWith('/models/report.json')) {
    e.respondWith(fetch(e.request)
      .then(r => {
        if (r.ok) caches.open(CACHE).then(c => c.put(e.request, r.clone()));
        return r;
      })
      .catch(() => caches.match(e.request).then(hit => hit ||
        new Response('{}', { status: 404, headers: { 'Content-Type': 'application/json' } }))));
    return;
  }
  e.respondWith(caches.match(e.request).then(hit => hit || fetch(e.request)));
});
