const CACHE_NAME = 'z-image-v2';
const SHELL = ['/', '/manifest.json'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE_NAME).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);

  // Don't intercept API calls — let them fail naturally so the app JS can handle it
  if (url.pathname.startsWith('/api/')) return;

  // Navigation requests (opening the app): cache first, network fallback
  if (e.request.mode === 'navigate') {
    e.respondWith(
      caches.open(CACHE_NAME).then(cache =>
        cache.match('/').then(cached => {
          // Always try to update the cache in background
          const fetchPromise = fetch(e.request).then(resp => {
            if (resp.ok) cache.put('/', resp.clone());
            return resp;
          }).catch(() => null);

          // Return cached immediately if available, otherwise wait for network
          if (cached) return cached;
          return fetchPromise.then(resp => resp || new Response(
            '<h1 style="font-family:system-ui;text-align:center;margin-top:40vh">Offline — open while server is running first</h1>',
            {headers: {'Content-Type': 'text/html'}}
          ));
        })
      )
    );
    return;
  }

  // Other resources: cache first
  e.respondWith(
    caches.match(e.request).then(cached =>
      cached || fetch(e.request).then(resp => {
        if (resp.ok) {
          const clone = resp.clone();
          caches.open(CACHE_NAME).then(c => c.put(e.request, clone));
        }
        return resp;
      }).catch(() => new Response('', {status: 408}))
    )
  );
});
