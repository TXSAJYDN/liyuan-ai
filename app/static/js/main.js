/* ============================================================
   梨园AI — 通用交互逻辑
   ============================================================ */

const API = {
  status:    () => fetch('/api/status').then(r => r.json()),
  genres:    () => fetch('/api/genres').then(r => r.json()),
  videos:    (genre) => fetch(`/api/videos?genre=${genre || ''}`).then(r => r.json()),
  analyzeUpload: (formData) => fetch('/api/analyze/upload', { method: 'POST', body: formData }),
  analyzeOpera:  (genre, name) => fetch('/api/analyze/opera', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ genre, video_name: name })
  }).then(r => r.json()),
  search: (query, videoName, topK = 10) => fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, video_name: videoName || null, top_k: topK })
  }).then(r => r.json()),
  qa: (question) => fetch('/api/qa', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: 5 })
  }).then(r => r.json()),
  initKB:   () => fetch('/api/init/knowledge_base', { method: 'POST' }).then(r => r.json()),
  initQwen: () => fetch('/api/init/qwen_model', { method: 'POST' }).then(r => r.json()),
};

/* ---------- Nav scroll effect ---------- */
const nav = document.getElementById('mainNav');
if (nav) {
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 20);
  }, { passive: true });
}

/* ---------- Mobile nav toggle ---------- */
const navToggle = document.getElementById('navToggle');
if (navToggle) {
  navToggle.addEventListener('click', () => {
    const links = document.querySelector('.nav-links');
    if (links) links.classList.toggle('open');
  });
}

/* ---------- Loading overlay ---------- */
function showLoading(text) {
  const overlay = document.getElementById('loadingOverlay');
  if (!overlay) return;
  const txt = overlay.querySelector('.spinner-text');
  if (txt && text) txt.textContent = text;
  overlay.classList.add('active');
}

function hideLoading() {
  const overlay = document.getElementById('loadingOverlay');
  if (overlay) overlay.classList.remove('active');
}

/* ---------- Intersection Observer for fade-in ---------- */
document.addEventListener('DOMContentLoaded', () => {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15 });

  document.querySelectorAll('.animate-on-scroll').forEach(el => {
    observer.observe(el);
  });
});

/* ---------- Utility: format seconds to mm:ss ---------- */
function formatTime(seconds) {
  if (seconds == null) return '--:--';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

/* ---------- Utility: parse analysis JSON from markdown ---------- */
function parseAnalysis(raw) {
  if (!raw) return null;
  const m = raw.match(/```(?:json)?\s*([\s\S]+?)\s*```/);
  if (!m) return null;
  try { return JSON.parse(m[1]); } catch { return null; }
}

/* ---------- Keyframe path → URL ---------- */
function keyframeUrl(path) {
  const parts = path.replace(/\\/g, '/').split('/');
  const kfIdx = parts.indexOf('keyframes');
  if (kfIdx >= 0) {
    return '/keyframes/' + parts.slice(kfIdx + 1).join('/');
  }
  return path;
}

/* ---------- Animate counter ---------- */
function animateCounter(el, target, duration = 1200) {
  const start = performance.now();
  const initial = 0;
  function update(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(initial + (target - initial) * eased);
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

/* ---------- Lightbox for images ---------- */
function openLightbox(src) {
  const existing = document.querySelector('.lightbox');
  if (existing) existing.remove();
  const lb = document.createElement('div');
  lb.className = 'lightbox';
  lb.innerHTML = `<img src="${src}" alt="preview">`;
  lb.addEventListener('click', () => lb.remove());
  document.body.appendChild(lb);
  document.addEventListener('keydown', function handler(e) {
    if (e.key === 'Escape') { lb.remove(); document.removeEventListener('keydown', handler); }
  });
}

/* ---------- Smooth nav hide/show on scroll ---------- */
let lastScrollY = 0;
window.addEventListener('scroll', () => {
  const nav = document.getElementById('mainNav');
  if (!nav) return;
  const currentY = window.scrollY;
  if (currentY > 300 && currentY > lastScrollY) {
    nav.style.transform = 'translateY(-100%)';
  } else {
    nav.style.transform = 'translateY(0)';
  }
  lastScrollY = currentY;
}, { passive: true });
