/**
 * WhatsApp bot (Baileys) + Catalog-first search + local Ollama fallback
 * - Product queries -> catalog ONLY (no LLM fallback on numeric/product-like)
 * - Chit-chat/general -> Mistral
 * - Safe math evaluator (before wiki)
 * - Weather/news/wiki/google/capital preserved
 * - Product lexicon loaded from rag-pdf/catalog_docs.json
 */

const {
  default: makeWASocket,
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  DisconnectReason,
} = require('@whiskeysockets/baileys');
const qrcode = require('qrcode-terminal');
const axios = require('axios');
const P = require('pino');
const Fuse = require('fuse.js');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

/* =========================
 * Config & paths
 * ========================= */
const BASE_DIR = __dirname;
const RAG_DIR = path.join(BASE_DIR, 'rag-pdf');
const SEARCH_SCRIPT = path.join(RAG_DIR, 'search_catalog.py');
const DOCS_JSON = path.join(RAG_DIR, 'catalog_docs.json');

let PYTHON_BIN = process.env.PYTHON_BIN || 'python';
try {
  const venvPython = path.join(
    RAG_DIR,
    '.venv',
    'Scripts',
    process.platform === 'win32' ? 'python.exe' : 'python'
  );
  if (!process.env.PYTHON_BIN && fs.existsSync(venvPython)) PYTHON_BIN = venvPython;
} catch {}

/* Optional APIs */
const WEATHER_API = process.env.OPENWEATHER_KEY || '';
const NEWS_API = process.env.NEWSAPI_KEY || '';
const GOOGLE_API = process.env.GOOGLE_API_KEY || '';
const CSE_ID = process.env.GOOGLE_CSE_ID || '';

const OLLAMA_URL = (process.env.OLLAMA_URL || 'http://localhost:11434').replace(/\/+$/,'');
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'mistral';
const OLLAMA_TIMEOUT_MS = parseInt(process.env.OLLAMA_TIMEOUT_MS || '60000', 10);
const DISABLE_OLLAMA = String(process.env.DISABLE_OLLAMA || 'false').toLowerCase() === 'true';

const WIKI_URL_EN = 'https://en.wikipedia.org/api/rest_v1/page/summary';
const WIKI_URL_DE = 'https://de.wikipedia.org/api/rest_v1/page/summary';

/* Simple topics for fuse fallback */
const topics = [
  'Kemmler Baustoffe', 'Germany', 'Concrete', 'Construction Materials',
  'Building Supplies', 'Bricks', 'Cement'
];
const fuse = new Fuse(topics, { includeScore: true, threshold: 0.4 });

/* =========================
 * Product lexicon (loaded once)
 * ========================= */
let PRODUCT_TOKENS = new Set();     // vocabulary from catalog (normalized)
let PRODUCT_HINTS = new Set([       // static hints for product-y words
  'hawe','veribor','tajima','fugenfux','kusto','hufa','ax65','dbgm',
  'kelle','kellen','spachtel','putz','putzkelle','fugenkelle','fliesenkelle',
  'glättekelle','glattscheibe','glättescheibe','glättekellen',
  'fugbrett','fugenbrett','reibebrett','scheibe','heber','saugheber',
  'hobel','messer','bohrer','pistole','halter','lifter','schneider',
  'presse','fux','rabo','rädchen','lochbohrer','kartuschenpresse',
  'mörtelpresse','moertelpresse','fugen','gipser','beton','porenbeton',
  'stiel','herzkelle','malerspachtel','fugenspachtel','ersatz','belag','ersatzbelag'
]);

// greetings set for extra safety (EN + DE, small)
const GREETINGS = new Set([
  'hi','hello','hey','hallo','servus','moin','yo','sup','whats','whatsup','what’s','what’sup',
  'good','morning','evening','night','gm','gn','bye','ciao','hru','how','are','you','u'
]);

function normalizeText(s = '') {
  return s
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[^a-z0-9äöüß.\s\-\/]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function splitTokens(s = '') {
  return normalizeText(s)
    .replace(/[-/]/g, ' ')    // split hyphen/slash too
    .split(' ')
    .filter(w => w && w.length >= 2); // allow 2 to catch 'ax', 'gn', etc. for greetings gate
}

function loadProductLexicon() {
  try {
    if (!fs.existsSync(DOCS_JSON)) return;
    const raw = JSON.parse(fs.readFileSync(DOCS_JSON, 'utf-8'));
    const toks = new Set();
    for (const d of raw) {
      const fields = [
        d.title || '', d.category || '', d.ausfuehrung || '',
        d.description || '', d.bestell_nr || ''
      ].join(' ');
      for (const t of splitTokens(fields)) toks.add(t);
    }
    PRODUCT_TOKENS = toks;
    console.log(`🧩 Loaded product lexicon: ${PRODUCT_TOKENS.size} tokens`);
  } catch (e) {
    console.warn('⚠️ Could not load product lexicon:', e.message);
  }
}
loadProductLexicon();

/* =========================
 * Helpers
 * ========================= */

// Keep dots so Bestell-Nr. like "13.500" / "50.150" are preserved
function cleanProductQuery(text) {
  return (text || '')
    .replace(/^\!search\s*/i, '')
    .replace(/do you have|do you sell|haben sie|verkaufst du|verkaufen sie|can i buy|please|bitte|gibt es/gi, '')
    .replace(/[?!,]/g, '') // keep '.'
    .trim();
}

// Chit-chat & utilities we should NOT send to catalog
function isChitChatOrUtility(text) {
  const t = (text || '').toLowerCase().trim();

  // super-short, no digits -> treat as chit-chat (prevents "hi"/"ok" from hitting catalog)
  if (t.length <= 3 && !/\d/.test(t)) return true;

  // greetings / small talk (covers "how r u", gm/gn, etc.)
  if (/\b(hi|hello|hey|hallo|servus|moin|yo|sup|whats? ?up)\b/.test(t)) return true;
  if (/\b(good\s*(morning|evening|night)|gm|gn|bye|ciao)\b/.test(t)) return true;
  if (/\b(thanks|thank\s*(you|u)|pls|please|help)\b/.test(t)) return true;
  if (/\b(how\s*are\s*(you|u)|how\s*r\s*u)\b/.test(t)) return true;

  // fun stuff
  if (/\b(joke|witz|poem|gedicht|story|geschichte|quote|spruch)\b/.test(t)) return true;

  // explicit tools/keywords
  if (/^!rebuild\b/i.test(t)) return true;
  if (/^(weather|temperature|forecast|rain)\b/i.test(t)) return true;
  if (/^news\b/i.test(t)) return true;
  if (/^search\s+/.test(t) || /^google\b/.test(t)) return true;
  if (/^(wiki|who is|what is)\b/i.test(t)) return true; // math runs earlier anyway

  return false;
}

// Strong signals of a catalog/product query
function looksLikeBestell(text) {
  const t = (text || '').trim();
  if (!t) return false;
  return /^\d[\d.]*$/.test(t) || /\b\d{1,3}\.\d{2,3}\b/.test(t);
}

function seemsProductByTokens(text) {
  const tokens = splitTokens(text);
  if (!tokens.length) return false;

  // if message is basically only greetings/stopwords, don't treat as product
  const allGreetings = tokens.every(w => GREETINGS.has(w));
  if (allGreetings) return false;

  // match any token in product lexicon or static hints
  for (const tok of tokens) {
    if (PRODUCT_TOKENS.has(tok) || PRODUCT_HINTS.has(tok)) return true;
  }

  // single compact token with hyphens/slashes like "Ersatz-Belag" / "Montage-/PU-Pistolen-Reiniger"
  if (!/\s/.test(text) && /[-/]/.test(text) && text.length <= 64) return true;

  // tooly suffixes
  if (/(kelle|spachtel|brett|scheibe|heber|hobel|messer|bohrer|pistole|halter|lifter|schneider|presse|fux|rabo)\b/i.test(text)) return true;

  // contains brand-y words
  if (/\b(hawe|veribor|tajima|hufa|kusto|fugenfux|dbgm)\b/i.test(text)) return true;

  return false;
}

function isLikelyProductQuery(text) {
  const t = (text || '').trim();
  if (!t) return false;

  // very short, non-numeric -> not a product
  if (t.length <= 3 && !/\d/.test(t)) return false;

  return looksLikeBestell(t) || seemsProductByTokens(t);
}

// Safe math: try to evaluate expressions like "what is 256+256" or "2*(128+64)"
function tryComputeMath(text) {
  const raw = (text || '').trim();

  // pull out expression after "what is / calculate / calc"
  const m = /^(?:what is|calculate|calc|compute)\s+(.+)$/i.exec(raw);
  let expr = m ? m[1] : raw;

  // only allow digits, operators, spaces, dots, parentheses
  if (!/^[\d+\-*/().\s]+$/.test(expr)) return null;
  if (!/[+\-*/]/.test(expr)) return null; // must have an operator (avoid Bestell-Nr.)
  try {
    // eslint-disable-next-line no-new-func
    const val = Function(`"use strict"; return (${expr});`)();
    if (typeof val === 'number' && Number.isFinite(val)) {
      return String(val);
    }
  } catch {}
  return null;
}

// Spawn Python search script
function runPythonSearch(query) {
  return new Promise((resolve) => {
    if (!fs.existsSync(SEARCH_SCRIPT)) {
      const msg = `Search script not found at ${SEARCH_SCRIPT}`;
      console.error('❌', msg);
      resolve({ ok: false, error: msg });
      return;
    }
    const args = [SEARCH_SCRIPT, query];
    const py = spawn(PYTHON_BIN, args, {
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
      cwd: RAG_DIR,
      shell: false,
      windowsHide: true
    });

    let out = '', err = '';
    const killTimer = setTimeout(() => { try { py.kill('SIGKILL'); } catch {} }, 60_000);

    py.stdout.on('data', d => out += d.toString('utf-8'));
    py.stderr.on('data', d => err += d.toString('utf-8'));
    py.on('error', (e) => { clearTimeout(killTimer); resolve({ ok: false, error: e.message || String(e) }); });
    py.on('close', (code) => {
      clearTimeout(killTimer);
      if (code !== 0) return resolve({ ok: false, error: err || `Python exited with code ${code}` });
      const txt = (out || '').trim();
      if (!txt || /^❌/.test(txt)) return resolve({ ok: false, error: txt || 'No results' });
      resolve({ ok: true, text: txt });
    });
  });
}

// Capital-of intent
function detectCapitalQuestion(text) {
  const m = /capital of\s+([a-zA-ZÀ-ÿ\s'-]+)\??$/i.exec(text || '');
  return m ? m[1].trim() : null;
}
const CAPITALS = {
  france: 'Paris', germany: 'Berlin', italy: 'Rom / Rome', spain: 'Madrid',
  india: 'Neu-Delhi / New Delhi', austria: 'Wien / Vienna',
  switzerland: 'Bern / Berne', belgium: 'Brüssel / Brussels',
};

/* WEATHER parsing */
function parseWeatherQuery(text) {
  const cityIn = text.match(/\bin\s+([a-zA-ZÀ-ÿ.'\- ]{2,})\??$/i);
  const cityAfterWord = text.match(/^(?:weather|temperature|forecast|rain)\s+(.+)/i);
  const city = (cityIn?.[1] || cityAfterWord?.[1] || '').trim();
  const tomorrow = /\btomorrow\b/i.test(text);
  return { city: city || null, tomorrow };
}
async function getCurrentWeather(city) {
  if (!WEATHER_API) throw new Error('Missing OPENWEATHER_KEY');
  const url = `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(city)}&appid=${WEATHER_API}&units=metric`;
  const r = await axios.get(url, { timeout: 20_000 });
  const { main, weather, name } = r.data;
  return `🌦 Weather in ${name}: ${weather?.[0]?.description || '-'}, ${main?.temp ?? '-'}°C.`;
}
async function getTomorrowRainForecast(city) {
  if (!WEATHER_API) throw new Error('Missing OPENWEATHER_KEY');
  const url = `https://api.openweathermap.org/data/2.5/forecast?q=${encodeURIComponent(city)}&appid=${WEATHER_API}&units=metric`;
  const r = await axios.get(url, { timeout: 20_000 });
  const list = r.data?.list || [];
  if (!list.length) throw new Error('No forecast data');

  const byDay = {};
  for (const item of list) {
    const dt = new Date(item.dt * 1000);
    const day = dt.toISOString().slice(0, 10);
    (byDay[day] ||= []).push(item);
  }
  const today = new Date();
  const tomorrowIso = new Date(today.getTime() + 24*60*60*1000).toISOString().slice(0,10);
  const slots = byDay[tomorrowIso] || [];
  if (!slots.length) return `☔ Forecast for tomorrow in ${r.data.city?.name || city}: (no data).`;

  const rainy = slots.some(s => {
    if (s.rain && (s.rain['3h'] || s.rain['1h'])) return true;
    const wid = s.weather?.[0]?.id || 0;
    return wid >= 200 && wid < 600;
  });
  const desc = slots[0]?.weather?.[0]?.description || '-';
  return rainy
    ? `☔ Yes, it looks like rain tomorrow in ${r.data.city?.name || city}. (${desc})`
    : `🌤 No, rain is unlikely tomorrow in ${r.data.city?.name || city}. (${desc})`;
}

/* =========================
 * Main WhatsApp logic
 * ========================= */
async function startSock() {
  const { state, saveCreds } = await useMultiFileAuthState('./auth_info_baileys');
  const { version } = await fetchLatestBaileysVersion();
  const sock = makeWASocket({ version, logger: P({ level: 'silent' }), auth: state });

  sock.ev.on('creds.update', saveCreds);

  sock.ev.on('connection.update', ({ connection, lastDisconnect, qr }) => {
    if (qr) {
      console.log('📱 Scan the QR code to login:');
      qrcode.generate(qr, { small: true });
    }
    if (connection === 'close') {
      const reason = lastDisconnect?.error;
      console.log('❌ Disconnected. Reason:', reason?.message || reason || '(unknown)');
      const shouldReconnect = reason?.output?.statusCode !== DisconnectReason.loggedOut;
      if (shouldReconnect) startSock();
    } else if (connection === 'open') {
      console.log('✅ WhatsApp connection established!');
      console.log(`🐍 Using Python: ${PYTHON_BIN}`);
      console.log(`📄 Search script: ${SEARCH_SCRIPT}`);
    }
  });

  sock.ev.on('messages.upsert', async ({ messages }) => {
    const msg = messages?.[0];
    if (!msg?.message || msg.key.fromMe) return;
    if (msg.key.remoteJid.endsWith('@g.us')) return; // ignore groups

    const sender = msg.key.remoteJid;
    const textRaw = msg.message.conversation || msg.message.extendedTextMessage?.text || '';
    const text = (textRaw || '').trim();
    if (!text) return;

    console.log(`📨 Message from ${sender}: ${text}`);

    try {
      // 0) Build command
      if (/^!rebuild\b/i.test(text)) {
        await sock.sendMessage(sender, { text: '🛠️ Rebuilding product index… this may take a minute.' });
        const res = await new Promise((resolve) => {
          const py = spawn(PYTHON_BIN, [SEARCH_SCRIPT, '--build'], {
            env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
            cwd: RAG_DIR,
            shell: false,
            windowsHide: true
          });
          let out = '', err = '';
          py.stdout.on('data', d => out += d.toString('utf-8'));
          py.stderr.on('data', d => err += d.toString('utf-8'));
          py.on('close', code => resolve({ code, out: out.trim(), err: err.trim() }));
          py.on('error', e => resolve({ code: -1, out: '', err: e.message || String(e) }));
        });
        if (res.code === 0) await sock.sendMessage(sender, { text: res.out || '✅ Rebuild finished.' });
        else {
          console.error('❌ Rebuild error:', res.err);
          await sock.sendMessage(sender, { text: `❌ Rebuild failed.\n${res.err || '(no error text)'}` });
        }
        // refresh lexicon after rebuild
        loadProductLexicon();
        return;
      }

      // 1) Explicit catalog search command: "!search ..."
      if (/^!search\s+/i.test(text)) {
        const q = cleanProductQuery(text);
        const res = await runPythonSearch(q);
        const reply = res.ok ? res.text : `❌ Kein Treffer im Katalog.\n${res.error ? 'Details: ' + res.error : ''}`;
        await sock.sendMessage(sender, { text: reply });
        return;
      }

      // 2) Weather
      if (/(^|\s)(weather|temperature|forecast|rain)\b/i.test(text) || /\btomorrow\b/i.test(text)) {
        const { city, tomorrow } = parseWeatherQuery(text);
        const cityName = city || 'Berlin';
        try {
          let reply;
          if (tomorrow || /forecast|rain/i.test(text)) reply = await getTomorrowRainForecast(cityName);
          else reply = await getCurrentWeather(cityName);
          await sock.sendMessage(sender, { text: reply });
        } catch (e) {
          const msgErr = e?.response?.data?.message || e.message || 'Could not fetch weather.';
          await sock.sendMessage(sender, { text: `⚠️ Weather error: ${msgErr}` });
        }
        return;
      }

      // 3) News
      if (/^news\b/i.test(text) || /news/i.test(text)) {
        const match = text.match(/news.*about\s+(.+)/i);
        let reply;
        try {
          if (match && match[1]) {
            const topic = match[1].trim().replace(/[?.!,]+$/, '');
            const r = await axios.get(`https://newsapi.org/v2/everything?q=${encodeURIComponent(topic)}&pageSize=3&apiKey=${NEWS_API}`, { timeout: 15_000 });
            reply = r.data.articles?.length
              ? r.data.articles.map((a, i) => `${i + 1}. ${a.title}`).join('\n')
              : `🔍 No news found about "${topic}".`;
          } else {
            const r = await axios.get(`https://newsapi.org/v2/top-headlines?country=de&pageSize=3&apiKey=${NEWS_API}`, { timeout: 15_000 });
            reply = r.data.articles?.map((a, i) => `${i + 1}. ${a.title}`).join('\n') || '🔍 No headlines.';
          }
        } catch (e) {
          reply = `⚠️ News error: ${e?.response?.data?.message || e.message}`;
        }
        await sock.sendMessage(sender, { text: reply });
        return;
      }

      // 4) Google
      if (/^search\s+/i.test(text) || /^google\b/i.test(text)) {
        const match = text.match(/search (.+)/i);
        const query = match ? match[1].trim().replace(/[?.!,]+$/, '') : text;
        try {
          const r = await axios.get(`https://www.googleapis.com/customsearch/v1?key=${GOOGLE_API}&cx=${CSE_ID}&q=${encodeURIComponent(query)}`, { timeout: 15_000 });
          const top = r.data.items?.[0];
          const reply = top ? `🔍 Top Google result: ${top.title}\n${top.link}` : '🔍 No results found.';
          await sock.sendMessage(sender, { text: reply });
        } catch (e) {
          await sock.sendMessage(sender, { text: `⚠️ Google error: ${e?.response?.data?.error?.message || e.message}` });
        }
        return;
      }

      // 5) Capital-of quick helper
      const countryAsked = detectCapitalQuestion(text);
      if (countryAsked) {
        const key = countryAsked.toLowerCase();
        if (CAPITALS[key]) {
          await sock.sendMessage(sender, { text: `🏛 The capital of ${countryAsked} is ${CAPITALS[key]}.` });
          return;
        }
        try {
          const rEN = await axios.get(`${WIKI_URL_EN}/${encodeURIComponent(countryAsked)}`, { timeout: 15_000 });
          const extract = rEN.data?.extract || '';
          const capital = /capital.*?\b([A-Z][A-Za-z -]+)/i.exec(extract);
          if (capital && capital[1]) {
            await sock.sendMessage(sender, { text: `🏛 The capital of ${countryAsked} is ${capital[1]}.` });
            return;
          }
        } catch {}
        await sock.sendMessage(sender, { text: `📚 Could not determine the capital of "${countryAsked}".` });
        return;
      }

      // 6) Math (before wiki)
      const mathAns = tryComputeMath(text);
      if (mathAns !== null) {
        await sock.sendMessage(sender, { text: `🧮 ${mathAns}` });
        return;
      }

      // 7) Wiki (explicit)
      if (/^(wiki|who is|what is)\b/i.test(text)) {
        const rawTopic = text.replace(/^(wiki|who is|what is)/i, '').trim().replace(/[?.!,]+$/, '');
        const fuseResult = fuse.search(rawTopic);
        const topic = fuseResult.length > 0 ? fuseResult[0].item : rawTopic;
        const enc = encodeURIComponent(topic);
        let reply;
        try {
          const rEN = await axios.get(`${WIKI_URL_EN}/${enc}`, { timeout: 15_000 });
          reply = `📖 ${rEN.data.extract}`;
        } catch (errEN) {
          if (errEN?.response?.status === 404) {
            try {
              const rDE = await axios.get(`${WIKI_URL_DE}/${enc}`, { timeout: 15_000 });
              reply = `📘 ${rDE.data.extract}`;
            } catch {
              reply = `📚 No Wikipedia entry found for "${topic}".`;
            }
          } else {
            reply = `⚠️ Wiki error: ${errEN?.message || 'unknown error'}`;
          }
        }
        await sock.sendMessage(sender, { text: reply });
        return;
      }

      // 8) Product router — product → catalog; everything else moves on to LLM/tools
      const looksProduct = isLikelyProductQuery(text);

      if (looksProduct) {
        const q = cleanProductQuery(text);
        const res = await runPythonSearch(q);
        if (res.ok) {
          await sock.sendMessage(sender, { text: res.text });
          return;
        }
        // If it looked like a product query, do NOT fall back to LLM
        const msgNoHit = res.error
          ? `❌ Kein Treffer im Katalog.\nDetails: ${res.error}`
          : '❌ Kein Treffer im Katalog.';
        await sock.sendMessage(sender, { text: msgNoHit });
        return;
      }

      // 9) Ollama fallback (chit-chat, general Q&A, etc.)
      if (!DISABLE_OLLAMA) {
        try {
          const r = await axios.post(`${OLLAMA_URL}/api/generate`, {
            model: OLLAMA_MODEL,
            prompt: text,
            stream: false
          }, { timeout: OLLAMA_TIMEOUT_MS });

          await sock.sendMessage(sender, { text: `🤖 ${r.data.response}` });
          return;
        } catch (e) {
          console.error('🧠 Ollama error:', e.message || e);
          await sock.sendMessage(sender, { text: '⚠️ Could not get a response from the AI model.' });
          return;
        }
      } else {
        await sock.sendMessage(sender, { text: '🙂' });
      }

    } catch (err) {
      console.error('❌ Handler error:', err);
      await sock.sendMessage(sender, { text: '⚠️ Could not fetch information. Please try again.' });
    }
  });
}

startSock();
