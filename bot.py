"""
Asistente personal en Telegram con modelo open-source vía HuggingFace.
Comandos:
  /start, /help       -> ayuda
  /ask <texto>        -> pregunta libre al LLM
  /nota <texto>       -> guarda una nota
  /notas              -> lista notas (cada una con botón "Leer en voz alta")
  /delnote <id>       -> borra una nota por ID
  /todo <texto>       -> agrega tarea
  /todos              -> lista tus tareas
  /done <id>          -> marca tarea completada
  /drop <id>          -> borra una tarea
  /remind <min> <txt> -> recordatorio en N minutos
  /remindat HH:MM <txt> -> recordatorio a una hora (hoy)
  /clima <ciudad>     -> clima actual
  /resume <url>       -> resumen de un artículo
  /modelo [nombre]    -> muestra o cambia el modelo (texto) — botones inline
  /voz <texto>        -> lee texto en voz alta (TTS, MP3)

Multimedia:
  - Foto enviada         -> BLIP caption + enriquecimiento con LLM
  - Audio / voice enviado -> Whisper transcribe y LLM responde
"""
import asyncio
import io
import os
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import aiohttp
from openai import OpenAI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters,
)

# --- Configuración / variables de entorno ---
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
HF_TOKEN        = os.getenv("HF_TOKEN")              # reemplaza a XAI_API_KEY
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0")) or None

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "assistant.db"

# Endpoint OpenAI-compatible de HuggingFace (verificado en docs oficiales)
HF_BASE_URL = "https://router.huggingface.co/v1"

# Modelo por defecto (defines cuál usar al arrancar o lo cambias con /modelo)
DEFAULT_TEXT_MODEL = os.getenv("HF_TEXT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
AVAILABLE_TEXT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",            # fuerte en español, recomendado
    "meta-llama/Meta-Llama-3-8B-Instruct", # estilo GPT-4
    "mistralai/Mistral-7B-Instruct-v0.3",  # rápido y conciso
]

# Archivo persistente (sin SQLite) del modelo elegido por el usuario
MODEL_FILE = DATA_DIR / "model.txt"
def _load_model() -> str:
    if MODEL_FILE.exists():
        m = MODEL_FILE.read_text().strip()
        if m:
            return m
    return DEFAULT_TEXT_MODEL
def _save_model(name: str) -> None:
    MODEL_FILE.write_text(name.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("assistant")

# --- Cliente OpenAI-compatible apuntando a HuggingFace ---
def hf_client() -> OpenAI:
    return OpenAI(api_key=HF_TOKEN, base_url=HF_BASE_URL)

SYSTEM_PROMPT = (
    "Eres un asistente personal útil y conciso. Responde en español a menos "
    "que el usuario escriba en otro idioma. Sé breve cuando sea posible. "
    "Si no sabes algo, dilo. Puedes resumir información cuando el usuario lo pida."
)

# --- Base de datos (notas / tareas) ---
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS notes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS todos(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            done INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """)

# --- Llamada LLM (texto, vía HuggingFace OpenAI-compatible) ---
def ask_llm(prompt: str, system: str = SYSTEM_PROMPT, model: str | None = None) -> str:
    c = hf_client()
    r = c.chat.completions.create(
        model=model or _load_model(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.4,
        max_tokens=800,
    )
    return (r.choices[0].message.content or "").strip()

async def ask_llm_async(prompt: str, system: str = SYSTEM_PROMPT, model: str | None = None) -> str:
    # SDK OpenAI es bloqueante -> lo mandamos a un hilo
    return await asyncio.to_thread(ask_llm, prompt, system, model)

# --- Visión: BLIP captioning (HF Inference API clásico) ---
async def describe_image_bytes(image: bytes) -> str:
    if not HF_TOKEN:
        return "(HF_TOKEN no configurado, no puedo analizar la imagen)"
    url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
        async with s.post(
            url,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            data=image,
        ) as r:
            if r.status != 200:
                txt = await r.text()
                raise RuntimeError(f"BLIP {r.status}: {txt[:200]}")
            data = await r.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "error" in data:
        return f"(BLIP no disponible: {data['error']})"
    return str(data)[:300]

# --- Voz: TTS con gTTS (sin FFmpeg, devuelve MP3) ---
async def tts_mp3_bytes(text: str) -> bytes:
    try:
        from gtts import gTTS
    except ImportError:
        raise RuntimeError("Falta el paquete gtts. Añádelo a requirements.txt")
    def _build() -> bytes:
        buf = io.BytesIO()
        gTTS(text=text, lang="es").write_to_fp(buf)
        return buf.getvalue()
    return await asyncio.to_thread(_build)

# --- Voz: Whisper (HuggingFace OpenAI-compatible) ---
async def transcribe_audio(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    c = hf_client()
    def _call() -> str:
        r = c.audio.transcriptions.create(
            model="openai/whisper-large-v3",
            file=(filename, audio_bytes),
        )
        return (r.text or "").strip()
    return await asyncio.to_thread(_call)

# --- Filtro de privacidad ---
def authorized(update: Update) -> bool:
    if ALLOWED_USER_ID is None:
        return True
    return update.effective_user and update.effective_user.id == ALLOWED_USER_ID

async def deny(update: Update):
    await update.effective_message.reply_text("⛔ Bot privado. Acceso denegado.")

# --- Comandos básicos ---
async def start_cmd(update: Update, _: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    model = _load_model()
    await update.effective_message.reply_text(
        f"👋 Soy tu asistente personal (HuggingFace).\n"
        f"Modelo activo: `{model}`\n"
        f"Tipos: texto, voz (TTS), visión, audio.\n\n"
        f"Envía /help para ver todo."
    )

async def help_cmd(update: Update, _: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    await update.effective_message.reply_text(
        "🧾 *Comandos disponibles:*\n\n"
        "*Texto / LLM:*\n"
        "• /ask _pregunta_ — pregunta libre\n"
        "• /modelo — ver/cambiar modelo\n"
        "• /resume _url_ — resumir artículo\n\n"
        "*Notas / tareas:*\n"
        "• /nota _texto_ — guardar nota\n"
        "• /notas — listar (cada nota con 🔊 Leer)\n"
        "• /delnote _id_ — borrar nota\n"
        "• /todo _texto_ — agregar tarea\n"
        "• /todos — listar tareas\n"
        "• /done _id_ — marcar tarea hecha\n"
        "• /drop _id_ — borrar tarea\n\n"
        "*Recordatorios:*\n"
        "• /remind _minutos_ _texto_ — en N min\n"
        "• /remindat _HH:MM_ _texto_ — a las HH:MM de hoy\n\n"
        "*Voz:*\n"
        "• /voz _texto_ — leer en voz alta (MP3)\n"
        "• Envíame un audio/voice — lo transcribo + respondo\n\n"
        "*Visión:*\n"
        "• Envíame una foto — la describo (BLIP + LLM)\n\n"
        "*Otros:*\n"
        "• /clima _ciudad_ — clima actual\n\n"
        "Cualquier mensaje sin '/' va al chat libre.",
        parse_mode="Markdown",
    )

# --- /modelo: ver o cambiar el modelo activo ---
async def modelo_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    if not ctx.args:
        actual = _load_model()
        buttons = [
            [InlineKeyboardButton(m.split("/")[-1], callback_data=f"setmodel|{m}")]
            for m in AVAILABLE_TEXT_MODELS
        ]
        await update.effective_message.reply_text(
            f"🤖 Modelo actual: `{actual}`\nToca uno para cambiar:",
            reply_markup=InlineKeyboardMarkup(buttons),
            parse_mode="Markdown",
        )
        return
    new = " ".join(ctx.args).strip()
    _save_model(new)
    await update.effective_message.reply_text(f"✅ Modelo cambiado a `{new}`.")

async def model_cb(update: Update, _: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if not authorized(update): return
    _, new = q.data.split("|", 1)
    _save_model(new)
    await q.edit_message_text(f"✅ Modelo activo: `{new}`", parse_mode="Markdown")

# --- Notas (con botón Leer en voz alta) ---
async def nota_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    text = " ".join(ctx.args).strip()
    if not text:
        return await update.effective_message.reply_text("Uso: /nota <texto>")
    uid = update.effective_user.id
    with db() as c:
        cur = c.execute("INSERT INTO notes(user_id,text) VALUES(?,?)", (uid, text))
        nid = cur.lastrowid
    await update.effective_message.reply_text(
        f"📝 Nota #{nid} guardada.",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("🔊 Leer en voz alta", callback_data=f"ttsnote|{nid}")
        ]]),
    )

async def notas_cmd(update: Update, _: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    uid = update.effective_user.id
    with db() as c:
        rows = c.execute(
            "SELECT id,text,created_at FROM notes WHERE user_id=? ORDER BY id DESC LIMIT 50",
            (uid,),
        ).fetchall()
    if not rows:
        return await update.effective_message.reply_text("No tienes notas guardadas.")
    lines = [f"#{r['id']} ({r['created_at']}) — {r['text']}" for r in rows]
    await update.effective_message.reply_text("\n".join(lines))

async def tts_note_cb(update: Update, _: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer("🎙️ Generando audio…")
    if not authorized(update): return
    _, nid = q.data.split("|", 1)
    uid = update.effective_user.id
    with db() as c:
        row = c.execute("SELECT text FROM notes WHERE id=? AND user_id=?",
                        (int(nid), uid)).fetchone()
    if not row:
        return await q.edit_message_text("Nota no encontrada.")
    try:
        mp3 = await tts_mp3_bytes(row["text"][:500])
        await q.message.reply_audio(
            audio=io.BytesIO(mp3), title=f"Nota #{nid}", filename=f"nota_{nid}.mp3"
        )
    except Exception as e:
        log.exception("tts note error")
        await q.message.reply_text(f"Error TTS: {e}")

async def delnote_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    if not ctx.args or not ctx.args[0].isdigit():
        return await update.effective_message.reply_text("Uso: /delnote <id>")
    nid = int(ctx.args[0])
    uid = update.effective_user.id
    with db() as c:
        cur = c.execute("DELETE FROM notes WHERE id=? AND user_id=?", (nid, uid))
    await update.effective_message.reply_text(
        "🗑️ Nota borrada." if cur.rowcount else "No se encontró esa nota."
    )

# --- Tareas ---
async def todo_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    text = " ".join(ctx.args).strip()
    if not text:
        return await update.effective_message.reply_text("Uso: /todo <texto>")
    uid = update.effective_user.id
    with db() as c:
        cur = c.execute("INSERT INTO todos(user_id,text) VALUES(?,?)", (uid, text))
        tid = cur.lastrowid
    await update.effective_message.reply_text(f"✅ Tarea #{tid} añadida.")

async def todos_cmd(update: Update, _: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    uid = update.effective_user.id
    with db() as c:
        rows = c.execute(
            "SELECT id,text,done FROM todos WHERE user_id=? ORDER BY done, id",
            (uid,),
        ).fetchall()
    if not rows:
        return await update.effective_message.reply_text("No tienes tareas.")
    lines = [f"{'✅' if r['done'] else '◻️'} #{r['id']} — {r['text']}" for r in rows]
    await update.effective_message.reply_text("\n".join(lines))

async def done_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    if not ctx.args or not ctx.args[0].isdigit():
        return await update.effective_message.reply_text("Uso: /done <id>")
    tid = int(ctx.args[0])
    uid = update.effective_user.id
    with db() as c:
        cur = c.execute("UPDATE todos SET done=1 WHERE id=? AND user_id=?", (tid, uid))
    await update.effective_message.reply_text(
        "🎉 Tarea completada." if cur.rowcount else "No se encontró esa tarea."
    )

async def drop_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    if not ctx.args or not ctx.args[0].isdigit():
        return await update.effective_message.reply_text("Uso: /drop <id>")
    tid = int(ctx.args[0])
    uid = update.effective_user.id
    with db() as c:
        cur = c.execute("DELETE FROM todos WHERE id=? AND user_id=?", (tid, uid))
    await update.effective_message.reply_text(
        "🗑️ Tarea borrada." if cur.rowcount else "No se encontró esa tarea."
    )

# --- Recordatorios (cron interno por JobQueue) ---
async def remind_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    if not ctx.args or not ctx.args[0].isdigit():
        return await update.effective_message.reply_text("Uso: /remind <minutos> <texto>")
    minutes = int(ctx.args[0])
    text = " ".join(ctx.args[1:]).strip()
    if not text:
        return await update.effective_message.reply_text("Falta el texto del recordatorio.")
    when = datetime.now() + timedelta(minutes=minutes)
    await schedule_reminder(ctx, update.effective_chat.id, text, when)
    await update.effective_message.reply_text(f"⏰ Te recordaré en {minutes} min: '{text}'.")

async def remindat_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    if not ctx.args:
        return await update.effective_message.reply_text("Uso: /remindat HH:MM <texto>")
    try:
        hh, mm = ctx.args[0].split(":")
        when = datetime.now().replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
    except Exception:
        return await update.effective_message.reply_text("Hora inválida. Usa HH:MM, ej. 20:30")
    if when < datetime.now():
        return await update.effective_message.reply_text("Esa hora ya pasó hoy.")
    text = " ".join(ctx.args[1:]).strip()
    if not text:
        return await update.effective_message.reply_text("Falta el texto del recordatorio.")
    await schedule_reminder(ctx, update.effective_chat.id, text, when)
    await update.effective_message.reply_text(f"⏰ Te recordaré a las {hh}:{mm}: '{text}'.")

async def schedule_reminder(ctx, chat_id, text, when_dt):
    delay = max(0, (when_dt - datetime.now()).total_seconds())
    ctx.job_queue.run_once(
        fire_reminder, when=delay,
        data={"chat_id": chat_id, "text": text},
    )
    log.info(f"Recordatorio programado: chat={chat_id} text={text!r} in {delay}s")

async def fire_reminder(ctx: ContextTypes.DEFAULT_TYPE):
    data = ctx.job.data
    await ctx.bot.send_message(
        chat_id=data["chat_id"],
        text=f"⏰ *Recordatorio:* {data['text']}",
        parse_mode="Markdown",
    )

# --- Voz: /voz ---
async def voz_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    text = " ".join(ctx.args).strip()
    if not text:
        return await update.effective_message.reply_text("Uso: /voz <texto>")
    try:
        mp3 = await tts_mp3_bytes(text[:500])
        await update.effective_message.reply_audio(
            audio=io.BytesIO(mp3), title="Voz", filename="voz.mp3"
        )
    except Exception as e:
        log.exception("voz error")
        await update.effective_message.reply_text(f"Error TTS: {e}")

# --- Multimedia: AUDIO/VOICE -> Whisper -> LLM ---
async def audio_handler(update: Update, _: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    waiting = await update.effective_message.reply_text("🎙️ Transcribiendo…")
    try:
        if update.effective_message.voice:
            file_id = update.effective_message.voice.file_id
            ext = "voice.ogg"
        elif update.effective_message.audio:
            file_id = update.effective_message.audio.file_id
            ext = "audio.m4a"
        else:
            return await waiting.edit_text("No detecté audio en ese mensaje.")
        tg_file = await update.effective_message.get_bot().get_file(file_id)
        buf = io.BytesIO()
        await tg_file.download_to_memory(buf)
        text = await transcribe_audio(buf.getvalue(), ext)
        await waiting.edit_text(
            f"📝 *Transcripción:*\n\n{text[:3500]}", parse_mode="Markdown"
        )
        if text.strip():
            try:
                reply = await ask_llm_async(
                    f"El usuario envió este audio transcrito: «{text}». "
                    f"Responde de forma útil y breve en español."
                )
                await update.effective_message.reply_text(reply[:3800])
            except Exception as e:
                log.info(f"respuesta LLM tras audio falló: {e}")
    except Exception as e:
        log.exception("audio error")
        await waiting.edit_text(f"Error al transcribir: {e}")

# --- Multimedia: FOTO -> BLIP + LLM ---
async def photo_handler(update: Update, _: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    waiting = await update.effective_message.reply_text("🖼️ Analizando imagen…")
    try:
        photo = update.effective_message.photo[-1]  # mayor resolución
        tg_file = await update.effective_message.get_bot().get_file(photo.file_id)
        buf = io.BytesIO()
        await tg_file.download_to_memory(buf)
        caption = await describe_image_bytes(buf.getvalue())
        user_q = update.effective_message.caption or "Describe la imagen en español."
        llm = await ask_llm_async(
            f"Descripción corta de la imagen (BLIP): «{caption}». "
            f"Usuario pregunta: {user_q}\nResponde en español, breve y útil."
        )
        await waiting.edit_text(
            f"🖼️ *Imagen:*\n{caption}\n\n🧠 *LLM:*\n{llm[:3500]}",
            parse_mode="Markdown",
        )
    except Exception as e:
        log.exception("photo error")
        await waiting.edit_text(f"Error al analizar imagen: {e}")

# --- Clima (Open-Meteo: gratis, sin API key) ---
WMO_ES = {
    0:"Despejado", 1:"Mayormente despejado", 2:"Parcialmente nublado", 3:"Nublado",
    45:"Neblina", 48:"Neblina con escarcha",
    51:"Llovizna ligera", 53:"Llovizna", 55:"Llovizna intensa",
    61:"Lluvia ligera", 63:"Lluvia", 65:"Lluvia fuerte",
    71:"Nieve ligera", 73:"Nieve", 75:"Nieve fuerte",
    80:"Chubascos ligeros", 81:"Chubascos", 82:"Chubascos fuertes",
    95:"Tormenta", 96:"Tormenta con granizo", 99:"Tormenta fuerte con granizo",
}

async def clima_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    city = " ".join(ctx.args).strip()
    if not city:
        return await update.effective_message.reply_text("Uso: /clima <ciudad>")
    url = "https://geocoding-api.open-meteo.com/v1/search?count=1&language=es&format=json"
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{url}&name={quote_plus(city)}") as r:
            geo = await r.json()
    if not geo.get("results"):
        return await update.effective_message.reply_text("No encontré esa ciudad.")
    g = geo["results"][0]
    lat, lon, name, country = g["latitude"], g["longitude"], g["name"], g.get("country","")
    async with aiohttp.ClientSession() as s:
        async with s.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
            f"wind_speed_10m,weather_code,is_day"
        ) as r:
            wx = await r.json()
    c = wx["current"]
    desc = WMO_ES.get(c["weather_code"], f"Código {c['weather_code']}")
    await update.effective_message.reply_text(
        f"🌤️ *{name}, {country}*\n"
        f"Estado: {desc}\n"
        f"Temperatura: {c['temperature_2m']}°C (sensación {c['apparent_temperature']}°C)\n"
        f"Humedad: {c['relative_humidity_2m']}%\n"
        f"Viento: {c['wind_speed_10m']} km/h",
        parse_mode="Markdown",
    )

# --- Resumen de artículo ---
ARTICLE_PROMPT = (
    "Resume en español el siguiente artículo en 5 viñetas claras y luego en 2 "
    "oraciones un párrafo final con la idea clave. Si el texto no parece un artículo, "
    "responde 'No pude identificar el contenido'.\n\n---ARTÍCULO---\n{body}"
)

async def resume_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    if not ctx.args:
        return await update.effective_message.reply_text("Uso: /resume <url>")

    url = ctx.args[0]
    waiting = await update.effective_message.reply_text("📥 Extrayendo y resumiendo…")
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
            async with s.get(url, headers={"User-Agent": "Mozilla/5.0"}) as r:
                html = await r.text(errors="ignore")

        import re
        m = re.search(r"<body[^>]*>(.*?)</body>", html, re.S | re.I)
        body = m.group(1) if m else html
        body = re.sub(r"<script[\s\S]*?</script>", " ", body, flags=re.I)
        body = re.sub(r"<style[\s\S]*?</style>",  " ", body, flags=re.I)
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        body = body[:6000]

        if not body:
            return await waiting.edit_text("No pude extraer texto de esa URL.")

        summary = await ask_llm_async(ARTICLE_PROMPT.format(body=body))
        await waiting.edit_text(
            f"📰 *Resumen:*\n\n{summary[:3800]}", parse_mode="Markdown"
        )
    except Exception as e:
        log.exception("resume error")
        await waiting.edit_text(f"Error al resumir: {e}")

# --- Chat libre (cualquier texto sin /) ---
async def chat_handler(update: Update, _: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    text = update.effective_message.text or ""
    if not text.strip():
        return
    model = _load_model()
    try:
        reply = await ask_llm_async(text)
        if len(reply) > 3800:
            reply = reply[:3800] + "…"
        await update.effective_message.reply_text(reply)
    except Exception as e:
        log.exception("llm error")
        await update.effective_message.reply_text(
            f"Error con el modelo `{model}`:\n{e}\n\n"
            f"¿Quieres probar otro? /modelo"
        )

# --- Arranque ---
async def post_init(app: Application):
    init_db()
    log.info(f"Bot listo. Modelo: {_load_model()}")

def main():
    if not TELEGRAM_TOKEN or not HF_TOKEN:
        log.error(
            "Faltan variables: TELEGRAM_TOKEN y/o HF_TOKEN. "
            "Define HF_TOKEN en Render (Settings -> Environment)."
        )
        raise SystemExit(1)

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    for cmd, fn in [
        ("start",   start_cmd),
        ("help",    help_cmd),
        ("ask",     chat_handler),
        ("nota",    nota_cmd),
        ("notas",   notas_cmd),
        ("delnote", delnote_cmd),
        ("todo",    todo_cmd),
        ("todos",   todos_cmd),
        ("done",    done_cmd),
        ("drop",    drop_cmd),
        ("remind",  remind_cmd),
        ("remindat",remindat_cmd),
        ("clima",   clima_cmd),
        ("resume",  resume_cmd),
        ("modelo",  modelo_cmd),
        ("voz",     voz_cmd),
    ]:
        app.add_handler(CommandHandler(cmd, fn))

    from telegram.ext import CallbackQueryHandler
    app.add_handler(CallbackQueryHandler(model_cb,    pattern=r"^setmodel\|"))
    app.add_handler(CallbackQueryHandler(tts_note_cb, pattern=r"^ttsnote\|\d+$"))

    # Multimedia
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, audio_handler))
    # Texto -> chat libre
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))

    log.info("Iniciando polling")
    # Python 3.14: crea loop explícito antes de run_polling()
    try:
        asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError:
        pass
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
