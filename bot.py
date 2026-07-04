"""
Asistente personal en Telegram con Grok.
Comandos:
  /start, /help   -> ayuda
  /ask <texto>     -> pregunta libre a Grok
  /nota <texto>    -> guarda una nota
  /notas           -> lista tus notas
  /delnote <id>    -> borra una nota por ID
  /todo <texto>    -> agrega tarea
  /todos           -> lista tus tareas
  /done <id>       -> marca tarea completada
  /drop <id>       -> borra una tarea
  /remind <min> <texto>  -> recordatorio en N minutos
  /remindat HH:MM <texto> -> recordatorio a una hora (hoy)
  /clima <ciudad>  -> clima actual
  /resume <url>     -> resumen de un artículo

Cualquier mensaje que NO sea un comando pasa a Grok como chat libre.
"""
import asyncio
import json
import os
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote_plus

import aiohttp
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters,
)

# --- Configuración ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0")) or None  # 0/None = cualquiera

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "assistant.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("assistant")

# --- Base de datos (notas y tareas) ---
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

# --- Cliente Grok (compatible con OpenAI) ---
def grok_client():
    return OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
    )

SYSTEM_PROMPT = (
    "Eres un asistente personal útil y conciso. Responde en español a menos "
    "que el usuario escriba en otro idioma. Sé breve cuando sea posible. "
    "Si no sabes algo, dilo. Puedes resumir información cuando el usuario lo pida."
)

async def ask_grok(prompt: str) -> str:
    # run blocking SDK in a thread
    def _call():
        c = grok_client()
        r = c.chat.completions.create(
            model="grok-3-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.4,
        )
        return r.choices[0].message.content.strip()
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
    await update.effective_message.reply_text(
        "👋 Soy tu asistente personal. Escribe /help para ver lo que puedo hacer.\n"
        "Puedes escribirme cualquier cosa y la respondo con Grok."
    )

async def help_cmd(update: Update, _: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    await update.effective_message.reply_text(
        "🧾 *Comandos disponibles:*\n\n"
        "• /ask _pregunta_ — pregunta libre a Grok\n"
        "• /nota _texto_ — guardar nota\n"
        "• /notas — listar notas\n"
        "• /delnote _id_ — borrar nota\n"
        "• /todo _texto_ — agregar tarea\n"
        "• /todos — listar tus tareas\n"
        "• /done _id_ — marcar tarea hecha\n"
        "• /drop _id_ — borrar tarea\n"
        "• /remind _minutos_ _texto_ — recordatorio en N min\n"
        "• /remindat _HH:MM_ _texto_ — recordatorio a las HH:MM (hoy)\n"
        "• /clima _ciudad_ — clima actual\n"
        "• /resume _url_ — resumen de un artículo\n\n"
        "Lo que escribas sin '/' lo tomo como chat normal con Grok.",
        parse_mode="Markdown",
    )

# --- Notas ---
async def nota_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    text = " ".join(ctx.args).strip()
    if not text:
        return await update.effective_message.reply_text("Uso: /nota <texto>")
    uid = update.effective_user.id
    with db() as c:
        cur = c.execute("INSERT INTO notes(user_id,text) VALUES(?,?)", (uid, text))
        nid = cur.lastrowid
    await update.effective_message.reply_text(f"📝 Nota #{nid} guardada.")

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
    lines = []
    for r in rows:
        mark = "✅" if r["done"] else "◻️"
        lines.append(f"{mark} #{r['id']} — {r['text']}")
    await update.effective_message.reply_text("\n".join(lines))

async def done_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    if not ctx.args or not ctx.args[0].isdigit():
        return await update.effective_message.reply_text("Uso: /done <id>")
    tid = int(ctx.args[0])
    uid = update.effective_user.id
    with db() as c:
        cur = c.execute(
            "UPDATE todos SET done=1 WHERE id=? AND user_id=?",
            (tid, uid),
        )
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

# --- Recordatorios ---
async def remind_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    if not ctx.args or not ctx.args[0].isdigit():
        return await update.effective_message.reply_text(
            "Uso: /remind <minutos> <texto>"
        )
    minutes = int(ctx.args[0])
    text = " ".join(ctx.args[1:]).strip()
    if not text:
        return await update.effective_message.reply_text("Falta el texto del recordatorio.")
    when = datetime.now() + timedelta(minutes=minutes)
    await schedule_reminder(ctx, update.effective_chat.id, text, when)
    await update.effective_message.reply_text(
        f"⏰ Te recordaré en {minutes} min: '{text}'."
    )

async def remindat_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    if not ctx.args:
        return await update.effective_message.reply_text(
            "Uso: /remindat HH:MM <texto>"
        )
    try:
        hh, mm = ctx.args[0].split(":")
        when = datetime.now().replace(
            hour=int(hh), minute=int(mm), second=0, microsecond=0
        )
    except Exception:
        return await update.effective_message.reply_text("Hora inválida. Usa HH:MM, ej. 20:30")
    if when < datetime.now():
        return await update.effective_message.reply_text("Esa hora ya pasó hoy.")
    text = " ".join(ctx.args[1:]).strip()
    if not text:
        return await update.effective_message.reply_text("Falta el texto del recordatorio.")
    await schedule_reminder(ctx, update.effective_chat.id, text, when)
    await update.effective_message.reply_text(
        f"⏰ Te recordaré a las {hh}:{mm}: '{text}'."
    )

async def schedule_reminder(ctx, chat_id, text, when_dt):
    delay = max(0, (when_dt - datetime.now()).total_seconds())
    job = ctx.job_queue.run_once(
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

# --- Clima (Open-Meteo: gratis sin API key) ---
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
            f"&current=temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,"
            f"weather_code,is_day"
        ) as r:
            wx = await r.json()
    c = wx["current"]
    code = c["weather_code"]
    desc = WMO_ES.get(code, f"Código {code}")
    await update.effective_message.reply_text(
        f"🌤️ *{name}, {country}*\n"
        f"Estado: {desc}\n"
        f"Temperatura: {c['temperature_2m']}°C (sensación {c['apparent_temperature']}°C)\n"
        f"Humedad: {c['relative_humidity_2m']}%\n"
        f"Viento: {c['wind_speed_10m']} km/h",
        parse_mode="Markdown",
    )

# --- Resumen de artículo (extracción + Grok) ---
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

        # extracción muy simple: corta entre <body ...> y </body>, luego quita tags
        import re
        m = re.search(r"<body[^>]*>(.*?)</body>", html, re.S | re.I)
        body = m.group(1) if m else html
        body = re.sub(r"<script[\s\S]*?</script>", " ", body, flags=re.I)
        body = re.sub(r"<style[\s\S]*?</style>",  " ", body, flags=re.I)
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()
        body = body[:6000]  # límite razonable

        if not body:
            return await waiting.edit_text("No pude extraer texto de esa URL.")

        summary = await ask_grok(ARTICLE_PROMPT.format(body=body))
        # límite de Telegram = 4096
        await waiting.edit_text(f"📰 *Resumen:*\n\n{summary[:3800]}")
    except Exception as e:
        log.exception("resume error")
        await waiting.edit_text(f"Error al resumir: {e}")

# --- Chat libre con Grok ---
async def chat_handler(update: Update, _: ContextTypes.DEFAULT_TYPE):
    if not authorized(update): return await deny(update)
    text = update.effective_message.text or ""
    if not text.strip():
        return
    try:
        reply = await ask_grok(text)
        if len(reply) > 4000:
            reply = reply[:4000] + "…"
        await update.effective_message.reply_text(reply)
    except Exception as e:
        log.exception("ask_grok error")
        await update.effective_message.reply_text(f"Error con Grok: {e}")

async def post_init(app: Application):
    init_db()
    log.info("Bot listo.")

def main():
    if not TELEGRAM_TOKEN or not XAI_API_KEY:
        log.error(
            "Faltan variables de entorno: TELEGRAM_TOKEN y/o XAI_API_KEY. "
            "Defínelas antes de iniciar."
        )
        raise SystemExit(1)

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start",    start_cmd))
    app.add_handler(CommandHandler("help",     help_cmd))
    app.add_handler(CommandHandler("ask",      chat_handler))  # /ask o chat libre
    app.add_handler(CommandHandler("nota",     nota_cmd))
    app.add_handler(CommandHandler("notas",    notas_cmd))
    app.add_handler(CommandHandler("delnote",  delnote_cmd))
    app.add_handler(CommandHandler("todo",     todo_cmd))
    app.add_handler(CommandHandler("todos",    todos_cmd))
    app.add_handler(CommandHandler("done",     done_cmd))
    app.add_handler(CommandHandler("drop",     drop_cmd))
    app.add_handler(CommandHandler("remind",   remind_cmd))
    app.add_handler(CommandHandler("remindat", remindat_cmd))
    app.add_handler(CommandHandler("clima",    clima_cmd))
    app.add_handler(CommandHandler("resume",   resume_cmd))

    # chat libre (todo lo que NO sea comando)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))

    log.info("Iniciando polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
