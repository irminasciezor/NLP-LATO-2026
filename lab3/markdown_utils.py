def escape_md(text: str) -> str:
    for ch in ['_', '*', '`', '[']:
        text = text.replace(ch, f'\\{ch}')
    return str(text)


def safe_send(bot, chat_id: int, text: str, parse_mode: str = "Markdown"):
    try:
        bot.send_message(chat_id, text, parse_mode=parse_mode)
    except Exception:
        clean = text.replace('*', '').replace('`', '').replace('_', '')
        bot.send_message(chat_id, clean)


def safe_reply(bot, message, text: str, parse_mode: str = "Markdown"):
    try:
        bot.reply_to(message, text, parse_mode=parse_mode)
    except Exception:
        clean = text.replace('*', '').replace('`', '').replace('_', '')
        bot.reply_to(message, clean)