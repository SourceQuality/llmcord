import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord import app_commands

from discord.ext import commands
import httpx
from openai import AsyncOpenAI
import yaml
import sqlite3

MEM_DB = "/opt/llmcord/db/memory.db"

def ensure_memory_table():
    with sqlite3.connect(MEM_DB) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER,
                sentence TEXT
)
        """)
        conn.commit()


def get_server_memories(guild_id: Optional[int]) -> str:
    try:
        with sqlite3.connect(MEM_DB) as conn:
            c = conn.cursor()
            c.execute("SELECT sentence FROM memories WHERE guild_id = ?", (guild_id,))
            rows = c.fetchall()
        return "\n".join(row[0] for row in rows)
    except Exception as e:
        logging.warning(f"Error fetching server memories: {e}")
        return ""




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " 🤔 Thinking..."
EDIT_DELAY_SECONDS = .2

MAX_MESSAGE_NODES = 500


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


config = get_config()
ensure_memory_table()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0
stop_flags = {}
active_streams = set()

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)
memory_group = app_commands.Group(name="memory", description="Manage your memories")
discord_bot.tree.add_command(memory_group)

httpx_client = httpx.AsyncClient()


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()][:24]
    choices += [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []

    return choices

@memory_group.command(name="add", description="Add a memory for this server.")
@app_commands.describe(sentence="What should the bot remember?")
async def memory_add(interaction: discord.Interaction, sentence: str):
    guild_id = interaction.guild.id if interaction.guild else None

    await interaction.response.defer(ephemeral=True)

    try:
        with sqlite3.connect(MEM_DB) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT OR IGNORE INTO memories (guild_id, sentence) VALUES (?, ?)",
                (guild_id, sentence),
            )
            conn.commit()

        await interaction.followup.send(f"✅ Memory added.")
        await interaction.channel.send(f"🧠 <@{interaction.user.id}> added: *{sentence}*")
    except Exception as e:
        logging.exception("Failed to add memory")
        await interaction.followup.send("❌ Failed to save memory.")


async def memory_autocomplete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    guild_id = interaction.guild.id if interaction.guild else None
    try:
        with sqlite3.connect(MEM_DB) as conn:
            c = conn.cursor()
            c.execute("SELECT id, sentence FROM memories WHERE guild_id = ?", (guild_id,))
            rows = c.fetchall()

        return [
            app_commands.Choice(name=sentence[:100], value=str(mem_id))  # 👈 value must be the ID
            for mem_id, sentence in rows if current.lower() in sentence.lower()
        ][:25]
    except Exception as e:
        logging.warning(f"Failed to autocomplete memory remove: {e}")
        return []




@memory_group.command(name="remove", description="Remove a memory sentence.")
@app_commands.describe(memory_id="The memory to remove")
@app_commands.autocomplete(memory_id=memory_autocomplete)
async def memory_remove(interaction: discord.Interaction, memory_id: str):
    guild_id = interaction.guild.id if interaction.guild else None

    if not interaction.response.is_done():
        await interaction.response.defer(ephemeral=True)

    try:
        with sqlite3.connect(MEM_DB) as conn:
            c = conn.cursor()
            c.execute("SELECT sentence FROM memories WHERE id = ? AND guild_id = ?", (memory_id, guild_id))
            row = c.fetchone()

            if not row:
                await interaction.followup.send("⚠️ That memory doesn't exist.", ephemeral=True)
                return

            sentence = row[0]
            c.execute("DELETE FROM memories WHERE id = ? AND guild_id = ?", (memory_id, guild_id))
            conn.commit()

        await interaction.followup.send("✅ Memory removed.", ephemeral=True)
        await interaction.channel.send(f"🧠 <@{interaction.user.id}> removed: *{sentence}*")

    except Exception as e:
        logging.exception("Failed to remove memory")
        await interaction.followup.send("❌ Failed to remove memory.", ephemeral=True)





@memory_group.command(name="list", description="List server memories.")
async def memory_list(interaction: discord.Interaction):
    guild_id = interaction.guild.id if interaction.guild else None

    await interaction.response.defer(ephemeral=True) 

    try:
        with sqlite3.connect(MEM_DB) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT sentence FROM memories WHERE guild_id = ?",
                (guild_id,),
            )
            rows = c.fetchall()

        if rows:
            sentences = "\n".join(f"- {row[0]}" for row in rows)
            await interaction.followup.send(f"🧠 Server memories:\n{sentences}")
        else:
            await interaction.followup.send("🧠 No memories set yet.")
    except Exception as e:
        logging.exception("Failed to list memories")
        await interaction.followup.send("❌ Failed to list memories.")
 

@discord_bot.tree.command(name="stop", description="Stop the bot's current response in this channel.")
async def stop_command(interaction: discord.Interaction):
    channel_id = interaction.channel.id
    if channel_id not in active_streams:
        await interaction.response.send_message("🤷 No active response to stop in this channel.", ephemeral=True)
        return

    stop_flags[channel_id] = True
    await interaction.response.send_message("🛑 Stopping response...", ephemeral=True)


@discord_bot.event
async def on_ready() -> None:
    if client_id := config["client_id"]:
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.split("/", 1)
    model_parameters = config["models"].get(provider_slash_model, None)

    base_url = config["providers"][provider]["base_url"]
    api_key = config["providers"][provider].get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := config["system_prompt"]:
        now = datetime.now().astimezone()
        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
    
        # Inject per-user/server memory
        memory_text = get_server_memories(new_msg.guild.id if new_msg.guild else None)
        if memory_text:
            system_prompt += f"\n\n# Server memory:\n{memory_text}"

    
        if accept_usernames:
            system_prompt += "\nUser's names are their Discord IDs and should be typed as '<@ID>'."
    
        logging.info(f"Final system prompt:\n{system_prompt}") 
    
        messages.insert(0, dict(role="system", content=system_prompt))  



    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = edit_task = None
    response_msgs = []
    response_contents = []

    use_plain_responses = config.get("use_plain_responses", False)
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    # Safety limits to prevent endless streaming
    max_stream_seconds = config.get("max_stream_seconds", 120)
    max_stream_idle_seconds = config.get("max_stream_idle_seconds", 120)
    max_total_response_chars = config.get("max_total_response_chars", 12300)

    # use monotonic clock for elapsed-time checks (faster and immune to system clock changes)
    stream_start_ts = time.monotonic()
    last_progress_ts = stream_start_ts
    total_response_chars = 0
    channel_id = new_msg.channel.id
    active_streams.add(channel_id)
    stop_flags.pop(channel_id, None)

    try:
        async with new_msg.channel.typing():
            # Send initial message with warnings if any
            if user_warnings and not use_plain_responses:
                embed = discord.Embed()
                for warning in sorted(user_warnings):
                    embed.add_field(name=warning, value="", inline=False)
                embed.color = EMBED_COLOR_INCOMPLETE
                response_msg = await new_msg.reply(embed=embed, silent=True)
                response_msgs.append(response_msg)
                msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                await msg_nodes[response_msg.id].lock.acquire()

            async for curr_chunk in await openai_client.chat.completions.create(model=model, messages=messages[::-1], stream=True, extra_body=model_parameters):
                if stop_flags.get(channel_id):
                    user_warnings.add("⚠️ Response stopped by user.")
                    break
                if finish_reason != None:
                    break

                if not (choice := curr_chunk.choices[0] if curr_chunk.choices else None):
                    continue

                # read monotonic once per iteration
                now_ts = time.monotonic()

                finish_reason = choice.finish_reason

                # Accumulate streaming content correctly; don't drop the first characters
                delta = choice.delta.content or ""
                logging.debug(f"API response chunk: {delta}")

                # Skip empty progress unless it's the final packet
                if response_contents == [] and delta == "" and finish_reason is None:
                    continue

                # Ensure we have a buffer to write into
                if response_contents == []:
                    response_contents.append("")

                # If adding this delta would overflow the current message, finalize it and start a new one
                if not use_plain_responses and len(response_contents[-1]) + len(delta) > max_message_length:
                    if response_msgs:
                        try:
                            finalize_embed = discord.Embed()
                            for warning in sorted(user_warnings):
                                finalize_embed.add_field(name=warning, value="", inline=False)
                            finalize_embed.description = response_contents[-1]
                            finalize_embed.color = EMBED_COLOR_COMPLETE
                            if edit_task is not None:
                                await edit_task
                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=finalize_embed))
                            await edit_task
                        except Exception:
                            logging.exception("Failed to finalize embed during overflow")

                    # Start a new buffer and seed it with the current delta
                    response_contents.append("")
                    if delta:
                        response_contents[-1] += delta

                    # Send a new message to continue streaming
                    new_embed = discord.Embed()
                    for warning in sorted(user_warnings):
                        new_embed.add_field(name=warning, value="", inline=False)
                    new_embed.description = response_contents[-1] + STREAMING_INDICATOR
                    new_embed.color = EMBED_COLOR_INCOMPLETE
                    reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                    response_msg = await reply_to_msg.reply(embed=new_embed, silent=True)
                    response_msgs.append(response_msg)
                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                    await msg_nodes[response_msg.id].lock.acquire()
                    last_task_time = now_ts
                else:
                    # Normal accumulation into current buffer
                    response_contents[-1] += delta

                # Track progress for safety checks
                added_len = len(delta)
                if added_len > 0:
                    total_response_chars += added_len
                    last_progress_ts = now_ts

                # Enforce limits: if tripped, force a graceful finish
                hit_time_limit = (now_ts - stream_start_ts) > max_stream_seconds
                hit_idle_limit = (now_ts - last_progress_ts) > max_stream_idle_seconds
                hit_char_limit = total_response_chars >= max_total_response_chars
                if hit_time_limit or hit_idle_limit or hit_char_limit:
                    if hit_time_limit:
                        user_warnings.add(f"⚠️ Stream timed out after {max_stream_seconds}s")
                    if hit_idle_limit:
                        user_warnings.add("⚠️ Stream ended due to inactivity")
                    if hit_char_limit:
                        user_warnings.add(f"⚠️ Max response length reached ({max_total_response_chars} chars)")
                    if finish_reason is None:
                        finish_reason = "length"  # trigger finalization below

                if not use_plain_responses:
                    # Re-create embed to ensure all warnings are present
                    embed = discord.Embed()
                    for warning in sorted(user_warnings):
                        embed.add_field(name=warning, value="", inline=False)

                    ready_to_edit = (edit_task == None or edit_task.done()) and (now_ts - last_task_time) >= EDIT_DELAY_SECONDS
                    is_final_edit = finish_reason != None
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    # Send first message or update existing one
                    if (not response_msgs) or ready_to_edit or is_final_edit:
                        if edit_task != None:
                            await edit_task

                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if is_good_finish else EMBED_COLOR_INCOMPLETE

                        if not response_msgs:
                            reply_to_msg = new_msg
                            response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                            response_msgs.append(response_msg)
                            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()
                        else:
                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))

                        last_task_time = now_ts

            if use_plain_responses:
                for content in response_contents:
                    reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                    response_msg = await reply_to_msg.reply(content=content, suppress_embeds=True)
                    response_msgs.append(response_msg)

                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                    await msg_nodes[response_msg.id].lock.acquire()
            else:
                # Finalize last embed (remove indicator, turn green)
                if response_msgs:
                    try:
                        final_embed = discord.Embed()
                        for warning in sorted(user_warnings):
                            final_embed.add_field(name=warning, value="", inline=False)
                        final_embed.description = response_contents[-1] if response_contents else ""
                        final_embed.color = EMBED_COLOR_COMPLETE
                        if edit_task is not None:
                            await edit_task
                        await response_msgs[-1].edit(embed=final_embed)
                    except Exception:
                        logging.exception("Failed to finalize embed after stream end")

    except Exception:
        logging.exception("Error while generating response")
    finally:
        active_streams.discard(channel_id)
        stop_flags.pop(channel_id, None)


    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass