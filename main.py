import io
import base64
import random
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import requests
import gradio as gr
from PIL import Image
from gradio_i18n import Translate, gettext as _

from grok_client import GrokClient
from i18n import get_text

# 配置
DB_PATH = Path(__file__).parent / "history.db"
MODEL_SERVER_URL = "http://127.0.0.1:8000"
TRANSLATIONS_PATH = Path(__file__).parent / "i18n" / "translations.yaml"

history_detail = []  # for history tab: [(image, prompt, seed, width, height, steps), ...]
grok_client = GrokClient()
current_lang = "zh"  # Default language


def init_db():
    """初始化数据库"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_data BLOB NOT NULL,
            prompt TEXT,
            seed INTEGER,
            width INTEGER,
            height INTEGER,
            steps INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def load_history():
    """从数据库加载历史记录"""
    global history_detail
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT image_data, prompt, seed, width, height, steps FROM images ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    history_detail = []
    for row in rows:
        img = Image.open(io.BytesIO(row[0]))
        history_detail.append({
            "image": img,
            "prompt": row[1] or "",
            "seed": row[2],
            "width": row[3],
            "height": row[4],
            "steps": row[5],
        })


def save_to_db(image, prompt, seed, width, height, steps):
    """保存图片到数据库"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_data = buffer.getvalue()

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO images (image_data, prompt, seed, width, height, steps) VALUES (?, ?, ?, ?, ?, ?)",
        (image_data, prompt, seed, width, height, steps)
    )
    conn.commit()
    conn.close()


def delete_from_db_by_index(index: int):
    """根据索引删除单张图片（索引对应 history_detail 的顺序）"""
    global history_detail
    if index < 0 or index >= len(history_detail):
        return False

    conn = sqlite3.connect(DB_PATH)
    # history_detail 是按 created_at DESC 排序的，获取所有 id
    rows = conn.execute("SELECT id FROM images ORDER BY created_at DESC").fetchall()
    if index < len(rows):
        image_id = rows[index][0]
        conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
        conn.commit()
    conn.close()

    # 从内存中删除
    history_detail.pop(index)
    return True


def clear_all_history():
    """清空所有历史记录"""
    global history_detail
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM images")
    conn.commit()
    conn.close()
    history_detail = []


RESOLUTION_OPTIONS = {
    "512": {
        "512x512 (1:1)": (512, 512),
        "640x384 (5:3)": (640, 384),
        "384x640 (3:5)": (384, 640),
        "512x384 (4:3)": (512, 384),
        "384x512 (3:4)": (384, 512),
        "640x368 (16:9)": (640, 368),
        "368x640 (9:16)": (368, 640),
    },
    "768": {
        "768x768 (1:1)": (768, 768),
        "960x576 (5:3)": (960, 576),
        "576x960 (3:5)": (576, 960),
        "768x576 (4:3)": (768, 576),
        "576x768 (3:4)": (576, 768),
        "960x544 (16:9)": (960, 544),
        "544x960 (9:16)": (544, 960),
    },
    "1024": {
        "1024x1024 (1:1)": (1024, 1024),
        "1280x768 (5:3)": (1280, 768),
        "768x1280 (3:5)": (768, 1280),
        "1024x768 (4:3)": (1024, 768),
        "768x1024 (3:4)": (768, 1024),
        "1280x720 (16:9)": (1280, 720),
        "720x1280 (9:16)": (720, 1280),
        "1024x576 (16:9)": (1024, 576),
        "576x1024 (9:16)": (576, 1024),
    },
    "1280": {
        "1280x1280 (1:1)": (1280, 1280),
        "1536x1024 (3:2)": (1536, 1024),
        "1024x1536 (2:3)": (1024, 1536),
        "1536x864 (16:9)": (1536, 864),
        "864x1536 (9:16)": (864, 1536),
    },
}

EXAMPLE_PROMPTS = [
    "一位男士和他的贵宾犬穿着配套的服装参加狗狗秀，室内灯光，背景中有观众。",
    "极具氛围感的暗调人像，一位优雅的中国美女在黑暗的房间里。一束强光通过遮光板，在她的脸上投射出一个清晰的闪电形状的光影，正好照亮一只眼睛。高对比度，明暗交界清晰，神秘感，莱卡相机色调。",
    "一张中景手机自拍照片拍摄了一位留着长黑发的年轻东亚女子在灯光明亮的电梯内对着镜子自拍。她穿着一件带有白色花朵图案的黑色露肩短上衣和深色牛仔裤。她的头微微倾斜，嘴唇嘟起做亲吻状，非常可爱俏皮。她右手拿着一部深灰色智能手机，遮住了部分脸，后置摄像头镜头对着镜子",
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.",
    """
    A vertical digital illustration depicting a serene and majestic Chinese landscape, rendered in a style reminiscent of traditional Shanshui painting but with a modern, clean aesthetic. The scene is dominated by towering, steep cliffs in various shades of blue and teal, which frame a central valley. In the distance, layers of mountains fade into a light blue and white mist, creating a strong sense of atmospheric perspective and depth. A calm, turquoise river flows through the center of the composition, with a small, traditional Chinese boat, possibly a sampan, navigating its waters. The boat has a bright yellow canopy and a red hull, and it leaves a gentle wake behind it. It carries several indistinct figures of people. Sparse vegetation, including green trees and some bare-branched trees, clings to the rocky ledges and peaks. The overall lighting is soft and diffused, casting a tranquil glow over the entire scene. Centered in the image is overlaid text. At the top of the text block is a small, red, circular seal-like logo containing stylized characters. Below it, in a smaller, black, sans-serif font, are the words 'Zao-Xiang * East Beauty & West Fashion * Z-Image'. Directly beneath this, in a larger, elegant black serif font, is the word 'SHOW & SHARE CREATIVITY WITH THE WORLD'. Among them, there are "SHOW & SHARE", "CREATIVITY", and "WITH THE WORLD"
    """,
    '一张虚构的英语电影《回忆之味》（The Taste of Memory）的电影海报。场景设置在一个质朴的19世纪风格厨房里。画面中央，一位红棕色头发、留着小胡子的中年男子（演员阿瑟·彭哈利根饰）站在一张木桌后，他身穿白色衬衫、黑色马甲和米色围裙，正看着一位女士，手中拿着一大块生红肉，下方是一个木制切菜板。在他的右边，一位梳着高髻的黑发女子（演员埃莉诺·万斯饰）倚靠在桌子上，温柔地对他微笑。她穿着浅色衬衫和一条上白下蓝的长裙。桌上除了放有切碎的葱和卷心菜丝的切菜板外，还有一个白色陶瓷盘、新鲜香草，左侧一个木箱上放着一串深色葡萄。背景是一面粗糙的灰白色抹灰墙，墙上挂着一幅风景画。最右边的一个台面上放着一盏复古油灯。海报上有大量的文字信息。左上角是白色的无衬线字体"ARTISAN FILMS PRESENTS"，其下方是"ELEANOR VANCE"和"ACADEMY AWARD® WINNER"。右上角写着"ARTHUR PENHALIGON"和"GOLDEN GLOBE® AWARD WINNER"。顶部中央是圣丹斯电影节的桂冠标志，下方写着"SUNDANCE FILM FESTIVAL GRAND JURY PRIZE 2024"。主标题"THE TASTE OF MEMORY"以白色的大号衬线字体醒目地显示在下半部分。标题下方注明了"A FILM BY Tongyi Interaction Lab"。底部区域用白色小字列出了完整的演职员名单，包括"SCREENPLAY BY ANNA REID"、"CULINARY DIRECTION BY JAMES CARTER"以及Artisan Films、Riverstone Pictures和Heritage Media等众多出品公司标志。整体风格是写实主义，采用温暖柔和的灯光方案，营造出一种亲密的氛围。色调以棕色、米色和柔和的绿色等大地色系为主。两位演员的身体都在腰部被截断。',
    '一张方形构图的特写照片，主体是一片巨大的、鲜绿色的植物叶片，并叠加了文字，使其具有海报或杂志封面的外观。主要拍摄对象是一片厚实、有蜡质感的叶子，从左下角到右上角呈对角线弯曲穿过画面。其表面反光性很强，捕捉到一个明亮的直射光源，形成了一道突出的高光，亮面下显露出平行的精细叶脉。背景由其他深绿色的叶子组成，这些叶子轻微失焦，营造出浅景深效果，突出了前景的主叶片。整体风格是写实摄影，明亮的叶片与黑暗的阴影背景之间形成高对比度。图像上有多处渲染文字。左上角是白色的衬线字体文字"PIXEL-PEEPERS GUILD Presents"。右上角同样是白色衬线字体的文字"[Instant Noodle] 泡面调料包"。左侧垂直排列着标题"Render Distance: Max"，为白色衬线字体。左下角是五个硕大的白色宋体汉字"显卡在...燃烧"。右下角是较小的白色衬线字体文字"Leica Glow™ Unobtanium X-1"，其正上方是用白色宋体字书写的名字"蔡几"。识别出的核心实体包括品牌像素偷窥者协会、其产品线泡面调料包、相机型号买不到™ X-1以及摄影师名字造相。',
]


def update_resolution_choices(category):
    choices = list(RESOLUTION_OPTIONS.get(category, {}).keys())
    return gr.update(choices=choices, value=choices[0] if choices else None)


def generate_image_internal(prompt, res_category, resolution, seed, random_seed, steps):
    """内部图片生成函数"""
    if random_seed:
        seed = random.randint(0, 2147483647)
    else:
        seed = int(seed)

    width, height = RESOLUTION_OPTIONS.get(res_category, {}).get(resolution, (1024, 1024))

    try:
        resp = requests.post(
            f"{MODEL_SERVER_URL}/generate",
            json={
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": int(steps),
                "seed": seed,
            },
            timeout=300,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise gr.Error(get_text("error_model_not_started", current_lang))
    except requests.exceptions.RequestException as e:
        raise gr.Error(get_text("error_request_failed", current_lang, error=str(e)))

    data = resp.json()
    image_bytes = base64.b64decode(data["image_base64"])
    image = Image.open(io.BytesIO(image_bytes))

    save_to_db(image, prompt, seed, width, height, int(steps))

    history_detail.insert(0, {
        "image": image,
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": int(steps),
    })

    return image, str(seed)


def chat_and_generate(user_message, chat_history, res_category, resolution, seed, random_seed, steps):
    """流式输出 Grok 回复，完成后自动生成图片"""
    if not user_message.strip():
        yield chat_history, None, "", user_message
        return

    if chat_history is None:
        chat_history = []

    # 添加用户消息
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": ""})

    # 流式输出 Grok 回复
    generated_prompt = ""
    try:
        for chunk in grok_client.chat_stream(user_message):
            generated_prompt += chunk
            chat_history[-1]["content"] = generated_prompt
            yield chat_history, None, get_text("status_generating_prompt", current_lang), ""
    except Exception as e:
        chat_history[-1]["content"] = get_text("error_grok_api", current_lang, error=str(e))
        yield chat_history, None, "", ""
        return

    # 流式输出完成，开始生成图片
    yield chat_history, None, get_text("status_generating_image", current_lang), ""

    try:
        image, seed_used = generate_image_internal(
            generated_prompt, res_category, resolution, seed, random_seed, steps
        )
    except gr.Error as e:
        yield chat_history, None, str(e), ""
        return

    yield chat_history, image, seed_used, ""


def clear_chat():
    """清空对话历史"""
    grok_client.clear_history()
    return [], None, ""


def regenerate_after_edit(chat_history, res_category, resolution, seed, random_seed, steps):
    """编辑消息后重新生成"""
    if not chat_history:
        yield chat_history, None, ""
        return

    # 找到最后一条用户消息
    last_user_idx = None
    for i in range(len(chat_history) - 1, -1, -1):
        if chat_history[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        yield chat_history, None, ""
        return

    user_message = chat_history[last_user_idx]["content"]

    # 添加 assistant 占位
    chat_history.append({"role": "assistant", "content": ""})

    # 流式输出 Grok 回复
    generated_prompt = ""
    try:
        for chunk in grok_client.chat_stream(user_message):
            generated_prompt += chunk
            chat_history[-1]["content"] = generated_prompt
            yield chat_history, None, get_text("status_generating_prompt", current_lang)
    except Exception as e:
        chat_history[-1]["content"] = get_text("error_grok_api", current_lang, error=str(e))
        yield chat_history, None, ""
        return

    yield chat_history, None, get_text("status_generating_image", current_lang)

    try:
        image, seed_used = generate_image_internal(
            generated_prompt, res_category, resolution, seed, random_seed, steps
        )
    except gr.Error as e:
        yield chat_history, None, str(e)
        return

    yield chat_history, image, seed_used


def direct_generate(prompt, res_category, resolution, seed, random_seed, steps):
    """直接使用 prompt 生成图片"""
    if not prompt.strip():
        return None, ""

    try:
        image, seed_used = generate_image_internal(
            prompt, res_category, resolution, seed, random_seed, steps
        )
        return image, seed_used
    except gr.Error as e:
        raise e


def fill_example(example_text):
    """将示例填入输入框"""
    return example_text


def get_history_gallery():
    """返回 history tab 的 gallery 数据"""
    return [(d["image"], get_text("history_seed_caption", current_lang, seed=d['seed'])) for d in history_detail]


def on_history_select(evt: gr.SelectData):
    """当在 history gallery 中选中一张图片时"""
    if evt.index < len(history_detail):
        item = history_detail[evt.index]
        info = get_text("history_info", current_lang, seed=item['seed'], width=item['width'], height=item['height'], steps=item['steps'])
        return item["image"], item["prompt"], info, evt.index
    return None, "", "", -1


def on_delete_selected(selected_index: int):
    """删除选中的图片"""
    if selected_index < 0:
        gr.Warning(get_text("warning_select_image", current_lang))
        return get_history_gallery(), None, "", "", -1

    if delete_from_db_by_index(selected_index):
        gr.Info(get_text("info_image_deleted", current_lang))
    return get_history_gallery(), None, "", "", -1


def on_clear_all_history():
    """清空所有历史"""
    clear_all_history()
    gr.Info(get_text("info_history_cleared", current_lang))
    return get_history_gallery(), None, "", "", -1


def on_language_change(lang: str):
    """Update the global current_lang when language changes."""
    global current_lang
    current_lang = lang
    # Refresh history gallery with new language
    return get_history_gallery()


with gr.Blocks(title="Z-Image-Turbo") as demo:
    # Language dropdown (render=False to place it manually)
    lang_selector = gr.Dropdown(
        choices=[("English", "en"), ("中文", "zh")],
        value="zh",
        label=_("language_label"),
        render=False,
        scale=0,
    )

    with Translate(str(TRANSLATIONS_PATH), lang_selector, placeholder_langs=["en", "zh"]) as lang:
        gr.HTML("""
        <style>
        /* Gradio 6.x Chatbot 气泡宽度 - 全局选择器 */
        .bubble-wrap .message-row.bubble {
            max-width: 100% !important;
        }
        .bubble-wrap .message {
            max-width: 100% !important;
        }
        .bubble-wrap .flex-wrap {
            max-width: 100% !important;
        }
        /* 隐藏 History Gallery 的选中效果 */
        .history-gallery .thumbnail-item.selected {
            border-color: transparent !important;
            outline: none !important;
        }
        .history-gallery .thumbnail-item:focus {
            outline: none !important;
        }
        </style>
        """)

        # Header with language selector
        with gr.Row():
            gr.Markdown(_("app_title"))
            lang_selector.render()

        gr.Markdown(_("app_subtitle"))

        with gr.Tabs():
            with gr.TabItem(_("tab_generate")):
                with gr.Row():
                    # 左侧控制面板
                    with gr.Column(scale=1):
                        with gr.Tabs(selected="direct_tab" if not grok_client.available else "chat_tab") as input_tabs:
                            with gr.TabItem(
                                _("tab_chat_mode") if grok_client.available else _("tab_chat_mode_disabled"),
                                id="chat_tab",
                                interactive=grok_client.available,
                            ):
                                chatbot = gr.Chatbot(
                                    label=_("chatbot_label"),
                                    height=300,
                                    editable="user",
                                    layout="panel",
                                )
                                user_input = gr.Textbox(
                                    placeholder=_("chat_placeholder"),
                                    lines=2,
                                    show_label=False,
                                )
                                with gr.Row():
                                    send_btn = gr.Button(_("btn_send"), variant="primary", scale=1)
                                    clear_btn = gr.Button(_("btn_clear_chat"), scale=1)

                            with gr.TabItem(_("tab_direct_input"), id="direct_tab"):
                                direct_prompt = gr.Textbox(
                                    label=_("direct_prompt_label"),
                                    placeholder=_("direct_prompt_placeholder"),
                                    lines=10,
                                )
                                direct_generate_btn = gr.Button(_("btn_generate"), variant="primary")

                        with gr.Row():
                            res_category = gr.Dropdown(
                                label=_("resolution_category_label"),
                                choices=["512", "768", "1024", "1280"],
                                value="512",
                            )
                            resolution = gr.Dropdown(
                                label=_("resolution_label"),
                                choices=list(RESOLUTION_OPTIONS["512"].keys()),
                                value="368x640 (9:16)",
                            )

                        with gr.Row():
                            seed_input = gr.Number(label=_("seed_label"), value=-1)
                            random_seed = gr.Checkbox(label=_("random_seed_label"), value=True)

                        steps = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=8,
                            step=1,
                            label=_("steps_label"),
                        )

                        gr.Markdown(_("examples_title"))
                        example_buttons = []
                        with gr.Row():
                            for i, example in enumerate(EXAMPLE_PROMPTS):
                                short_label = example[:20] + "..." if len(example) > 20 else example
                                btn = gr.Button(short_label, size="sm")
                                example_buttons.append((btn, example))

                    # 右侧图片展示
                    with gr.Column(scale=1):
                        output_image = gr.Image(label=_("generated_image_label"), type="pil")
                        seed_used = gr.Textbox(label=_("seed_used_label"), interactive=False)

            with gr.TabItem(_("tab_history")) as history_tab:
                with gr.Row():
                    clear_all_btn = gr.Button(_("btn_clear_all_history"), variant="stop", size="sm")
                with gr.Row():
                    with gr.Column(scale=1):
                        history_gallery = gr.Gallery(label=_("history_gallery_label"), columns=3, height=400, preview=False, allow_preview=False)
                    with gr.Column(scale=1):
                        preview_image = gr.Image(label=_("preview_label"), type="pil")
                        preview_info = gr.Textbox(label=_("info_label"), interactive=False)
                        preview_prompt = gr.Textbox(label=_("prompt_label"), lines=5, interactive=False)
                        delete_btn = gr.Button(_("btn_delete"), variant="secondary")
                        selected_index = gr.State(value=-1)

        # 事件绑定

        # Language change handler
        lang_selector.change(
            fn=on_language_change,
            inputs=[lang_selector],
            outputs=[history_gallery],
        )

        # 示例按钮点击事件（同时填入两个输入框）
        for btn, example in example_buttons:
            btn.click(fn=lambda e=example: (e, e), outputs=[user_input, direct_prompt])

        res_category.change(
            fn=update_resolution_choices,
            inputs=[res_category],
            outputs=[resolution],
        )

        # Chat 模式：发送消息 → Grok 生成 prompt → 自动生成图片
        send_btn.click(
            fn=chat_and_generate,
            inputs=[user_input, chatbot, res_category, resolution, seed_input, random_seed, steps],
            outputs=[chatbot, output_image, seed_used, user_input],
        )

        # Chat 模式：支持回车发送
        user_input.submit(
            fn=chat_and_generate,
            inputs=[user_input, chatbot, res_category, resolution, seed_input, random_seed, steps],
            outputs=[chatbot, output_image, seed_used, user_input],
        )

        # Chat 模式：清空对话
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, output_image, seed_used],
        )

        # 直接模式：生成图片
        direct_generate_btn.click(
            fn=direct_generate,
            inputs=[direct_prompt, res_category, resolution, seed_input, random_seed, steps],
            outputs=[output_image, seed_used],
        )

        # 直接模式：支持回车生成
        direct_prompt.submit(
            fn=direct_generate,
            inputs=[direct_prompt, res_category, resolution, seed_input, random_seed, steps],
            outputs=[output_image, seed_used],
        )

        # 编辑消息后重新生成
        def on_edit(chat_history, evt: gr.EditData):
            """处理编辑事件：截断历史"""
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index

            # 只处理用户消息
            if row_idx >= len(chat_history) or chat_history[row_idx].get("role") != "user":
                return chat_history

            # 截断到编辑的消息（保留该消息，更新内容）
            truncated = chat_history[:row_idx + 1]
            truncated[row_idx]["content"] = evt.value

            # 同步截断 GrokClient 的历史
            grok_client.history = grok_client.history[:row_idx]

            return truncated

        chatbot.edit(
            fn=on_edit,
            inputs=[chatbot],
            outputs=[chatbot],
        ).then(
            fn=regenerate_after_edit,
            inputs=[chatbot, res_category, resolution, seed_input, random_seed, steps],
            outputs=[chatbot, output_image, seed_used],
        )

        # History tab 切换时刷新
        history_tab.select(fn=get_history_gallery, outputs=[history_gallery])

        # History gallery 选中图片时显示详情
        history_gallery.select(
            fn=on_history_select,
            outputs=[preview_image, preview_prompt, preview_info, selected_index],
        )

        # 删除选中的图片
        delete_btn.click(
            fn=on_delete_selected,
            inputs=[selected_index],
            outputs=[history_gallery, preview_image, preview_prompt, preview_info, selected_index],
        )

        # 清空全部历史
        clear_all_btn.click(
            fn=on_clear_all_history,
            outputs=[history_gallery, preview_image, preview_prompt, preview_info, selected_index],
        )

if __name__ == "__main__":
    init_db()
    load_history()
    demo.launch()
