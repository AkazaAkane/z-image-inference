import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import random
import torch
import gradio as gr
from diffusers import ZImagePipeline

pipe = None
history = []

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


def load_model():
    global pipe
    if pipe is None:
        print("Loading model...")
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
        pipe.to(torch.device("mps"))
        pipe.enable_attention_slicing()
        print("Model loaded!")
    return pipe


def update_resolution_choices(category):
    choices = list(RESOLUTION_OPTIONS.get(category, {}).keys())
    return gr.update(choices=choices, value=choices[0] if choices else None)


def generate_image(prompt, res_category, resolution, seed, random_seed, steps):
    if not prompt:
        raise gr.Error("请输入 Prompt")

    model = load_model()

    # 处理种子
    if random_seed:
        seed = random.randint(0, 2147483647)
    else:
        seed = int(seed)

    # 获取分辨率
    width, height = RESOLUTION_OPTIONS.get(res_category, {}).get(resolution, (1024, 1024))

    generator = torch.Generator("mps").manual_seed(seed)
    image = model(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=int(steps),
        guidance_scale=0.0,
        max_sequence_length=512,
        generator=generator,
    ).images[0]

    history.insert(0, (image, f"seed: {seed}"))
    torch.mps.empty_cache()

    return image, str(seed), history


def get_history():
    return history


with gr.Blocks(title="Z-Image-Turbo") as demo:
    gr.Markdown("# Z-Image-Turbo\n*An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer*")

    with gr.Tabs():
        with gr.TabItem("Generate"):
            with gr.Row():
                # 左侧控制面板
                with gr.Column(scale=1):
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3,
                    )

                    with gr.Row():
                        res_category = gr.Dropdown(
                            label="Resolution Category",
                            choices=["512", "768", "1024", "1280"],
                            value="1024",
                        )
                        resolution = gr.Dropdown(
                            label="Resolution",
                            choices=list(RESOLUTION_OPTIONS["1024"].keys()),
                            value="1024x1024 (1:1)",
                        )

                    with gr.Row():
                        seed = gr.Number(label="Seed", value=-1)
                        random_seed = gr.Checkbox(label="Random Seed", value=True)

                    steps = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=8,
                        step=1,
                        label="Steps",
                    )

                    generate_btn = gr.Button("Generate", variant="primary")

                    gr.Markdown("### Example Prompts")
                    examples = gr.Examples(
                        examples=[[p] for p in EXAMPLE_PROMPTS],
                        inputs=[prompt],
                        label="Examples",
                    )

                # 右侧图片展示
                with gr.Column(scale=1):
                    output_image = gr.Image(label="Generated Image", type="pil")
                    seed_used = gr.Textbox(label="Seed Used", interactive=False)
                    gen_gallery = gr.Gallery(label="History", columns=4, height="auto")

        with gr.TabItem("History") as history_tab:
            history_gallery = gr.Gallery(label="History", columns=4, height="auto")

    # 事件绑定
    res_category.change(
        fn=update_resolution_choices,
        inputs=[res_category],
        outputs=[resolution],
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, res_category, resolution, seed, random_seed, steps],
        outputs=[output_image, seed_used, gen_gallery],
    )

    # History tab 切换时刷新
    history_tab.select(fn=get_history, outputs=[history_gallery])

if __name__ == "__main__":
    demo.launch()
