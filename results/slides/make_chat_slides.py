"""Two 1920x1080 slide images of the web-chat interactions (text verbatim from the chat)."""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/nangbui/Documents/ArthropodModel"
OUT = "/home/nangbui/Documents/BeetleID/results/slides"   # chat images come from ArthropodModel (ROOT)
F = "/usr/share/fonts/truetype/dejavu/"
font = lambda size, bold=False: ImageFont.truetype(F + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)
MONO = ImageFont.truetype(F + "DejaVuSansMono.ttf", 19)

BG, CARD, INK, MUTED, ACCENT = "#f4f6f9", "#ffffff", "#1f2328", "#57606a", "#1a5fb4"
USER_BG, TOOL_BG, BORDER = "#e3eefc", "#f3f4f6", "#d0d7de"


def wrap_rich(draw, segments, width, size):
    """segments: [(text, bold)] -> lines of [(word, bold)] fitting width."""
    lines, line, x = [], [], 0
    space = draw.textlength(" ", font=font(size))
    for text, bold in segments:
        for i, word in enumerate(text.split(" ")):
            if word == "":
                continue
            w = draw.textlength(word, font=font(size, bold))
            if line and x + w > width:
                lines.append(line)
                line, x = [], 0
            line.append((word, bold))
            x += w + space
    if line:
        lines.append(line)
    return lines


def draw_rich(draw, xy, segments, width, size, color=INK, gap=8):
    x0, y = xy
    space = draw.textlength(" ", font=font(size))
    for line in wrap_rich(draw, segments, width, size):
        x = x0
        for word, bold in line:
            draw.text((x, y), word, font=font(size, bold), fill=color)
            x += draw.textlength(word, font=font(size, bold)) + space
        y += size + gap
    return y


def fit(img, w, h):
    s = min(w / img.width, h / img.height)
    return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)


def slide(question, upload, tool_header, tool_lines, answer_blocks, overlay, caption, out, json_line):
    im = Image.new("RGB", (1920, 1080), BG)
    d = ImageDraw.Draw(im)
    d.text((60, 36), "Arthropod tray assistant", font=font(38, True), fill=INK)
    d.text((60, 86), "web chat  ·  nanobot + Gemini, calling our detector and classifier through MCP",
           font=font(22), fill=MUTED)

    # chat card (left)
    d.rounded_rectangle((50, 135, 1110, 1045), 18, fill=CARD, outline=BORDER, width=2)
    # user turn: photo + question, right-aligned
    thumb = fit(Image.open(upload).convert("RGB"), 210, 118)
    qlines = [" ".join(w for w, _ in line) for line in wrap_rich(d, [(question, False)], 660, 26)]
    qw = max(d.textlength(q, font=font(26)) for q in qlines)
    bx2 = 1080
    bx1 = bx2 - max(qw, thumb.width) - 40
    bh = thumb.height + 30 + 34 * len(qlines) + 18
    d.rounded_rectangle((bx1, 160, bx2, 160 + bh), 16, fill=USER_BG)
    im.paste(thumb, (int(bx2 - 20 - thumb.width), 175))
    for i, q in enumerate(qlines):
        d.text((bx2 - 20 - d.textlength(q, font=font(26)), 175 + thumb.height + 14 + 34 * i), q, font=font(26), fill=INK)
    d.text((bx1 - 60, 170), "You", font=font(20, True), fill=ACCENT)
    y = 160 + bh + 20

    # tool panel
    d.text((80, y), "Assistant", font=font(20, True), fill=ACCENT)
    y += 34
    th = 52 + 27 * len(tool_lines)
    d.rounded_rectangle((80, y, 1080, y + th), 12, fill=TOOL_BG, outline=BORDER, width=1)
    d.text((100, y + 12), tool_header, font=font(22, True), fill=INK)
    for i, line in enumerate(tool_lines):
        d.text((100, y + 48 + 27 * i), line, font=MONO, fill=MUTED)
    y += th + 20

    # answer
    for block in answer_blocks:
        y = draw_rich(d, (90, y), block, 980, 22, gap=7) + 10

    # overlay card (right)
    d.rounded_rectangle((1140, 135, 1870, 1045), 18, fill=CARD, outline=BORDER, width=2)
    d.text((1170, 160), "Detector + classifier output", font=font(26, True), fill=INK)
    d.text((1170, 198), "computed locally on the lab GPU; the photo never leaves the computer",
           font=font(17), fill=MUTED)
    ov = fit(Image.open(overlay).convert("RGB"), 670, 760)
    im.paste(ov, (1140 + (730 - ov.width) // 2, 250))
    d.text((1170, 250 + ov.height + 20), caption, font=font(20), fill=MUTED)
    up = fit(Image.open(upload).convert("RGB"), 670, 300)
    d.text((1170, 250 + ov.height + 75), "The language model got only the photo's path and this JSON (abridged)", font=font(17), fill=MUTED)
    d.rounded_rectangle((1170, 250 + ov.height + 105, 1840, 250 + ov.height + 105 + 46), 8, fill=TOOL_BG, outline=BORDER)
    d.text((1185, 250 + ov.height + 117), json_line, font=font(16), fill=INK)
    im.save(out, quality=95)
    print("wrote", out, "- answer ends at y =", y)


os.makedirs(OUT, exist_ok=True)
slide(
    "Is this insect harmful?",
    f"{ROOT}/runs/mcp/uploads/8a5eb02c_PICA0084.jpg",
    "Used 1 tool call  (122.5 s)",
    ["Tool 1: analyze_tray — 8a5eb02c_PICA0084.jpg (RT-DETR)",
     "├ RT-DETR (960): 1 specimen(s) found",
     "├ keypoints kp4-kp8 predicted for 1 specimen(s)",
     "├ 1 of 1 pose-aligned",
     "├ classified 1 crop(s)",
     "├ 1 specimen(s): 1 SPB_Dendroctonus_frontalis (0 below 0.5 probability)",
     "├ overlay saved: 8a5eb02c_PICA0084_rtdetr.jpg",
     "└ finished in 0.2 s"],
    [[("Yes, ecologically and economically, it is considered very harmful.", True)],
     [("The photo contains 1 specimen identified as", False), ("Southern Pine Beetle (Dendroctonus frontalis)", True),
      ("with 88% confidence.", False)],
     [("Impact:", True)],
     [("Forests & Pine Trees:", True), ("Highly destructive. Southern pine beetles bore into the inner bark of pine "
      "trees to feed and reproduce, girdling the tree and introducing blue-stain fungi. During outbreaks, they can "
      "kill large stands of healthy pine timber rapidly.", False)],
     [("Humans & Pets:", True), ("Harmless directly—they do not bite, sting, or transmit diseases to humans or "
      "animals.", False)]],
    f"{ROOT}/runs/mcp/outputs/8a5eb02c_PICA0084_rtdetr.jpg",
    "SPB, Dendroctonus frontalis  ·  held-out validation photo PICA0084",
    f"{OUT}/chat_1_southern_pine_beetle.png",
    '{"n_specimens": 1, "counts_by_class": {"SPB_Dendroctonus_frontalis": 1}, ...}')

slide(
    "How about this one?",
    f"{ROOT}/runs/mcp/uploads/d769c202_PICA0120.jpg",
    "Used 1 tool call  (164.8 s)",
    [],
    [[("No, this insect is beneficial rather than harmful.", True)],
     [("The photo contains 1 specimen identified as a", False), ("checkered beetle (Cleridae)", True),
      ("with 93% confidence.", False)],
     [("Natural predator of bark beetles:", True), ("Many clerids (such as Thanasimus dubius) are primary natural "
      "predators of pine bark beetles, including the Southern Pine Beetle from your previous photo. Both adults and "
      "larvae actively hunt and feed on them, playing an important role in natural pest control.", False)],
     [("Humans & Pets:", True), ("Harmless—they do not sting or transmit diseases.", False)]],
    f"{ROOT}/runs/mcp/outputs/d769c202_PICA0120_rtdetr.jpg",
    "Cleridae (checkered beetle)  ·  held-out validation photo PICA0120",
    f"{OUT}/chat_2_checkered_beetle.png",
    '{"n_specimens": 1, "counts_by_class": {"Cleridae": 1}, ...}')

slide(
    "How many specimens are in here? And what kind of insects are the majority?",
    f"{ROOT}/runs/mcp/uploads/b0a3746f_D405_Multiple_Specimens.jpg",
    "Used 1 tool call  (158.2 s)",
    [],
    [[("There are 26 specimens detected in this image.", True)],
     [("Breakdown by Taxon", True)],
     [("•", False), ("Acari (mites):", True), ("24 (vast majority)", False)],
     [("•", False), ("Araneae (spiders):", True), ("1", False)],
     [("•", False), ("Opiliones (harvestmen):", True), ("1", False)],
     [("(Note: taxonomically, these are arachnids rather than true insects, with mites dominating the sample.)", False)],
     [("Uncertain Identifications (< 50% confidence)", True)],
     [("•", False), ("Specimen #10:", True), ("Opiliones (38.3%, runner-up Archaeognatha at 12.0%)", False)],
     [("•", False), ("Specimen #17:", True), ("Acari (33.1%, runner-up Collembola at 19.2%)", False)],
     [("•", False), ("Specimen #24:", True), ("Acari (45.2%, runner-up Diplopoda at 20.4%)", False)],
     [("Keep in mind that the classifier was trained on isolated single specimens, so classifications on "
       "multi-specimen trays are somewhat less reliable than the overall detection count.", False)]],
    f"{ROOT}/runs/mcp/outputs/b0a3746f_D405_Multiple_Specimens_rtdetr.jpg",
    "Tray D405  ·  real multi-specimen tray, never used in training",
    f"{OUT}/chat_3_tray_D405.png",
    '{"n_specimens": 26, "counts_by_class": {"Acari": 24, "Araneae": 1, ...}, ...}')
