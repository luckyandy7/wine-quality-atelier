from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "static" / "assets" / "cellar.png"


def lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def draw_bottle(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float, color: tuple[int, int, int]) -> None:
    w = int(44 * scale)
    h = int(178 * scale)
    neck_w = int(14 * scale)
    shoulder = int(32 * scale)
    neck_h = int(48 * scale)
    body_top = y + neck_h
    body = [
        (x + (w - neck_w) // 2, y),
        (x + (w + neck_w) // 2, y),
        (x + (w + neck_w) // 2, body_top - int(10 * scale)),
        (x + w - int(4 * scale), body_top),
        (x + w, y + h - int(8 * scale)),
        (x + int(5 * scale), y + h),
        (x, y + h - int(8 * scale)),
        (x + int(4 * scale), body_top),
        (x + (w - shoulder) // 2, body_top - int(10 * scale)),
    ]
    draw.polygon(body, fill=color)
    draw.rounded_rectangle(
        [x + int(10 * scale), y + int(84 * scale), x + w - int(10 * scale), y + int(123 * scale)],
        radius=int(3 * scale),
        fill=(205, 175, 101),
    )
    draw.line(
        [x + int(11 * scale), y + int(16 * scale), x + int(11 * scale), y + h - int(16 * scale)],
        fill=(255, 244, 205, 52),
        width=max(1, int(2 * scale)),
    )


def main() -> None:
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), (21, 12, 14))
    pixels = image.load()
    top = (86, 38, 48)
    bottom = (22, 14, 15)
    for y in range(height):
        t = y / (height - 1)
        for x in range(width):
            side = abs((x / width) - 0.5) * 0.54
            r = max(0, lerp(top[0], bottom[0], t) - int(side * 80))
            g = max(0, lerp(top[1], bottom[1], t) - int(side * 45))
            b = max(0, lerp(top[2], bottom[2], t) - int(side * 48))
            pixels[x, y] = (r, g, b)

    draw = ImageDraw.Draw(image, "RGBA")

    for y in range(90, height, 142):
        draw.line([(80, y), (width - 80, y - 22)], fill=(226, 192, 118, 66), width=2)
        draw.line([(80, y + 44), (width - 80, y + 22)], fill=(226, 192, 118, 34), width=1)

    for x in range(140, width, 145):
        draw.line([(x, 84), (x - 58, height - 80)], fill=(226, 192, 118, 28), width=1)

    colors = [
        (122, 32, 47),
        (42, 105, 96),
        (116, 49, 72),
        (62, 45, 34),
    ]
    for row, y in enumerate([135, 300, 465, 630]):
        for i, x in enumerate(range(120, width - 120, 92)):
            offset = (row % 2) * 36
            scale = 0.78 + ((i + row) % 3) * 0.08
            draw_bottle(draw, x + offset, y + ((i % 2) * 10), scale, colors[(i + row) % len(colors)])

    for x in range(90, width, 176):
        draw.arc([x, 86, x + 260, 380], 190, 285, fill=(229, 207, 150, 46), width=2)

    draw.rectangle([0, height - 160, width, height], fill=(12, 9, 9, 130))
    draw.line([(110, height - 142), (width - 110, height - 170)], fill=(226, 192, 118, 62), width=2)
    draw.line([(110, height - 96), (width - 110, height - 122)], fill=(226, 192, 118, 34), width=1)

    image = image.filter(ImageFilter.GaussianBlur(0.25))
    vignette = Image.new("L", (width, height), 0)
    mask = ImageDraw.Draw(vignette)
    mask.ellipse([-260, -130, width + 260, height + 220], fill=232)
    vignette = vignette.filter(ImageFilter.GaussianBlur(90))
    dark = Image.new("RGB", (width, height), (9, 7, 7))
    image = Image.composite(image, dark, Image.eval(vignette, lambda p: 255 - p))
    image = ImageEnhance.Brightness(image).enhance(1.38)
    image = ImageEnhance.Contrast(image).enhance(1.08)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, "PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
