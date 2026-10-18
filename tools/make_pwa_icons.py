"""生成 PWA 需要的 PNG 图标（192 / 512 / maskable 512 / apple-touch 180）。

复用 make_icon.py 的绘制与降采样逻辑，纯标准库，无 PIL 依赖。
用法：python tools/make_pwa_icons.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_icon as mi  # noqa: E402


def composite_over(px, bg):
    """把带透明通道的像素合成到纯色背景上（maskable 图标不能有透明角）。"""
    out = bytearray(len(px))
    for i in range(0, len(px), 4):
        a = px[i + 3] / 255.0
        out[i] = int(px[i] * a + bg[0] * (1 - a))
        out[i + 1] = int(px[i + 1] * a + bg[1] * (1 - a))
        out[i + 2] = int(px[i + 2] * a + bg[2] * (1 - a))
        out[i + 3] = 255
    return out


def write_png(path, px, size):
    data = mi.png_entry(px, size)
    with open(path, 'wb') as f:
        f.write(data)
    return len(data)


def main():
    out_dir = os.path.join(HERE, '..', 'assets')
    os.makedirs(out_dir, exist_ok=True)

    master = mi.render_master(mi.MASTER)

    jobs = [
        ('icon-192.png', 192),
        ('icon-512.png', 512),
        ('apple-touch-icon.png', 180),
    ]
    for name, size in jobs:
        px = mi.downsample(master, mi.MASTER, size)
        n = write_png(os.path.join(out_dir, name), px, size)
        print(f'{name:24s} {size}x{size}  {n} bytes')

    px512 = mi.downsample(master, mi.MASTER, 512)
    mask = composite_over(px512, mi.C1)
    n = write_png(os.path.join(out_dir, 'icon-maskable-512.png'), mask, 512)
    print(f'{"icon-maskable-512.png":24s} 512x512  {n} bytes')


if __name__ == '__main__':
    main()
