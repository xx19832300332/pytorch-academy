"""生成 PyTorch 学习站图标（纯标准库，无 PIL 依赖）。
设计：紫罗兰渐变圆角方块 + 白色神经网络图形（3-2 全连接 + 连边）。
"""
import os
import struct
import zlib

MASTER = 512          # 超采样主图尺寸
SIZES = [256, 128, 64, 48, 32, 24, 16]

# 品牌色：#4F46E5 -> #7C3AED
C1 = (79, 70, 229)
C2 = (124, 58, 237)
WHITE = (255, 255, 255)


def lerp(a, b, t):
    return a + (b - a) * t


def rounded_alpha(x, y, size, radius_ratio=0.235, inset=0.015):
    """返回该点在圆角矩形内的覆盖率（0..1），用小子采样近似抗锯齿。"""
    m = inset * size
    r = radius_ratio * size
    x0, y0 = m, m
    x1, y1 = size - m, size - m
    cx = min(max(x, x0 + r), x1 - r)
    cy = min(max(y, y0 + r), y1 - r)
    if x0 <= x <= x1 and y0 <= y <= y1:
        d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        if d <= r:
            return 1.0
        if d <= r + 1.2:
            return max(0.0, 1.0 - (d - r) / 1.2)
        return 0.0
    return 0.0


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return ((px - (ax + t * dx)) ** 2 + (py - (ay + t * dy)) ** 2) ** 0.5


def render_master(S=MASTER):
    """返回 S*S 的 RGBA 像素（自底向上不是这里的事，这里按 y 向下）。"""
    left_x, right_x = 0.315 * S, 0.695 * S
    left_y = [0.295 * S, 0.5 * S, 0.705 * S]
    right_y = [0.375 * S, 0.625 * S]
    node_r = 0.072 * S
    line_r = 0.019 * S

    buf = bytearray(S * S * 4)

    for y in range(S):
        for x in range(S):
            a = rounded_alpha(x, y, S)
            i = (y * S + x) * 4
            if a <= 0:
                buf[i:i + 4] = b'\x00\x00\x00\x00'
                continue
            # 背景渐变（左上 -> 右下）
            t = max(0.0, min(1.0, (x / S * 0.55 + y / S * 0.45)))
            r = int(lerp(C1[0], C2[0], t))
            g = int(lerp(C1[1], C2[1], t))
            b = int(lerp(C1[2], C2[2], t))

            # 只在图形区域内做像素级判定，省时间
            if 0.22 * S < x < 0.79 * S and 0.22 * S < y < 0.79 * S:
                w = 0.0
                for ly in left_y:
                    for ry in right_y:
                        if seg_dist(x, y, left_x, ly, right_x, ry) <= line_r:
                            w = max(w, 0.62)
                for ly in left_y:
                    d = ((x - left_x) ** 2 + (y - ly) ** 2) ** 0.5
                    if d <= node_r + 1.0:
                        w = max(w, min(1.0, (node_r + 1.0 - d)))
                        if d <= node_r:
                            w = 1.0
                for ry in right_y:
                    d = ((x - right_x) ** 2 + (y - ry) ** 2) ** 0.5
                    if d <= node_r + 1.0:
                        w = max(w, min(1.0, (node_r + 1.0 - d)))
                        if d <= node_r:
                            w = 1.0

                if w > 0:
                    r = int(lerp(r, WHITE[0], w))
                    g = int(lerp(g, WHITE[1], w))
                    b = int(lerp(b, WHITE[2], w))

            buf[i] = r
            buf[i + 1] = g
            buf[i + 2] = b
            buf[i + 3] = int(round(a * 255))
    return buf


def downsample(src, S, size):
    """盒式滤波缩放（面积平均），同时预乘 alpha 避免边缘发黑。"""
    factor = S / size
    out = bytearray(size * size * 4)
    for y in range(size):
        y0 = int(y * factor)
        y1 = max(y0 + 1, int((y + 1) * factor))
        for x in range(size):
            x0 = int(x * factor)
            x1 = max(x0 + 1, int((x + 1) * factor))
            rs = gs = bs = as_ = 0.0
            n = 0
            for yy in range(y0, min(y1, S)):
                base = yy * S
                for xx in range(x0, min(x1, S)):
                    i = (base + xx) * 4
                    a = src[i + 3] / 255.0
                    rs += src[i] * a
                    gs += src[i + 1] * a
                    bs += src[i + 2] * a
                    as_ += a
                    n += 1
            i = (y * size + x) * 4
            if n == 0 or as_ == 0:
                out[i:i + 4] = b'\x00\x00\x00\x00'
                continue
            out[i] = int(round(rs / as_))
            out[i + 1] = int(round(gs / as_))
            out[i + 2] = int(round(bs / as_))
            out[i + 3] = int(round(as_ / n * 255))
    return out


def bmp_entry(px, size):
    """32bpp BGRA + AND 掩码，bottom-up。"""
    hdr = struct.pack('<IiiHHIIiiII', 40, size, size * 2, 1, 32, 0,
                      size * size * 4, 0, 0, 0, 0)
    rows = []
    for y in range(size - 1, -1, -1):
        row = bytearray()
        for x in range(size):
            i = (y * size + x) * 4
            row += bytes((px[i + 2], px[i + 1], px[i], px[i + 3]))
        rows.append(bytes(row))
    xor = b''.join(rows)
    and_row = ((size + 31) // 32) * 4
    andmask = b'\x00' * (and_row * size)
    return hdr + xor + andmask


def png_entry(px, size):
    raw = bytearray()
    for y in range(size):
        raw.append(0)
        for x in range(size):
            i = (y * size + x) * 4
            raw += bytes((px[i], px[i + 1], px[i + 2], px[i + 3]))

    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)

    ihdr = struct.pack('>IIBBBBB', size, size, 8, 6, 0, 0, 0)
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', ihdr)
            + chunk(b'IDAT', zlib.compress(bytes(raw), 9))
            + chunk(b'IEND', b''))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, '..', 'assets')
    os.makedirs(out_dir, exist_ok=True)

    master = render_master()
    entries = []
    for size in SIZES:
        px = master if size == MASTER else downsample(master, MASTER, size)
        data = png_entry(px, size) if size >= 256 else bmp_entry(px, size)
        entries.append((size, data))
        print('rendered', size, len(data), 'bytes')

    count = len(entries)
    header = struct.pack('<HHH', 0, 1, count)
    offset = 6 + 16 * count
    dirs, blobs = b'', b''
    for size, data in entries:
        w = 0 if size >= 256 else size
        dirs += struct.pack('<BBBBHHII', w, w, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)

    ico_path = os.path.join(out_dir, 'icon.ico')
    with open(ico_path, 'wb') as f:
        f.write(header + dirs + blobs)
    print('saved', ico_path, os.path.getsize(ico_path), 'bytes')

    prev = downsample(master, MASTER, 256)
    with open(os.path.join(out_dir, 'icon-preview.png'), 'wb') as f:
        f.write(png_entry(prev, 256))


if __name__ == '__main__':
    main()
