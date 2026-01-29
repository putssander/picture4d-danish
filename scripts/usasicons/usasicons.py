#!/usr/bin/env python3
import os
from PIL import Image

# 1) Define your 21 icons and their grid positions (col, row) in a 4×6 grid
positions_4x6 = {
    "A": (0, 0), "B": (1, 0), "C": (2, 0), "E": (3, 0),
    "F": (0, 1), "G": (1, 1), "H": (2, 1), "I": (3, 1),
    "K": (0, 2), "L": (1, 2), "M": (2, 2), "N": (3, 2),
    "O": (0, 3), "P": (1, 3), "Q": (2, 3), "S": (3, 3),
    "T": (0, 4), "W": (1, 4), "X": (2, 4), "Y": (3, 4),
    "Z": (0, 5),  # bottom-left cell
}

# 2) Load all icons and record their sizes
target_letters = list(positions_4x6.keys())
icons = {}
max_w = max_h = 0
for letter in target_letters:
    fname = f"img/{letter}.png"
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Icon file not found: {fname}")
    img = Image.open(fname).convert("RGBA")
    icons[letter] = img
    w, h = img.size
    max_w = max(max_w, w)
    max_h = max(max_h, h)

# Helper: paste grid

def create_grid(positions, cols, rows, output_name):
    canvas_w = cols * max_w
    canvas_h = rows * max_h
    out = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    for letter, (col, row) in positions.items():
        img = icons[letter]
        w, h = img.size
        x = col * max_w + (max_w - w) // 2
        y = row * max_h + (max_h - h) // 2
        out.paste(img, (x, y), img)
    out.save(output_name, format="PNG")
    print(f"Saved {output_name}")

# 3) Create 4×6 grid image
title_4x6 = "usas_icons_grid_4x6.png"
create_grid(positions_4x6, cols=4, rows=6, output_name=title_4x6)

# 4) Build positions for 5×5 grid (row-major filling first 21 slots)
positions_5x5 = {}
for idx, letter in enumerate(target_letters):
    row = idx // 5
    col = idx % 5
    positions_5x5[letter] = (col, row)

# 5) Create 5×5 grid image
# Empty slots (4 remaining) will be left transparent

title_5x5 = "usas_icons_grid_5x5.png"
create_grid(positions_5x5, cols=5, rows=5, output_name=title_5x5)
