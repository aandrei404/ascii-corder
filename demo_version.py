import cv2
import tkinter as tk
import numpy as np
from tkinter import font as tkfont

# Character brightness ramp
# Used to assign a character to each cell
# based on its brightness
# RAMP = " .:-=+*#%@"
RAMP = r"""$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`'. """[::-1]

# Pick preview character width
COLUMNS = 72
PREVIEW_WIDTH = 960

CAMERA_INDEX = 0
WINDOW_TITLE = "ascii-croder"

# Simulate terminal back background
BACKGROUND = "#000000"
FONT_NAME = "TkFixedFont"
FONT_SIZE = 10

# Frame rate for live updates
REFRESH_DELAY_MS = 16

# Webcam tuning
CHANNEL_MAX = 255
BLACK_POINT = 10
WHITE_POINT = 190
GAMMA = 0.99
values = np.arange(CHANNEL_MAX + 1, dtype=np.float32)
scaled = (values - BLACK_POINT) / (WHITE_POINT - BLACK_POINT)
scaled = np.clip(scaled, 0.0, 1.0)
TONE_LUT = np.rint(scaled ** GAMMA * CHANNEL_MAX).astype(np.uint8)


def brightness_to_char(brightness):
    # Formula divides the brightness values into buckets based on how
    # many characters are in the brightness RAMP
    #  val * 4 // 256 <=> val // 64 (so we divided in 4 buckets of 64 each)
    index = int(brightness) * len(RAMP) // 256
    return RAMP[index]


# Check camera is working
camera = cv2.VideoCapture(CAMERA_INDEX)
try:
    ok, frame = camera.read()
    if not ok:
        raise SystemExit("Could not read a camera frame.")

    # Make terminal-like window
    window = tk.Tk()
    window.title(WINDOW_TITLE)

    # Configure viewing ratio's based on the selected font
    preview_font = tkfont.nametofont(FONT_NAME).copy()
    preview_font.configure(size=FONT_SIZE)
    cell_width = preview_font.measure("M")
    cell_height = preview_font.metrics("linespace")
    columns = max(1, PREVIEW_WIDTH // cell_width)

    # Keep image proportions
    # If we want, for example, 80 cells width (so 80 chars), that means
    # width / 80 pixels per cell. We assume the cells are square,
    # so we divide height by (width / 80), which gives the
    # number of cells for the rows. Characters in the terminal
    # are twice as tall as they are wide, so we scale by the ratio
    # between the cell width and height. Then we  round everything,
    # making sure there is at least one row by clamping with the max
    height, width = frame.shape[:2]
    rows = max(1, round(columns * height / width * cell_width / cell_height))

    # Make drawing area, scaled to our proportions
    canvas = tk.Canvas(
        window,
        width=columns * cell_width,
        height=rows * cell_height,
        background=BACKGROUND
    )
    canvas.pack()

    # Create text input for each cell
    cells = []
    for y in range(rows):
        row_items = []
        for x in range(columns):
            item = canvas.create_text(
                x * cell_width, y * cell_height,
                font=preview_font,
                anchor=tk.NW,
            )
            row_items.append(item)
        cells.append(row_items)

    def update_preview():
        ok, frame = camera.read()
        if not ok:
            print("Could not read a camera frame.")
            window.destroy()
            return

        # Create the grid for image processing
        grid = cv2.resize(frame, (columns, rows), interpolation=cv2.INTER_AREA)
        # Tune camera frame
        grid = cv2.LUT(grid, TONE_LUT)
        # We parse the grid and extract a brightness value for each cell
        # then assign for each cell a character from the brightness ramp
        gray_grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)

        # Live feed loop
        for y, row in enumerate(gray_grid):
            for x, brightness in enumerate(row):
                # from the original grid, get the colours
                blue, green, red = grid[y, x]
                color = f"#{red:02X}{green:02X}{blue:02X}"
                canvas.itemconfigure(
                    cells[y][x], text=brightness_to_char(brightness), fill=color)

        window.after(REFRESH_DELAY_MS, update_preview)

    window.after(REFRESH_DELAY_MS, update_preview)
    window.mainloop()

finally:
    camera.release()
