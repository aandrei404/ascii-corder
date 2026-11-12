import tkinter as tk

import cv2
import numpy as np

CAMERA_INDEX = 0
CAMERA_BUFFER_SIZE = 1
WINDOW_TITLE = "ascii-corder"
PREVIEW_WIDTH = 1920
REFRESH_DELAY_MS = 1

# Each character is drawn into a fixed-size cell.
CELL_WIDTH = 8
CELL_HEIGHT = 13
FONT_FACE = cv2.FONT_HERSHEY_PLAIN
FONT_SCALE = 0.5
FONT_THICKNESS = 1
RAMP = r" .,:;!|/\~+-=iozxuvncrsawMWB#%&@$"

CHANNEL_MAX = 255
BLACK_POINT = 5
WHITE_POINT = 220
GAMMA = 0.72
INK_GAMMA = 0.45
SATURATION_GAIN = 1.5

channel_values = np.arange(CHANNEL_MAX + 1, dtype=np.float32)
scaled = (channel_values - BLACK_POINT) / (WHITE_POINT - BLACK_POINT)
scaled = np.clip(scaled, 0.0, 1.0)
TONE_LUT = np.rint(scaled ** GAMMA * CHANNEL_MAX).astype(np.uint8)
INK_LUT = np.rint((channel_values / CHANNEL_MAX) **
                  INK_GAMMA * CHANNEL_MAX).astype(np.uint8)


def build_glyph_masks():
    masks = np.zeros((len(RAMP), CELL_HEIGHT, CELL_WIDTH), dtype=np.uint8)
    (_, text_height), baseline = cv2.getTextSize(
        RAMP, FONT_FACE, FONT_SCALE, FONT_THICKNESS
    )
    baseline_y = (CELL_HEIGHT + text_height - baseline) // 2

    for char, mask in zip(RAMP, masks):
        (text_width, _), _ = cv2.getTextSize(
            char, FONT_FACE, FONT_SCALE, FONT_THICKNESS
        )
        position = ((CELL_WIDTH - text_width) // 2, baseline_y)
        cv2.putText(
            mask, char, position, FONT_FACE, FONT_SCALE,
            CHANNEL_MAX, FONT_THICKNESS, cv2.LINE_AA
        )

    # Order characters by their actual ink coverage at the preview font size.
    coverage = masks.sum(axis=(1, 2))
    return masks[np.argsort(coverage, kind="stable")]


def render_frame(frame, columns, rows, glyph_masks):
    grid = cv2.resize(frame, (columns, rows), interpolation=cv2.INTER_AREA)
    grid = cv2.LUT(grid, TONE_LUT)
    brightness = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY).astype(np.uint16)
    indices = brightness * len(glyph_masks) // (CHANNEL_MAX + 1)

    # Interleave the glyph rows to assemble the full preview in one operation.
    width = columns * CELL_WIDTH
    height = rows * CELL_HEIGHT
    alpha = glyph_masks[indices].transpose(0, 2, 1, 3).reshape(height, width)

    ink_bgr = cv2.LUT(grid, INK_LUT)
    hsv = cv2.cvtColor(ink_bgr, cv2.COLOR_BGR2HSV)

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] * SATURATION_GAIN, 0, CHANNEL_MAX
    ).astype(np.uint8)

    ink = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    ink = cv2.resize(ink, (width, height), interpolation=cv2.INTER_NEAREST)
    return cv2.multiply(ink, cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB), scale=1 / CHANNEL_MAX)


def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    try:
        camera.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        ok, frame = camera.read()
        if not ok:
            raise SystemExit("Could not read a camera frame.")

        frame_height, frame_width = frame.shape[:2]
        columns = max(1, PREVIEW_WIDTH // CELL_WIDTH)
        rows = max(1, round(columns * frame_height /
                   frame_width * CELL_WIDTH / CELL_HEIGHT))
        width = columns * CELL_WIDTH
        height = rows * CELL_HEIGHT
        glyph_masks = build_glyph_masks()
        ppm_header = f"P6\n{width} {height}\n{CHANNEL_MAX}\n".encode("ascii")

        window = tk.Tk()
        window.title(WINDOW_TITLE)
        photo = tk.PhotoImage(master=window, width=width, height=height)
        preview = tk.Label(window, image=photo, borderwidth=0)
        preview.pack()

        def update_preview():
            ok, frame = camera.read()
            if not ok:
                print("Could not read a camera frame.")
                window.destroy()
                return

            rgb = render_frame(frame, columns, rows, glyph_masks)
            photo.configure(data=ppm_header + rgb.tobytes(), format="PPM")
            window.after(REFRESH_DELAY_MS, update_preview)

        window.after(REFRESH_DELAY_MS, update_preview)
        window.mainloop()
    finally:
        camera.release()


if __name__ == "__main__":
    main()
