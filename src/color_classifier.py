
import cv2
import numpy as np


def estimate_vehicle_color(frame, box):
    """
    Estimate a broad vehicle color from its bounding box.

    This is a lightweight heuristic, not a trained color classifier.
    """
    x1, y1, x2, y2 = map(int, box)
    height, width = frame.shape[:2]

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return "unknown"

    # Avoid some background by examining the central region.
    h, w = crop.shape[:2]
    crop = crop[int(h * 0.25):int(h * 0.80),
                int(w * 0.20):int(w * 0.80)]

    if crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    # Median reduces the influence of isolated bright pixels.
    hue = float(np.median(pixels[:, 0]))
    saturation = float(np.median(pixels[:, 1]))
    value = float(np.median(pixels[:, 2]))

    if value < 55:
        return "black"
    if saturation < 35:
        if value > 190:
            return "white"
        if value > 110:
            return "silver/grey"
        return "dark grey"

    if hue < 10 or hue >= 170:
        return "red"
    if hue < 22:
        return "orange/brown"
    if hue < 35:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 130:
        return "blue"
    return "other"
