import cv2
import numpy as np
import mediapipe as mp
from collections import deque

CANVAS_ALPHA = 0.6
BRUSH_THICKNESS = 8
ERASER_THICKNESS = 40
SMOOTHING = 5


COLORS = {
    "Blue": (255, 100, 20),
    "Green": (50, 220, 50),
    "Red": (30, 30, 220),
    "Yellow": (0, 230, 230),
    "Purple": (200, 50, 200),
    "White": (255, 255, 255),
}

BUTTON_H = 55
BUTTON_W = 80
PANEL_H = BUTTON_H + 10


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_det = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)


def get_finger_states(hand_landmarks):
    """Return (index_up, middle_up) boolean tuple."""
    tips = [8, 12]
    pips = [6, 10]
    results = []
    for tip, pip in zip(tips, pips):
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[pip].y
        results.append(tip_y < pip_y)
    return results[0], results[1]


def get_index_tip(hand_landmarks, w, h):
    lm = hand_landmarks.landmark[8]
    return int(lm.x * w), int(lm.y * h)


# ─────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────
def build_buttons(frame_w):
    """Return list of (label, x1, y1, x2, y2, color_bgr)."""
    buttons = []
    x = 5
    for name, bgr in COLORS.items():
        buttons.append((name, x, 5, x + BUTTON_W, 5 + BUTTON_H, bgr))
        x += BUTTON_W + 5
    # Eraser
    buttons.append(("ERASER", x, 5, x + BUTTON_W + 10, 5 + BUTTON_H, (80, 80, 80)))
    x += BUTTON_W + 15
    # Clear
    buttons.append(("CLEAR", x, 5, x + BUTTON_W + 10, 5 + BUTTON_H, (40, 40, 40)))
    return buttons


def draw_toolbar(frame, buttons, active_color, eraser_on):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], PANEL_H + 5), (30, 30, 30), -1)
    for label, x1, y1, x2, y2, bgr in buttons:

        is_active = (label == "ERASER" and eraser_on) or (
            label not in ("ERASER", "CLEAR")
            and COLORS.get(label) == active_color
            and not eraser_on
        )
        border = 3 if is_active else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), border)

        font_scale = 0.38 if label not in ("ERASER", "CLEAR") else 0.42
        tx = x1 + 4
        ty = y2 - 8
        cv2.putText(
            frame,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
        )


def hit_test(px, py, buttons):
    """Return button label if (px,py) is inside a button, else None."""
    for label, x1, y1, x2, y2, _ in buttons:
        if x1 <= px <= x2 and y1 <= py <= y2:
            return label
    return None


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("❌  Cannot open webcam.")
        return

    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    active_color = list(COLORS.values())[0]
    eraser_mode = False
    prev_pt = None
    smooth_pts = deque(maxlen=SMOOTHING)
    buttons = build_buttons(w)

    print("AIR BOARD , AUGMENTED REALITY DRAWING TOOL")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_det.process(rgb)

        draw_mode = False
        finger_tip = None

        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                index_up, middle_up = get_finger_states(hand_lms)
                tip_x, tip_y = get_index_tip(hand_lms, w, h)

                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1),
                )

                if index_up and not middle_up:
                    draw_mode = True
                    finger_tip = (tip_x, tip_y)

                    if tip_y < PANEL_H + 5:
                        hit = hit_test(tip_x, tip_y, buttons)
                        if hit == "CLEAR":
                            canvas[:] = 0
                            prev_pt = None
                        elif hit == "ERASER":
                            eraser_mode = True
                        elif hit and hit in COLORS:
                            active_color = COLORS[hit]
                            eraser_mode = False
                        draw_mode = False

                elif index_up and middle_up:

                    prev_pt = None

        if draw_mode and finger_tip:
            smooth_pts.append(finger_tip)
            sx = int(np.mean([p[0] for p in smooth_pts]))
            sy = int(np.mean([p[1] for p in smooth_pts]))
            smoothed = (sx, sy)

            if prev_pt:
                color = (0, 0, 0) if eraser_mode else active_color
                thickness = ERASER_THICKNESS if eraser_mode else BRUSH_THICKNESS
                cv2.line(canvas, prev_pt, smoothed, color, thickness, cv2.LINE_AA)
            prev_pt = smoothed
        else:
            prev_pt = None
            smooth_pts.clear()

        mask = canvas.astype(bool)
        output = frame.copy()
        output[mask] = cv2.addWeighted(
            frame, 1 - CANVAS_ALPHA, canvas, CANVAS_ALPHA, 0
        )[mask]

        draw_toolbar(output, buttons, active_color, eraser_mode)

        if finger_tip:
            color = (0, 0, 0) if eraser_mode else active_color
            size = ERASER_THICKNESS // 2 if eraser_mode else BRUSH_THICKNESS // 2 + 4
            cv2.circle(output, finger_tip, size, color, -1)
            cv2.circle(output, finger_tip, size + 2, (255, 255, 255), 2)

        status = (
            "🖊  DRAW"
            if draw_mode
            else ("HOVER" if result.multi_hand_landmarks else "Searching…")
        )
        mode_t = "ERASER" if eraser_mode else "BRUSH"
        cv2.putText(
            output,
            f"{status}  |  {mode_t}",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        cv2.imshow("AIR BOARD - AUGEMENTED REALITY TOOL DPSI", output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands_det.close()
    print("Air Board closed.")


if __name__ == "__main__":
    main()
