import cv2

def get_video_capture(source=0):
    return cv2.VideoCapture(source)

def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame
