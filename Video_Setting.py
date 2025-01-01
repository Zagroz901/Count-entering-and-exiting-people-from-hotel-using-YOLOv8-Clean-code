import cv2
class Video_Setting():
    def __init__(self, video_path):
        self.video_path=video_path
        # self.output_path=output_path
        # self.frame_size = None
        # self.fps=fps
        # self.cap = None
        self.out = None
    def load_video(self):
        self.cap=cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
    def release_widnow(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()    