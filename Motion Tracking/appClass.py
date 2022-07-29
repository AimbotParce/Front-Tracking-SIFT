import cv2


class App:
    def __init__(self, video: cv2.VideoCapture):
        """Class to hold information about the main app and video."""
        self.frameId = 0 # Current frame being displayed on screen
        self.playing = False

        # Get video properties
        self.width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(video.get(cv2.CAP_PROP_FPS))
        self.length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create empty lists for each frame keypoints and descriptors
        self.kps = [[] for _ in range(self.length)]
        self.dess = [[] for _ in range(self.length)]

        # All the text that will be displayed on screen
        # To add new information to the screen, just make an entry in this dictionary with any key you want
        self.announceText = {
            "play": "Press SPACE to play/pause",
            "select": "Press Q to open selection window",
            "save": "Press S to save tracker data",
        }

    def reset_announce_text(self):
        """Reset the text displayed on screen to the default"""
        self.announceText["play"] = "Press SPACE to play/pause"
        self.announceText["select"] = "Press Q to open selection window"
        self.announceText["save"] = "Press S to save tracker data"
