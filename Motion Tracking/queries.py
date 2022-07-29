import cv2
import numpy as np

from appClass import App
from boxes import Box, Clickable
from overlay import draw_query_box, draw_query_overlay
from SIFT import *


# To make this less of a mess, I'll be using classes
class Query:
    def __init__(self, frame: np.ndarray, id: int) -> None:
        # Each querry needs an id to identify it
        self.id = id
        # Create selection window
        cv2.namedWindow(f"Query {id}")
        cv2.setMouseCallback(f"Query {id}", self.mouse_click)
        # Store the frame
        self.frame = frame.copy()
        # Only one selection per query can be done. Box class is an expandable dynamic box selection.
        # Clickables are just points on the screen that can be clicked and dragged
        self.box: Box = None
        self.selectedClickable: Clickable = None
        # Find frame features
        self.kp, self.des = get_features(frame)

        # All the text that will be displayed on screen
        # To add new information to the screen, just make an entry in this dictionary with any key you want
        self.announceText = {"base": "Click and drag to make a selection"}

        # Initialize object and background keypoints and descriptors. Background features aren't used on this iteration
        # of the program, but may be used on future iterations.
        self.objectKp = []
        self.objectDes = []
        self.backgroundKp = []
        self.backgroundDes = []

        # The history of the positions of the selection on the main frame. This will be used to save tracker data later
        # Key will each frame, and value will be the top left and bottom right coordinates of the found selection in frame
        self.trackerHistory = {}

    def update(self):
        """Update the tracker window"""
        # Do not overwrite the frame, just a copy
        frame = self.frame.copy()
        if self.box is not None:
            draw_query_box(frame, self.box, self.id)
        draw_query_overlay(frame, ". ".join(list(self.announceText.values())))
        cv2.imshow(f"Query {self.id}", frame)

    def select_features(self):
        """Identify features in the selection as object features and others as background features"""
        if self.box is None:
            return
        # Reset the lists
        self.objectKp = []
        self.objectDes = []
        self.backgroundKp = []
        self.backgroundDes = []

        for i in range(len(self.kp)):
            if self.box.is_inside(self.kp[i].pt[0], self.kp[i].pt[1]):
                self.objectKp.append(self.kp[i])
                self.objectDes.append(self.des[i])
            else:
                self.backgroundKp.append(self.kp[i])
                self.backgroundDes.append(self.des[i])

        # For BFF.KnnMatcher to work, these must be numpy arrays
        self.objectDes = np.array(self.objectDes)
        self.backgroundDes = np.array(self.backgroundDes)

        # Update on screen text to show how many features were found inside the selection
        self.announceText["selected"] = f"{len(self.objectKp)} features found"

    def mouse_click(self, event, x, y, flags, param):
        """Handle mouse clicks on the tracker window"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # See if there was a clickable already on the mouse position
            self.selectedClickable = self.find_clickable_by_position(x, y)

            if self.selectedClickable is None:
                # If no clickable was selected, recreate the box selection on current mouse position
                self.box = Box(x, y, x, y)
                # We save the clickable as selected to be able to move it
                self.selectedClickable = self.box.cornerB
            # The selector needs to know it was selected so it can also update its parent
            # (Box logic is explained in the Box document)
            self.selectedClickable.selected = True
        elif event == cv2.EVENT_LBUTTONUP:
            # If the mouse was released, we need to unselect the clickable
            self.selectedClickable = None
            # And update the object/background feature list
            self.select_features()
        elif event == cv2.EVENT_MOUSEMOVE:
            # If the mouse was moved with a selected clickable, move it aswell
            if self.selectedClickable is not None:
                self.selectedClickable.x = x
                self.selectedClickable.y = y
                # As the clickable knows it is selected, the box knows wich corner to move
                self.box.update()

    def find_clickable_by_position(self, x: int, y: int):
        """Find the clickable on the mouse position, if any"""
        if self.box is None:
            return None
        if self.box.cornerA.is_in_range(x, y):
            return self.box.cornerA
        if self.box.cornerB.is_in_range(x, y):
            return self.box.cornerB
        return None

    def save(self, app: App):
        """Save the tracker data to a file. Need app variable, to save some miscellaneous data aswell"""
        with open(f"tracker_data/{self.id}.txt", "w") as doc:
            # Info about the video and the query
            info = {
                "length (frames)": app.length,
                "width": app.width,
                "height": app.height,
                "fps": app.fps,
                "tracker_id": self.id,
            }
            doc.write("\n".join(["# %s: %s" % (k, v) for k, v in info.items()]))
            doc.write("\n# Columns: frame, x, y, stdx, stdy\n")

            # We'll save them sorted by frame number
            frames = np.sort(list(self.trackerHistory.keys()))
            for frame in frames:
                pt = self.trackerHistory[frame]
                doc.write(f"{frame}\t{pt[0]}\t{pt[1]}\t{pt[2]}\t{pt[3]}\n")

    def find_in_frame(self, frameKp: list, frameDes: list, frameId: int):
        """
        Match the features on the selected box to the features in the frame.
        Returns and saves to the class trackerHistory dictionary the center of mass of the matched points, as the expected
        position of the selection in the frame, and the standard deviation of it as a measure of the certainty of the match.
        Returns: x0, y0, stdx, stdy
        """
        if len(self.objectKp) == 0 or len(self.backgroundKp) == 0:
            self.announceText["selected"] = "No features found"
            return

        objectMatches = match_features(self.objectDes, frameDes)
        # backgroundMatches = match_features(self.backgroundDes, frameDes)

        if len(objectMatches) == 0:  # or len(backgroundMatches) == 0
            self.announceText["matches"] = "No matches found, please modify selection"
            return

        # Nothing really needs to be displayed if everything went okay
        self.announceText["matches"] = ""

        # Get object feature coordinates on the video frame
        objectPts = np.uint([frameKp[m.trainIdx].pt for m in objectMatches])

        # Compute the mean and std and save it and return it
        x0, y0 = np.int64(objectPts.mean(axis=0))
        stdx, stdy = np.int64(objectPts.std(axis=0))

        self.trackerHistory[frameId] = [x0, y0, stdx, stdy]
        return x0, y0, stdx, stdy
