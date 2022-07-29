import cv2
import numpy as np

from appClass import App
from overlay import *
from queries import *
from SIFT import *

cv2.namedWindow("Video Tracker")

# Open file (CHANGE THE FILE PATH TO ANY FILE YOU WANT TO TRY THIS ON)
# BE CAREFUL WITH THE FILE EXTENSION
file_path = "./habitacio.mp4"
video = cv2.VideoCapture(file_path)

if not video.isOpened():

    class FileError(Exception):
        def __init__(self):
            self.message = "Couldn't open video file, please check the file path."

        def __str__(self):
            return self.message

    raise FileError()

app = App(video)

videoArray = np.zeros((app.length, app.height, app.width, 3), np.uint8)

# Load video (We will load the entire video at once to simplify some work)
loadImgBase = cv2.imread("loading.png")
while True:
    ret, frame = video.read()
    if not ret:
        break
    frameId = int(video.get(cv2.CAP_PROP_POS_FRAMES))

    # A facny loading screen to give the user some feedback
    percentage = int(frameId / app.length * 100)
    loadImg = cv2.putText(
        loadImgBase.copy(),
        f"{percentage}%",
        (230 - int(len(str(percentage)) * 30), 280),
        cv2.FONT_HERSHEY_TRIPLEX,
        3,
        (255, 255, 255),
        2,
    )
    cv2.imshow("Video Tracker", loadImg)

    videoArray[frameId - 1] = frame

    # If escape is pressed or window is closed, break program
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or cv2.getWindowProperty("Video Tracker", cv2.WND_PROP_VISIBLE) < 1:
        exit()


# List for all the selections that need to be tracked troughuout the video
queries = []

# Main program loop
while True:

    # If some query exists, compute curren frame features
    if len(queries) > 0:
        # Small Optmization: If frame features have already been computed, skip this step
        if len(app.kps[app.frameId]) == 0:
            app.kps[app.frameId], app.dess[app.frameId] = get_features(frame)

    # Dictionary to save each query's most likely position in the current frame. keys will be the query's id
    # This will be used soly to draw the found targets on the current frame.
    # Actual tracker data and history is saved in the query class to be later exported.
    findings = {}
    for query in queries:
        query.update()
        found = query.find_in_frame(
            app.kps[app.frameId], app.dess[app.frameId], app.frameId
        )
        # If nothing was found, don't save anything
        if found is not None:
            findings[query.id] = found

    # Draw an overlay to the current frame and display it
    frame = draw_main_overlay(videoArray, app, findings)
    cv2.imshow("Video Tracker", frame)

    # Playing logic
    if app.playing:
        app.frameId += 1
        if app.frameId >= app.length:
            app.frameId = 0
            app.playing = False

    # Key press logic + fps control
    key = cv2.waitKey(int(1000 / app.fps)) & 0xFF
    if key == 27:  # Escape
        exit()
    elif key == 32:  # Space
        app.playing = not app.playing
        app.reset_announce_text()  # Reset the information displayed by text on screen
    elif key == ord("q"):  # Add query
        app.playing = False
        queries.append(Query(videoArray[app.frameId], len(queries)))
    elif key == ord("s"):  # Save tracker data
        app.playing = False
        # A folder should be created with the name tracker_data, but to do that we need the os module
        # So we'll just assume it already exists.
        for query in queries:
            # Just to be sure the folder exists, check if the save raised an exception
            try:
                query.save(app)
            except FileNotFoundError as e:
                print(e)
                print(
                    "Error saving tracker data, please make sure a folder called tracker_data exists"
                )
        app.announceText[
            "save"
        ] = "Tracker data saved"  # Display on screen the trackers were saved

    if cv2.getWindowProperty("Video Tracker", cv2.WND_PROP_VISIBLE) < 1:
        # Window closed
        exit()
