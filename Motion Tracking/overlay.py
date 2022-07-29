import cv2
import numpy as np
from numpy import ndarray

from appClass import App
from boxes import Box


def draw_query_box(frame: np.ndarray, box: Box, id: int) -> None:
    """Draw the box selection on the query window"""
    cv2.rectangle(
        frame,
        (box.x0, box.y0),
        (box.x1, box.y1),
        (0, 255, 0),
        1,
    )
    cv2.rectangle(
        frame,
        (box.x0 - 3, box.y0 - 3),
        (box.x0 + 3, box.y0 + 3),
        (0, 255, 0),
        -1,
    )
    cv2.rectangle(
        frame,
        (box.x1 - 3, box.y1 - 3),
        (box.x1 + 3, box.y1 + 3),
        (0, 255, 0),
        -1,
    )

    bbox = box.get_bounding_box()
    cv2.putText(
        frame,
        f"Selection {id}",
        (bbox[0] + 10, bbox[1] - 10),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        0.8,
        (0, 255, 0),
        1,
    )


def draw_query_overlay(frame: np.ndarray, text: str) -> None:
    """Draws the query overlay on the frame except the box selection. Shows text on screen."""
    # To make the text position and its background responsive to the text itself, we need to find
    # the size of it.
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1)[0]

    textX = frame.shape[1] // 2
    textY = frame.shape[0] - size[1] - 30

    # Translucent black rectangle for text background
    frame[
        textY - size[1] // 2 - 10 : textY + size[1] // 2 + 10,
        textX - size[0] // 2 - 10 : textX + size[0] // 2 + 10,
    ] = np.uint8(
        0.5
        * frame[
            textY - size[1] // 2 - 10 : textY + size[1] // 2 + 10,
            textX - size[0] // 2 - 10 : textX + size[0] // 2 + 10,
        ]
    )

    cv2.putText(
        frame,
        text,
        (textX - size[0] // 2, textY + size[1] // 2),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        0.8,
        (255, 255, 255),
        1,
    )


def draw_main_overlay(videoArray: ndarray, app: App, findings: dict) -> ndarray:
    """
    Draws the overlay on the main frame:
    - Draws the found query targets in frame.
    - Displays on screen the text on the app.announceText dictionary.

    Returns the overlayed frame.
    """

    # For this function, the frame is passed inside teh videoArray variable, so we shall copy it
    # to prevent overwritting it.
    frame = videoArray[app.frameId].copy()

    # Draw found queries
    for id, point in findings.items():
        cv2.rectangle(
            frame,
            (point[0] - 5, point[1] - 5),
            (point[0] + 5, point[1] + 5),
            (0, 255, 0),
            1,
        )
        cv2.line(
            frame,
            (point[0] - point[2], point[1]),
            (point[0] + point[2], point[1]),
            (0, 255, 0),
            1,
        )
        cv2.line(
            frame,
            (point[0], point[1] - point[3]),
            (point[0], point[1] + point[3]),
            (0, 255, 0),
            1,
        )
        cv2.putText(
            frame,
            f"Query {id}",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.8,
            (0, 255, 0),
            1,
        )

    displayText = ". ".join(app.announceText.values())

    size = cv2.getTextSize(displayText, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1)[0]

    textX = app.width // 2
    textY = app.height - size[1] - 30

    # Translucent black rectangle for text background
    frame[
        textY - size[1] // 2 - 10 : textY + size[1] // 2 + 10,
        textX - size[0] // 2 - 10 : textX + size[0] // 2 + 10,
    ] = np.uint8(
        0.5
        * frame[
            textY - size[1] // 2 - 10 : textY + size[1] // 2 + 10,
            textX - size[0] // 2 - 10 : textX + size[0] // 2 + 10,
        ]
    )

    cv2.putText(
        frame,
        displayText,
        (textX - size[0] // 2, textY + size[1] // 2),
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        0.8,
        (255, 255, 255),
        1,
    )

    return frame
