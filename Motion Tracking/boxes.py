import numpy as np


class Clickable:
    def __init__(self, x, y, parent) -> None:
        """Simple clickable class that stores a point that can be selected and moved trhough a frame with the mouse"""
        self.x = x
        self.y = y
        # The parent box will check if the clickable is selected to know how to update its shape
        self.selected = False

        # This isn't really used as by now, but may be usefull in the future
        self.parent: "Box" = parent

    def is_in_range(self, x: int, y: int) -> bool:
        """Check if the clickable is in the range of the mouse position"""
        size = 7
        return self.x - size < x < self.x + size and self.y - size < y < self.y + size


class Box:
    def __init__(self, x0, y0, x1, y1) -> None:
        """A box selection on a frame. It has two top left and bottom right corners that can be moved with the mouse"""
        self.cornerA = Clickable(x0, y0, self)
        self.cornerB = Clickable(x1, y1, self)
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def update(self):
        """Update the box selection, by simply changing its corner points based on the corner clickable that was selected"""
        if self.cornerA.selected:
            self.x0 = self.cornerA.x
            self.y0 = self.cornerA.y
        if self.cornerB.selected:
            self.x1 = self.cornerB.x
            self.y1 = self.cornerB.y

        # Update clickable positions:
        self.cornerA.x = self.x0
        self.cornerA.y = self.y0
        self.cornerB.x = self.x1
        self.cornerB.y = self.y1

    def get_bounding_box(self):
        """Get the bounding box of the box selection, as not always the corners are ordered correctly"""
        self.BBx0 = np.min([self.x0, self.x1])
        self.BBx1 = np.max([self.x0, self.x1])
        self.BBy0 = np.min([self.y0, self.y1])
        self.BBy1 = np.max([self.y0, self.y1])
        return self.BBx0, self.BBy0, self.BBx1, self.BBy1

    def is_inside(self, x: int, y: int) -> bool:
        """Check if the point is inside the box selection"""
        self.get_bounding_box()
        return self.BBx0 < x < self.BBx1 and self.BBy0 < y < self.BBy1

    def __repr__(self) -> str:
        return f"Query ({self.x0}, {self.y0}) -> ({self.x1}, {self.y1})"
