class Node:
    def __init__(self, x_, y_, parent_=None) -> None:
        self.x = x_
        self.y = y_
        self.parent = parent_
        self.g = 0
        self.h = 0
        self.f = 0
        
    def __eq__(self, other) -> bool:
        return (self.x == other.x) and (self.y == other.y)