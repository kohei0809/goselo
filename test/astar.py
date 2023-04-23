import numpy as np
import math

from node import Node

class Astar:
    def __init__(self, map_, start_, goal_) -> None:
        self.map = map_
        self.start = start_
        self.goal = goal_
    
    def getRoot(self) -> list:
        start_node = Node(self.start[0], self.start[1], None)
        end_node = Node(self.goal[0], self.goal[1], None)
        #print("map")
        #print("(" + str(self.map.shape[1]) + ", " + str(self.map.shape[0]) + ")")
        
        # Initialize both open and closed list
        open_list = []
        closed_list = []
        
        # Add the start node
        open_list.append(start_node)
        
        while len(open_list) > 0:
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                #open_listの中でfが最小のノードを選択
                #print("n1(" + str(current_node.x) + ", " + str(current_node.y) + "):" + str(current_node.f))
                #print("n2(" + str(item.x) + ", " + str(item.y) + "):" + str(item.f))
                
                if item.f < current_node.f:
                    current_node = item
                    current_index = index
                    
            #current_nodeをopen_listから削除して、closed_listに追加
            open_list.pop(current_index)
            closed_list.append(current_node)
            
            #print("current_node=(" + str(current_node.x) + ", " + str(current_node.y) + ")")
            #goalに到達していれば終了
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append((current.x, current.y))
                    #print("current=(" + str(current.x) + ", " + str(current.y) + ")")
                    #print("parent=(" + str(current.parent.x) + ", " + str(current.parent.y) + ")")
                    current = current.parent
                
                #pathを逆順にして返す
                return path[::-1]
        
            #Generate children
            children = []
                                #上         下      左      右      左上        左下       右上     右下                 
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                # Get node position
                node_position = (current_node.x + new_position[0], current_node.y + new_position[1])
                
                #map内の移動か
                if (node_position[0] > (self.map.shape[1] - 1)) or (node_position[0] < 0) or (node_position[1] > (self.map.shape[0] - 1)) or (node_position[1] < 0):
                    continue
                
                #壁ではないか
                if self.map[node_position[1]][node_position[0]] != 0:
                    continue
                
                #移動できる位置のノードのみを生成
                new_node = Node(node_position[0], node_position[1], parent_=current_node)
                
                #childrenに追加
                children.append(new_node)
                
            #各子ノードでg, h, fを計算
            current_g = current_node.g
            for child in children:
                d = math.sqrt((child.x - current_node.x) ** 2 + (child.y - current_node.y) ** 2)
                g_ = current_g + d
                if (child in closed_list) and (closed_list[closed_list.index(child)].g < g_):
                    continue
                
                #print("child=(" + str(child.x) + ", " + str(child.y) + "):" + str(g_))
                child.g = g_
                child.h = math.sqrt(((child.x - end_node.x) ** 2) + ((child.y - end_node.y) ** 2))
                #child.h = max((child.x - end_node.x), (child.y - end_node.y))
                child.f = child.g + child.h
                
                if child in open_list:
                    continue
                
                open_list.append(child)
        
        
def example1() -> None:
    my_map = np.array([[0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 1, 0]])        
    start = (0, 0)
    end = (4, 4)
    
    path = Astar(my_map, start, end).getRoot()
    print(path)
    
def example2() -> None:
    my_map = np.array([[0, 0, 0, 0, 0],
                       [0, 1, 0, 1, 0],
                       [0, 1, 0, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 1, 0]])        
    start = (1, 4)
    end = (4, 4)
    
    path = Astar(my_map, start, end).getRoot()
    print(path)
     
def example3() -> None:
    my_map = np.array([[0, 0, 0, 0, 0],
                       [0, 1, 0, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 1, 0]])        
    start = (1, 3)
    end = (4, 3)
    
    path = Astar(my_map, start, end).getRoot()
    print(path)
        
if __name__ == '__main__':
    #example1()
    example2()
    #example3()