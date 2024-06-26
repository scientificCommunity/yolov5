import torch
import pyautogui
import cv2
import heapq
import numpy as np
import time

import networkx as nx

import detect2


def create_grid(image, detections, grid_size=50):
    """
    将图像转换为网格，每个单元格包含是否有障碍物的信息。
    """
    height, width, _ = image.shape
    grid_height = height // grid_size
    grid_width = width // grid_size

    grid = np.zeros((grid_height, grid_width))

    for *xyxy, conf, cls in detections.xyxy[0].tolist():
        x1, y1, x2, y2 = map(int, xyxy)
        grid_x1, grid_y1 = x1 // grid_size, y1 // grid_size
        grid_x2, grid_y2 = x2 // grid_size, y2 // grid_size
        grid[grid_y1:grid_y2 + 1, grid_x1:grid_x2 + 1] = 1

    return grid


def a_star_search(grid, start, goal):
    """
    实现A*算法进行路径规划。
    """

    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)

            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False


# 加载模型
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model = torch.hub.load(".", "custom", path="/Users/kuntang/PycharmProjects/yolov5/runs/train/exp8/weights/best.pt"
                       , source="local")  # or yolov5n - yolov5x6, custom
grid_size = 50
current_player_position_x = 10
current_player_position_y = 10


# 截图函数
def capture_screenshot():
    screenshot = pyautogui.screenshot()
    # screenshot.save("./test.png")
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screenshot


# 获取资源节点位置
def get_resource_position(detections):
    for *xyxy, conf, cls in detections.xyxy[0].tolist():
        if conf > 0.5 and int(cls) == 16:  # 确保是资源节点
            x_center = int((xyxy[0] + xyxy[2]) / 2)
            y_center = int((xyxy[1] + xyxy[3]) / 2)
            return x_center, y_center
    return None


names = ['40cutton', '40cutton_flag', '40tree', '40tree_flag', '40tree_leaf', '41cutton', '41tree', '41tree_flag',
         '42cutton', '42cutton_flag', '42tree', '50cutton', '50cutton_flag', '50cutton_sprite',
         '50cutton_sprite_attack',
         '50cutton_sprite_dead', '50tree', '50tree_leaf', '51cutton', '51cutton_flag', '51tree', '53tree',
         'another_area', 'been_attacted', 'big_tree_root', 'bridge_margin', 'exhausted', 'fire_basket', 'green_hollow',
         'horse_down', 'horse_up', 'lake', 'lake_margin', 'land_margin', 'map_location', 'pier', 'player', 'processing',
         'stone', 'tower', 'wall', 'yellow_hollow']
resource_flag_set = {16, names.index('40cutton'), names.index('40tree'), names.index('41cutton'), names.index('41tree'),
                     names.index('42cutton'), names.index('42tree'), names.index('50cutton'), names.index('50tree'),
                     names.index('51cutton'), names.index('51tree')}


def get_resources_position(detections):
    resource_location_set = {}
    i = 0
    for *xyxy, conf, cls in detections.xyxy[0].tolist():
        if conf > 0.5 and int(cls) in resource_flag_set:  # 确保是资源节点
            x_center = int((xyxy[0] + xyxy[2]) / 2)
            y_center = int((xyxy[1] + xyxy[3]) / 2)
            resource_location = [x_center, y_center]
            resource_location_set[i] = resource_location
            i += 1
    return resource_location_set


def main_logic():
    # screenshot = capture_screenshot()
    img = "dataset/tmp/albion/4000.png"  # or file, Path, PIL, OpenCV, numpy, list
    # results = model(img)
    results = detect2.main1("/Users/kuntang/PycharmProjects/yolov5/runs/train/exp8/weights/best.pt", img, False)
    # results.show()

    # grid = create_grid(screenshot, results)
    # start = (current_player_position_x, current_player_position_y)
    resources_pos = get_resources_position(results)

    return resources_pos
    # for resources_po in resources_pos:
    #     # todo 遍历资源并找到离中心点最近的资源&寻路
    #     print()
    # # if resource_pos:
    # #     goal = (resource_pos[1] // grid_size, resource_pos[0] // grid_size)
    # #     path = a_star_search(grid, start, goal)
    # #
    # #     if path:
    # #         for step in path:
    # #             pyautogui.click(step[1] * grid_size, step[0] * grid_size)
    # #             time.sleep(0.1)  # 确保动作完成
    #
    # time.sleep(1)


# 主循环
# while True:
#     main_logic()
if __name__ == '__main__':
    main_logic()
