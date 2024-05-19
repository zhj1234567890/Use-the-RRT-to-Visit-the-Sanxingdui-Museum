import copy
import math
import random
import time

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import numpy as np

show_animation = True


class RRT:

    def __init__(self, obstacleList, randArea,
                 expandDis=2.0, goalSampleRate=10, maxIter=200):

        self.start = None
        self.goal = None
        self.min_rand = randArea[0]
        self.max_rand = randArea[1]
        self.expand_dis = expandDis
        self.goal_sample_rate = goalSampleRate
        self.max_iter = maxIter
        self.obstacle_list = obstacleList
        self.node_list = None

    def rrt_planning(self, start, goal, animation=True):
        start_time = time.time()
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.node_list = [self.start]
        path = None

        for i in range(self.max_iter):
            rnd = self.sample()
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearestNode = self.node_list[n_ind]

            # steer
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
            newNode = self.get_new_node(theta, n_ind, nearestNode)

            noCollision = self.check_segment_collision(newNode.x, newNode.y, nearestNode.x, nearestNode.y)
            if noCollision:
                self.node_list.append(newNode)
                if animation:
                    self.draw_graph(newNode, path)

                if self.is_near_goal(newNode):
                    if self.check_segment_collision(newNode.x, newNode.y,
                                                    self.goal.x, self.goal.y):
                        lastIndex = len(self.node_list) - 1
                        path = self.get_final_course(lastIndex)
                        pathLen = self.get_path_len(path)
                        print("current path length: {}, It costs {} s".format(pathLen, time.time()-start_time))

                        if animation:
                            self.draw_graph(newNode, path)
                        return path

    def rrt_star_planning(self, start, goal, animation=True):
        start_time = time.time()
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.node_list = [self.start]
        path = None
        lastPathLength = float('inf')

        for i in range(self.max_iter):
            rnd = self.sample()
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearestNode = self.node_list[n_ind]

            # steer
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
            newNode = self.get_new_node(theta, n_ind, nearestNode)

            noCollision = self.check_segment_collision(newNode.x, newNode.y, nearestNode.x, nearestNode.y)
            if noCollision:
                nearInds = self.find_near_nodes(newNode)
                newNode = self.choose_parent(newNode, nearInds)

                self.node_list.append(newNode)
                self.rewire(newNode, nearInds)

                if animation:
                    self.draw_graph(newNode, path)

                if self.is_near_goal(newNode):
                    if self.check_segment_collision(newNode.x, newNode.y,
                                                    self.goal.x, self.goal.y):
                        lastIndex = len(self.node_list) - 1

                        tempPath = self.get_final_course(lastIndex)
                        tempPathLen = self.get_path_len(tempPath)
                        if lastPathLength > tempPathLen:
                            path = tempPath
                            lastPathLength = tempPathLen
                            print("current path length: {}, It costs {} s".format(tempPathLen, time.time()-start_time))

        return path

    def informed_rrt_star_planning(self, start, goal, animation=True):
        start_time = time.time()
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.node_list = [self.start]
        # max length we expect to find in our 'informed' sample space,
        # starts as infinite
        cBest = float('inf')
        path = None

        # Computing the sampling space
        cMin = math.sqrt(pow(self.start.x - self.goal.x, 2)
                         + pow(self.start.y - self.goal.y, 2))
        xCenter = np.array([[(self.start.x + self.goal.x) / 2.0],
                            [(self.start.y + self.goal.y) / 2.0], [0]])
        a1 = np.array([[(self.goal.x - self.start.x) / cMin],
                       [(self.goal.y - self.start.y) / cMin], [0]])

        e_theta = math.atan2(a1[1], a1[0])

        # 论文方法求旋转矩阵（2选1）
        # first column of identity matrix transposed
        # id1_t = np.array([1.0, 0.0, 0.0]).reshape(1, 3)
        # M = a1 @ id1_t
        # U, S, Vh = np.linalg.svd(M, True, True)
        # C = np.dot(np.dot(U, np.diag(
        #     [1.0, 1.0, np.linalg.det(U) * np.linalg.det(np.transpose(Vh))])),
        #            Vh)

        # 直接用二维平面上的公式（2选1）
        C = np.array([[math.cos(e_theta), -math.sin(e_theta), 0],
                      [math.sin(e_theta), math.cos(e_theta),  0],
                      [0,                 0,                  1]])

        for i in range(self.max_iter):
            # Sample space is defined by cBest
            # cMin is the minimum distance between the start point and the goal
            # xCenter is the midpoint between the start and the goal
            # cBest changes when a new path is found

            rnd = self.informed_sample(cBest, cMin, xCenter, C)
            n_ind = self.get_nearest_list_index(self.node_list, rnd)
            nearestNode = self.node_list[n_ind]

            # steer
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
            newNode = self.get_new_node(theta, n_ind, nearestNode)

            noCollision = self.check_segment_collision(newNode.x, newNode.y, nearestNode.x, nearestNode.y)
            if noCollision:
                nearInds = self.find_near_nodes(newNode)
                newNode = self.choose_parent(newNode, nearInds)

                self.node_list.append(newNode)
                self.rewire(newNode, nearInds)

                if self.is_near_goal(newNode):
                    if self.check_segment_collision(newNode.x, newNode.y,
                                                    self.goal.x, self.goal.y):
                        lastIndex = len(self.node_list) - 1
                        tempPath = self.get_final_course(lastIndex)
                        tempPathLen = self.get_path_len(tempPath)
                        if tempPathLen < cBest:
                            path = tempPath
                            cBest = tempPathLen
                            print("current path length: {}, It costs {} s".format(tempPathLen, time.time()-start_time))
            if animation:
                self.draw_graph_informed_RRTStar(xCenter=xCenter,
                                                cBest=cBest, cMin=cMin,
                                                e_theta=e_theta, rnd=rnd, path=path)

        return path

    def sample(self):
        if random.randint(0, 300) > self.goal_sample_rate:
            rnd = [random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand)]
        else:  # goal point sampling
            rnd = [self.goal.x, self.goal.y]
        return rnd

    def choose_parent(self, newNode, nearInds):
        if len(nearInds) == 0:
            return newNode

        dList = []
        for i in nearInds:
            dx = newNode.x - self.node_list[i].x
            dy = newNode.y - self.node_list[i].y
            d = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            if self.check_collision(self.node_list[i], theta, d):
                dList.append(self.node_list[i].cost + d)
            else:
                dList.append(float('inf'))

        minCost = min(dList)
        minInd = nearInds[dList.index(minCost)]

        if minCost == float('inf'):
            print("min cost is inf")
            return newNode

        newNode.cost = minCost
        newNode.parent = minInd

        return newNode

    def find_near_nodes(self, newNode):
        n_node = len(self.node_list)
        r = 50.0 * math.sqrt((math.log(n_node) / n_node))
        d_list = [(node.x - newNode.x) ** 2 + (node.y - newNode.y) ** 2
                  for node in self.node_list]
        near_inds = [d_list.index(i) for i in d_list if i <= r ** 2]
        return near_inds

    def informed_sample(self, cMax, cMin, xCenter, C):
        if cMax < float('inf'):
            r = [cMax / 2.0,
                 math.sqrt(cMax ** 2 - cMin ** 2) / 2.0,
                 math.sqrt(cMax ** 2 - cMin ** 2) / 2.0]
            L = np.diag(r)
            xBall = self.sample_unit_ball()
            rnd = np.dot(np.dot(C, L), xBall) + xCenter
            rnd = [rnd[(0, 0)], rnd[(1, 0)]]
        else:
            rnd = self.sample()

        return rnd

    @staticmethod
    def sample_unit_ball():
        a = random.random()
        b = random.random()

        if b < a:
            a, b = b, a

        sample = (b * math.cos(2 * math.pi * a / b),
                  b * math.sin(2 * math.pi * a / b))
        return np.array([[sample[0]], [sample[1]], [0]])

    @staticmethod
    def get_path_len(path):
        pathLen = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            pathLen += math.sqrt((node1_x - node2_x)
                                 ** 2 + (node1_y - node2_y) ** 2)

        return pathLen

    @staticmethod
    def line_cost(node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    @staticmethod
    def get_nearest_list_index(nodes, rnd):
        dList = [(node.x - rnd[0]) ** 2
                 + (node.y - rnd[1]) ** 2 for node in nodes]
        minIndex = dList.index(min(dList))
        return minIndex

    def get_new_node(self, theta, n_ind, nearestNode):
        newNode = copy.deepcopy(nearestNode)

        newNode.x += self.expand_dis * math.cos(theta)
        newNode.y += self.expand_dis * math.sin(theta)

        newNode.cost += self.expand_dis
        newNode.parent = n_ind
        return newNode

    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)
        if d < self.expand_dis:
            return True
        return False

    def rewire(self, newNode, nearInds):
        n_node = len(self.node_list)
        for i in nearInds:
            nearNode = self.node_list[i]

            d = math.sqrt((nearNode.x - newNode.x) ** 2
                          + (nearNode.y - newNode.y) ** 2)

            s_cost = newNode.cost + d

            if nearNode.cost > s_cost:
                theta = math.atan2(newNode.y - nearNode.y,
                                   newNode.x - nearNode.x)
                if self.check_collision(nearNode, theta, d):
                    nearNode.parent = n_node - 1
                    nearNode.cost = s_cost

    @staticmethod
    def distance_squared_point_to_segment(v, w, p):
        # Return minimum distance between line segment vw and point p
        if np.array_equal(v, w):
            return (p - v).dot(p - v)  # v == w case
        l2 = (w - v).dot(w - v)  # i.e. |w-v|^2 -  avoid a sqrt
        # Consider the line extending the segment,
        # parameterized as v + t (w - v).
        # We find projection of point p onto the line.
        # It falls where t = [(p-v) . (w-v)] / |w-v|^2
        # We clamp t from [0,1] to handle points outside the segment vw.
        t = max(0, min(1, (p - v).dot(w - v) / l2))
        projection = v + t * (w - v)  # Projection falls on the segment
        return (p - projection).dot(p - projection)

    def check_segment_collision(self, x1, y1, x2, y2):
        for (ox, oy, size) in self.obstacle_list:
            dd = self.distance_squared_point_to_segment(
                np.array([x1, y1]),
                np.array([x2, y2]),
                np.array([ox, oy]))
            if dd <= size ** 2:
                return False  # collision
        return True

    def check_collision(self, nearNode, theta, d):
        tmpNode = copy.deepcopy(nearNode)
        end_x = tmpNode.x + math.cos(theta) * d
        end_y = tmpNode.y + math.sin(theta) * d
        return self.check_segment_collision(tmpNode.x, tmpNode.y, end_x, end_y)

    def get_final_course(self, lastIndex):
        path = [[self.goal.x, self.goal.y]]
        while self.node_list[lastIndex].parent is not None:
            node = self.node_list[lastIndex]
            path.append([node.x, node.y])
            lastIndex = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def draw_graph_informed_RRTStar(self, xCenter=None, cBest=None, cMin=None, e_theta=None, rnd=None, path=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
            if cBest != float('inf'):
                self.plot_ellipse(xCenter, cBest, cMin, e_theta)

        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x, self.node_list[node.parent].x], [
                        node.y, self.node_list[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacle_list:
            plt.plot(ox, oy, "ok", ms=30 * size)

        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        plt.axis([-2, 18, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_ellipse(xCenter, cBest, cMin, e_theta):  # pragma: no cover

        a = math.sqrt(cBest ** 2 - cMin ** 2) / 2.0
        b = cBest / 2.0
        angle = math.pi / 2.0 - e_theta
        cx = xCenter[0]
        cy = xCenter[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
        fx = rot @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, "xc")
        plt.plot(px, py, "--c")

    def draw_graph(self, rnd=None, path=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")

        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x, self.node_list[node.parent].x], [
                        node.y, self.node_list[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacle_list:
            # self.plot_circle(ox, oy, size)
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")

        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')

        plt.axis([-2, 130, -2, 250])
        plt.grid(True)
        plt.pause(0.01)


class Node:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None


def main():
    print("Start rrt planning")

    # create obstacles
    obstacleList = [
        (50,  109,  0.4),
        (45, 101,  1),
        (47,  92,  1),
        (51,  84, 0.4),
        (54, 78, 0.3),
        (57, 76, 0.3),
        (60, 70, 0.3),
        (63, 67, 0.3),
        (65, 64, 0.3),
        (69, 62, 0.3),
        (73, 64, 0.3),
        (77, 65, 0.3),
        (81, 68, 0.3),
        (84, 70, 0.3),
        (87, 72, 0.3),
        (70, 80, 0.4),
        (68, 75, 0.2),
        (80, 93, 0.5),
        (83, 86, 0.4),
        (84, 81, 0.2),
        (74, 110, 0.2),
        (73, 141, 0.5),
        (73, 134, 0.3),
        (76, 129, 0.5),
        (78, 122, 0.3),
        (79, 118, 0.3),
        (57, 138, 0.1),
        (59, 134, 0.1),
        (53, 163, 0.2),
        (64, 170, 0.2),
        (65, 177, 0.3),
        (69, 173, 0.3),
        (72, 169, 0.3),
        (75, 164, 0.3),
        (60, 183, 0.2),
        (63, 185, 0.2),
        (69, 184, 0.5),
        (44, 221, 0.2),
        (49, 223, 0.3),
        (56, 221, 0.4),
        (63, 216, 0.6),
        (59, 211, 0.3),
        (65, 208, 0.4),
        (64, 238.00200000000007, 0.3),
        (65, 230.91500000000008, 0.3),
        (66, 223.8280000000001, 0.3),
        (67, 216.74100000000004, 0.3),
        (68, 209.65400000000005, 0.3),
        (69, 202.56700000000006, 0.3),
        (70, 195.48000000000008, 0.3),
        (71, 188.3930000000001, 0.3),
        (72, 181.30600000000004, 0.3),
        (73, 174.21900000000005, 0.3),
        (74, 167.13200000000006, 0.3),
        (75, 160.04500000000007, 0.3),
        (76, 152.95800000000008, 0.3),
        (77, 145.8710000000001, 0.3),
        (78, 138.7840000000001, 0.3),
        (79, 131.69700000000012, 0.3),
        (80, 124.61000000000001, 0.3),
        (81, 117.52300000000002, 0.3),
        (82, 110.43600000000004, 0.3),
        (83, 103.34900000000005, 0.3),
        (84, 96.26200000000006, 0.3),
        (85, 89.17500000000007, 0.3),
        (86, 82.08800000000008, 0.3),
        (87, 75.00100000000009, 0.3),
        (88, 67.9140000000001, 0.3),
        (89, 60.82700000000011, 0.3),

        (45, 143.005, 0.2),
        (46, 141.394, 0.2),
        (47, 139.78300000000002, 0.2),
        (48, 138.172, 0.2),
        (49, 136.561, 0.2),
        (50, 134.95, 0.2),
        (51, 133.339, 0.2),
        (52, 131.728, 0.2),
        (53, 130.11700000000002, 0.2),
        (54, 128.506, 0.2),
        (55, 126.895, 0.2),
        (56, 125.284, 0.2),
        (57, 123.673, 0.2),
        (58, 122.062, 0.2),
        (59, 120.45100000000001, 0.2),
        (60, 118.84, 0.2),
        (61, 117.229, 0.2),
        (62, 115.618, 0.2),

        (25, 195.993, 0.2),
        (26, 194.441, 0.2),
        (27, 192.889, 0.2),
        (28, 191.337, 0.2),
        (29, 189.785, 0.2),
        (30, 188.233, 0.2),
        (31, 186.681, 0.2),
        (32, 185.12900000000002, 0.2),
        (33, 183.577, 0.2),
        (34, 182.025, 0.2),
        (35, 180.473, 0.2),
        (36, 178.921, 0.2),
        (37, 177.369, 0.2),
        (38, 175.817, 0.2),
        (39, 174.26500000000001, 0.2),
        (40, 172.71300000000002, 0.2),
        (41, 171.161, 0.2),
        (42, 169.609, 0.2),
        (43, 168.05700000000002, 0.2),
        (44, 166.505, 0.2),
        (45, 164.953, 0.2),
        (46, 163.401, 0.2),
        (47, 161.849, 0.2),
        (48, 160.297, 0.2),
        (49, 158.745, 0.2),
        (50, 157.19299999999998, 0.2),
        (51, 155.64100000000002, 0.2),
        (52, 154.089, 0.2),
        (53, 152.537, 0.2),

        (48, 215, 0.03),
        (51, 217, 0.03),
        (50, 214, 0.03),
        (57, 201, 0.03),
        (55, 197, 0.03),
        (63, 192, 0.03),
        (47, 183, 0.03),
        (49, 184, 0.03),
        (49, 180, 0.03),
        (50, 182, 0.03),
        (51, 178, 0.03),
        (52, 179, 0.03),
        (52, 175, 0.03),
        (54, 176, 0.03),
        (54, 173, 0.03),
        (55, 175, 0.03),
        (58, 168, 0.03),
        (62, 162, 0.03),
        (63, 160, 0.03),
        (64, 158, 0.03),
        (65, 156, 0.03),
        (66, 149, 0.03),
        (68, 146, 0.03),
        (60, 139, 0.03),
        (55, 136, 0.03),
        (63, 126, 0.03),
        (65, 127, 0.03),
        (67, 128, 0.03),
        (65, 124, 0.03),
        (67, 125, 0.03),
        (69, 126, 0.03),
        (73, 102, 0.03),
        (76, 104, 0.03),
        (79, 106, 0.03),
        (75, 100, 0.03),
        (78, 102, 0.03),
        (81, 104, 0.03),
        (32, 202.0, 0.05),
        (32, 202.0, 0.05),
        (32, 202.0, 0.05),
        (33, 200.39999999999998, 0.05),
        (34, 198.79999999999998, 0.05),
        (35, 197.2, 0.05),
        (36, 195.6, 0.05),
        (37, 194.0, 0.05),
        (38, 192.39999999999998, 0.05),
        (39, 190.79999999999998, 0.05),
        (40, 189.2, 0.05),
        (44, 182.79999999999998, 0.05),
        (45, 181.2, 0.05),
        (46, 179.59999999999997, 0.05),
        (49, 174.79999999999998, 0.05),
        (50, 173.2, 0.05),
        (57, 162.0, 0.05),
        (61, 155.59999999999997, 0.05),
        (45, 200.2, 0.05),
        (46, 198.59999999999997, 0.05),
        (47, 197.0, 0.05),
        (48, 195.39999999999998, 0.05),
        (52, 189.0, 0.05),
        (53, 187.39999999999998, 0.05),
        (54, 185.79999999999998, 0.05),
        (56, 182.59999999999997, 0.05),
        (57, 181.0, 0.05),
        (64, 169.79999999999998, 0.05),
        (65, 168.2, 0.05),
        (68, 163.39999999999998, 0.05),
        (69, 161.79999999999998, 0.05),
        (45, 174.00015, 0.05),
        (46, 174.66682, 0.05),
        (47, 175.33348999999998, 0.05),
        (48, 176.00016, 0.05),
        (54, 180.00018, 0.05),
        (55, 180.66685, 0.05),
        (56, 181.33352, 0.05),
        (57, 182.00019, 0.05),
        (45, 187.00015, 0.05),
        (46, 187.66682, 0.05),
        (47, 188.33348999999998, 0.05),
        (48, 189.00016, 0.05),
        (40, 212.6668, 0.05),
        (41, 213.33347, 0.05),
        (42, 214.00014, 0.05),
        (43, 214.66681, 0.05),
        (43, 209.66681, 0.05),
        (44, 210.33348, 0.05),
        (45, 211.00015, 0.05),
        (46, 211.66682, 0.05),
        (51, 205.00017, 0.05),
        (52, 205.66684, 0.05),
        (53, 206.33351, 0.05),
        (54, 207.00018, 0.05),
        (66, 120.00022, 0.05),
        (67, 120.66689, 0.05),
        (68, 121.33356, 0.05),
        (65, 112.33355, 0.05),
        (66, 113.00022, 0.05),
        (67, 113.66689, 0.05),
        (68, 114.33356, 0.05),
        (71, 116.33357000000001, 0.05),
        (72, 117.00023999999999, 0.05),
        (73, 117.66691, 0.05),
        (74, 118.33358, 0.05),
        (75, 119.00025, 0.05),
        (76, 119.66692, 0.05),
        (40, 213.2, 0.05),
        (41, 211.59999999999997, 0.05),
        (42, 210.0, 0.05),
        (43, 116.66681, 0.2),
        (44, 117.33348, 0.2),
        (45, 118.00014999999999, 0.2),
        (46, 118.66682, 0.2),
        (47, 119.33349, 0.2),
        (48, 120.00016, 0.2),
        (49, 120.66683, 0.2),
        (50, 121.3335, 0.2),
        (51, 122.00017, 0.2),
        (52, 122.66684000000001, 0.2),
        (53, 123.33350999999999, 0.2),
        (54, 124.00018, 0.2),
        (63, 107, 0.05),
        (60, 93.0, 0.05),
        (61, 94.25, 0.05),
        (62, 95.5, 0.05),
        (63, 96.75, 0.05),
        (64, 96.0, 0.05),
        (65, 93.19999999999999, 0.05),
        (66, 90.4, 0.05),
        (67, 87.6, 0.05),
        (68, 84.80000000000001, 0.05),
        (45, 96, 0.5),
        (45, 105, 0.5),
    ]
    # obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
    #                 (9, 5, 2), (8, 10, 1)]

    # Set params
    rrt = RRT(randArea=[-2, 300], obstacleList=obstacleList, maxIter=20000)
    path = rrt.rrt_planning(start=[52, 118], goal=[33, 206], animation=show_animation)
    # path = rrt.rrt_star_planning(start=[0, 0], goal=[15, 12], animation=show_animation)
    # path = rrt.informed_rrt_star_planning(start=[0, 0], goal=[15, 12], animation=show_animation)
    print("Done!!")

    if show_animation and path:
        plt.show()


if __name__ == '__main__':
    main()
