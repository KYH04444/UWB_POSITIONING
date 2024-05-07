import random
import numpy as np
import matplotlib.pyplot as plt

class Problem1:
    def __init__(self):
        self.map_size = [100, 100]
        self.uwb_anchors = [[random.randint(0, self.map_size[0]), random.randint(0, self.map_size[1])] for _ in range(5)]
    def cal_PEB(self, anchors):
        PEB_map = np.zeros((self.map_size[0], self.map_size[1]))
        PEB_map_sum = 0
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                J = np.zeros((2, 2))
                for anchor in anchors:
                    r = np.array([i, j]) - np.array(anchor)
                    if np.linalg.norm(np.array([i, j]) - np.array(anchor)) == 0:  
                        continue
                    unit_r = r / np.linalg.norm(r)
                    J += np.outer(unit_r, unit_r.transpose()) / 2
                PEB_map[i, j] = np.sqrt(np.trace(np.linalg.inv(J))) 
                PEB_map_sum += np.sqrt(np.trace(np.linalg.inv(J)))
        return PEB_map, PEB_map_sum
    
    def plot_PEB(self):
        PEB_map = self.cal_PEB(self.uwb_anchors)[0]
        im = plt.imshow(PEB_map, cmap='jet', origin='lower')
        plt.colorbar(im)
        plt.title('PEB TOA [m]')
        plt.xlabel('X [m]]')
        plt.ylabel('Y [m]')
        plt.show()

    def find_best_anchors(self):
        best_anchors = None
        best_average_PEB = float('inf')
        cnt = 0
        for _ in range(200):
            cnt += 1
            tmp_anchors = [[random.randint(0, self.map_size[0]), random.randint(0, self.map_size[1])] for _ in range(5)]
            PEB_map_sum = self.cal_PEB(tmp_anchors)[1]
            average_PEB = PEB_map_sum / (self.map_size[0] * self.map_size[1])

            if average_PEB < best_average_PEB:
                best_average_PEB = average_PEB
                best_anchors = tmp_anchors
            print("Left count: ",cnt,"/200")
        print("Best anchor positions:", best_anchors)
        return best_anchors
    
    def A(self):
        self.plot_PEB()

    def B(self):
        best_anchors = self.find_best_anchors()
        self.uwb_anchors = best_anchors
        self.plot_PEB()
        
class Problem2():

    def __init__(self):
        pass


if __name__ == '__main__':
    prob = Problem1()
    # prob.A()
    prob.B()