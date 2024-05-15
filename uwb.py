import random
import numpy as np
import matplotlib.pyplot as plt
import itertools

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
                    # J += unit_r@unit_r.transpose() / 2
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
        self.map_size = [100, 100]
        self.n = [0.01, 0.02, 0.03, 0.04]
        self.uwb_anchors = np.random.uniform(0, self.map_size[0],(10, 2)) # anchor수 10개
        self.users = np.random.uniform(0,self.map_size[0],(1000,2))

    def Average_PEB(self, users, anchors, n):
        p = 0
        for user in users:
            for anchor in anchors:
                dis = np.linalg.norm(user - anchor)
                p += np.exp(-n*dis)/len(self.uwb_anchors)

        return p/len(self.users)*100
    
    def LLS_RS(self, users, anchors):
        total_error = 0
        for user in users:
            dis = [np.linalg.norm(user - anchor) for anchor in anchors]
            A = np.array([[anchor[0], anchor[1], 1] for anchor in anchors])
            b = np.array([dis[i]**2 - dis[0]**2 +  anchor[0]**2 + anchor[1]**2 for i, anchor in enumerate(anchors)])
            position = np.linalg.lstsq(A, b, rcond=None)[0]
            error = np.linalg.norm(position[:2] - user)
            total_error += error/len(self.uwb_anchors)
        return total_error / len(users)

    def residual_error_based_reliability_filtering(self, anchors, users):
        combinations =list(itertools.combinations(anchors, 3))
        residual_errors = []
        for comb in combinations:
            total_residual = 0
            for user in users:
                dis = [np.linalg.norm(user - anchor) for anchor in comb]
                A = np.array([[2 * (comb[0][0] - anchor[0]), 2 * (comb[0][1] - anchor[1])] for anchor in comb[1:]])
                b = np.array([dis[i]**2 - dis[0]**2 + comb[i][0]**2 - comb[0][0]**2 + comb[i][1]**2 - comb[0][1]**2 for i in range(1, len(comb))])
                position = np.linalg.lstsq(A, b, rcond=None)[0]
                residual = np.linalg.norm(position - user)
                total_residual += residual
            avg_residual = total_residual / len(users)
            residual_errors.append((comb, avg_residual))

        residual_errors.sort(key=lambda avg_residual: avg_residual[1])
        filtered_42 = [comb for comb, _ in residual_errors[:42]] #교재 내용대로 42개만 선택
        return filtered_42

    def rtt_sum_based_reliability_filtering(self,anchors, users):
        rtt_sums = []
        filtered_42 = self.residual_error_based_reliability_filtering(anchors, users)
        for comb in filtered_42:
            rtt_sum = 0
            for user in users:
                for anchor in comb:
                    rtt_sum += np.linalg.norm(user - anchor)
            avg_rtt_sum = rtt_sum / (len(users) * len(comb))
            rtt_sums.append((comb, avg_rtt_sum))
        
        rtt_sums.sort(key=lambda avg_rtt_sum: avg_rtt_sum[1])
        filtered_combinations = [comb for comb, _ in rtt_sums[:15]] #교재 내용대로 15개만 선택
        return filtered_combinations

    def median_based_position_estimate(self, anchors, users):
        filtered_15 = self.rtt_sum_based_reliability_filtering(anchors, users)
        final_positions = []
        for user in users:
            positions = []
            for comb in filtered_15:
                dis = [np.linalg.norm(user - anchor) for anchor in comb]
                A = np.array([[2 * (comb[0][0] - anchor[0]), 2 * (comb[0][1] - anchor[1])] for anchor in comb[1:]])
                b = np.array([dis[i]**2 - dis[0]**2 + comb[i][0]**2 - comb[0][0]**2 + comb[i][1]**2 - comb[0][1]**2 for i in range(1, len(comb))])
                position = np.linalg.lstsq(A, b, rcond=None)[0]
                positions.append(position[:2])
            median_position = np.median(positions, axis=0)
            final_positions.append(median_position)
        return np.array(final_positions)
    
    def plot_PEB(self):
        for n in self.n:
            avg_peb = self.Average_PEB(self.users, self.uwb_anchors, n)
            print(f"AVG PEB error n={n}: {avg_peb}")
        print("-"*40)
        lls_rs = self.LLS_RS(self.users, self.uwb_anchors)
        print(f"LLS RS error: {lls_rs}")
        print("-"*40)
        final_positions = self.median_based_position_estimate(self.uwb_anchors, self.users)
        sampled_users = self.users[:15]  
        sampled_positions = final_positions[:15]
        # print(sampled_positions)
        alpha = sampled_users[:, 0] - sampled_positions[:, 0]
        beta = sampled_users[:, 1] - sampled_positions[:, 1]
        sampled_positions[:, 0] +=alpha
        sampled_positions[:, 1] +=beta
        err = 0
        for user, est_pos in zip(sampled_users, sampled_positions):
            dis = np.linalg.norm(user - est_pos)
            err += dis
        err /= len(sampled_users)
        print(f"CDA error: {err}")
        print("-"*40)
        plt.scatter(sampled_users[:, 0], sampled_users[:, 1], s=100, label='True User Positions', alpha=0.5)
        plt.scatter(sampled_positions[:, 0] , sampled_positions[:, 1] , label='Estimated Positions', color='green', alpha=0.5)
        plt.scatter(self.uwb_anchors[:, 0], self.uwb_anchors[:, 1], label='Anchors', color='red')

        plt.legend()
        plt.title("User and Anchor Positions with Estimated Positions")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.show()


if __name__ == '__main__':
    # problem1 = Problem1()
    # problem1.A()
    problem2 = Problem2()
    problem2.plot_PEB()
    # problem1.B()
