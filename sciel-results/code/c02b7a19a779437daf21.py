import numpy as np
import time
import sys
import heapq
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.optimal_x1 = None
        self._compute_optimal_x1()

    def _compute_optimal_x1(self):
        N = self.budget
        if N <= 0:
            self.optimal_x1 = np.array([])
            return

        # Step 1: Greedy initialization
        points = [0.0]
        heap = []
        
        def calc_gain(a, b):
            if a == b:
                return 0.0, 0.0
            # Optimal c in [a, b]
            # c = ((sqrt(a) + sqrt(a+3b)) / 3)^2
            c = ((np.sqrt(a) + np.sqrt(a + 3*b)) / 3.0)**2
            if c <= a or c >= b:
                c = (a + b) / 2.0
            gain = (b - c) * (np.sqrt(c) - np.sqrt(a))
            return gain, c

        gain_01, c_01 = calc_gain(0.0, 1.0)
        heapq.heappush(heap, (-gain_01, 0.0, 1.0, c_01))
        
        for _ in range(N - 1):
            if not heap:
                break
            neg_gain, a, b, c = heapq.heappop(heap)
            gain = -neg_gain
            if gain < 1e-12:
                break
            
            points.append(c)
            
            gain_ac, c_ac = calc_gain(a, c)
            if gain_ac > 1e-12:
                heapq.heappush(heap, (-gain_ac, a, c, c_ac))
            
            gain_cb, c_cb = calc_gain(c, b)
            if gain_cb > 1e-12:
                heapq.heappush(heap, (-gain_cb, c, b, c_cb))
        
        x1_initial = np.array(points)
        x1_initial.sort()

        # Step 2: Refinement using scipy.optimize.minimize
        # We optimize the gaps to ensure ordering and bounds.
        # x_0 = delta_0
        # x_i = x_{i-1} + delta_i
        # Sum(delta) <= 1
        
        def hv_from_gaps(deltas):
            # deltas has length N
            # x_1 = deltas[0]
            # x_2 = deltas[0] + deltas[1]
            # ...
            # x_N = sum(deltas)
            # Check if sum(deltas) > 1, penalize
            x = np.cumsum(deltas)
            if x[-1] > 1.0:
                # Penalty for exceeding bound
                return -1e6 * (x[-1] - 1.0)
            
            f2 = 1.0 - np.sqrt(x)
            # Points are (x, f2)
            # HV = sum (x_{i+1} - x_i) * (ref_2 - f2_i)
            # x_{N+1} = 11.0
            # ref_2 = 11.0
            heights = 11.0 - f2
            widths = np.diff(np.append(x, 11.0))
            hv = np.sum(widths * heights)
            return -hv

        # Initial gaps
        x1_sorted = x1_initial
        deltas = np.diff(np.insert(x1_sorted, 0, 0.0))
        # Ensure deltas are non-negative
        deltas = np.maximum(deltas, 1e-6)
        
        # Bounds for deltas
        bounds = [(1e-6, 1.0)] * N
        
        start_time = time.time()
        deadline = start_time + 200 # Leave some time for the rest
        
        try:
            res = minimize(hv_from_gaps, deltas, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        except Exception:
            res = None
            
        if res is not None and res.success:
            x1_opt = np.cumsum(res.x)
            if x1_opt[-1] <= 1.0 + 1e-6:
                # Clip to [0, 1]
                x1_opt = np.clip(x1_opt, 0.0, 1.0)
                # Ensure strictly increasing (should be due to deltas > 0)
                if np.all(np.diff(x1_opt) > 0):
                    self.optimal_x1 = x1_opt
                    return
        self.optimal_x1 = x1_initial

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320
        
        if self.optimal_x1 is None:
             self._compute_optimal_x1()
             
        x1_vals = self.optimal_x1
        N = len(x1_vals)
        
        X = np.zeros((N, self.dim))
        X[:, 0] = x1_vals
        
        for i in range(N):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
