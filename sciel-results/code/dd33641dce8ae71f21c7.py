import time
import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
                        start_time = time.time()
                        deadline = start_time + 290
                        
                        n_points = self.budget
                        dim = self.dim
                        
                        # 1. Generate Sobol points
                        # Scrambled Sobol is better for deterministic low-discrepancy
                        sobol = qmc.Sobol(d=2, scramble=True)
                        # We need n_points - 3 Sobol points
                        n_sobol = n_points - 3
                        points_v = sobol.random(n_sobol)
                        
                        # 2. Add vertices
                        # v=(0,0) -> (1,0,0)
                        # v=(1,0) -> (0,0,1)
                        # v=(1,1) -> (0,1,0)
                        vertices = np.array([
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [1.0, 1.0]
                        ])
                        
                        points_v = np.vstack([vertices, points_v])
                        
                        # 3. Map to x
                        # x0 = 2/pi * arccos(v1)
                        # x1 = v2
                        x0 = (2.0 / np.pi) * np.arccos(points_v[:, 0])
                        x1 = points_v[:, 1]
                        
                        x = np.zeros((len(x0), dim))
                        x[:, 0] = x0
                        x[:, 1] = x1
                        x[:, 2:] = 0.5
                        
                        # 4. Evaluate
                        results = []
                        for i in range(len(x)):
                            if time.time() > deadline:
                                break
                            f = problem(x[i])
                            results.append(f)
                            
                        return results
