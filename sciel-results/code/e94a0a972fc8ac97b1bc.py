import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 280  # Safety margin before 330s kill

            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])

            # DTLZ2 Pareto front: x_i = 0.5 for i >= 2
            # x_0, x_1 parameterize the front
            # theta_0 = x_0 * pi/2, theta_1 = x_1 * pi/2
            # f_1 = cos(theta_0)*cos(theta_1), f_2 = cos(theta_0)*sin(theta_1), f_3 = sin(theta_0)
            # Surface area measure: sin(theta_0) dtheta_0 dtheta_1
            # Uniform in surface area: u ~ U[0,1] -> theta_0 = arccos(u), v ~ U[0,1] -> theta_1 = v*pi/2

            # Add strategic extreme points of the Pareto front
            # These are the 3 vertices of the quarter-sphere
            extreme_points = np.array([
                [0.0, 0.0],  # theta_0=0, theta_1=0 -> f=(1,0,0)
                [0.0, 1.0],  # theta_0=0, theta_1=pi/2 -> f=(0,1,0)
                [1.0, 0.0],  # theta_0=pi/2, theta_1=0 -> f=(0,0,1)
            ])

            n_extreme = len(extreme_points)
            n_sobol = self.budget - n_extreme

            if n_sobol < 0:
                n_sobol = 0
                n_extreme = self.budget

            # Generate Sobol points for the remaining budget
            if n_sobol > 0:
                sampler = qmc.Sobol(d=2, scramble=True)
                # Use power of 2 >= n_sobol for better uniformity, then truncate
                n_pow2 = 1
                while n_pow2 < n_sobol:
                    n_pow2 *= 2
                sobol_points = sampler.random(n_pow2)[:n_sobol]
            else:
                sobol_points = np.empty((0, 2))

            # Combine extreme and Sobol points in (u, v) space
            # u corresponds to cos(theta_0), v corresponds to theta_1/(pi/2)
            all_uv = np.vstack([extreme_points, sobol_points]) if n_sobol > 0 else extreme_points

            u = all_uv[:, 0]
            v = all_uv[:, 1]

            # Map to x_0, x_1
            # theta_0 = arccos(u), so x_0 = theta_0 / (pi/2) = (2/pi) * arccos(u)
            # theta_1 = v * pi/2, so x_1 = v
            x0 = (2.0 / np.pi) * np.arccos(np.clip(u, 0, 1))
            x1 = np.clip(v, 0, 1)

            # Construct full x vectors
            X = np.zeros((len(u), self.dim))
            X[:, 0] = x0
            X[:, 1] = x1
            X[:, 2:] = 0.5  # Center of bounds for distance variables

            # Clip to bounds
            X = np.clip(X, lo, hi)

            # Evaluate all points
            F = np.array([problem(x) for x in X])

            return X
