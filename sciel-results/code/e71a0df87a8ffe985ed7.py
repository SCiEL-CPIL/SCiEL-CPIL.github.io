"""Improved program using Structure-Aware Sampling for DTLZ2."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc

class Optimizer:
    """Memetic optimizer for DTLZ2 using local search around Sobol centers."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        M = self.n_obj
        # Use a subset of budget for initial diverse centers
        # and the rest for local refinement.
        n_centers = 20
        if self.budget < 2 * n_centers:
            n_centers = self.budget // 2
            steps_per_center = self.budget - n_centers
        else:
            steps_per_center = (self.budget - n_centers) // n_centers
            
        # 1. Generate initial centers using Sobol sequence in position space
        sampler = qmc.Sobol(d=M, seed=42)
        centers_pos = sampler.random(n_centers)
        
        X_centers = np.zeros((n_centers, self.dim))
        X_centers[:, :M] = centers_pos
        X_centers[:, M:] = 0.5 # On the Pareto front
        
        recorded_points = []
        recorded_objs = []
        
        # Evaluate centers
        for i in range(n_centers):
            f = problem(X_centers[i])
            recorded_points.append(X_centers[i])
            recorded_objs.append(f)
            
        # 2. Local Search (Random Walk with Refinement) around each center
        for i in range(n_centers):
            current_x = X_centers[i].copy()
            current_f = recorded_objs[i]
            
            for _ in range(steps_per_center):
                # Perturb position variables
                delta_pos = np.random.normal(0, 0.15, M)
                new_pos = current_x[:M] + delta_pos
                new_pos = np.clip(new_pos, lo[:M], hi[:M])
                
                # Perturb distance variables, strongly biased towards 0.5
                # This ensures we stay close to the Pareto front (g ~ 0)
                delta_dist = np.random.normal(0, 0.05, self.dim - M)
                new_dist = 0.5 + delta_dist
                new_dist = np.clip(new_dist, lo[M:], hi[M:])
                
                new_x = np.concatenate([new_pos, new_dist])
                new_f = problem(new_x)
                
                recorded_points.append(new_x)
                recorded_objs.append(new_f)
                
                # Greedy update: move to new point if it improves the objective sum
                # This helps converge to the front while exploring
                if np.sum(new_f) < np.sum(current_f):
                    current_x = new_x
                    current_f = new_f
                # If not better, we keep the previous current_x for the next step
                # This creates a cluster around the best found point in the neighborhood.
                
        return np.array(recorded_objs)
# EVOLVE-BLOCK-END
