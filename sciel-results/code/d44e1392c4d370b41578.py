"""Derivative-Free Coordinate Descent for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-resolution coordinate descent with bi-modal aware initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def _eval(self, func, x):
        """Evaluate and track best."""
        x = np.clip(x, -5.0, 5.0)
        fx = float(func(x))
        self.calls += 1
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return x, fx

    def _coordinate_descent(self, func, x, step_sizes, max_evals, deadline):
        """Perform coordinate descent with multiple step sizes per dimension."""
        n = self.dim
        current_x = x.copy()
        current_f = self.best_f
        
        # For each step size, sweep through dimensions
        for step in step_sizes:
            if self.calls >= max_evals or time.time() >= deadline:
                break
            
            # Multiple passes over dimensions for this step size
            for _pass in range(3):
                if self.calls >= max_evals or time.time() >= deadline:
                    break
                
                for d in range(n):
                    if self.calls >= max_evals or time.time() >= deadline:
                        break
                    
                    # Try moving in positive and negative directions
                    for direction in [1.0, -1.0]:
                        if self.calls >= max_evals or time.time() >= deadline:
                            break
                        
                        trial_x = current_x.copy()
                        trial_x[d] += direction * step
                        trial_x = np.clip(trial_x, -5.0, 5.0)
                        
                        _, trial_f = self._eval(func, trial_x)
                        
                        if trial_f < current_f:
                            current_x = trial_x
                            current_f = trial_f
        
        return current_x, current_f

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit
        
        # Phase 1: Coarse grid scan to find promising regions
        # Use a coarse grid in each dimension to handle bi-modal nature
        coarse_points = []
        
        # Center point
        coarse_points.append(np.zeros(self.dim))
        
        # Points near the two known minima of Lunacek (approximate locations)
        # The Lunacek function has minima near 0 and near ±(some value)
        # We'll sample around several candidate regions
        candidate_centers = [
            np.zeros(self.dim),
            np.array([2.0] * self.dim),
            np.array([-2.0] * self.dim),
            np.array([0.5, -0.5, 0.5, -0.5, 0.5]) * 4,
            np.array([-0.5, 0.5, -0.5, 0.5, -0.5]) * 4,
        ]
        
        for center in candidate_centers:
            center = np.clip(center, -4.5, 4.5)
            coarse_points.append(center)
            # Add some jittered variants
            for _ in range(2):
                jittered = center + rng.uniform(-1.0, 1.0, self.dim)
                jittered = np.clip(jittered, -4.5, 4.5)
                coarse_points.append(jittered)
        
        # Add random points
        for _ in range(10):
            rand_point = rng.uniform(-4.0, 4.0, self.dim)
            coarse_points.append(rand_point)
        
        # Evaluate all coarse points
        for point in coarse_points:
            if self.calls >= self.budget or time.time() >= deadline:
                break
            self._eval(func, point)
        
        # Phase 2: Local refinement around the best point found
        if self.best_x is not None and self.calls < self.budget and time.time() < deadline:
            # Use coordinate descent with decreasing step sizes
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
            
            # Allocate remaining budget to refinement
            remaining = self.budget - self.calls
            if remaining > 0:
                self._coordinate_descent(func, self.best_x, step_sizes, self.budget, deadline)
        
        # Phase 3: If we have budget left, try another local search from a different starting point
        if self.calls < self.budget and time.time() < deadline:
            # Pick a second-best point from our coarse evaluation
            # We'll just do another coordinate descent from a slightly perturbed best
            if self.best_x is not None:
                x2 = self.best_x + rng.uniform(-0.3, 0.3, self.dim)
                x2 = np.clip(x2, -4.9, 4.9)
                self._eval(func, x2)
                
                step_sizes_fine = [0.3, 0.15, 0.07, 0.03, 0.01, 0.005]
                self._coordinate_descent(func, self.best_x, step_sizes_fine, self.budget, deadline)
        
        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
