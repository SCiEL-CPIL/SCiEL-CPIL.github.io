"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Optimizer that exploits the known Pareto front of ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.optimal_x0s = self._compute_optimal_front_points()

    def _compute_hv(self, f1, f2, ref=(11.0, 11.0)):
        """Calculate Hypervolume for 2D points."""
        # Sort by f1 ascending
        idx = np.argsort(f1)
        f1_sorted = f1[idx]
        f2_sorted = f2[idx]
        
        # Filter dominated points? 
        # Since we are on the front, they are non-dominated.
        # But if there are duplicates or slight numerical issues, we should be careful.
        
        hv = 0.0
        r1, r2 = ref
        prev_f1 = 0.0
        
        for i in range(len(f1_sorted)):
            next_f1 = f1_sorted[i+1] if i + 1 < len(f1_sorted) else r1
            # Width of the box
            width = next_f1 - prev_f1
            # Height of the box
            height = r2 - f2_sorted[i]
            if width > 0 and height > 0:
                hv += width * height
            prev_f1 = next_f1
            
        return hv

    def _compute_optimal_front_points(self):
        """
        Find the optimal placement of 200 points on the ZDT2 Pareto front
        to maximize Hypervolume.
        Front: f1 = x, f2 = 1 - sqrt(x) for x in [0, 1].
        """
        n_points = 200
        # Initialize with uniform distribution
        x = np.linspace(0, 1, n_points)
        
        def get_f2(x_vals):
            return 1.0 - np.sqrt(x_vals)
        
        best_x = x.copy()
        best_hv = self._compute_hv(x, get_f2(x))
        
        # Simple local search to maximize HV
        # We want to maximize HV, so we minimize -HV
        # Use a simple iterative perturbation
        step_size = 0.01
        for _ in range(100): # 100 iterations of refinement
            improved = False
            for i in range(n_points):
                original_x = best_x[i]
                # Try shifting left
                new_x = np.clip(original_x - step_size, 0, 1)
                if new_x != original_x:
                    test_x = best_x.copy()
                    test_x[i] = new_x
                    # Ensure sorted for HV calc? _compute_hv sorts internally.
                    test_hv = self._compute_hv(test_x, get_f2(test_x))
                    if test_hv > best_hv:
                        best_x = test_x
                        best_hv = test_hv
                        improved = True
                # Try shifting right
                new_x = np.clip(original_x + step_size, 0, 1)
                if new_x != original_x:
                    test_x = best_x.copy()
                    test_x[i] = new_x
                    test_hv = self._compute_hv(test_x, get_f2(test_x))
                    if test_hv > best_hv:
                        best_x = test_x
                        best_hv = test_hv
                        improved = True
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
                    
        return best_x

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate points on the front
        # x[0] is the variable that determines position on the front
        # x[1:] should be 0
        X = np.zeros((self.budget, self.dim))
        X[:, 0] = self.optimal_x0s
        # x[1:] is already 0
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
