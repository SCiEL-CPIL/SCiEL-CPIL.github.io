"""Deterministic multi-start coordinate-wise descent for Katsuura-like f23."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic multi-start coordinate-wise descent.
    
    Phase 1: Evaluate a diverse set of initial points to avoid local minima.
    Phase 2: Adaptive coordinate-wise descent with shrinking step sizes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0

        def evaluate(x):
            nonlocal calls
            calls += 1
            return float(func(x))

        # Phase 1: Multi-start initialization
        # Generate a set of diverse candidate points.
        # 1. Center
        # 2. Diagonal points (all coords equal) - good for coupled functions
        # 3. Axis points (one coord non-zero) - good for separable functions
        
        candidates = []
        
        # Center
        candidates.append(np.zeros(dim))
        
        # Diagonal points: x_i = v for all i
        diag_vals = [2.0, -2.0, 4.0, -4.0, 1.0, -1.0, 3.0, -3.0, 0.5, -0.5]
        for v in diag_vals:
            candidates.append(np.full(dim, v))
            
        # Axis points: x_d = v, others 0
        # We limit the number of axis points to stay within budget
        # 2 points per dimension (v=2, v=-2)
        for d in range(dim):
            for v in [2.0, -2.0]:
                x = np.zeros(dim)
                x[d] = v
                candidates.append(x)
                
        # Evaluate all candidates and keep the best
        best_x = None
        best_f = float('inf')
        
        for x in candidates:
            if calls >= budget:
                break
            fx = evaluate(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Adaptive Coordinate-wise Descent
        # Start with a moderate step size and shrink if no improvement is found.
        current_x = best_x.copy()
        current_f = best_f
        step = 1.0
        
        # We will iterate until budget is exhausted or step becomes very small
        while calls < budget and step > 1e-6:
            improved = False
            # Try to improve each coordinate
            for d in range(dim):
                if calls >= budget:
                    break
                # Try both directions
                for direction in [-1, 1]:
                    if calls >= budget:
                        break
                    candidate = current_x.copy()
                    candidate[d] += direction * step
                    # Clip to valid box
                    candidate[d] = np.clip(candidate[d], -5.0, 5.0)
                    
                    f_val = evaluate(candidate)
                    if f_val < current_f:
                        current_f = f_val
                        current_x = candidate
                        improved = True
            
            # If no improvement was made in this full pass, shrink the step
            if not improved:
                step *= 0.5
            else:
                # Optional: If we improved, we could keep the step or shrink slightly
                # Keeping the step allows for larger moves in the basin
                # But shrinking slightly is safer to avoid oscillation
                step *= 0.8 

        return current_x, current_f
# EVOLVE-BLOCK-END
