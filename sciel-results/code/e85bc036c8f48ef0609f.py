"""Model-and-optimize approach for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB-style f5 (Linear slope).
    
    Strategy:
    1. Evaluate origin and each axis-boundary pair (±5) to determine sign per coordinate.
    2. Jump to the vertex of the box that minimizes the estimated linear model.
    3. Refine using (1+1)-ES with adaptive step size and remaining budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin (330s limit)
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        budget_used = 0
        best_f = np.inf
        best_x = np.zeros(dim)

        # Phase 1: Direct sign determination via boundary evaluation
        # Evaluate origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        budget_used += 1
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()

        # For each dimension, evaluate both +5 and -5 to determine the minimizing direction
        signs = np.zeros(dim)  # 1 means go to -5, -1 means go to +5, 0 means stay
        for i in range(dim):
            if time.time() > deadline:
                break
            
            e_i = np.zeros(dim)
            e_i[i] = 1.0
            
            # Evaluate +5 in dimension i
            x_plus = np.clip(x0 + 5.0 * e_i, -5, 5)
            f_plus = func(x_plus)
            budget_used += 1
            if f_plus < best_f:
                best_f = f_plus
                best_x = x_plus.copy()
            
            # Evaluate -5 in dimension i
            x_minus = np.clip(x0 - 5.0 * e_i, -5, 5)
            f_minus = func(x_minus)
            budget_used += 1
            if f_minus < best_f:
                best_f = f_minus
                best_x = x_minus.copy()
            
            # Determine sign: if f_minus < f_plus, go to -5; else go to +5
            if f_minus < f_plus:
                signs[i] = 1.0
            elif f_plus < f_minus:
                signs[i] = -1.0
            else:
                signs[i] = 0.0

        # Phase 2: Informed Jump to vertex
        x_jump = np.zeros(dim)
        for i in range(dim):
            if signs[i] > 0:
                x_jump[i] = -5.0
            elif signs[i] < 0:
                x_jump[i] = 5.0
            else:
                x_jump[i] = 0.0

        if budget_used < budget:
            f_jump = func(x_jump)
            budget_used += 1
            if f_jump < best_f:
                best_f = f_jump
                best_x = x_jump.copy()

        # Phase 3: Local Refinement with (1+1)-ES
        x = best_x.copy()
        fx = best_f
        sigma = 0.3
        
        # Track consecutive failures for adaptive sigma
        fail_count = 0
        max_fails = 15
        
        while budget_used < budget:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            budget_used += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.05, 2.0)
                fail_count = 0
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.95
                fail_count += 1
                
            # Reset if too many failures
            if fail_count > max_fails:
                sigma = max(sigma, 0.05)
                fail_count = 0
                
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
