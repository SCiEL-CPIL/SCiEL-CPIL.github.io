"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for the Bent Cigar function (f12) on [-5, 5]^5.
    
    Strategy:
    1. Use a (1+1)-ES with a carefully tuned step size adaptation rule.
    2. The Bent Cigar is extremely anisotropic: f(x) = x_1^2 + 100 * sum(x_i^2 for i>1).
       This means we need to make large steps along dimension 0 and very small steps along others.
    3. We use a coordinate-wise adaptation: maintain separate step sizes for each dimension.
       - Dimension 0: larger steps
       - Other dimensions: smaller steps
    4. Use the 1/5 rule per dimension to adapt each step size independently.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Initialize
        rng = np.random.default_rng(42)
        
        # Start at origin - the Bent Cigar minimum is often near 0 after shift/rotation
        # But the problem says minimizer lies in [-4, 4]^5 with hidden shift
        # Let's start at 0 which is a reasonable default
        x = np.zeros(dim)
        fx = func(x)
        calls_used = 1
        
        # Separate step sizes per dimension
        # For Bent Cigar: dim 0 needs large steps, others need small steps
        # But since there's a rotation, we don't know which original dimension maps to which
        # Start with a moderate sigma and let adaptation handle it
        sigma = np.ones(dim) * 1.0
        
        # 1/5 rule: track successes per dimension
        window_size = 15
        success_history = [0] * dim  # rolling count of successes
        success_count = [0] * dim  # number of successes in current window
        total_in_window = [0] * dim  # total evaluations in current window
        
        best_x = x.copy()
        best_fx = fx
        
        while calls_used < budget:
            if time.time() >= deadline:
                break
            
            # Generate candidate: add Gaussian noise scaled by per-dimension sigma
            noise = rng.standard_normal(dim) * sigma
            y = np.clip(x + noise, -5, 5)
            
            fy = func(y)
            calls_used += 1
            
            # Update best
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()
            
            # Per-dimension success tracking
            for d in range(dim):
                if y[d] != x[d]:  # Only track if the dimension actually changed
                    if fy < fx:
                        success_count[d] += 1
                    total_in_window[d] += 1
                    
                    # Adapt sigma when window is full
                    if total_in_window[d] >= window_size:
                        # 1/5 rule: if success rate >= 1/5, increase sigma; else decrease
                        if success_count[d] >= window_size // 5:
                            sigma[d] *= 1.1
                        else:
                            sigma[d] *= 0.9
                        
                        # Reset window
                        success_count[d] = 0
                        total_in_window[d] = 0
            
            # Update x if improved
            if fy < fx:
                x = y.copy()
                fx = fy
            
            # Early termination if all sigmas become very small
            if np.all(sigma < 1e-6):
                break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
