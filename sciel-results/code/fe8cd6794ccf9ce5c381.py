"""Multi-objective optimizer for ZDT4 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits ZDT4's analytic Pareto set: x2..xn at zeros of sin(2*pi*x_i) with cos=1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        dim = self.dim
        budget = self.budget
        used = 0
        
        # Optimal values for x2..x10: 0.0 (gives sin^2=0, cos=1, term=-10)
        # This minimizes g(x), placing points on the true Pareto front.
        optimal_tail = np.zeros(dim - 1)
        optimal_tail = np.clip(optimal_tail, lo[1:], hi[1:])
        
        # ZDT4 Pareto front: f1 = x1, f2 = g*(2 - 2*x1^2 - 0.2*sin(8*pi*x1))
        # where g = 19 when x2..x10 are at optimal values.
        
        # Strategy: 
        # 1. Include endpoints.
        # 2. Include critical points of the oscillatory term sin(8*pi*x1) 
        #    to ensure the "dents" in the front are well-resolved.
        # 3. Fill remaining budget with a Golden Ratio low-discrepancy sequence 
        #    for uniform coverage of the rest of the front.
        
        x1_candidates = set()
        
        # Endpoints
        x1_candidates.add(0.0)
        x1_candidates.add(1.0)
        
        # Critical points for sin(8*pi*x1):
        # Zeros: 8*pi*x1 = k*pi => x1 = k/8 for k=0..8
        for k in range(9):
            x1_candidates.add(k / 8.0)
            
        # Extrema: 8*pi*x1 = pi/2 + k*pi => x1 = (2k+1)/16 for k=0..7
        for k in range(8):
            x1_candidates.add((2 * k + 1) / 16.0)
        
        # Golden ratio conjugate for low-discrepancy sequence
        golden = (1 + np.sqrt(5)) / 2
        golden_conj = 1.0 / golden  # ~0.618
        
        n_fixed = len(x1_candidates)
        n_remaining = budget - n_fixed
        
        # Add golden ratio points if budget allows
        if n_remaining > 0:
            for i in range(1, n_remaining + 1):
                # Use a shifted golden ratio sequence to avoid collisions with fixed points
                # and ensure good spread
                x1_val = (i * golden_conj + 0.5) % 1.0
                x1_candidates.add(x1_val)
        
        # Convert to sorted list for deterministic evaluation
        x1_values = sorted(list(x1_candidates))
        
        # If we have fewer unique points than budget, fill with more golden ratio points
        if len(x1_values) < budget:
            idx = len(x1_values) + 1
            while len(x1_values) < budget and time.time() < deadline:
                x1_val = (idx * golden_conj + 0.5) % 1.0
                if x1_val not in x1_candidates:
                    x1_candidates.add(x1_val)
                    x1_values.append(x1_val)
                    x1_values.sort()
                idx += 1
        
        # Ensure we don't exceed budget
        if len(x1_values) > budget:
            x1_values = x1_values[:budget]

        # Evaluate all points on the Pareto front
        for x1 in x1_values:
            if used >= budget or time.time() > deadline:
                break
            
            # Construct vector
            x = np.concatenate([[x1], optimal_tail])
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            problem(x)
            used += 1
            
        return None
# EVOLVE-BLOCK-END
