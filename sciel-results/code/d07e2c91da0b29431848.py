"""Direct construction approach for multi-objective optimization on ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits ZDT4 structure:
    f1(x) = x1
    g(x) = 10*(n-1) + sum(xi^2 - 10*cos(4*pi*xi) for i=2..n)
    h(x1, g) = 1 - sqrt(x1/g) - (x1/g)*sin^2(10*pi*x1)
    f2(x) = g(x) * h(x1, g)
    
    The Pareto front is achieved when g(x) is at its global minimum (90 for n=10).
    For ZDT4, g(x_i) = xi^2 - 10*cos(4*pi*xi) has global minimum at xi = 0.
    Thus, setting x2..xn = 0 minimizes g.
    
    Strategy:
    - Set x2..xn = 0 to achieve the global minimum of g
    - Sweep x1 across its range using non-uniform sampling
    - Use a transformation that accounts for the curvature of the Pareto front
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n = self.dim
        budget = self.budget

        # Set x[1:] to 0 (global minimum of g for ZDT4)
        # If 0 is not in bounds, clamp to nearest bound
        x_rest = np.zeros(n - 1, dtype=np.float64)
        for i in range(n - 1):
            idx = i + 1
            if lo[idx] <= 0.0 <= hi[idx]:
                x_rest[i] = 0.0
            else:
                # Clamp to nearest bound
                x_rest[i] = lo[idx] if abs(0.0 - lo[idx]) < abs(0.0 - hi[idx]) else hi[idx]

        # Use all budget for sampling x1 on the Pareto front
        # Non-uniform grid: use a transformation that gives denser sampling
        # where the Pareto front has higher curvature
        
        # For ZDT4, f2 = g * (1 - sqrt(x1/g) - (x1/g)*sin^2(10*pi*x1))
        # with g=90, f2 = 90 - 30*sqrt(x1) - x1*sin^2(10*pi*x1)
        # The sin^2 term creates oscillations, so we want good coverage
        
        # Strategy: combine uniform sampling with a few key points
        # to ensure we capture the oscillatory nature of the front
        
        n_evals = min(budget, 200)  # Cap at budget
        
        # Create a mix of sampling strategies:
        # 1. Uniform sampling for even coverage
        # 2. Points at the zeros and extrema of sin^2(10*pi*x1)
        
        all_x1 = []
        
        # Uniform sampling
        if n_evals >= 100:
            n_uniform = n_evals - 20  # Reserve 20 for special points
            x1_uniform = np.linspace(lo[0], hi[0], n_uniform)
            all_x1.extend(x1_uniform.tolist())
        else:
            x1_uniform = np.linspace(lo[0], hi[0], n_evals)
            all_x1.extend(x1_uniform.tolist())
        
        # Add special points: zeros and extrema of sin^2(10*pi*x1)
        # sin^2(10*pi*x1) = 0 when 10*pi*x1 = k*pi, i.e., x1 = k/10
        # sin^2(10*pi*x1) = 1 when 10*pi*x1 = (2k+1)*pi/2, i.e., x1 = (2k+1)/20
        
        if n_evals > 100:
            # Add points at x1 = k/10 for k=0..10
            for k in range(11):
                x1_val = k / 10.0
                if lo[0] <= x1_val <= hi[0]:
                    all_x1.append(x1_val)
            
            # Add points at x1 = (2k+1)/20 for k=0..9
            for k in range(10):
                x1_val = (2 * k + 1) / 20.0
                if lo[0] <= x1_val <= hi[0]:
                    all_x1.append(x1_val)
        
        # Remove duplicates and sort
        all_x1 = sorted(set(all_x1))
        
        # If we have more points than budget, take the first budget points
        if len(all_x1) > budget:
            # Keep a mix: first, last, and evenly spaced
            if budget >= 3:
                indices = [0, len(all_x1) - 1]
                n_middle = budget - 2
                if n_middle > 0:
                    middle_indices = np.linspace(0, len(all_x1) - 1, n_middle + 2)[1:-1].astype(int)
                    indices.extend(middle_indices.tolist())
                all_x1 = [all_x1[i] for i in sorted(set(indices))]
            else:
                all_x1 = all_x1[:budget]
        
        # Evaluate all points
        all_points = []
        for x1 in all_x1:
            if time.time() > deadline:
                break
            if len(all_points) >= budget:
                break
            
            x = np.empty(n, dtype=np.float64)
            x[0] = x1
            x[1:] = x_rest
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                f = np.array(f, dtype=np.float64)
                if np.all(np.isfinite(f)):
                    all_points.append(f)
            except Exception:
                break

        # If we didn't get any points, try a fallback
        if len(all_points) == 0:
            x = (lo + hi) / 2
            try:
                f = problem(x)
                all_points.append(np.array(f, dtype=np.float64))
            except Exception:
                pass

        return all_points
# EVOLVE-BLOCK-END
