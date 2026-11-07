"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start CMA-ES with adaptive restarts for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # State
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Helper to safely call func and update state
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            try:
                f_val = func(x)
                calls_made += 1
                if np.isnan(f_val) or np.isinf(f_val):
                    return np.inf
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return np.inf

        # Phase 1: Global Sparse Sampling to find promising basins
        # Rastrigin is multimodal. We need to sample broadly to find the global minimum's basin.
        # We use a fixed set of points spread across the domain.
        global_budget = int(budget * 0.15)  # 15% for global exploration
        
        # Generate a deterministic set of points.
        # A simple grid might be too coarse or fine. Let's use a Halton-like or just random with seed.
        # Since the shift is hidden, random is safer than a fixed grid aligned to axes.
        rng = np.random.default_rng(123)
        
        # Evaluate center first
        x_center = np.zeros(n)
        evaluate(x_center)
        
        # Evaluate a set of random points
        for _ in range(global_budget - 1):
            if calls_made >= budget or time.time() > deadline:
                break
            x_try = rng.uniform(-5.0, 5.0, n)
            evaluate(x_try)
        
        if best_x is None:
            return np.zeros(n), np.inf

        # Phase 2: Local Refinement using CMA-ES
        # We will run CMA-ES from the best point found.
        # CMA-ES is effective for continuous optimization and can handle multimodality better than NM if started well.
        # However, standard CMA-ES is single-start. We will implement a simplified CMA-ES loop.
        
        # Simple CMA-ES implementation
        # Parameters
        pop_size = max(4, int(4 + 3 * np.log(n)))
        sigma = 1.0  # Initial step size
        min_sigma = 1e-8
        max_sigma = 10.0
        
        # Mean vector
        mean = best_x.copy()
        
        # Covariance matrix
        C = np.eye(n)
        # Eigen-decomposition for sampling
        # We'll use a simplified version: sample from N(mean, sigma^2 * C)
        # For simplicity and speed, we might just use diagonal C initially or a simple adaptation.
        # Let's implement a basic (mu+lambda)-CMA-ES variant.
        
        # We will restart CMA-ES if it stagnates.
        # Max iterations per run
        max_iter_per_run = 50
        
        # Main loop for multiple CMA-ES runs
        while calls_made < budget and time.time() < deadline:
            # Initialize CMA-ES state
            mean = best_x.copy()
            sigma = 0.5  # Start with a smaller sigma for local refinement
            C = np.eye(n)
            
            # Precompute eigenvalues/vectors for C (identity initially)
            # For simplicity, we'll just use Cholesky or just random directions if C is diagonal.
            # Since C starts as identity, we can just use random normal vectors.
            # As C adapts, we need to update.
            # Let's use a simplified CMA-ES that adapts sigma and mean, and keeps C diagonal (Separable CMA-ES).
            # This is faster and often sufficient for BBOB functions.
            
            # Diagonal CMA-ES (Separable)
            sigma = 0.5
            # We don't really need C if we assume separability, but Rastrigin rotated is not separable.
            # Let's stick to a basic CMA-ES with full covariance but simplified updates.
            
            # Actually, let's use a simpler approach: Nelder-Mead is often robust for low-dim.
            # Let's try Nelder-Mead again but with more careful initialization and restarts.
            
            # Restart NM from best point
            nm_calls = 0
            nm_max_calls = 100
            
            # Define NM function wrapper
            def nm_func(x):
                nonlocal nm_calls
                if nm_calls >= nm_max_calls or time.time() > deadline or calls_made >= budget:
                    return np.inf
                nm_calls += 1
                return evaluate(x)
            
            # Initial simplex
            x0 = best_x.copy()
            f0 = best_f
            delta = 0.5
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += delta
                simplex.append(xi)
            
            # Evaluate simplex
            simplex_f = [evaluate(x) for x in simplex]
            
            # NM iterations
            for _ in range(200):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Check convergence
                if abs(simplex_f[-1] - simplex_f[0]) < 1e-6:
                    break
                
                # Centroid
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                gamma = 1.0
                x_r = centroid + gamma * (centroid - simplex[-1])
                f_r = evaluate(x_r)
                
                if f_r < simplex_f[0]:
                    # Expansion
                    gamma_e = 2.0
                    x_e = centroid + gamma_e * (centroid - simplex[-1])
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    gamma_c = 0.5
                    x_c = centroid + gamma_c * (simplex[-1] - centroid)
                    f_c = evaluate(x_c)
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = evaluate(simplex[i])
            
            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()
            
            # If we made progress, continue. Otherwise, break to avoid wasting time.
            # Simple check: if no improvement in last few iterations, break.
            # For now, just break if budget is low.
            if budget - calls_made < 50:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
