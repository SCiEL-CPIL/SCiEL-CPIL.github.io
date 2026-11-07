import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill

        def _time_ok():
            return time.time() - start_time < time_limit

        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0

        def safe_eval(x):
            nonlocal calls
            if calls >= budget or not _time_ok():
                return np.inf
            try:
                fx = func(x)
                calls += 1
                if not np.isfinite(fx):
                    return np.inf
                return float(fx)
            except Exception:
                calls += 1
                return np.inf

        # PSO Parameters
        n_particles = 15
        # We have 300 calls. 15 particles * 20 iterations = 300.
        # We will run exactly 20 iterations if possible.
        max_iters = budget // n_particles
        
        # Initialize particles
        # Positions in [-5, 5]
        X = rng.uniform(-5, 5, (n_particles, dim))
        # Velocities initialized to 0
        V = np.zeros((n_particles, dim))
        
        # Personal bests
        P_best = X.copy()
        P_best_f = np.array([safe_eval(x) for x in X])
        
        # Global best
        g_idx = np.argmin(P_best_f)
        G_best = P_best[g_idx].copy()
        G_best_f = P_best_f[g_idx]
        
        best_x = G_best.copy()
        best_f = G_best_f

        # PSO constants
        c1 = 1.49445  # Cognitive
        c2 = 1.49445  # Social
        w_start = 0.9
        w_end = 0.4
        
        stagnation_count = 0
        stagnation_limit = 5 # Iterations without improvement before re-injecting

        for it in range(max_iters):
            if not _time_ok() or calls >= budget:
                break
                
            # Linearly decay inertia weight
            w = w_start - (w_start - w_end) * (it / max(max_iters - 1, 1))
            
            r1 = rng.random((n_particles, dim))
            r2 = rng.random((n_particles, dim))
            
            # Update velocities
            V = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (G_best[None, :] - X)
            
            # Update positions
            X = X + V
            
            # Clip to box [-5, 5]
            X = np.clip(X, -5, 5)
            
            # Evaluate new positions
            F = np.array([safe_eval(x) for x in X])
            
            # Update personal bests
            improved_mask = F < P_best_f
            P_best[improved_mask] = X[improved_mask]
            P_best_f[improved_mask] = F[improved_mask]
            
            # Update global best
            curr_g_idx = np.argmin(P_best_f)
            if P_best_f[curr_g_idx] < G_best_f:
                G_best = P_best[curr_g_idx].copy()
                G_best_f = P_best_f[curr_g_idx]
                best_x = G_best.copy()
                best_f = G_best_f
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Stagnation handling: Re-inject worst particles if stuck
            if stagnation_count >= stagnation_limit:
                # Identify worst particles
                worst_indices = np.argsort(P_best_f)[-5:] # Worst 5
                for idx in worst_indices:
                    X[idx] = rng.uniform(-5, 5, dim)
                    V[idx] = np.zeros(dim)
                    f_new = safe_eval(X[idx])
                    P_best[idx] = X[idx]
                    P_best_f[idx] = f_new
                    if f_new < G_best_f:
                        G_best = X[idx].copy()
                        G_best_f = f_new
                        best_x = G_best.copy()
                        best_f = G_best_f
                stagnation_count = 0

        return best_x, best_f
