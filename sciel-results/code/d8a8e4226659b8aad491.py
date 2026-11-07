import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 280  # 20s margin before 300s kill
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if self.calls >= budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            self.calls += 1
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            if np.isfinite(f) and f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        def nelder_mead(x0, max_iters=200, initial_alpha=1.0, alpha_decay=0.5, min_alpha=1e-6):
            """Nelder-Mead simplex optimization with adaptive step size."""
            n = dim
            # Initialize simplex
            simplex = [x0.copy() + 1e-3 * np.ones(n) if i == 0 else x0.copy() for i in range(n)]
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += 0.05 * (abs(x0[i - 1]) + 1.0)
            
            # Evaluate initial simplex
            simplex_f = [wrapped_func(s) for s in simplex]
            
            alpha = initial_alpha
            best_f_local = min(simplex_f)
            best_x_local = simplex[simplex_f.index(best_f_local)].copy()
            
            for it in range(max_iters):
                if time.time() > self.deadline - 2:
                    break
                if self.calls >= budget - 5:
                    break
                
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # If best point is good enough, shrink and continue
                centroid = np.mean(simplex[1:], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[0])
                fr = wrapped_func(xr)
                
                if fr < simplex_f[1]:
                    # Expansion
                    xe = centroid + 2 * alpha * (centroid - simplex[0])
                    fe = wrapped_func(xe)
                    if fe < fr:
                        simplex[0] = xe
                        simplex_f[0] = fe
                    else:
                        simplex[0] = xr
                        simplex_f[0] = fr
                elif fr < simplex_f[-1]:
                    simplex[0] = xr
                    simplex_f[0] = fr
                else:
                    # Contraction
                    xc = centroid - 0.5 * alpha * (centroid - simplex[0])
                    fc = wrapped_func(xc)
                    if fc < simplex_f[0]:
                        simplex[0] = xc
                        simplex_f[0] = fc
                        alpha *= alpha_decay
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = wrapped_func(simplex[i])
                        alpha *= alpha_decay
                
                # Track best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f_local:
                    best_f_local = simplex_f[min_idx]
                    best_x_local = simplex[min_idx].copy()
                
                # Check convergence
                if alpha < min_alpha and np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                    break
            
            return best_x_local, best_f_local

        def hill_climb(x0, max_steps=100, initial_step=0.1):
            """Simple hill climbing with adaptive step size."""
            x = x0.copy()
            f = wrapped_func(x)
            step = initial_step
            best_f = f
            best_x = x.copy()
            
            for _ in range(max_steps):
                if time.time() > self.deadline - 2:
                    break
                if self.calls >= budget - 5:
                    break
                
                improved = False
                for i in range(dim):
                    for direction in [1, -1]:
                        x_trial = x.copy()
                        x_trial[i] += direction * step
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = wrapped_func(x_trial)
                        
                        if f_trial < f - 1e-12:
                            x = x_trial
                            f = f_trial
                            improved = True
                            break
                    if improved:
                        break
                
                if improved:
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    step = min(step * 1.2, 1.0)
                else:
                    step *= 0.7
                    if step < 1e-8:
                        break
            
            return best_x, best_f

        # Phase 1: Generate diverse starting points
        np.random.seed(123)
        starts = []
        starts.append(np.zeros(dim))
        
        # Random points in the box
        for _ in range(10):
            starts.append(np.random.uniform(-4, 4, dim))
        
        # Points near corners
        for corner in range(2 ** dim):
            if self.calls >= budget - 50:
                break
            point = np.array([(corner >> i) & 1 for i in range(dim)], dtype=float)
            point = point * 4 - 2  # Map to [-2, 2]
            starts.append(point)
        
        # Limit starts to avoid using too much budget
        max_starts = min(len(starts), 15)
        starts = starts[:max_starts]
        
        # Phase 2: For each start, run Nelder-Mead with limited budget
        calls_per_nm = max(20, (budget - 50) // max_starts)
        
        for i, x0 in enumerate(starts):
            if self.calls >= budget - 20:
                break
            if time.time() > self.deadline - 15:
                break
            
            x0 = np.clip(x0, -5, 5)
            
            # Evaluate starting point
            f0 = wrapped_func(x0)
            if not np.isfinite(f0):
                continue
            
            # Run Nelder-Mead
            try:
                max_iters = min(calls_per_nm // 10, 100)
                x_opt, f_opt = nelder_mead(x0, max_iters=max_iters)
                
                if np.isfinite(f_opt) and f_opt < self.best_f:
                    self.best_f = f_opt
                    self.best_x = np.clip(x_opt, -5, 5).copy()
            except Exception:
                pass
        
        # Phase 3: Polish the best point with hill climbing
        if self.calls < budget - 10 and time.time() < self.deadline - 10:
            try:
                max_steps = min((budget - self.calls) // 5, 100)
                x_polished, f_polished = hill_climb(self.best_x, max_steps=max_steps, initial_step=0.05)
                
                if np.isfinite(f_polished) and f_polished < self.best_f:
                    self.best_f = f_polished
                    self.best_x = np.clip(x_polished, -5, 5).copy()
            except Exception:
                pass
        
        # Phase 4: If budget remains, do more restarts around the best point
        if self.calls < budget - 15 and time.time() < self.deadline - 15:
            for _ in range(8):
                if self.calls >= budget - 10 or time.time() > self.deadline - 10:
                    break
                
                # Perturb best point
                noise_scale = 0.2 * np.exp(-self.calls / 100)
                noise = np.random.randn(dim) * noise_scale
                x_pert = np.clip(self.best_x + noise, -5, 5)
                f_pert = wrapped_func(x_pert)
                
                if np.isfinite(f_pert) and f_pert < self.best_f:
                    self.best_f = f_pert
                    self.best_x = x_pert.copy()
                
                # Quick Nelder-Mead polish
                try:
                    max_iters = min((budget - self.calls) // 15, 30)
                    if max_iters > 2:
                        x_opt, f_opt = nelder_mead(self.best_x, max_iters=max_iters, initial_alpha=0.5)
                        if np.isfinite(f_opt) and f_opt < self.best_f:
                            self.best_f = f_opt
                            self.best_x = np.clip(x_opt, -5, 5).copy()
                except Exception:
                    pass
        
        return self.best_x, self.best_f
