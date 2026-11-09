# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Sobol-grid global search + BFGS local refinement on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        self.max_calls = self.budget - 2
        self._deadline = None

    def __call__(self, func):
        start = time.time()
        self._deadline = start + 280.0
        d = self.dim
        rng = np.random.default_rng(42)

        self._calls = 0
        self._best_x = None
        self._best_f = np.inf

        def call(x):
            if self._calls >= self.max_calls:
                return None
            self._calls += 1
            return func(np.asarray(x, dtype=float))

        def time_ok():
            return time.time() < self._deadline

        # ---------------- Phase A: Sobol grid for global basin discovery ----------------
        n_sobol = 32
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            sobol_pts = sampler.random(n_sobol)
        except Exception:
            sobol_pts = rng.random((n_sobol, d))
        
        sobol_pts = sobol_pts * 10.0 - 5.0
        sobol_pts = np.clip(sobol_pts, -5.0, 5.0)

        candidates = []
        for pt in sobol_pts:
            if not time_ok() or self._calls >= self.max_calls:
                break
            fx = call(pt)
            if fx is None:
                break
            candidates.append((fx, pt.copy()))
            if fx < self._best_f:
                self._best_f, self._best_x = fx, pt.copy()

        # Sort candidates by function value, take top few for local refinement
        candidates.sort(key=lambda x: x[0])
        top_k = min(5, len(candidates))
        
        # ---------------- Phase B: BFGS local refinement ----------------
        for i in range(top_k):
            if not time_ok() or self._calls >= self.max_calls:
                break
            x0 = candidates[i][1]
            self._bfgs(func, call, x0, time_ok)

        return self._best_x, self._best_f

    def _bfgs(self, func, call, x0, time_ok):
        d = self.dim
        x = x0.copy()
        
        # Initial function value
        if self._calls >= self.max_calls or not time_ok():
            return
        
        fx = call(x)
        if fx is None:
            return
        
        if fx < self._best_f:
            self._best_f, self._best_x = fx, x.copy()

        # Compute initial gradient numerically
        eps = 1e-5
        grad = np.zeros(d)
        for i in range(d):
            xp = x.copy()
            xp[i] += eps
            xp = np.clip(xp, -5.0, 5.0)
            f_plus = call(xp)
            if f_plus is None:
                return
            
            xm = x.copy()
            xm[i] -= eps
            xm = np.clip(xm, -5.0, 5.0)
            f_minus = call(xm)
            if f_minus is None:
                return
            
            grad[i] = (f_plus - f_minus) / (2 * eps)

        # Hessian approximation H (inverse of Hessian)
        H = np.eye(d)
        
        n_iter = 100
        for _ in range(n_iter):
            if not time_ok() or self._calls >= self.max_calls:
                break
            
            # Check gradient norm for convergence
            if np.linalg.norm(grad) < 1e-8:
                break
            
            # Search direction
            direction = -H @ grad
            
            # Line search (backtracking)
            step = 1.0
            improved = False
            for _ in range(20):
                x_new = x + step * direction
                x_new = np.clip(x_new, -5.0, 5.0)
                
                f_new = call(x_new)
                if f_new is None:
                    return
                
                if f_new < fx - 1e-4 * step * np.dot(grad, direction):
                    # Compute new gradient
                    grad_new = np.zeros(d)
                    for i in range(d):
                        xp = x_new.copy()
                        xp[i] += eps
                        xp = np.clip(xp, -5.0, 5.0)
                        f_plus = call(xp)
                        if f_plus is None:
                            return
                        
                        xm = x_new.copy()
                        xm[i] -= eps
                        xm = np.clip(xm, -5.0, 5.0)
                        f_minus = call(xm)
                        if f_minus is None:
                            return
                        
                        grad_new[i] = (f_plus - f_minus) / (2 * eps)
                    
                    # Update BFGS
                    s = x_new - x
                    y = grad_new - grad
                    sy = np.dot(s, y)
                    if sy > 1e-12:
                        Hy = H @ y
                        H = H + ((sy + np.dot(y, Hy)) / (sy ** 2)) * np.outer(s, s) - (np.outer(s, Hy) + np.outer(Hy, s)) / sy
                    
                    x, fx, grad = x_new, f_new, grad_new
                    improved = True
                    if fx < self._best_f:
                        self._best_f, self._best_x = fx, x.copy()
                    break
                else:
                    step *= 0.5
            
            if not improved:
                break

# EVOLVE-BLOCK-END
