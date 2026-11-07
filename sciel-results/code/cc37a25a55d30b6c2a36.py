import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f13 (Sharp Ridge) in 5D with budget 500.
    Strategy:
    1. Adaptive random search to locate the ridge basin.
    2. Nelder-Mead local refinement with adaptive step size.
    3. Multiple restarts from perturbed best points if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Safety margin: Evaluator kills at 330s. We target finishing well before.
        deadline = time.time() + 290.0

        best_x = None
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f_val = func(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Adaptive Random Search
        # Start with coarse sampling, then refine around best point
        n_coarse = 40
        for _ in range(n_coarse):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0
            best_f = f0

        # Phase 2: Local refinement around best point with multiple scales
        # Try different step sizes to adapt to ridge width
        step_sizes = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
        
        for step_size in step_sizes:
            if calls >= budget - 10 or time.time() > deadline:
                break
            
            # Run Nelder-Mead from best point with this step size
            x_start = best_x.copy()
            f_start = best_f
            nelder_mead(x_start, f_start, step_size, max_iter=80)
            
            if best_x is None or np.isinf(best_f):
                break

        # Phase 3: Additional restarts from perturbed best points
        if calls < budget - 20 and time.time() < deadline:
            for restart in range(5):
                if calls >= budget - 10 or time.time() > deadline:
                    break
                
                # Perturb the best point with decreasing noise
                noise_scale = 0.3 / (restart + 1)
                perturbation = rng.normal(0, noise_scale, size=dim)
                x_restart = np.clip(best_x + perturbation, -5.0, 5.0)
                
                f_restart = safe_call(x_restart)
                if f_restart is None:
                    break
                
                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()
                
                # Run NM from this point with small step
                nelder_mead(x_restart.copy(), f_restart, step_size=0.05, max_iter=40)
        
        return best_x, best_f

    def nelder_mead(self, x_start, f_start, step_size, max_iter=100):
        """
        Nelder-Mead simplex method with adaptive parameters.
        """
        dim = self.dim
        budget = self.budget
        deadline = self._deadline if hasattr(self, '_deadline') else time.time() + 290.0
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex
        simplex = [x_start.copy()]
        f_values = [f_start]
        
        for i in range(dim):
            xi = x_start.copy()
            xi[i] += step_size
            fi = self._safe_call(xi)
            if fi is None:
                return
            simplex.append(xi)
            f_values.append(fi)
        
        if len(simplex) < dim + 1:
            return

        simplex = np.array(simplex)
        f_values = np.array(f_values)
        
        no_improve_count = 0
        patience = 20
        
        for iteration in range(max_iter):
            if self._calls >= budget or time.time() > deadline:
                break
            
            # Sort simplex by function value
            idx = np.argsort(f_values)
            simplex = simplex[idx]
            f_values = f_values[idx]
            
            # Update global best
            if f_values[0] < self._best_f:
                self._best_f = f_values[0]
                self._best_x = simplex[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Check for convergence
            spread_x = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            spread_f = np.max(f_values) - np.min(f_values)
            
            if np.max(spread_x) < 1e-6 or spread_f < 1e-10:
                break
            
            if no_improve_count > patience:
                break
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = self._safe_call(xr)
            if fr is None:
                break
            
            if fr < f_values[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = self._safe_call(xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            elif fr < f_values[-2]:
                # Reflection accepted
                simplex[-1] = xr
                f_values[-1] = fr
            else:
                # Contraction
                if fr < f_values[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = self._safe_call(xc)
                if fc is None:
                    break
                if fc <= min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        if self._calls >= budget or time.time() > deadline:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fi = self._safe_call(simplex[i])
                        if fi is None:
                            break
                        f_values[i] = fi
                    if fi is None:
                        break

    def _safe_call(self, x):
        """Internal safe call that tracks state."""
        nonlocal_calls = self._calls
        if nonlocal_calls >= self.budget:
            return None
        if time.time() > self._deadline:
            return None
        self._calls += 1
        try:
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            f_val = self._func(x)
            if f_val < self._best_f:
                self._best_f = f_val
                self._best_x = x.copy()
            return f_val
        except Exception:
            return None

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0
        
        self._calls = 0
        self._best_f = float('inf')
        self._best_x = None
        self._func = func
        self._deadline = deadline
        self._budget = budget
        self._dim = dim
        
        rng = np.random.default_rng(42)
        
        def safe_call(x):
            nonlocal self
            if self._calls >= budget:
                return None
            if time.time() > deadline:
                return None
            self._calls += 1
            try:
                x = np.asarray(x, dtype=np.float64)
                x = np.clip(x, -5.0, 5.0)
                f_val = func(x)
                if f_val < self._best_f:
                    self._best_f = f_val
                    self._best_x = x.copy()
                return f_val
            except Exception:
                return None
        
        # Phase 1: Coarse random search
        n_coarse = 40
        for _ in range(n_coarse):
            if self._calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break
        
        if self._best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return self._best_x, self._best_f
            self._best_x = x0
            self._best_f = f0
        
        # Phase 2: Local refinement with adaptive step sizes
        step_sizes = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]
        
        for step_size in step_sizes:
            if self._calls >= budget - 10 or time.time() > deadline:
                break
            
            x_start = self._best_x.copy()
            f_start = self._best_f
            
            # Inline Nelder-Mead to avoid recursion issues
            self._nelder_mead_inline(x_start, f_start, step_size, safe_call, max_iter=80)
            
            if self._best_x is None or np.isinf(self._best_f):
                break
        
        # Phase 3: Restart from perturbed best points
        if self._calls < budget - 20 and time.time() < deadline:
            for restart in range(5):
                if self._calls >= budget - 10 or time.time() > deadline:
                    break
                
                noise_scale = 0.3 / (restart + 1)
                perturbation = rng.normal(0, noise_scale, size=dim)
                x_restart = np.clip(self._best_x + perturbation, -5.0, 5.0)
                
                f_restart = safe_call(x_restart)
                if f_restart is None:
                    break
                
                if f_restart < self._best_f:
                    self._best_f = f_restart
                    self._best_x = x_restart.copy()
                
                self._nelder_mead_inline(x_restart.copy(), f_restart, 0.05, safe_call, max_iter=40)
        
        return self._best_x, self._best_f

    def _nelder_mead_inline(self, x_start, f_start, step_size, safe_call, max_iter=100):
        """Inline Nelder-Mead to avoid state management issues."""
        dim = self._dim
        budget = self._budget
        deadline = self._deadline
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        simplex = [x_start.copy()]
        f_values = [f_start]
        
        for i in range(dim):
            xi = x_start.copy()
            xi[i] += step_size
            fi = safe_call(xi)
            if fi is None:
                return
            simplex.append(xi)
            f_values.append(fi)
        
        if len(simplex) < dim + 1:
            return

        simplex = np.array(simplex)
        f_values = np.array(f_values)
        
        no_improve_count = 0
        patience = 20
        
        for iteration in range(max_iter):
            if self._calls >= budget or time.time() > deadline:
                break
            
            idx = np.argsort(f_values)
            simplex = simplex[idx]
            f_values = f_values[idx]
            
            if f_values[0] < self._best_f:
                self._best_f = f_values[0]
                self._best_x = simplex[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            spread_x = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            spread_f = np.max(f_values) - np.min(f_values)
            
            if np.max(spread_x) < 1e-6 or spread_f < 1e-10:
                break
            
            if no_improve_count > patience:
                break
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = safe_call(xr)
            if fr is None:
                break
            
            if fr < f_values[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = safe_call(xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            elif fr < f_values[-2]:
                simplex[-1] = xr
                f_values[-1] = fr
            else:
                if fr < f_values[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_call(xc)
                if fc is None:
                    break
                if fc <= min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    for i in range(1, len(simplex)):
                        if self._calls >= budget or time.time() > deadline:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fi = safe_call(simplex[i])
                        if fi is None:
                            break
                        f_values[i] = fi
                    if fi is None:
                        break
