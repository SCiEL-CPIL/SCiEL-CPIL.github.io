import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None

    def _evaluate(self, x):
        if self.calls >= self.budget:
            return None
        self.calls += 1
        return self.func(x)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin for 300s limit
        
        d = self.dim
        self.func = func
        self.calls = 0
        
        best_x = None
        best_f = float('inf')
        
        # Multiple restarts with different initializations
        n_restarts = 3
        budget_per_restart = max(100, self.budget // n_restarts)
        
        for restart in range(n_restarts):
            if self.calls >= self.budget:
                break
            if time.time() > deadline - 30:  # leave 30s for final polish
                break
                
            # Different initialization strategies
            if restart == 0:
                # Start near center
                rng = np.random.default_rng(42)
                x0 = rng.uniform(-2, 2, d)
                step = 1.0
            elif restart == 1:
                # Start at origin with larger steps
                x0 = np.zeros(d)
                step = 2.0
            else:
                # Random start
                rng = np.random.default_rng(123 + restart)
                x0 = rng.uniform(-4, 4, d)
                step = 1.5
            
            # Initial simplex
            x = [x0.copy()]
            fx = [self._evaluate(x[0])]
            
            if fx[0] is None:
                break
            
            if fx[0] < best_f:
                best_f = fx[0]
                best_x = x[0].copy()
            
            for i in range(d):
                if self.calls >= self.budget:
                    break
                xi = x0.copy()
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                x.append(xi)
                fx.append(self._evaluate(xi))
                if fx[-1] is None:
                    break
                if fx[-1] < best_f:
                    best_f = fx[-1]
                    best_x = xi.copy()
            
            if len(x) < d + 1 or self.calls >= self.budget:
                continue
            
            # Nelder-Mead with adaptive parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Track progress for stagnation detection
            no_improve_count = 0
            best_f_in_restart = min(fx)
            
            while self.calls < self.budget:
                if time.time() > deadline - 30:
                    break
                    
                # Sort by function value
                order = np.argsort(fx)
                x = [x[i] for i in order]
                fx = [fx[i] for i in order]
                
                # Check for convergence
                if fx[-1] - fx[0] < 1e-10:
                    break
                
                # Centroid of best d points (excluding worst)
                centroid = np.mean(x[:-1], axis=0)
                
                # Reflect
                xr = centroid + alpha * (centroid - x[-1])
                xr = np.clip(xr, -5, 5)
                
                fxr = self._evaluate(xr)
                if fxr is None:
                    break

                if fxr < fx[0]:
                    # Expand
                    xe = centroid + gamma * (centroid - x[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = self._evaluate(xe)
                    if fxe is None:
                        if fxr < fx[-1]:
                            x[-1] = xr
                            fx[-1] = fxr
                        break
                    if fxe < fxr:
                        x[-1] = xe
                        fx[-1] = fxe
                    else:
                        x[-1] = xr
                        fx[-1] = fxr
                elif fxr < fx[-2]:
                    # Accept reflection
                    x[-1] = xr
                    fx[-1] = fxr
                else:
                    # Contract
                    if fxr < fx[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fxc = self._evaluate(xc)
                        if fxc is None:
                            break
                        if fxc <= fxr:
                            x[-1] = xc
                            fx[-1] = fxc
                        else:
                            # Shrink
                            for i in range(1, d + 1):
                                if self.calls >= self.budget:
                                    break
                                x[i] = x[0] + sigma * (x[i] - x[0])
                                x[i] = np.clip(x[i], -5, 5)
                                fx[i] = self._evaluate(x[i])
                                if fx[i] is None:
                                    break
                    else:
                        # Inside contraction
                        xc = centroid + rho * (x[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fxc = self._evaluate(xc)
                        if fxc is None:
                            break
                        if fxc < fx[-1]:
                            x[-1] = xc
                            fx[-1] = fxc
                        else:
                            # Shrink
                            for i in range(1, d + 1):
                                if self.calls >= self.budget:
                                    break
                                x[i] = x[0] + sigma * (x[i] - x[0])
                                x[i] = np.clip(x[i], -5, 5)
                                fx[i] = self._evaluate(x[i])
                                if fx[i] is None:
                                    break
                
                # Update best
                current_best_idx = np.argmin(fx)
                if fx[current_best_idx] < best_f:
                    best_f = fx[current_best_idx]
                    best_x = x[current_best_idx].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Stagnation detection - if no improvement for many iterations, break
                if no_improve_count > 30:
                    break
        
        # Final polish: coordinate descent refinement
        if best_x is not None and self.calls < self.budget and time.time() < deadline - 5:
            x_polish = best_x.copy()
            f_polish = best_f
            step_size = 0.5
            
            for _ in range(20):  # max 20 refinement iterations
                if self.calls >= self.budget or time.time() > deadline - 2:
                    break
                
                improved = False
                for i in range(d):
                    if self.calls >= self.budget:
                        break
                    
                    # Try positive step
                    x_try = x_polish.copy()
                    x_try[i] += step_size
                    x_try = np.clip(x_try, -5, 5)
                    f_try = self._evaluate(x_try)
                    if f_try is not None and f_try < f_polish:
                        x_polish = x_try
                        f_polish = f_try
                        improved = True
                        continue
                    
                    # Try negative step
                    x_try = x_polish.copy()
                    x_try[i] -= step_size
                    x_try = np.clip(x_try, -5, 5)
                    f_try = self._evaluate(x_try)
                    if f_try is not None and f_try < f_polish:
                        x_polish = x_try
                        f_polish = f_try
                        improved = True
                
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-4:
                        break
                
                if f_polish < best_f:
                    best_f = f_polish
                    best_x = x_polish.copy()
        
        if best_x is None:
            # Fallback: return the first point we evaluated
            best_x = np.zeros(d)
            best_f = self.func(best_x)
        
        return best_x, best_f
