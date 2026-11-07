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
        deadline = start_time + 290.0 # 10s safety margin
        
        d = self.dim
        self.func = func
        self.calls = 0
        
        # Initial simplex
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5, 5, d)
        
        # Initial simplex vertices
        x = [x0.copy()]
        fx = [self._evaluate(x[0])]
        
        if fx[0] is None:
            return x0, 0.0 # Should not happen with budget 500 and d=5

        for i in range(d):
            xi = x0.copy()
            step = 1.0
            xi[i] += step
            xi = np.clip(xi, -5, 5)
            x.append(xi)
            fx.append(self._evaluate(xi))
            if fx[-1] is None:
                break
            
        if self.calls >= self.budget:
             best_idx = np.argmin(fx)
             return x[best_idx], fx[best_idx]

        best_x = x[np.argmin(fx)]
        best_f = min(fx)
        
        # Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while self.calls < self.budget:
            if time.time() > deadline:
                break
                
            # Sort by function value
            order = np.argsort(fx)
            x = [x[i] for i in order]
            fx = [fx[i] for i in order]
            
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
                    # Couldn't expand, accept reflection if it was good
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
                best_x = x[current_best_idx]
                
        return best_x, best_f
