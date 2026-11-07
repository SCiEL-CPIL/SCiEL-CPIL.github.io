import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None
        self.best_x = None
        self.best_f = float('inf')

    def _evaluate(self, x):
        if self.calls >= self.budget:
            return None
        self.calls += 1
        val = self.func(x)
        if val < self.best_f:
            self.best_f = val
            self.best_x = x.copy()
        return val

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 300s limit
        
        d = self.dim
        self.func = func
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = None
        
        # Strategy:
        # 1. L-BFGS-B from origin to find a good basin
        # 2. Adaptive Coordinate Descent with parabolic/golden section 1D minimization
        # 3. Nelder-Mead polish
        
        def lbfgsb_func(x):
            if self.calls >= self.budget or time.time() > deadline - 5:
                return 1e10
            f_val = self._evaluate(x)
            if f_val is None:
                return 1e10
            return f_val

        x0 = np.zeros(d)
        f0 = self._evaluate(x0)
        if f0 is None:
            return np.zeros(d), 0.0

        # Phase 1: L-BFGS-B
        try:
            res = minimize(lbfgsb_func, x0, method='L-BFGS-B', 
                           bounds=[(-5, 5)] * d, 
                           options={'maxiter': 30, 'ftol': 1e-10, 'gtol': 1e-6})
        except Exception:
            pass

        if self.best_x is None:
            self.best_x = x0.copy()
            self.best_f = f0

        # Phase 2: Adaptive Coordinate Descent
        # Key insight: Bent Cigar has a narrow valley. We need to find the valley first,
        # then converge precisely along it.
        
        def _minimize_1d_adaptive(axis, x_center, f_center, window_size):
            """
            Minimizes func along 'axis' centered at x_center.
            Uses adaptive window: starts wide, shrinks based on where min is found.
            Uses parabolic interpolation when possible for faster convergence.
            """
            if self.calls >= self.budget - 5 or time.time() > deadline - 3:
                return x_center[axis], f_center
            
            # Try to find the minimum along this axis
            # Start with a window around the current point
            a = max(-5.0, x_center[axis] - window_size)
            b = min(5.0, x_center[axis] + window_size)
            
            if b - a < 1e-14:
                return x_center[axis], f_center
            
            # Evaluate at three points: left, center, right
            mid = (a + b) / 2.0
            
            x_left = x_center.copy()
            x_left[axis] = a
            f_left = self._evaluate(x_left)
            
            x_right = x_center.copy()
            x_right[axis] = b
            f_right = self._evaluate(x_right)
            
            if f_left is None or f_right is None:
                return x_center[axis], f_center
            
            # Keep track of best
            best_pos = x_center[axis]
            best_val = f_center
            if f_left < best_val:
                best_val = f_left
                best_pos = a
            if f_right < best_val:
                best_val = f_right
                best_pos = b
            
            # Now we have three points: (a, f_left), (x_center[axis], f_center), (b, f_right)
            # Use parabolic interpolation to estimate the minimum
            # Parabolic interpolation formula for points (x1, f1), (x2, f2), (x3, f3):
            # x_min = x2 - 0.5 * ((x2-x1)^2*(f2-f3) - (x2-x3)^2*(f2-f1)) / ((x2-x1)*(f2-f3) - (x2-x3)*(f2-f1))
            
            x1, f1 = a, f_left
            x2, f2 = x_center[axis], f_center
            x3, f3 = b, f_right
            
            denom = (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)
            if abs(denom) > 1e-15:
                x_min = x2 - 0.5 * ((x2 - x1)**2 * (f2 - f3) - (x2 - x3)**2 * (f2 - f1)) / denom
            else:
                x_min = x2
            
            # Clip to bounds
            x_min = max(a, min(b, x_min))
            
            # Evaluate at the parabolic estimate
            x_par = x_center.copy()
            x_par[axis] = x_min
            f_par = self._evaluate(x_par)
            
            if f_par is None:
                return best_pos, best_val
            
            if f_par < best_val:
                best_val = f_par
                best_pos = x_min
            
            # Now we have four points. Use the three best to refine.
            # For simplicity, let's do a few more iterations with golden section
            # on the interval containing the best point
            
            # Determine a tighter interval around the best point
            # Use the two points that bracket the best
            points = [(a, f_left), (x_center[axis], f_center), (b, f_right), (x_min, f_par)]
            points.sort(key=lambda p: p[0])
            
            # Find the interval that contains the best point
            best_idx = 0
            for i, (px, pf) in enumerate(points):
                if pf == best_val:
                    best_idx = i
                    break
            
            # Determine left and right bounds for golden section
            if best_idx > 0:
                left_bound = points[best_idx - 1][0]
            else:
                left_bound = a
            
            if best_idx < len(points) - 1:
                right_bound = points[best_idx + 1][0]
            else:
                right_bound = b
            
            # Ensure we have a valid interval
            if right_bound - left_bound < 1e-14:
                return best_pos, best_val
            
            # Golden section search on [left_bound, right_bound]
            gr = (np.sqrt(5) - 1) / 2
            inv_gr = 1.0 / gr
            
            a_gs = left_bound
            b_gs = right_bound
            
            c = b_gs - inv_gr * (b_gs - a_gs)
            d_pt = a_gs + inv_gr * (b_gs - a_gs)
            
            x_c = x_center.copy()
            x_c[axis] = c
            f_c = self._evaluate(x_c)
            
            x_d = x_center.copy()
            x_d[axis] = d_pt
            f_d = self._evaluate(x_d)
            
            if f_c is None or f_d is None:
                return best_pos, best_val
            
            if f_c < best_val:
                best_val = f_c
                best_pos = c
            if f_d < best_val:
                best_val = f_d
                best_pos = d_pt
            
            # Iterate golden section
            max_iter = 30
            for _ in range(max_iter):
                if self.calls >= self.budget - 5 or time.time() > deadline - 3:
                    break
                
                if f_c < f_d:
                    b_gs = d_pt
                    d_pt = c
                    f_d = f_c
                    c = b_gs - inv_gr * (b_gs - a_gs)
                    x_c = x_center.copy()
                    x_c[axis] = c
                    f_c = self._evaluate(x_c)
                    if f_c is None:
                        break
                    if f_c < best_val:
                        best_val = f_c
                        best_pos = c
                else:
                    a_gs = c
                    c = d_pt
                    f_c = f_d
                    d_pt = a_gs + inv_gr * (b_gs - a_gs)
                    x_d = x_center.copy()
                    x_d[axis] = d_pt
                    f_d = self._evaluate(x_d)
                    if f_d is None:
                        break
                    if f_d < best_val:
                        best_val = f_d
                        best_pos = d_pt
                
                if abs(b_gs - a_gs) < 1e-12:
                    break
            
            return best_pos, best_val

        # Run adaptive coordinate descent
        # Start with a moderate window, shrink as we converge
        current_x = self.best_x.copy()
        current_f = self.best_f
        
        # Multiple passes with shrinking windows
        windows = [2.0, 0.5, 0.1, 0.02]
        
        for window in windows:
            if self.calls >= self.budget - 50 or time.time() > deadline - 10:
                break
            
            for _ in range(3):  # 3 passes per window size
                if self.calls >= self.budget - 20 or time.time() > deadline - 5:
                    break
                
                for axis in range(d):
                    if self.calls >= self.budget - 10 or time.time() > deadline - 3:
                        break
                    
                    best_pos, best_val = _minimize_1d_adaptive(axis, current_x, current_f, window)
                    current_x[axis] = best_pos
                    current_f = best_val
                    
                    if best_val < self.best_f:
                        self.best_f = best_val
                        self.best_x = current_x.copy()
        
        # Phase 3: Nelder-Mead polish
        if self.calls < self.budget - 10 and time.time() < deadline - 10:
            
            def nm_func(x):
                if self.calls >= self.budget or time.time() > deadline - 2:
                    return 1e10
                f_val = self._evaluate(x)
                if f_val is None:
                    return 1e10
                return f_val

            x_start = self.best_x if self.best_x is not None else x0
            
            try:
                res2 = minimize(nm_func, x_start, method='Nelder-Mead',
                                options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-14})
            except Exception:
                pass

        if self.best_x is None:
            self.best_x = np.zeros(d)
            self.best_f = 0.0
            
        return self.best_x, self.best_f
