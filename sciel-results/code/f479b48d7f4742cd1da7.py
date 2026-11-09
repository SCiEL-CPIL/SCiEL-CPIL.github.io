import numpy as np
import time

class Optimizer:
    """DF-TR optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        n = self.dim
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(n)
        
        def eval_point(x):
            nonlocal calls, best_f, best_x
            if calls >= self.budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Phase 1: Brief random initialization to find a good starting point
        n_init = min(15, self.budget // 10)
        for _ in range(n_init):
            if calls >= self.budget or time.time() > deadline:
                break
            x_rand = np.random.uniform(-4.5, 4.5, size=n)
            f = eval_point(x_rand)
            if f is None:
                break
        
        # If no valid evaluations, return what we have
        if calls == 0:
            return best_x, best_f
        
        # Phase 2: Quadratic Model Fitting (DF-TR)
        n_points = n * (n + 1) // 2 + n + 1
        
        x_center = best_x.copy()
        delta = 1.5
        
        # Helper to get Hessian index in theta vector
        def h_idx(i, j):
            if i > j: i, j = j, i
            return n + i * n - i * (i - 1) // 2 + (j - i)
            
        max_iters = 30
        
        for iter_idx in range(max_iters):
            if calls + n_points > self.budget:
                break
            if time.time() > deadline - 5:
                break
                
            # Generate points
            points = [x_center.copy()]
            
            # Axis points
            for i in range(n):
                x_plus = x_center.copy()
                x_plus[i] += delta
                points.append(x_plus)
                
                x_minus = x_center.copy()
                x_minus[i] -= delta
                points.append(x_minus)
                
            # Pair points
            for i in range(n):
                for j in range(i + 1, n):
                    x_pair = x_center.copy()
                    x_pair[i] += delta / np.sqrt(2)
                    x_pair[j] += delta / np.sqrt(2)
                    points.append(x_pair)
                    
            # Evaluate
            fvals = []
            valid = True
            for p in points:
                f = eval_point(p)
                if f is None:
                    valid = False
                    break
                fvals.append(f)
                
            if not valid:
                break
                
            # Fit
            m = len(points) - 1
            num_unknowns = n + n * (n + 1) // 2
            A = np.zeros((m, num_unknowns))
            y = np.array(fvals[1:]) - fvals[0]
            
            for k in range(m):
                d = points[k+1] - x_center
                for i in range(n):
                    A[k, i] = d[i]
                for i in range(n):
                    for j in range(i, n):
                        if i == j:
                            coeff = 0.5 * d[i] * d[j]
                        else:
                            coeff = d[i] * d[j]
                        A[k, h_idx(i, j)] = coeff
                        
            try:
                theta, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
            except:
                break
                
            g = theta[:n]
            H = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    val = theta[h_idx(i, j)]
                    H[i, j] = val
                    H[j, i] = val
            
            # Regularize Hessian if needed for stability
            eigvals = np.linalg.eigvalsh(H)
            if np.min(eigvals) < 1e-8:
                H += (1e-8 - np.min(eigvals)) * np.eye(n)
                    
            try:
                step = np.linalg.solve(H, -g)
                # Limit step size to avoid overshooting
                step_norm = np.linalg.norm(step)
                if step_norm > delta:
                    step = step * delta / step_norm
            except:
                step = -g * 0.1
                step_norm = np.linalg.norm(step)
                if step_norm > delta:
                    step = step * delta / step_norm
                
            x_next = x_center + step
            x_next = np.clip(x_next, -5, 5)
            
            f_next = eval_point(x_next)
            if f_next is None:
                break
                
            if f_next < fvals[0]:
                x_center = x_next
                delta *= 0.7
            else:
                delta *= 0.5
                
            if delta < 1e-5:
                break
                
        # Phase 3: Final polish with coordinate descent using remaining budget
        if calls < self.budget and time.time() < deadline - 2:
            x_polish = best_x.copy()
            step_size = 0.1
            for _ in range(20):
                if calls >= self.budget or time.time() > deadline - 1:
                    break
                improved = False
                for i in range(n):
                    if calls >= self.budget or time.time() > deadline - 1:
                        break
                    for direction in [1, -1]:
                        x_trial = x_polish.copy()
                        x_trial[i] += direction * step_size
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = eval_point(x_trial)
                        if f_trial is None:
                            break
                        if f_trial < best_f:
                            x_polish = x_trial
                            improved = True
                            break
                    if time.time() > deadline - 1:
                        break
                if not improved:
                    step_size *= 0.5
                    if step_size < 1e-6:
                        break
        
        return best_x, best_f
