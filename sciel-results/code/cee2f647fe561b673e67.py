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

        # Helper to get Hessian index in theta vector
        def h_idx(i, j):
            if i > j: i, j = j, i
            return n + i * n - i * (i - 1) // 2 + (j - i)

        # Phase 1: Build initial quadratic model
        x_center = np.zeros(n)
        delta = 1.0
        
        # Evaluate center
        f_center = eval_point(x_center)
        if f_center is None:
            return best_x, best_f
            
        # Generate structured points
        points = [x_center.copy()]
        for i in range(n):
            x_plus = x_center.copy(); x_plus[i] += delta; points.append(x_plus)
            x_minus = x_center.copy(); x_minus[i] -= delta; points.append(x_minus)
        for i in range(n):
            for j in range(i + 1, n):
                x_pair = x_center.copy()
                x_pair[i] += delta / np.sqrt(2)
                x_pair[j] += delta / np.sqrt(2)
                points.append(x_pair)
        
        # Evaluate all
        fvals = [f_center]
        for p in points[1:]:
            f = eval_point(p)
            if f is None:
                return best_x, best_f
            fvals.append(f)
            
        # Fit quadratic
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
                    coeff = 0.5 * d[i] * d[j] if i == j else d[i] * d[j]
                    A[k, h_idx(i, j)] = coeff
                    
        try:
            theta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        except:
            return best_x, best_f
            
        g = theta[:n]
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                val = theta[h_idx(i, j)]
                H[i, j] = val; H[j, i] = val
                
        # Phase 2: Newton iterations with adaptive delta
        max_iters = 30
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            try:
                eigvals = np.linalg.eigvalsh(H)
                if np.min(eigvals) > 1e-8:
                    step = np.linalg.solve(H, -g)
                else:
                    step = -g * 0.05
            except:
                step = -g * 0.05
                
            step_norm = np.linalg.norm(step)
            if step_norm > delta:
                step = step * (delta / step_norm)
                
            x_next = np.clip(x_center + step, -5, 5)
            f_next = eval_point(x_next)
            if f_next is None:
                break
                
            if f_next < f_center:
                x_center = x_next
                f_center = f_next
                delta *= 0.6
            else:
                delta *= 0.3
                
            if delta < 1e-6:
                break
                
            # Refit model every 3 iterations if budget allows
            if (it + 1) % 3 == 0 and calls + 21 <= self.budget and delta > 1e-3:
                points = [x_center.copy()]
                for i in range(n):
                    x_plus = x_center.copy(); x_plus[i] += delta; points.append(x_plus)
                    x_minus = x_center.copy(); x_minus[i] -= delta; points.append(x_minus)
                for i in range(n):
                    for j in range(i + 1, n):
                        x_pair = x_center.copy()
                        x_pair[i] += delta / np.sqrt(2)
                        x_pair[j] += delta / np.sqrt(2)
                        points.append(x_pair)
                
                fvals = [f_center]
                for p in points[1:]:
                    f = eval_point(p)
                    if f is None:
                        break
                    fvals.append(f)
                else:
                    m = len(points) - 1
                    A = np.zeros((m, num_unknowns))
                    y = np.array(fvals[1:]) - fvals[0]
                    for k in range(m):
                        d = points[k+1] - x_center
                        for i in range(n):
                            A[k, i] = d[i]
                        for i in range(n):
                            for j in range(i, n):
                                coeff = 0.5 * d[i] * d[j] if i == j else d[i] * d[j]
                                A[k, h_idx(i, j)] = coeff
                    try:
                        theta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                        g = theta[:n]
                        H = np.zeros((n, n))
                        for i in range(n):
                            for j in range(i, n):
                                val = theta[h_idx(i, j)]
                                H[i, j] = val; H[j, i] = val
                    except:
                        pass

        # Phase 3: Final polish with remaining budget
        if calls < self.budget and time.time() < deadline - 2:
            polish_delta = max(delta * 0.05, 1e-7)
            for i in range(n):
                for sign in [1, -1]:
                    if calls >= self.budget or time.time() > deadline:
                        break
                    x_try = best_x.copy()
                    x_try[i] += sign * polish_delta
                    x_try = np.clip(x_try, -5, 5)
                    eval_point(x_try)

        return best_x, best_f
