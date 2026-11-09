import numpy as np
import time

class Optimizer:
    """Quadratic Model Fitting optimizer for BBOB-style functions."""

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
        # This helps avoid getting stuck in a bad local region if the function
        # were more complex, but for Sphere it's mostly to establish a baseline.
        # We use a small number of points to conserve budget for the quadratic fit.
        n_init = 10
        for _ in range(n_init):
            if calls >= self.budget or time.time() > deadline:
                break
            x_rand = np.random.uniform(-5, 5, size=n)
            f = eval_point(x_rand)
            if f is None:
                break
        
        if calls == 0:
            return best_x, best_f

        # Phase 2: Quadratic Model Fitting (Newton's Method)
        # For n=5, we need 21 points (1 center + 10 axis + 10 pair) to fit a full quadratic model.
        # f(x) = f0 + g^T (x-x0) + 0.5 (x-x0)^T H (x-x0)
        # This converges in 1-2 steps for quadratic functions like Sphere.
        
        # Number of points needed: 1 (center) + 2n (axes) + n(n-1)/2 (pairs)
        # Total = 1 + 2*5 + 10 = 21 for n=5.
        # General formula: 1 + 2n + n(n-1)/2 = 1 + 1.5n + 0.5n^2
        # Alternatively, n(n+3)/2 + 1
        n_points = 1 + 2 * n + n * (n - 1) // 2
        
        x_center = best_x.copy()
        delta = 2.0  # Initial trust region radius
        
        # Helper to get Hessian index in theta vector
        # theta = [g_1, ..., g_n, H_11, H_12, ..., H_nn]
        # H is symmetric, so we store upper triangle.
        def h_idx(i, j):
            if i > j:
                i, j = j, i
            # Index in the Hessian part (starting at n)
            # Number of elements before row i in upper triangle: i*n - i*(i-1)/2
            # Actually, let's just count:
            # H_11 is at index n
            # H_12 is at index n+1
            # ...
            # H_1n is at index n + (n-1)
            # H_22 is at index n + n
            # Let's use a formula:
            # index = n + (i * (2*n - i + 1)) // 2 - (j - i) - 1 ? No, that's messy.
            # Simpler:
            # k = 0
            # for a in range(n):
            #   for b in range(a, n):
            #     if a==i and b==j: return n+k
            #     k+=1
            # For n=5, this is fast enough.
            k = 0
            for a in range(n):
                for b in range(a, n):
                    if a == i and b == j:
                        return n + k
                    k += 1
            return -1 # Should not happen

        max_iters = 20  # Max iterations of the quadratic fit loop
        
        for iter_idx in range(max_iters):
            # Check if we have enough budget for a full cycle
            # Each cycle costs n_points evaluations
            if calls + n_points > self.budget:
                break
            if time.time() > deadline - 5:
                break
                
            # Generate points
            points = [x_center.copy()]
            
            # Axis points: x_center +/- delta * e_i
            for i in range(n):
                x_plus = x_center.copy()
                x_plus[i] += delta
                points.append(x_plus)
                
                x_minus = x_center.copy()
                x_minus[i] -= delta
                points.append(x_minus)
                
            # Pair points: x_center + delta/sqrt(2) * (e_i + e_j) for i < j
            # This allows estimating cross-terms H_ij
            for i in range(n):
                for j in range(i + 1, n):
                    x_pair = x_center.copy()
                    x_pair[i] += delta / np.sqrt(2)
                    x_pair[j] += delta / np.sqrt(2)
                    points.append(x_pair)
                    
            # Evaluate all points
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
                
            # Fit the quadratic model
            # We have m = n_points - 1 points relative to center
            m = len(points) - 1
            num_unknowns = n + n * (n + 1) // 2  # n gradient terms + n(n+1)/2 Hessian terms
            
            A = np.zeros((m, num_unknowns))
            y = np.array(fvals[1:]) - fvals[0]
            
            for k in range(m):
                d = points[k + 1] - x_center
                # Gradient terms
                for i in range(n):
                    A[k, i] = d[i]
                # Hessian terms
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
                    
            # Solve for step: H * step = -g
            # Add regularization if H is ill-conditioned
            try:
                step = np.linalg.solve(H, -g)
            except:
                # Fallback to steepest descent or small step
                step = -g * 0.1
                
            # Clip step to trust region
            step_norm = np.linalg.norm(step)
            if step_norm > delta:
                step = step * (delta / step_norm)
                
            x_next = x_center + step
            x_next = np.clip(x_next, -5, 5)
            
            f_next = eval_point(x_next)
            if f_next is None:
                break
                
            # Update center
            # If we made progress, keep the new point as center.
            # The quadratic model is exact for Sphere, so we should converge quickly.
            x_center = x_next
            
            # Shrink delta for next iteration to refine the model
            # Since we converged (or should have) to the minimum, the next model
            # will be built around the new center.
            # For Sphere, the minimum is found in 1 step.
            # We shrink delta to ensure we stay in the basin of attraction if the
            # function were slightly non-quadratic, but for Sphere it's fine.
            delta *= 0.5
            
            if delta < 1e-6:
                break
                
        return best_x, best_f
