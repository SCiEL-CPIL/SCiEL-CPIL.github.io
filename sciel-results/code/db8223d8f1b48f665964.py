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
            
        # Phase 1: Initial exploration to find a good starting point
        # Use a small set of structured points around the origin
        x_center = np.zeros(n)
        delta = 2.0
        
        # Evaluate center
        f_center = eval_point(x_center)
        if f_center is None:
            return best_x, best_f
        
        # Evaluate axis points to get gradient estimate
        axis_points = []
        for i in range(n):
            x_plus = x_center.copy()
            x_plus[i] += delta
            f_plus = eval_point(x_plus)
            if f_plus is None:
                break
            axis_points.append((f_plus, i, 1))
            
            x_minus = x_center.copy()
            x_minus[i] -= delta
            f_minus = eval_point(x_minus)
            if f_minus is None:
                break
            axis_points.append((f_minus, i, -1))
        
        if calls >= self.budget or time.time() > deadline:
            return best_x, best_f
        
        # Phase 2: DF-TR with trust region
        # Trust region radius
        tr_radius = 1.5
        max_tr_radius = 4.0
        min_tr_radius = 1e-6
        max_iters = 30
        
        for iter_idx in range(max_iters):
            if time.time() > deadline:
                break
            remaining = self.budget - calls
            if remaining < 25:  # Need at least 21 points + a few
                break
                
            # Generate points for quadratic model
            points = [x_center.copy()]
            
            # Axis points
            for i in range(n):
                x_plus = x_center.copy()
                x_plus[i] += tr_radius
                points.append(x_plus)
                
                x_minus = x_center.copy()
                x_minus[i] -= tr_radius
                points.append(x_minus)
                
            # Pair points
            for i in range(n):
                for j in range(i + 1, n):
                    x_pair = x_center.copy()
                    x_pair[i] += tr_radius / np.sqrt(2)
                    x_pair[j] += tr_radius / np.sqrt(2)
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
                
            if not valid or len(fvals) < 2:
                break
                
            # Fit quadratic model
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
                    
            # Check if Hessian is positive definite enough
            try:
                eigvals = np.linalg.eigvalsh(H)
                if np.min(eigvals) < -1e-6:
                    # Add regularization to make it PD
                    H += (abs(np.min(eigvals)) + 1e-6) * np.eye(n)
            except:
                H = np.eye(n) * 0.1
                
            # Compute step
            try:
                step = np.linalg.solve(H, -g)
            except:
                step = -g * 0.1
                
            # Scale step to trust region
            step_norm = np.linalg.norm(step)
            if step_norm > tr_radius:
                step = step * (tr_radius / step_norm)
            elif step_norm < 1e-10:
                # Gradient is small, reduce trust region
                tr_radius *= 0.5
                if tr_radius < min_tr_radius:
                    break
                continue
                
            x_next = x_center + step
            x_next = np.clip(x_next, -5, 5)
            
            f_next = eval_point(x_next)
            if f_next is None:
                break
                
            # Accept or reject
            predicted_reduction = fvals[0] - (fvals[0] + g @ step + 0.5 * step @ H @ step)
            actual_reduction = fvals[0] - f_next
            
            if actual_reduction > 0.25 * predicted_reduction and f_next < fvals[0]:
                # Successful step
                x_center = x_next
                tr_radius = min(tr_radius * 1.5, max_tr_radius)
            else:
                # Unsuccessful step
                tr_radius *= 0.5
                
            if tr_radius < min_tr_radius:
                break
                
        # Phase 3: Final polish using Nelder-Mead simplex method
        if time.time() < deadline and calls < self.budget - 10:
            # Initialize simplex around best_x
            alpha = 1.0  # expansion
            gamma = 0.5  # contraction
            rho = 2.0    # reflection
            sigma = 0.5  # shrinkage
            
            simplex = [best_x.copy()]
            for i in range(n):
                v = best_x.copy()
                v[i] += 0.1
                simplex.append(v)
            
            # Evaluate simplex
            simplex_f = []
            for v in simplex:
                f = eval_point(v)
                if f is None:
                    break
                simplex_f.append(f)
            
            if len(simplex_f) == n + 1:
                max_polish_iters = 50
                for _ in range(max_polish_iters):
                    if time.time() > deadline or calls >= self.budget:
                        break
                    
                    # Sort by function value
                    idx = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in idx]
                    simplex_f = [simplex_f[i] for i in idx]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_reflect = centroid + rho * (centroid - simplex[-1])
                    x_reflect = np.clip(x_reflect, -5, 5)
                    f_reflect = eval_point(x_reflect)
                    if f_reflect is None:
                        break
                    
                    if simplex_f[0] <= f_reflect < simplex_f[-2]:
                        # Accept reflection
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                    elif f_reflect < simplex_f[0]:
                        # Expansion
                        x_expand = centroid + alpha * (x_reflect - centroid)
                        x_expand = np.clip(x_expand, -5, 5)
                        f_expand = eval_point(x_expand)
                        if f_expand is None:
                            break
                        if f_expand < f_reflect:
                            simplex[-1] = x_expand
                            simplex_f[-1] = f_expand
                        else:
                            simplex[-1] = x_reflect
                            simplex_f[-1] = f_reflect
                    else:
                        # Contraction
                        x_contract = centroid + gamma * (simplex[-1] - centroid)
                        x_contract = np.clip(x_contract, -5, 5)
                        f_contract = eval_point(x_contract)
                        if f_contract is None:
                            break
                        if f_contract < simplex_f[-1]:
                            simplex[-1] = x_contract
                            simplex_f[-1] = f_contract
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                f = eval_point(simplex[i])
                                if f is None:
                                    break
                                simplex_f[i] = f
                            if f is None:
                                break
                            
                            # Check if simplex is too small
                            if np.max(np.array(simplex_f)) - np.min(np.array(simplex_f)) < 1e-10:
                                break

        return best_x, best_f
