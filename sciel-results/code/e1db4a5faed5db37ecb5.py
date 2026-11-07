import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280.0

        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def check_time():
            return time.time() < deadline and calls < budget

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if not check_time():
                return None
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)
            return f

        # Phase 0: Evaluate center
        center = np.zeros(n)
        safe_call(center)

        if not check_time():
            return best_x, best_f

        # Phase 1: Coordinate Descent to find ridge direction
        # The sharp ridge means the function is flat along the ridge and steep
        # perpendicular to it. CD will quickly converge along the steep directions
        # and reveal the ridge direction.
        
        # First, do a quick scan to find a good starting point
        # Try a few random points
        for _ in range(min(5, budget - calls)):
            if not check_time():
                break
            p = rng.uniform(-4, 4, n)
            safe_call(p)

        # Coordinate descent: sweep each coordinate, find best along each axis
        step = 1.0
        for cd_round in range(20):
            if not check_time():
                break
            improved = False
            for i in range(n):
                if not check_time():
                    break
                # Scan coordinate i from current best position
                # Use a fine grid around the current best
                base = best_x.copy()
                # Try a range of values for coordinate i
                # Start with a wider search, then narrow
                if cd_round < 3:
                    vals = np.linspace(-5, 5, 11)
                else:
                    center_val = best_x[i]
                    spread = step * 2
                    vals = np.linspace(max(-5, center_val - spread), min(5, center_val + spread), 11)
                
                local_best_val = best_f
                local_best_i = best_x[i]
                for v in vals:
                    if not check_time():
                        break
                    test = base.copy()
                    test[i] = v
                    f = safe_call(test)
                    if f is not None and f < local_best_val:
                        local_best_val = f
                        local_best_i = v
                
                if local_best_i != best_x[i]:
                    best_x[i] = local_best_i
                    improved = True
            
            if not improved:
                step *= 0.5
                if step < 0.01:
                    break

        if not check_time():
            return best_x, best_f

        # Phase 2: Nelder-Mead for local refinement
        # Initialize simplex around the best point found by CD
        delta = 0.3
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x.copy()
        for i in range(n):
            simplex[i + 1] = best_x.copy()
            simplex[i + 1, i] += delta
        simplex = np.clip(simplex, -5, 5)

        f_simplex = np.zeros(n + 1)
        for i in range(n + 1):
            if not check_time():
                break
            f_simplex[i] = func(simplex[i])
            calls += 1
            if f_simplex[i] < best_f:
                best_f = f_simplex[i]
                best_x = simplex[i].copy()

        if not check_time():
            return best_x, best_f

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        # Main Nelder-Mead loop
        while check_time():
            # Sort simplex
            idx = np.argsort(f_simplex)
            simplex_sorted = simplex[idx]
            f_sorted = f_simplex[idx]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = simplex_sorted[0].copy()

            # Check convergence
            if np.max(f_sorted) - np.min(f_sorted) < 1e-14:
                break

            centroid = np.mean(simplex_sorted[:n], axis=0)
            worst = simplex_sorted[n]

            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)

            if not check_time():
                break
            f_r = func(x_r)
            calls += 1

            if f_r < f_sorted[0]:
                # Expansion
                if not check_time():
                    break
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1

                if f_e < f_r:
                    simplex_sorted[n] = x_e
                    f_sorted[n] = f_e
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                else:
                    simplex_sorted[n] = x_r
                    f_sorted[n] = f_r
                    if f_r < best_f:
                        best_f = f_r
                        best_x = x_r.copy()
            elif f_r < f_sorted[n - 1]:
                # Accept reflection
                simplex_sorted[n] = x_r
                f_sorted[n] = f_r
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
            else:
                # Contraction
                if f_r < f_sorted[n]:
                    # Outside contraction
                    x_c = centroid + rho * (worst - centroid)
                else:
                    # Inside contraction
                    x_c = centroid + rho * (x_r - centroid)

                x_c = np.clip(x_c, -5, 5)

                if not check_time():
                    break
                f_c = func(x_c)
                calls += 1

                if f_c < min(f_r, f_sorted[n]):
                    simplex_sorted[n] = x_c
                    f_sorted[n] = f_c
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex_sorted[i] = simplex_sorted[0] + sigma * (simplex_sorted[i] - simplex_sorted[0])
                        simplex_sorted[i] = np.clip(simplex_sorted[i], -5, 5)

                        if not check_time():
                            break
                        f_sorted[i] = func(simplex_sorted[i])
                        calls += 1
                        if f_sorted[i] < best_f:
                            best_f = f_sorted[i]
                            best_x = simplex_sorted[i].copy()

                    if not check_time():
                        break

            simplex = simplex_sorted
            f_simplex = f_sorted

        return best_x, best_f
