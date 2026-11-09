"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Model-based optimizer with paraboloid fitting and Lunacek-aware basin jumping."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def fit_paraboloid(center, h):
            """
            Fit a local paraboloid using 2n+1 points.
            Returns the predicted minimizer or None.
            """
            if calls_made + 2 * n >= total_budget:
                return None
            
            points = [center]
            values = []
            
            # Evaluate center
            fc = evaluate(center)
            values.append(fc)
            
            # Evaluate neighbors
            for i in range(n):
                for sign in [1, -1]:
                    p = center.copy()
                    p[i] += sign * h
                    p = np.clip(p, -5, 5)
                    if calls_made >= total_budget:
                        return None
                    fp = evaluate(p)
                    points.append(p)
                    values.append(fp)
            
            # Now we have 2n+1 points.
            # Fit f(x) = c + g^T(x-c) + 0.5 (x-c)^T H (x-c)
            # Using the 2n+1 points, we can solve for g and H efficiently.
            # g_i = (f(c + h e_i) - f(c - h e_i)) / (2h)
            # H_ii = (f(c + h e_i) + f(c - h e_i) - 2 f(c)) / h^2
            # H_ij (i != j) cannot be determined with only axis-aligned points.
            # We assume H is diagonal for the jump. This is valid if the basin is roughly spherical or aligned.
            # For Lunacek, the basins are rotated, but a diagonal approximation often works for a good jump.
            # Alternatively, we can use a full quadratic fit if we have enough points, but 2n+1 is minimal for diagonal.
            # Let's try a full quadratic fit if we have budget for more points? No, stick to 2n+1 for speed and robustness.
            # Actually, a diagonal Hessian assumption might fail if the basin is elongated.
            # Let's use the points to fit a general quadratic via least squares if we have enough points (which we do, 2n+1 >= n+1 for n>=1).
            # But 2n+1 points in n dimensions is sufficient to determine a diagonal quadratic.
            # Let's just use the diagonal approximation for the jump direction.
            
            g = np.zeros(n)
            H_diag = np.zeros(n)
            
            for i in range(n):
                # points[1 + 2*i] is c + h e_i
                # points[2 + 2*i] is c - h e_i
                fp = values[1 + 2 * i]
                fm = values[2 + 2 * i]
                g[i] = (fp - fm) / (2 * h)
                H_diag[i] = (fp + fm - 2 * fc) / (h * h)
            
            # If H is not positive definite, regularize
            if np.any(H_diag <= 0):
                H_diag = np.where(H_diag <= 0, 1e-6, H_diag)
            
            # Solve H dx = -g => dx = -g / H_diag
            dx = -g / H_diag
            
            # Check if the step improves the model
            # f(c+dx) approx fc + g^T dx + 0.5 dx^T H dx
            # = fc - g^T g / H + 0.5 g^T g / H = fc - 0.5 g^T g / H
            # This is always < fc if g != 0 and H > 0.
            
            new_x = center + dx
            new_x = np.clip(new_x, -5, 5)
            
            # Check if new_x is significantly different
            if np.allclose(new_x, center):
                return None
                
            return new_x

        def do_es_around(x, fx, budget, sigma_init=1.0, sigma_min=1e-4, sigma_max=5.0, lam=8, max_no_improve=20):
            """Run (1+lambda)-ES around a starting point."""
            nonlocal calls_made, best_x, best_f
            
            sigma = sigma_init
            no_improve = 0
            budget_used = 0
            
            while budget_used < budget and calls_made < total_budget:
                if time.time() > deadline:
                    break
                
                # Generate offspring
                offspring = []
                offspring_f = []
                for i in range(lam):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * z
                    fy = evaluate(y)
                    offspring.append(y)
                    offspring_f.append(fy)
                    budget_used += 1
                
                if not offspring:
                    break
                
                best_idx = np.argmin(offspring_f)
                best_y = offspring[best_idx]
                best_yf = offspring_f[best_idx]
                
                if best_yf < fx:
                    x = best_y
                    fx = best_yf
                    no_improve = 0
                    sigma *= 1.1
                else:
                    no_improve += 1
                    sigma *= 0.9
                
                sigma = np.clip(sigma, sigma_min, sigma_max)
                
                if no_improve >= max_no_improve:
                    break
            
            return x, fx

        # Strategy:
        # 1. Explore with ES from a few diverse starts.
        # 2. After each ES run, try a paraboloid jump to refine the basin.
        # 3. Specifically try to find the second Lunacek basin by reflecting the best point.
        
        starts = [
            np.zeros(n),
            rng.uniform(-3, 3, n),
            rng.uniform(-5, 5, n)
        ]
        
        for start in starts:
            if calls_made >= total_budget or time.time() > deadline:
                break
            
            # Budget for this phase: 30% of remaining
            remaining = total_budget - calls_made
            es_budget = int(remaining * 0.3)
            if es_budget < 10:
                continue
            
            x = start
            fx = evaluate(x)
            
            # Explore
            x, fx = do_es_around(x, fx, es_budget, sigma_init=1.5)
            
            # Refine with paraboloid jump
            if calls_made < total_budget and time.time() < deadline:
                h = 0.5
                jump = fit_paraboloid(x, h)
                if jump is not None:
                    fj = evaluate(jump)
                    if fj < fx:
                        x = jump
                        fx = fj
                        # Fine ES around the jump
                        x, fx = do_es_around(x, fx, 10, sigma_init=0.2, sigma_min=1e-5, sigma_max=1.0, lam=4, max_no_improve=10)
                        
                        # Another paraboloid jump
                        if calls_made < total_budget:
                            h = 0.1
                            jump2 = fit_paraboloid(x, h)
                            if jump2 is not None:
                                fj2 = evaluate(jump2)
                                if fj2 < fx:
                                    x = jump2
                                    fx = fj2
                                    do_es_around(x, fx, 5, sigma_init=0.05, sigma_min=1e-6, sigma_max=0.5, lam=3, max_no_improve=5)

        # Lunacek-specific: Try to find the other basin
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            # The two minima of Lunacek are often related by a shift.
            # Try reflecting best_x across the origin.
            reflected = -best_x
            reflected = np.clip(reflected, -5, 5)
            
            fr = evaluate(reflected)
            if fr < best_f + 10: # If it's not terrible, explore it
                x_ref = reflected
                # Explore around the reflected point
                x_ref, fr = do_es_around(x_ref, fr, 20, sigma_init=1.0, sigma_min=1e-4, sigma_max=3.0, lam=6, max_no_improve=15)
                
                # Refine
                if calls_made < total_budget:
                    h = 0.5
                    jump = fit_paraboloid(x_ref, h)
                    if jump is not None:
                        fj = evaluate(jump)
                        if fj < fr:
                            x_ref = jump
                            fr = fj
                            do_es_around(x_ref, fr, 10, sigma_init=0.2, sigma_min=1e-5, sigma_max=1.0, lam=4, max_no_improve=10)
                            
                            if calls_made < total_budget:
                                h = 0.1
                                jump2 = fit_paraboloid(x_ref, h)
                                if jump2 is not None:
                                    fj2 = evaluate(jump2)
                                    if fj2 < fr:
                                        x_ref = jump2
                                        fr = fj2
                                        do_es_around(x_ref, fr, 5, sigma_init=0.05, sigma_min=1e-6, sigma_max=0.5, lam=3, max_no_improve=5)

        # Final polish
        if best_x is not None and calls_made < total_budget and time.time() < deadline:
            h = 0.05
            jump = fit_paraboloid(best_x, h)
            if jump is not None:
                fj = evaluate(jump)
                if fj < best_f:
                    best_x = jump
                    best_f = fj
                    do_es_around(best_x, best_f, 5, sigma_init=0.02, sigma_min=1e-7, sigma_max=0.2, lam=3, max_no_improve=5)

        # Ensure we return valid values
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
