"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES with coordinate descent, golden-section polish, and deadline guard for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 30s safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Diversified initialization
        n_init = min(10, budget)
        candidates = []
        # Origin
        candidates.append(np.zeros(dim))
        # Corners (sample a few)
        for i in range(min(4, dim)):
            c = np.zeros(dim)
            c[i] = rng.choice([-4.0, 4.0])
            candidates.append(c)
        # Random
        while len(candidates) < n_init:
            candidates.append(rng.uniform(-4.5, 4.5, dim))

        best_x = None
        best_f = np.inf
        for c in candidates:
            c = np.clip(c, -5, 5)
            fc = func(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()
            if time.time() > deadline:
                break

        calls_made = n_init
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: (1+1)-ES with adaptive sigma + periodic coordinate descent
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        success_rate = 0.3
        coord_descent_interval = 40
        evals_in_phase = 0
        stagnation_count = 0
        best_f_so_far = fx

        while calls_made < budget and time.time() < deadline:
            # Check for deadline
            if time.time() > deadline:
                break

            # Coordinate descent every N evals
            if evals_in_phase % coord_descent_interval == 0 and evals_in_phase > 0:
                # Coordinate descent sweep
                improved_in_cd = False
                for d in range(dim):
                    for direction in [-1, 1]:
                        trial = x.copy()
                        trial[d] = x[d] + direction * sigma
                        trial[d] = np.clip(trial[d], -5, 5)
                        if time.time() > deadline or calls_made >= budget:
                            break
                        ft = func(trial)
                        calls_made += 1
                        evals_in_phase += 1
                        if ft < fx:
                            x = trial
                            fx = ft
                            improved_in_cd = True
                        else:
                            # Try smaller step
                            trial2 = x.copy()
                            trial2[d] = x[d] + direction * sigma * 0.5
                            trial2[d] = np.clip(trial2[d], -5, 5)
                            if time.time() > deadline or calls_made >= budget:
                                break
                            ft2 = func(trial2)
                            calls_made += 1
                            evals_in_phase += 1
                            if ft2 < fx:
                                x = trial2
                                fx = ft2
                                improved_in_cd = True
                    if time.time() > deadline or calls_made >= budget:
                        break
                if improved_in_cd:
                    stagnation_count = 0
                    best_f_so_far = fx
                else:
                    stagnation_count += 1
                # Adapt sigma based on coordinate descent success
                sigma *= 0.9
                continue

            # (1+1)-ES step
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1
            evals_in_phase += 1

            if fy < fx:
                success_rate = 0.7 * success_rate + 0.3 * 1.0
                x = y
                fx = fy
                stagnation_count = 0
                best_f_so_far = fx
                # Increase sigma if success rate is high
                if success_rate > 0.5:
                    sigma *= 1.2
                else:
                    sigma *= 1.05
            else:
                success_rate = 0.7 * success_rate + 0.3 * 0.0
                # Decrease sigma if success rate is low
                if success_rate < 0.3:
                    sigma *= 0.85
                else:
                    sigma *= 0.95

            # Prevent sigma from getting too small or too large
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Check if stuck — try a larger jump
            if evals_in_phase % 100 == 0 and success_rate < 0.1:
                # Try a random restart from current best
                jump = x + sigma * 3.0 * rng.standard_normal(dim)
                jump = np.clip(jump, -5, 5)
                if time.time() > deadline or calls_made >= budget:
                    break
                fj = func(jump)
                calls_made += 1
                evals_in_phase += 1
                if fj < fx:
                    x = jump
                    fx = fj
                    success_rate = 0.5
                    stagnation_count = 0
                    best_f_so_far = fx

        # Phase 3: Final high-precision polish using golden-section search on each coordinate
        # Only if we have remaining budget and time
        if calls_made < budget and time.time() < deadline:
            # Determine how many polish iterations we can afford
            # Each coordinate polish with golden section takes ~15 evals per dimension
            # We'll do a few passes
            polish_passes = 0
            max_polish_passes = 3
            
            while polish_passes < max_polish_passes and calls_made < budget and time.time() < deadline:
                polish_passes += 1
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    # Golden-section search on coordinate d
                    # First, find an interval [a, b] that brackets the minimum
                    # Start with current x[d] and expand
                    step = max(sigma * 0.1, 0.1)
                    
                    # Try to find a better point in direction +step
                    trial_p = x.copy()
                    trial_p[d] = np.clip(x[d] + step, -5, 5)
                    fp = func(trial_p)
                    calls_made += 1
                    
                    # Try to find a better point in direction -step
                    trial_m = x.copy()
                    trial_m[d] = np.clip(x[d] - step, -5, 5)
                    fm = func(trial_m)
                    calls_made += 1
                    
                    # Determine bracketing interval
                    if fp < fx and fm < fx:
                        # Both sides are better, expand
                        a = x[d] - step
                        b = x[d] + step
                    elif fp < fx:
                        a = x[d]
                        b = x[d] + step
                    elif fm < fx:
                        a = x[d] - step
                        b = x[d]
                    else:
                        # No improvement, skip this coordinate
                        continue
                    
                    # Golden section search within [a, b]
                    # Ensure a < b
                    if a > b:
                        a, b = b, a
                    
                    # Don't go outside bounds
                    a = max(a, -5.0)
                    b = min(b, 5.0)
                    
                    if b - a < 1e-10:
                        continue
                    
                    gr = (np.sqrt(5) - 1) / 2  # golden ratio ~0.618
                    c = b - gr * (b - a)
                    d_val = a + gr * (b - a)
                    
                    # Evaluate at c and d
                    if calls_made >= budget or time.time() > deadline:
                        break
                    xc = x.copy()
                    xc[d] = c
                    fc = func(xc)
                    calls_made += 1
                    
                    if calls_made >= budget or time.time() > deadline:
                        break
                    xd = x.copy()
                    xd[d] = d_val
                    fd = func(xd)
                    calls_made += 1
                    
                    # Golden section iterations
                    for _ in range(12):
                        if calls_made >= budget or time.time() > deadline:
                            break
                        if fc < fd:
                            b = d_val
                            d_val = c
                            fd = fc
                            c = b - gr * (b - a)
                            xc = x.copy()
                            xc[d] = c
                            fc = func(xc)
                            calls_made += 1
                        else:
                            a = c
                            c = d_val
                            fc = fd
                            d_val = a + gr * (b - a)
                            xd = x.copy()
                            xd[d] = d_val
                            fd = func(xd)
                            calls_made += 1
                    
                    # Update x[d] to the best found
                    mid = (a + b) / 2
                    if calls_made < budget and time.time() < deadline:
                        xmid = x.copy()
                        xmid[d] = mid
                        fmid = func(xmid)
                        calls_made += 1
                        if fmid < fx:
                            x[d] = mid
                            fx = fmid
                        elif fc < fd:
                            x[d] = c
                            fx = fc
                        else:
                            x[d] = d_val
                            fx = fd

        return x, fx
# EVOLVE-BLOCK-END
