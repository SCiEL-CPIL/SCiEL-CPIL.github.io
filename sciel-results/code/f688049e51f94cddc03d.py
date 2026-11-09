# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer for BBOB-style f13 (Sharp Ridge).
    
    Combines:
    1. Nelder-Mead simplex for global-ish convergence and robustness.
    2. Coordinate-descent line search (Powell-like) for exploiting ridge structure.
       This is crucial for f13 where the ridge is sharp in one direction and flat in others.
    3. Adaptive step sizes and stall detection with random restarts/perturbations.
    4. Strict time and budget guards.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, aim for 285s to be safe
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        def timed_call(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= self.budget:
                return None
            if time.time() > deadline - 2:
                return None
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initialization
        # Use a small set of random points to find a good starting basin.
        # 15 points is enough to get a reasonable start without wasting budget.
        n_init = 15
        for _ in range(n_init):
            if calls_made >= self.budget:
                break
            x = rng.uniform(-5, 5, self.dim)
            timed_call(x)
            
        if best_x is None:
            # Fallback: center point
            x = np.zeros(self.dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        x = best_x.copy()
        fx = best_f

        # Phase 2: Main Loop - Alternating Nelder-Mead and Coordinate Descent
        # We will alternate between these two methods to balance exploration and exploitation.
        # NM is good for general convergence, CD is good for ridges.
        
        # NM State
        n = self.dim
        simplex = np.zeros((n + 1, n))
        simplex[0] = x
        for i in range(n):
            simplex[i + 1] = x.copy()
            # Initial simplex size
            step_init = 0.5
            simplex[i + 1][i] += step_init if x[i] < 4.5 else -step_init
            simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)
        
        fvals = np.zeros(n + 1)
        for i in range(n + 1):
            if calls_made >= self.budget or time.time() > deadline:
                break
            fvals[i] = timed_call(simplex[i])
        
        # CD State
        steps = np.ones(self.dim) * 1.0
        no_improvement_count = 0
        max_no_improvement = 30 # Stall threshold

        # NM Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        use_nm = True
        
        while calls_made < self.budget and time.time() < deadline - 2:
            
            if use_nm:
                # --- Nelder-Mead Step ---
                if no_improvement_count > max_no_improvement:
                    # Stall detected in NM, switch to CD or restart
                    use_nm = False
                    continue

                # Sort simplex
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                if calls_made >= self.budget or time.time() > deadline:
                    break
                fr = timed_call(xr)
                if fr is None:
                    break

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    fe = timed_call(xe)
                    if fe is None:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    no_improvement_count = 0
                elif fr < fvals[-2]:
                    # Reflection accepted
                    simplex[-1] = xr
                    fvals[-1] = fr
                    no_improvement_count = 0
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    fc = timed_call(xc)
                    if fc is None:
                        break

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                        no_improvement_count = 0
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls_made >= self.budget or time.time() > deadline:
                                break
                            fvals[i] = timed_call(simplex[i])
                            if fvals[i] is None:
                                break
                        no_improvement_count += 1

                # Update best
                idx = np.argmin(fvals)
                if fvals[idx] < best_f:
                    best_f = fvals[idx]
                    best_x = simplex[idx].copy()
                    x = best_x.copy()
                    fx = best_f
                else:
                    # Even if best didn't improve, check if simplex moved
                    # Update x to the best point in simplex for CD start
                    x = simplex[idx].copy()
                    fx = fvals[idx]

                # Switch to CD periodically to exploit ridge structure
                # Every 10 NM iterations or so
                if no_improvement_count % 10 == 0 and no_improvement_count > 0:
                    use_nm = False
                else:
                    use_nm = True

            else:
                # --- Coordinate Descent Step ---
                if no_improvement_count > max_no_improvement:
                    # Stall detected in CD, switch to NM or restart
                    # Reset simplex around current best
                    simplex[0] = best_x
                    for i in range(n):
                        simplex[i + 1] = best_x.copy()
                        simplex[i + 1][i] += 0.5 if best_x[i] < 4.5 else -0.5
                        simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)
                    for i in range(n + 1):
                        if calls_made >= self.budget or time.time() > deadline:
                            break
                        fvals[i] = timed_call(simplex[i])
                        if fvals[i] is None:
                            break
                    no_improvement_count = 0
                    use_nm = True
                    continue

                improved_in_cycle = False
                
                for d in range(self.dim):
                    if calls_made >= self.budget:
                        break
                    if time.time() > deadline - 2:
                        break
                    
                    # Perform a 1D search along axis d
                    # We will try to find a better point along this axis using a few evaluations.
                    # Simple strategy: try x + step, x - step.
                    # If one is better, move there and expand step.
                    # If neither is better, shrink step.
                    
                    x_plus = x.copy()
                    x_plus[d] += steps[d]
                    if x_plus[d] > 5: x_plus[d] = 5
                    if x_plus[d] < -5: x_plus[d] = -5
                    
                    x_minus = x.copy()
                    x_minus[d] -= steps[d]
                    if x_minus[d] > 5: x_minus[d] = 5
                    if x_minus[d] < -5: x_minus[d] = -5
                    
                    # Evaluate both (if budget allows)
                    fx_plus = timed_call(x_plus)
                    if fx_plus is None:
                        break
                    
                    fx_minus = timed_call(x_minus)
                    if fx_minus is None:
                        break
                    
                    # Compare
                    if fx_plus < fx and fx_plus < fx_minus:
                        x = x_plus
                        fx = fx_plus
                        steps[d] *= 1.5 # Expand
                        improved_in_cycle = True
                    elif fx_minus < fx and fx_minus < fx_plus:
                        x = x_minus
                        fx = fx_minus
                        steps[d] *= 1.5 # Expand
                        improved_in_cycle = True
                    else:
                        # Neither improved
                        steps[d] *= 0.5 # Shrink
                        if steps[d] < 1e-4:
                            steps[d] = 1e-4 # Prevent underflow

                if improved_in_cycle:
                    no_improvement_count = 0
                    # Update global best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    use_nm = True # Switch back to NM after a successful CD cycle
                else:
                    no_improvement_count += 1
                    if no_improvement_count > max_no_improvement:
                        # Stagnation: try a random perturbation
                        # Pick a random axis and make a random jump
                        d = rng.integers(self.dim)
                        jump = rng.uniform(-2, 2)
                        x[d] += jump
                        if x[d] > 5: x[d] = 5
                        if x[d] < -5: x[d] = -5
                        fx = timed_call(x)
                        if fx is None:
                            break
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        # Reset steps for the jumped axis
                        steps[d] = 1.0
                        no_improvement_count = 0
                        use_nm = True

        # Phase 3: Final Polish with Powell's Method (if budget remains)
        # Powell's method is particularly effective for ridge-like functions
        # as it iteratively searches along conjugate directions.
        
        if best_x is not None and calls_made < self.budget - 10 and time.time() < deadline - 5:
            # Implement a simple version of Powell's method
            # Start with coordinate axes
            directions = np.eye(self.dim)
            x_powell = best_x.copy()
            f_powell = best_f
            
            max_powell_iter = 5
            for it in range(max_powell_iter):
                if calls_made >= self.budget or time.time() > deadline - 2:
                    break
                
                f_old = f_powell
                x_old = x_powell.copy()
                best_dir_idx = 0
                best_dir_gain = 0
                
                for d in range(self.dim):
                    if calls_made >= self.budget or time.time() > deadline - 2:
                        break
                    
                    dir_vec = directions[d]
                    
                    # Find min along direction
                    # Use golden section search or simple ternary
                    # For simplicity, use a few probes
                    # First, find a bracket
                    step = 0.1
                    x_try = x_powell.copy()
                    x_try += dir_vec * step
                    x_try = np.clip(x_try, -5, 5)
                    f_try = timed_call(x_try)
                    if f_try is None:
                        break
                    
                    if f_try < f_powell:
                        # Move in this direction
                        # Simple line search: try increasing steps
                        best_x_dir = x_try
                        best_f_dir = f_try
                        for _ in range(5):
                            if calls_made >= self.budget or time.time() > deadline - 2:
                                break
                            x_try2 = best_x_dir.copy()
                            x_try2 += dir_vec * step
                            x_try2 = np.clip(x_try2, -5, 5)
                            f_try2 = timed_call(x_try2)
                            if f_try2 is None:
                                break
                            if f_try2 < best_f_dir:
                                best_x_dir = x_try2
                                best_f_dir = f_try2
                                step *= 1.5
                            else:
                                break
                        
                        x_powell = best_x_dir
                        f_powell = best_f_dir
                        gain = f_old - f_powell
                        if gain > best_dir_gain:
                            best_dir_gain = gain
                            best_dir_idx = d
                    else:
                        # Try opposite direction
                        x_try = x_powell.copy()
                        x_try -= dir_vec * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = timed_call(x_try)
                        if f_try is None:
                            break
                        
                        if f_try < f_powell:
                            best_x_dir = x_try
                            best_f_dir = f_try
                            for _ in range(5):
                                if calls_made >= self.budget or time.time() > deadline - 2:
                                    break
                                x_try2 = best_x_dir.copy()
                                x_try2 -= dir_vec * step
                                x_try2 = np.clip(x_try2, -5, 5)
                                f_try2 = timed_call(x_try2)
                                if f_try2 is None:
                                    break
                                if f_try2 < best_f_dir:
                                    best_x_dir = x_try2
                                    best_f_dir = f_try2
                                    step *= 1.5
                                else:
                                    break
                            
                            x_powell = best_x_dir
                            f_powell = best_f_dir
                            gain = f_old - f_powell
                            if gain > best_dir_gain:
                                best_dir_gain = gain
                                best_dir_idx = d

                # Update global best
                if f_powell < best_f:
                    best_f = f_powell
                    best_x = x_powell.copy()
                
                # Update directions: replace worst direction with new conjugate direction
                if best_dir_gain > 1e-10:
                    # The new direction is from x_old to x_powell
                    new_dir = x_powell - x_old
                    norm_new_dir = np.linalg.norm(new_dir)
                    if norm_new_dir > 1e-10:
                        new_dir /= norm_new_dir
                        directions[best_dir_idx] = new_dir

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1
            
        return best_x, best_f

# EVOLVE-BLOCK-END
