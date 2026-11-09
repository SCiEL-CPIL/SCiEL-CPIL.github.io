"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _halton_point(dim, i):
    """Generate a deterministic point in [0, 1)^dim using the Halton sequence."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    point = np.zeros(dim)
    for d in range(dim):
        base = primes[d % len(primes)]
        val = 0.0
        f = 1.0 / base
        n_val = i + 1
        while n_val > 0:
            val += f * (n_val % base)
            n_val //= base
            f /= base
        point[d] = val
    return point

class Optimizer:
    """
    Ridge-Aligned Nelder-Mead Optimizer for BBOB f13 (Sharp Ridge).
    
    Phase 1: Halton sequence for broad early coverage (maximizes AOCC).
    Phase 2: Nelder-Mead with geometrically stretched simplex aligned to ridge direction.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead_stretched(self, func, x0, direction, budget_remaining, deadline, calls_counter):
        """
        Nelder-Mead with initial simplex stretched along 'direction'.
        Returns (best_x, best_f, calls_used).
        """
        dim = len(x0)
        # Create stretched simplex: base points along axes, one point stretched along direction
        alpha = 0.5  # initial step size
        beta = 3.0   # stretch factor along direction
        
        # Build simplex vertices
        simplex = [x0.copy()]
        for i in range(dim):
            v = x0.copy()
            v[i] += alpha
            simplex.append(v)
        
        # Stretch the last vertex along the ridge direction
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-10:
            dir_unit = direction / dir_norm
            simplex[-1] = x0 + beta * alpha * dir_unit
        
        # Evaluate initial simplex
        fvals = []
        for v in simplex:
            if calls_counter[0] >= budget_remaining or time.time() > deadline:
                break
            try:
                f = func(v)
            except Exception:
                f = np.inf
            calls_counter[0] += 1
            fvals.append(f)
        
        if len(fvals) < dim + 1:
            # Not enough budget for full simplex, return best so far
            best_idx = np.argmin(fvals) if fvals else 0
            best_x = simplex[best_idx].copy() if fvals else x0.copy()
            best_f = min(fvals) if fvals else np.inf
            return best_x, best_f, calls_counter[0]
        
        # Nelder-Mead iterations
        gamma = 2.0  # reflection
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        max_iter = min(200, budget_remaining - calls_counter[0])
        for _ in range(max_iter):
            if calls_counter[0] >= budget_remaining or time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = fvals[-1]
            
            # Reflection
            x_r = centroid + gamma * (centroid - worst)
            if calls_counter[0] >= budget_remaining or time.time() > deadline:
                break
            try:
                f_r = func(x_r)
            except Exception:
                f_r = np.inf
            calls_counter[0] += 1
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + 2.0 * (x_r - centroid)
                if calls_counter[0] >= budget_remaining or time.time() > deadline:
                    break
                try:
                    f_e = func(x_e)
                except Exception:
                    f_e = np.inf
                calls_counter[0] += 1
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < worst_f:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (worst - centroid)
                
                if calls_counter[0] >= budget_remaining or time.time() > deadline:
                    break
                try:
                    f_c = func(x_c)
                except Exception:
                    f_c = np.inf
                calls_counter[0] += 1
                
                if f_c < min(f_r, worst_f):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        if calls_counter[0] >= budget_remaining or time.time() > deadline:
                            break
                        try:
                            fvals[i] = func(simplex[i])
                        except Exception:
                            fvals[i] = np.inf
                        calls_counter[0] += 1
            
            # Check convergence
            if len(set([round(f, 10) for f in fvals])) == 1:
                break
        
        # Find best in final simplex
        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]
        return best_x, best_f, calls_counter[0]

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Global Exploration with Halton Sequence
        n_explore = 40
        n_explore = min(n_explore, budget - 1)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        candidates = []  # List of (f, x)
        
        for i in range(n_explore):
            if time.time() > deadline or calls_made >= budget:
                break
            
            p_unit = _halton_point(dim, i)
            p = p_unit * 10.0 - 5.0
            p = np.clip(p, -5.0, 5.0)
            
            try:
                f = func(p)
            except Exception:
                f = np.inf
            calls_made += 1
            
            if f < best_f:
                best_f = f
                best_x = p.copy()
            
            candidates.append((f, p))
        
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except Exception:
                best_f = np.inf
            calls_made += 1
            candidates.append((best_f, best_x))
        
        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Estimate ridge direction from top candidates
        # Use the two best distinct points to estimate direction
        direction = np.zeros(dim)
        if len(candidates) >= 2:
            x1 = candidates[0][1]
            x2 = candidates[1][1]
            direction = x2 - x1
            if np.linalg.norm(direction) < 1e-8:
                # Fallback: use difference from origin
                direction = x1 - np.zeros(dim)
        
        # Phase 2: Ridge-Aligned Nelder-Mead Refinement
        max_local_starts = 3
        starts_to_try = []
        
        for f_init, x_init in candidates:
            if len(starts_to_try) >= max_local_starts:
                break
            is_duplicate = False
            for _, x_sel in starts_to_try:
                if np.linalg.norm(x_init - x_sel) < 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                starts_to_try.append((f_init, x_init))
        
        budget_remaining = budget - calls_made
        calls_counter = [0]  # Mutable counter for NM
        
        for f_init, x_init in starts_to_try:
            if time.time() > deadline or calls_counter[0] >= budget_remaining:
                break
            
            best_x_nm, best_f_nm, _ = self._nelder_mead_stretched(
                func, x_init, direction, budget_remaining, deadline, calls_counter
            )
            
            if best_f_nm < best_f:
                best_f = best_f_nm
                best_x = best_x_nm.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
