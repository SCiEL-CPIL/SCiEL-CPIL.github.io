"""Hybrid CMA-ES with GP-informed restarts for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with adaptive restarts and GP-informed exploration for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        n_calls = 0
        
        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls
            if n_calls >= budget or check_time():
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(x, -5, 5)
                f = float(func(x_clipped))
                return f, x_clipped
            except Exception:
                return None, None

        def cma_es(x0, sigma0, max_calls, rng):
            """
            Runs CMA-ES.
            Returns (best_x, best_f, calls_used, stagnation_detected).
            """
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            x = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            local_best_x = x.copy()
            local_best_f = np.inf
            calls_used = 0
            stagnation_counter = 0
            no_improve_threshold = 5 * lam
            
            pop = np.zeros((lam, n))
            fvals = np.zeros(lam)
            for i in range(lam):
                if calls_used >= max_calls or check_time():
                    return local_best_x, local_best_f, calls_used, True
                z = rng.standard_normal(n)
                x_trial = x + sigma * z
                f, x_eval = call_func(x_trial)
                if f is None:
                    return local_best_x, local_best_f, calls_used, True
                calls_used += 1
                pop[i] = x_eval
                fvals[i] = f
                if f < local_best_f:
                    local_best_f = f
                    local_best_x = x_eval.copy()
            
            while calls_used < max_calls and not check_time():
                order = np.argsort(fvals)
                pop = pop[order]
                fvals = fvals[order]
                
                x_new = np.sum(weights[:, None] * pop[:mu], axis=0)
                
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-20, None)
                    Cinv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    Cinv_sqrt = np.eye(n)

                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * Cinv_sqrt @ (x_new - x)
                x = x_new

                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls_used // lam)))) < (1.4 + 2.0 / (n + 1)) * chi_n
                pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (x_new - x)

                C = (1.0 + c1 * hsig * (1.0 - hsig) - c1 - cmu) * C + c1 * np.outer(pc, pc)
                for i in range(mu):
                    y_i = (pop[i] - x) / sigma
                    C += cmu * weights[i] * np.outer(y_i, y_i)
                C = (C + C.T) / 2.0

                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-15, 1e6)

                for i in range(lam):
                    if calls_used >= max_calls or check_time():
                        break
                    z = rng.standard_normal(n)
                    x_trial = x + sigma * (Cinv_sqrt @ z)
                    f, x_eval = call_func(x_trial)
                    if f is None:
                        return local_best_x, local_best_f, calls_used, True
                    calls_used += 1
                    pop[i] = x_eval
                    fvals[i] = f
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x_eval.copy()
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1

                if stagnation_counter > no_improve_threshold:
                    return local_best_x, local_best_f, calls_used, True

                if sigma < 1e-10:
                    return local_best_x, local_best_f, calls_used, True

            return local_best_x, local_best_f, calls_used, True

        def gp_informed_restart(history_x, history_f, best_x, best_f, rng):
            """
            Uses a simple GP-like heuristic to pick a restart point.
            We model the function as smooth with length scale L.
            We want to maximize expected improvement.
            EI(x) = (f_best - mu(x)) * Phi(Z) + sigma(x) * phi(Z)
            where Z = (f_best - mu(x)) / sigma(x)
            
            Since we don't have a full GP, we approximate:
            1. mu(x) is interpolated from nearby points (inverse distance weighting).
            2. sigma(x) is larger where data is sparse (distance to nearest neighbor).
            
            We sample candidate points and pick the one with highest EI.
            """
            n = len(history_x)
            if n == 0:
                return rng.uniform(-4, 4, dim)
            
            # Normalize history to [0, 1] for distance calculations
            x_min = np.min(history_x, axis=0)
            x_max = np.max(history_x, axis=0)
            x_range = np.max(x_max - x_min, axis=0)
            x_range[x_range == 0] = 1.0
            
            # Length scale heuristic: roughly 1/3 of the domain range
            length_scale = 0.5 
            
            candidates = []
            # Generate a few random candidates
            for _ in range(20):
                x_cand = rng.uniform(-4, 4, dim)
                candidates.append(x_cand)
            
            # Also consider points far from current best
            for _ in range(5):
                x_cand = best_x + rng.normal(0, 2.0, dim)
                x_cand = np.clip(x_cand, -4, 4)
                candidates.append(x_cand)
                
            best_ei = -np.inf
            best_cand = candidates[0]
            
            # Simple EI approximation
            for x_cand in candidates:
                # Distance to all history points
                dists = np.linalg.norm(history_x - x_cand, axis=1)
                min_dist = np.min(dists)
                
                # Inverse distance weighting for mean
                weights = 1.0 / (dists ** 2 + 1e-6)
                mu = np.sum(weights * history_f) / np.sum(weights)
                
                # Uncertainty increases with distance
                sigma = length_scale * min_dist
                
                # Expected Improvement
                if sigma < 1e-6:
                    ei = max(0, best_f - mu)
                else:
                    z = (best_f - mu) / sigma
                    # Approximate Phi and phi
                    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
                    # Simple approximation for Phi (normal CDF)
                    if z > 0:
                        phi_cdf = 1.0 - 0.5 * np.exp(-z ** 2 / 2) * (1.0 + 1.0/(1.0 + 0.33267 * z) ** 2) # Crude
                        # Better: use erf approximation
                        phi_cdf = 0.5 * (1 + np.tanh(0.7978845608028654 * (z + 0.044715 * z ** 3) / (1 + 0.0378568 * z ** 2)))
                    else:
                        phi_cdf = 0.5 * (1 - np.tanh(0.7978845608028654 * (-z + 0.044715 * (-z) ** 3) / (1 + 0.0378568 * (-z) ** 2)))
                    
                    ei = (best_f - mu) * phi_cdf + sigma * phi
                
                if ei > best_ei:
                    best_ei = ei
                    best_cand = x_cand
            
            return best_cand

        rng = np.random.default_rng(42)
        
        # Phase 1: Initial Sampling
        # Use diverse points to find a good starting location
        initial_candidates = [
            np.zeros(dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-2, 2, dim),
            rng.uniform(-4, 4, dim)
        ]
        
        history_x = []
        history_f = []
        
        for x in initial_candidates:
            f, x_eval = call_func(x)
            if f is not None:
                if f < best_f:
                    best_f = f
                    best_x = x_eval.copy()
                history_x.append(x_eval)
                history_f.append(f)
        
        if best_f == np.inf:
             best_x = np.zeros(dim)
             best_f = 0.0

        # Phase 2: CMA-ES with Adaptive Restarts and GP-Informed Exploration
        current_x = best_x.copy()
        current_sigma = 1.5
        restart_count = 0
        max_restarts = 5
        
        while n_calls < budget and not check_time() and restart_count < max_restarts:
            remaining = budget - n_calls
            if remaining < 10:
                break
            
            if restart_count == max_restarts - 1:
                alloc = remaining
            else:
                alloc = int(remaining * 0.6)
            
            if alloc < 20:
                alloc = remaining
            
            res_x, res_f, used, stagnated = cma_es(current_x, current_sigma, alloc, rng)
            
            # Update history with the best point from this run (approximation)
            # In a real implementation, we'd track all points, but here we just track the best
            # to keep memory and computation low.
            if res_f < best_f:
                best_f = res_f
                best_x = res_x.copy()
                history_x.append(res_x)
                history_f.append(res_f)
            
            if not stagnated or n_calls >= budget or check_time():
                break
                
            # If stagnated, decide next move
            restart_count += 1
            
            # Use GP-informed restart if we have enough history
            if len(history_x) > 5:
                current_x = gp_informed_restart(np.array(history_x), np.array(history_f), best_x, best_f, rng)
                current_sigma = 1.0
            else:
                # Fallback to random or best point with larger sigma
                if restart_count > 2:
                    current_x = rng.uniform(-4, 4, dim)
                    f, x_eval = call_func(current_x)
                    if f is not None:
                        if f < best_f:
                            best_f = f
                            best_x = x_eval.copy()
                        history_x.append(x_eval)
                        history_f.append(f)
                else:
                    current_x = best_x.copy()
                    current_sigma *= 1.5

        # Phase 3: Local Polish
        if n_calls < budget and not check_time():
            local_sigma = 0.1
            for _ in range(budget - n_calls):
                if check_time() or n_calls >= budget:
                    break
                x_trial = best_x + local_sigma * rng.standard_normal(dim)
                f, x_eval = call_func(x_trial)
                if f is not None:
                    if f < best_f:
                        best_f = f
                        best_x = x_eval.copy()
                        local_sigma *= 0.8
                    else:
                        local_sigma *= 0.95
                        if local_sigma < 1e-4:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
