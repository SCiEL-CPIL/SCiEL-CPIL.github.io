import random
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 108.0  # Conservative safety margin before 120s limit

    def score_raw(h):
        """Compute R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) for non-negative h."""
        h = np.asarray(h, dtype=float)
        if len(h) == 0:
            return 0.0
        h1 = np.sum(h)
        if h1 <= 0:
            return 0.0
        h2 = np.sum(h * h)
        c = np.convolve(h, h)
        cmax = np.max(c)
        if cmax <= 0:
            return 0.0
        return h2 / (h1 * cmax)

    def soft_max(c, tau=1.0):
        """Smooth approximation of max(c) using log-sum-exp."""
        cmax = np.max(c)
        if cmax <= 0:
            return 0.0
        # Stable log-sum-exp
        shifted = c / tau
        shifted = shifted - np.max(shifted)
        lse = np.max(c) + tau * np.log(np.sum(np.exp(shifted)))
        return lse

    def optimize_smooth(h_init, max_iter=1000, ftol=1e-10, tau=0.01):
        """
        Optimize h using smooth epigraph with log-sum-exp surrogate.
        Variables: x = [h_0, ..., h_{N-1}]
        Minimize soft_max(conv(h,h)) subject to sum(h)=1, h >= 0
        """
        def obj(x):
            h = np.clip(x, 1e-12, None)
            s = np.sum(h)
            if s <= 0:
                return 1e6
            h = h / s
            c = np.convolve(h, h)
            return soft_max(c, tau)

        def obj_grad(x):
            h = np.clip(x, 1e-12, None)
            s = np.sum(h)
            if s <= 0:
                return np.zeros(N)
            h = h / s
            c = np.convolve(h, h)
            cmax = np.max(c)
            if cmax <= 0:
                return np.zeros(N)
            
            # Gradient of soft_max w.r.t. c
            shifted = c / tau
            shifted = shifted - np.max(shifted)
            weights = np.exp(shifted) / np.sum(np.exp(shifted))
            
            # Vectorized gradient: grad_c[k,i] = 2*h[k-i] if 0<=k-i<N else 0
            # grad_h[i] = sum_k weights[k] * 2*h[k-i] (where valid)
            # This is a correlation: grad_h = 2 * (weights * h) convolved appropriately
            # c = convolve(h, h), so dc_k/dh_i = h_{k-i} + h_{k-i} = 2*h_{k-i} when valid
            # grad_h[i] = sum_k weights[k] * 2*h[k-i]
            # = 2 * sum_k weights[k] * h[k-i]
            # This is 2 * (reverse(weights) * h) at appropriate indices
            # Actually: let's compute directly using the correlation structure
            # grad_h[i] = 2 * sum_{k} weights[k] * h[k-i] for k-i in [0,N)
            # = 2 * (flip(weights) correlated with h)
            # More precisely: grad_h[i] = 2 * sum_{j=0}^{N-1} weights[i+j] * h[j]
            # where we sum over valid j
            # This is like a cross-correlation
            grad_h = np.zeros(N)
            for i in range(N):
                # j ranges such that k = i+j is in [0, 2N-2]
                j_min = 0
                j_max = min(N - 1, 2 * N - 2 - i)
                if j_max >= j_min:
                    grad_h[i] = 2.0 * np.dot(weights[i + np.arange(j_min, j_max + 1)], 
                                              h[j_min:j_max + 1])
            
            # Chain rule: grad_x = grad_h / s - (obj/s) * ones
            obj_val = soft_max(c, tau)
            grad_x = grad_h / s - (obj_val / s) * np.ones(N)
            
            return grad_x

        h_init = np.asarray(h_init, dtype=float)
        s = np.sum(h_init)
        if s > 0:
            h_init = h_init / s
        else:
            h_init = np.ones(N) / N
        h_init = np.clip(h_init, 1e-12, None)
        s = np.sum(h_init)
        h_init = h_init / s

        bounds = [(1e-12, None)] * N

        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
             'jac': lambda x: np.ones(N)}
        ]

        try:
            res = minimize(
                obj,
                h_init,
                method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': ftol}
            )

            if res.success or res.fun < 1e6:
                h_res = np.clip(res.x, 1e-12, None)
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    return None, 0.0

                ratio_res = score_raw(h_res)
                return h_res, ratio_res
        except Exception:
            pass
        return None, 0.0

    def optimize_epigraph(h_init, max_iter=500, ftol=1e-8):
        """
        Optimize h using SLSQP with epigraph formulation.
        Variables: x = [h_0, ..., h_{N-1}, M]
        Minimize M subject to sum(h)=1, M >= c_k for all k, h >= 0
        """
        def obj(x):
            return x[N]

        def obj_grad(x):
            grad = np.zeros(N + 1)
            grad[N] = 1.0
            return grad

        def constraint_sum(x):
            return np.sum(x[:N]) - 1.0

        def constraint_sum_grad(x):
            grad = np.zeros(N + 1)
            grad[:N] = 1.0
            return grad

        def constraints_autoconv(x):
            h = x[:N]
            M = x[N]
            c = np.convolve(h, h)
            return M - c

        def constraints_autoconv_grad(x):
            h = x[:N]
            grad = np.zeros((2 * N - 1, N + 1))
            for k in range(2 * N - 1):
                derivs = np.zeros(N)
                for i in range(N):
                    j = k - i
                    if 0 <= j < N:
                        derivs[i] = 2.0 * h[j]
                grad[k, :N] = -derivs
                grad[k, N] = 1.0
            return grad

        h_init = np.asarray(h_init, dtype=float)
        c_init = np.convolve(h_init, h_init)
        M_init = np.max(c_init)
        x0 = np.concatenate([h_init, [M_init]])

        bounds = [(0.0, None)] * N + [(0.0, None)]

        cons = [
            {'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad},
            {'type': 'ineq', 'fun': constraints_autoconv, 'jac': constraints_autoconv_grad}
        ]

        try:
            res = minimize(
                obj,
                x0,
                method='SLSQP',
                jac=obj_grad,
                bounds=bounds,
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': ftol}
            )

            if res.success or res.fun < np.max(c_init):
                h_res = res.x[:N]
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    return None, 0.0
                h_res = np.clip(h_res, 0, None)
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                else:
                    return None, 0.0

                ratio_res = score_raw(h_res)
                return h_res, ratio_res
        except Exception:
            pass
        return None, 0.0

    def find_sidon_support(target_size):
        """
        Greedy construction of a near-Sidon set to minimize pairwise sum collisions.
        Returns a list of indices.
        """
        if target_size <= 0:
            return []

        support = [0]
        used_sums = {0: 1}
        support_set = set(support)

        for _ in range(target_size - 1):
            best_candidate = None
            best_collisions = float('inf')

            for candidate in range(N):
                if candidate in support_set:
                    continue

                # Count new collisions
                new_collisions = 0
                for s in support:
                    pair_sum = candidate + s
                    if pair_sum in used_sums:
                        new_collisions += 1

                if new_collisions < best_collisions:
                    best_collisions = new_collisions
                    best_candidate = candidate

            if best_candidate is not None:
                support.append(best_candidate)
                support_set.add(best_candidate)
                for s in support:
                    pair_sum = best_candidate + s
                    used_sums[pair_sum] = used_sums.get(pair_sum, 0) + 1
            else:
                break

        return support

    def find_sidon_support_randomized(target_size, num_trials=50):
        """
        Find the best Sidon-like support among multiple random trials.
        """
        best_support = []
        best_collisions = float('inf')

        for _ in range(num_trials):
            if time.time() > deadline - 10:
                break

            support = [0]
            used_sums = {0: 1}
            collisions = 0

            # Randomize the order of candidates
            candidates = list(range(1, N))
            random.shuffle(candidates)

            for candidate in candidates:
                if len(support) >= target_size:
                    break

                # Count new collisions
                new_collisions = 0
                for s in support:
                    pair_sum = candidate + s
                    if pair_sum in used_sums:
                        new_collisions += 1

                # Heuristic: accept if collisions are low
                # We prioritize adding elements with fewer collisions
                if new_collisions < 2:  # Threshold can be tuned
                    support.append(candidate)
                    for s in support:
                        pair_sum = candidate + s
                        used_sums[pair_sum] = used_sums.get(pair_sum, 0) + 1
                        if used_sums[pair_sum] > 1:
                            collisions += 1

            if len(support) >= 3:
                if collisions < best_collisions:
                    best_collisions = collisions
                    best_support = support[:]

        if not best_support:
            # Fallback to deterministic greedy
            best_support = find_sidon_support(target_size)

        return best_support

    def find_best_support_by_score(target_size, num_trials=100):
        """
        Find the best support by directly evaluating the ratio for uniform weights.
        """
        best_support = []
        best_ratio = 0.0

        for _ in range(num_trials):
            if time.time() > deadline - 10:
                break

            # Random subset
            support = sorted(random.sample(range(N), target_size))
            h = np.zeros(N)
            h[support] = 1.0 / target_size
            ratio = score_raw(h)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_support = support[:]

        # Also try greedy Sidon
        support = find_sidon_support(target_size)
        if len(support) >= 3:
            h = np.zeros(N)
            h[support] = 1.0 / len(support)
            ratio = score_raw(h)
            if ratio > best_ratio:
                best_ratio = ratio
                best_support = support[:]

        return best_support, best_ratio

    def block_coordinate_refine(h, max_iters=200):
        """
        Block-coordinate ascent: optimize small subsets of coordinates while
        fixing others. Helps escape local optima in the nonsmooth landscape.
        """
        h = np.array(h, dtype=float)
        s = np.sum(h)
        if s <= 0:
            return h
        
        for iteration in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try optimizing pairs of coordinates
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, N):
                    if time.time() > deadline - 3:
                        break
                    
                    h_old = h[i] + h[j]
                    if h_old <= 0:
                        continue
                    
                    # Optimize h[i] and h[j] with fixed sum
                    # Maximize h2 / (h1 * cmax) locally
                    # Fix other coordinates, vary h[i] = t, h[j] = h_old - t
                    best_t = h[i]
                    best_local_score = score_raw(h)
                    
                    # Try a few values of t
                    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        t = frac * h_old
                        h_test = h.copy()
                        h_test[i] = t
                        h_test[j] = h_old - t
                        s_test = np.sum(h_test)
                        if s_test <= 0:
                            continue
                        h_test = h_test / s_test
                        local_score = score_raw(h_test)
                        if local_score > best_local_score:
                            best_local_score = local_score
                            best_t = t
                    
                    if best_t != h[i]:
                        h[i] = best_t
                        h[j] = h_old - best_t
                        improved = True
            
            if not improved:
                break
        
        return h

    best_h = None
    best_score = 0.0

    # Generate candidate initializations and sort by initial score
    candidates = []
    
    # 1. Uniform distribution
    h_uniform = np.ones(N) / N
    candidates.append(('uniform', h_uniform))
    
    # 2. Sidon-like structured supports
    for target_k in [4, 5, 6, 7, 8, 9, 10, 12]:
        if time.time() > deadline - 40:
            break
        support = find_sidon_support(target_k)
        if len(support) >= 3:
            h_sidon = np.zeros(N)
            h_sidon[support] = 1.0 / len(support)
            candidates.append((f'sidon_{target_k}', h_sidon))
    
    # 3. Geometric progressions
    for base in [2, 3]:
        support = []
        val = 0
        while val < N:
            support.append(val)
            val = val * base + 1
        if len(support) > 2:
            h_geom = np.zeros(N)
            h_geom[support] = 1.0 / len(support)
            candidates.append((f'geom_{base}', h_geom))
    
    # 4. Random sparse starts
    num_random = 15
    for i in range(num_random):
        if time.time() > deadline - 35:
            break
        K = random.randint(3, 12)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        candidates.append((f'rand_{i}', h_rand))
    
    # 5. Alternating patterns
    for period in [2, 3, 4, 5]:
        h_alt = np.zeros(N)
        for i in range(N):
            if i % period == 0:
                h_alt[i] = 1.0
        s_alt = np.sum(h_alt)
        if s_alt > 0:
            h_alt /= s_alt
            candidates.append((f'alt_{period}', h_alt))
    
    # Sort candidates by initial score (best first)
    scored_candidates = []
    for name, h in candidates:
        s = score_raw(h)
        scored_candidates.append((s, name, h))
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Optimize top candidates
    max_optimizations = 20
    opt_count = 0
    for init_score, name, h_init in scored_candidates:
        if time.time() > deadline - 30:
            break
        if opt_count >= max_optimizations:
            break
        
        # Try epigraph optimization
        h_opt, s_opt = optimize_epigraph(h_init, max_iter=500)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        opt_count += 1
        
        # Try smooth optimization for top candidates
        if init_score > 0.5 and time.time() > deadline - 40:
            continue
        if init_score > 0.5:
            h_opt, s_opt = optimize_smooth(h_init, max_iter=500, tau=0.05)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
            opt_count += 1

    # Additional random restarts if time allows
    while time.time() < deadline - 25:
        K = random.randint(2, 15)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K

        h_opt, s_opt = optimize_epigraph(h_rand, max_iter=300)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()
        
        if random.random() < 0.4:
            h_opt, s_opt = optimize_smooth(h_rand, max_iter=300, tau=0.05)
            if h_opt is not None and s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()

    # Block-coordinate refinement on best solution
    if best_h is not None and time.time() < deadline - 8:
        h_bc = block_coordinate_refine(best_h, max_iters=100)
        s_bc = score_raw(h_bc)
        if s_bc > best_score:
            best_score = s_bc
            best_h = h_bc.copy()

    # Final high-precision polish pass
    if best_h is not None and time.time() < deadline - 5:
        h_polished, s_polished = optimize_epigraph(best_h, max_iter=2000, ftol=1e-12)
        if h_polished is not None and s_polished > best_score:
            best_score = s_polished
            best_h = h_polished.copy()
    
    # High-precision smooth polish
    if best_h is not None and time.time() < deadline - 3:
        h_smooth_polished, s_smooth = optimize_smooth(best_h, max_iter=2000, ftol=1e-14, tau=0.001)
        if h_smooth_polished is not None and s_smooth > best_score:
            best_score = s_smooth
            best_h = h_smooth_polished.copy()
        
        # One more epigraph polish after smooth
        if time.time() < deadline - 1:
            h_final, s_final = optimize_epigraph(best_h, max_iter=1000, ftol=1e-14)
            if h_final is not None and s_final > best_score:
                best_score = s_final
                best_h = h_final.copy()
    
    # Final block-coordinate refinement
    if best_h is not None and time.time() < deadline - 0.5:
        h_bc_final = block_coordinate_refine(best_h, max_iters=50)
        s_bc_final = score_raw(h_bc_final)
        if s_bc_final > best_score:
            best_score = s_bc_final
            best_h = h_bc_final.copy()

    # Fallback
    if best_h is None:
        best_h = np.ones(N) / N
        best_score = score_raw(best_h)

    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
# EVOLVE-BLOCK-END
