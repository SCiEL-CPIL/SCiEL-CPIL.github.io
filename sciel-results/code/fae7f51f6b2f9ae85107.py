import random
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

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

    def find_best_sidon_set(max_time):
        """
        Finds a large subset of {0, ..., N-1} that is as close to a Sidon set as possible.
        A Sidon set has all pairwise sums i+j distinct.
        Returns the best set found and its size.
        """
        best_set = []
        best_size = 0
        best_collisions = float('inf')
        
        # We want to maximize size, but also minimize collisions.
        # Score = size * 1000 - collisions (heuristic)
        best_score = -1
        
        # Number of trials depends on time available
        # Each trial is very fast (greedy O(N^2))
        num_trials = 10000 
        current_time = time.time()
        
        for _ in range(num_trials):
            if time.time() - current_time > max_time:
                break
            
            # Randomized Greedy Construction
            support = []
            used_sums = set()
            
            # Start with a random element or 0
            first = random.randint(0, N - 1)
            support.append(first)
            used_sums.add(2 * first)
            
            # Candidates
            candidates = [i for i in range(N) if i != first]
            random.shuffle(candidates)
            
            for cand in candidates:
                if time.time() - current_time > max_time:
                    break
                    
                # Check collisions
                new_collisions = 0
                new_sums_to_add = []
                
                for s in support:
                    pair_sum = cand + s
                    # Note: pair_sum == s + cand is the same
                    # We need to check if this sum already exists in used_sums
                    # AND if it's a "new" sum for this candidate
                    # Actually, for Sidon property, we just need all sums from the *entire* set to be unique.
                    # So we check if cand+s is in used_sums.
                    if pair_sum in used_sums:
                        new_collisions += 1
                    else:
                        new_sums_to_add.append(pair_sum)
                
                # Heuristic: Accept if no collisions, or if collisions are very low
                # To find large sets, we might need to tolerate some collisions initially
                # but we prefer 0.
                if new_collisions == 0:
                    support.append(cand)
                    for ns in new_sums_to_add:
                        used_sums.add(ns)
                elif new_collisions <= 1 and len(support) < 6:
                    # Allow slight imperfection to grow the set, but penalize
                    # We'll track total collisions for the whole set later
                    support.append(cand)
                    for ns in new_sums_to_add:
                        used_sums.add(ns)
            
            # Evaluate the final set
            if len(support) > best_size or (len(support) == best_size):
                # Count total collisions in the final set
                total_collisions = 0
                sums_count = {}
                for i in range(len(support)):
                    for j in range(i, len(support)):
                        s_val = support[i] + support[j]
                        sums_count[s_val] = sums_count.get(s_val, 0) + 1
                for count in sums_count.values():
                    if count > 1:
                        total_collisions += count - 1
                        
                # Score: prioritize size, then minimize collisions
                current_score = len(support) * 1000 - total_collisions
                if current_score > best_score:
                    best_score = current_score
                    best_size = len(support)
                    best_collisions = total_collisions
                    best_set = support[:]

        return best_set, best_size

    best_h = None
    best_score_val = 0.0

    # Phase 1: Combinatorial Search for Best Sidon-like Support
    # Allocate 60% of time to this
    search_time_budget = 70.0
    start_search = time.time()
    
    # We want to find sets of size 5, 6, 7, etc.
    # The find_best_sidon_set function above runs multiple trials.
    # Let's run it with a time limit.
    sidon_set, sidon_size = find_best_sidon_set(search_time_budget)
    
    search_elapsed = time.time() - start_search
    if sidon_size > 0:
        # Construct h from this set
        h_sidon = np.zeros(N)
        h_sidon[sidon_set] = 1.0 / len(sidon_set)
        s_val = score_raw(h_sidon)
        
        if s_val > best_score_val:
            best_score_val = s_val
            best_h = h_sidon.copy()

    # Phase 2: Refinement using SLSQP
    # If we found a good discrete structure, use it as a start for continuous optimization.
    # This can sometimes squeeze out a bit more if the "Sidon" set isn't perfectly optimal
    # or if weights can be adjusted.
    
    remaining_time = deadline - time.time()
    
    if best_h is not None and remaining_time > 10:
        # Try polishing the best discrete solution
        h_opt, s_opt = optimize_epigraph(best_h, max_iter=1000, ftol=1e-9)
        if h_opt is not None and s_opt > best_score_val:
            best_score_val = s_opt
            best_h = h_opt.copy()
            
    # If the combinatorial search didn't yield a good result, or if we have lots of time left,
    # fall back to random restarts with SLSQP, but guided by sparse starts.
    if time.time() < deadline - 15:
        # Try a few random sparse starts just in case the combinatorial search missed a local optimum
        # that SLSQP could exploit.
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            K = random.randint(4, 8)
            support = np.random.choice(N, K, replace=False)
            h_rand = np.zeros(N)
            h_rand[support] = 1.0 / K
            
            h_opt, s_opt = optimize_epigraph(h_rand, max_iter=300, ftol=1e-8)
            if h_opt is not None and s_opt > best_score_val:
                best_score_val = s_opt
                best_h = h_opt.copy()

    # Final Polish if time allows
    if best_h is not None and time.time() < deadline - 2:
        h_polished, s_polished = optimize_epigraph(best_h, max_iter=500, ftol=1e-10)
        if h_polished is not None and s_polished > best_score_val:
            best_score_val = s_polished
            best_h = h_polished.copy()

    # Fallback
    if best_h is None:
        best_h = np.ones(N) / N
        best_score_val = score_raw(best_h)

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
