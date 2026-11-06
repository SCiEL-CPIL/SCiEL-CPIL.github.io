import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    # Precompute APs: all arithmetic progressions of length >= 2 within [0, N)
    ap_terms = []
    for d in range(1, N + 1):
        for s in range(0, N - d + 1):
            terms = list(range(s, N, d))
            if len(terms) > 1:
                ap_terms.append(terms)

    num_aps = len(ap_terms)

    # Build incidence matrix M (num_aps x N) as float64 for precision
    M = np.zeros((num_aps, N), dtype=np.float64)
    pos_to_aps = [[] for _ in range(N)]
    for j, terms in enumerate(ap_terms):
        for i in terms:
            M[j, i] = 1.0
            pos_to_aps[i].append(j)

    pos_to_aps_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_aps]

    # C = M.T @ M
    C = (M.T @ M).astype(np.float64)
    K = np.diag(C).copy()

    def hard_max_of_signs(signs):
        sums = M @ signs
        return int(np.max(np.abs(sums)))

    best_signs = None
    best_hard_max = N + 1

    # Phase 1: Continuous subgradient descent to find a low-discrepancy basin
    # We minimize the smooth surrogate: sum of squares of all AP sums = x^T C x
    # subject to ||x||_inf <= 1 (relaxation of {+/-1}^N)
    # Use projected subgradient descent with step size ~ 1/sqrt(t)
    
    if time.time() < deadline - 15:
        # Multiple random starts for continuous optimization
        cont_seeds = [101, 202, 303, 404, 505]
        
        for cseed in cont_seeds:
            if time.time() > deadline - 15:
                break
                
            rng = random.Random(cseed)
            # Start with random continuous values in [-1, 1]
            x = np.random.RandomState(cseed).uniform(-1, 1, size=N).astype(np.float64)
            
            # Project to [-1, 1]
            x = np.clip(x, -1, 1)
            
            # Compute gradient: grad = 2 * C @ x
            t = 0
            max_steps = 2000
            step_size_base = 1.0
            
            for step in range(max_steps):
                if time.time() > deadline - 15:
                    break
                    
                # Gradient of x^T C x is 2 C x
                grad = 2.0 * (C @ x)
                
                # Adaptive step size: decrease over time
                t = step + 1
                step_size = step_size_base / math.sqrt(t)
                
                # Subgradient step
                x = x - step_size * grad
                
                # Project back to [-1, 1]
                x = np.clip(x, -1, 1)
                
                # Every 100 steps, evaluate the rounded signs and update best
                if step % 100 == 99:
                    # Project to nearest +/-1
                    signs_rounded = np.sign(x)
                    signs_rounded[signs_rounded == 0] = 1.0  # break ties
                    hm = hard_max_of_signs(signs_rounded)
                    if hm < best_hard_max:
                        best_hard_max = hm
                        best_signs = signs_rounded.copy()
                        
                # Early stop if we've converged
                if step > 500:
                    x_new = np.clip(x - step_size * grad, -1, 1)
                    if np.linalg.norm(x_new - x) < 1e-6:
                        break

    # Phase 2: Local search on the best continuous projection
    if best_signs is not None and time.time() < deadline - 5:
        signs = best_signs.copy().astype(np.float64)
        sums = M @ signs
        G = C @ signs
        
        # Iterate local search: flip signs that reduce the max absolute AP sum
        improved = True
        max_rounds = 50
        for round_num in range(max_rounds):
            if time.time() > deadline - 5:
                break
            if not improved:
                break
            improved = False
            for k in range(N):
                delta = -2.0 * signs[k]
                # New sums for APs containing k
                idxs = pos_to_aps_arr[k]
                new_sums = sums[idxs] + delta
                old_max = np.max(np.abs(sums))
                new_max = np.max(np.abs(new_sums))
                # Also need to check if removing k from some AP reduces max
                # More precise: compute the new global max
                # But that's expensive, so use a heuristic:
                # Only flip if the max of affected APs decreases AND
                # the global max doesn't increase
                if new_max < old_max - 1e-9:
                    signs[k] = -signs[k]
                    sums[idxs] += delta
                    G += delta * C[:, k]
                    improved = True
            
            if improved:
                current_hm = int(np.max(np.abs(sums)))
                if current_hm < best_hard_max:
                    best_hard_max = current_hm
                    best_signs = signs.copy()
                else:
                    # Revert to best
                    signs = best_signs.copy().astype(np.float64)
                    sums = M @ signs
                    G = C @ signs
                    break

    # Phase 3: Multi-restart SA to escape local optima
    if time.time() < deadline - 3:
        seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000]
        
        for seed in seeds:
            if time.time() > deadline - 3:
                break
                
            rng = random.Random(seed)
            
            # Initialize from best_signs with some random perturbations
            if best_signs is not None:
                signs = best_signs.copy().astype(np.float64)
                # Flip a few random signs
                for _ in range(rng.randint(3, 8)):
                    kk = rng.randint(0, N - 1)
                    signs[kk] = -signs[kk]
            else:
                signs = np.zeros(N, dtype=np.float64)
                for i in range(N):
                    signs[i] = 1.0 if rng.random() < 0.5 else -1.0
            
            sums = M @ signs
            G = C @ signs
            
            current_hard_max = int(np.max(np.abs(sums)))
            if current_hard_max < best_hard_max:
                best_hard_max = current_hard_max
                best_signs = signs.copy()
            
            # SA parameters
            T = 5.0
            T_min = 0.2
            T_decay = 0.998
            steps = 0
            no_improve = 0
            
            while time.time() < deadline - 3:
                k = rng.randint(0, N - 1)
                delta = -2.0 * signs[k]
                
                delta_J = 2.0 * delta * G[k] + (delta * delta) * K[k]
                
                accept = False
                if delta_J < 0:
                    accept = True
                else:
                    if delta_J / T > 80.0:
                        accept = False
                    else:
                        accept = rng.random() < math.exp(-delta_J / T)
                
                if accept:
                    signs[k] = -signs[k]
                    idxs = pos_to_aps_arr[k]
                    sums[idxs] += delta
                    G += delta * C[:, k]
                    steps += 1
                    
                    if steps % 30 == 0:
                        current_hard_max = int(np.max(np.abs(sums)))
                        if current_hard_max < best_hard_max:
                            best_hard_max = current_hard_max
                            best_signs = signs.copy()
                            no_improve = 0
                        else:
                            no_improve += 1
                        
                        if no_improve > 150:
                            for _ in range(rng.randint(2, 5)):
                                kk = rng.randint(0, N - 1)
                                dd = -2.0 * signs[kk]
                                signs[kk] = -signs[kk]
                                sums[pos_to_aps_arr[kk]] += dd
                                G += dd * C[:, kk]
                            no_improve = 0
                
                if steps % 500 == 0:
                    T *= T_decay
                    if T < T_min:
                        T = T_min
                
                if best_hard_max <= 1:
                    break

    # Final polish
    if best_signs is not None and time.time() < deadline - 1:
        signs = best_signs.copy().astype(np.float64)
        sums = M @ signs
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for k in range(N):
                delta = -2.0 * signs[k]
                idxs = pos_to_aps_arr[k]
                new_sums = sums[idxs] + delta
                old_max = np.max(np.abs(sums))
                new_max = np.max(np.abs(new_sums))
                if new_max < old_max - 1e-9:
                    signs[k] = -signs[k]
                    sums[idxs] += delta
                    improved = True
            
            if not improved:
                current_hm = int(np.max(np.abs(sums)))
                if current_hm < best_hard_max:
                    best_hard_max = current_hm
                    best_signs = signs.copy()

    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]

    return [int(s) for s in best_signs]
