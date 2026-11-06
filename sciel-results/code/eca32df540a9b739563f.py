import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute APs: all arithmetic progressions of length >= 2 within [0, N)
    ap_terms = []
    for d in range(1, N + 1):
        for s in range(0, N - d + 1):
            terms = list(range(s, N, d))
            if len(terms) > 1:
                ap_terms.append(terms)

    num_aps = len(ap_terms)

    # Build incidence matrix M (num_aps x N) as float64
    M = np.zeros((num_aps, N), dtype=np.float64)
    pos_to_aps = [[] for _ in range(N)]
    for j, terms in enumerate(ap_terms):
        for i in terms:
            M[j, i] = 1.0
            pos_to_aps[i].append(j)

    pos_to_aps_arr = [np.array(aps, dtype=np.int32) for aps in pos_to_aps]

    # C = M.T @ M, K = diag(C)
    C = (M.T @ M).astype(np.float64)
    K = np.diag(C).copy()

    def hard_max_of_signs(signs):
        sums = M @ signs
        return int(np.max(np.abs(sums)))

    best_signs = None
    best_hard_max = N + 1

    # ============================================================
    # PHASE 1: Greedy construction (multiple variants)
    # Build sequence position by position, choosing the sign that
    # minimizes the maximum |AP sum| over all APs that include
    # the new position.
    # ============================================================

    def greedy_construct(seed, mode=0):
        """
        mode 0: minimize max |sum| greedily
        mode 1: minimize sum of squares of affected AP sums
        mode 2: minimize max |sum| with tie-breaking by sum of squares
        """
        rng = random.Random(seed)
        signs = np.zeros(N, dtype=np.float64)
        sums = np.zeros(num_aps, dtype=np.float64)
        
        # For each position, precompute which APs it belongs to
        # and track current sums incrementally
        
        for pos in range(N):
            # Determine which sign (+1 or -1) is better
            idxs = pos_to_aps_arr[pos]
            
            if mode == 0:
                # Greedy: pick sign that minimizes max |sum| over affected APs
                best_val = None
                best_sign = 1.0
                for s in [1.0, -1.0]:
                    new_sums = sums[idxs] + s
                    m = np.max(np.abs(new_sums)) if len(new_sums) > 0 else 0.0
                    if best_val is None or m < best_val:
                        best_val = m
                        best_sign = s
                signs[pos] = best_sign
                sums[idxs] += best_sign
                
            elif mode == 1:
                # Minimize sum of squares of affected AP sums
                best_val = None
                best_sign = 1.0
                for s in [1.0, -1.0]:
                    new_sums = sums[idxs] + s
                    ss = np.sum(new_sums ** 2) if len(new_sums) > 0 else 0.0
                    if best_val is None or ss < best_val:
                        best_val = ss
                        best_sign = s
                signs[pos] = best_sign
                sums[idxs] += best_sign
                
            else:
                # mode 2: minimize max, tie-break by sum of squares
                best_key = None
                best_sign = 1.0
                for s in [1.0, -1.0]:
                    new_sums = sums[idxs] + s
                    m = np.max(np.abs(new_sums)) if len(new_sums) > 0 else 0.0
                    ss = np.sum(new_sums ** 2) if len(new_sums) > 0 else 0.0
                    key = (m, ss)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_sign = s
                signs[pos] = best_sign
                sums[idxs] += best_sign
        
        hm = hard_max_of_signs(signs)
        return signs, hm

    # Try multiple greedy constructions
    greedy_configs = [
        (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),
        (10, 1), (20, 1), (30, 1),
        (100, 2), (200, 2), (300, 2),
    ]
    
    for seed, mode in greedy_configs:
        if time.time() > deadline - 30:
            break
        signs, hm = greedy_construct(seed, mode)
        if hm < best_hard_max:
            best_hard_max = hm
            best_signs = signs.copy()

    # ============================================================
    # PHASE 2: Iterative improvement with single-flip + pair-flip
    # ============================================================
    
    def improve(signs_in, time_limit):
        """
        Iteratively improve signs using:
        - Single flips (greedy)
        - Pair flips (swap two positions)
        Returns improved signs and their hard max.
        """
        signs = signs_in.copy().astype(np.float64)
        sums = M @ signs
        G = C @ signs
        current_hm = int(np.max(np.abs(sums)))
        
        improved = True
        max_rounds = 200
        for rnd in range(max_rounds):
            if time.time() > time_limit:
                break
            if not improved:
                break
            improved = False
            
            # --- Single flips ---
            for k in range(N):
                if time.time() > time_limit:
                    break
                delta = -2.0 * signs[k]
                idxs = pos_to_aps_arr[k]
                new_sums = sums[idxs] + delta
                old_max = np.max(np.abs(sums))
                new_max = np.max(np.abs(new_sums))
                if new_max < old_max - 1e-9:
                    signs[k] = -signs[k]
                    sums[idxs] += delta
                    G += delta * C[:, k]
                    improved = True
            
            if not improved and time.time() < time_limit:
                # --- Pair flips: try swapping signs at two positions ---
                # Only try a subset to keep it fast
                for k in range(N):
                    if time.time() > time_limit:
                        break
                    for l in range(k + 1, N):
                        if time.time() > time_limit:
                            break
                        # Check if flipping both k and l helps
                        dk = -2.0 * signs[k]
                        dl = -2.0 * signs[l]
                        
                        # Affected APs: union of APs containing k or l
                        idxs_k = pos_to_aps_arr[k]
                        idxs_l = pos_to_aps_arr[l]
                        
                        # Combined affected indices
                        combined = np.unique(np.concatenate([idxs_k, idxs_l]))
                        
                        new_sums = sums.copy()
                        new_sums[idxs_k] += dk
                        new_sums[idxs_l] += dl
                        
                        old_max = np.max(np.abs(sums))
                        new_max = np.max(np.abs(new_sums))
                        
                        if new_max < old_max - 1e-9:
                            signs[k] = -signs[k]
                            signs[l] = -signs[l]
                            sums[idxs_k] += dk
                            sums[idxs_l] += dl
                            G += dk * C[:, k] + dl * C[:, l]
                            improved = True
        
        final_hm = int(np.max(np.abs(sums)))
        return signs, final_hm

    if best_signs is not None:
        signs, hm = improve(best_signs, deadline - 15)
        if hm < best_hard_max:
            best_hard_max = hm
            best_signs = signs.copy()

    # ============================================================
    # PHASE 3: Multi-start with random perturbations + improvement
    # ============================================================
    
    if time.time() < deadline - 10:
        seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
        
        for seed in seeds:
            if time.time() > deadline - 10:
                break
            
            rng = random.Random(seed)
            
            if best_signs is not None:
                signs = best_signs.copy().astype(np.float64)
                # Flip a random subset
                n_flips = rng.randint(2, 12)
                for _ in range(n_flips):
                    kk = rng.randint(0, N - 1)
                    signs[kk] = -signs[kk]
            else:
                signs = np.ones(N, dtype=np.float64)
                for i in range(N):
                    if rng.random() < 0.5:
                        signs[i] = -1.0
            
            # Quick improvement
            signs, hm = improve(signs, deadline - 10)
            if hm < best_hard_max:
                best_hard_max = hm
                best_signs = signs.copy()

    # ============================================================
    # PHASE 4: Final polish - exhaustive single-flip pass
    # ============================================================
    
    if best_signs is not None and time.time() < deadline - 2:
        signs = best_signs.copy().astype(np.float64)
        sums = M @ signs
        improved = True
        while improved and time.time() < deadline - 2:
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
        
        current_hm = int(np.max(np.abs(sums)))
        if current_hm < best_hard_max:
            best_hard_max = current_hm
            best_signs = signs.copy()

    if best_signs is None:
        return [1 if random.random() < 0.5 else -1 for _ in range(N)]

    return [int(s) for s in best_signs]
