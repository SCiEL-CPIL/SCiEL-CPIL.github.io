"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # Precompute all arithmetic progressions of length >= 2 within [1, 256]
    ap_indices = []
    for start in range(1, n + 1):
        for step in range(1, n + 1):
            terms = list(range(start, n + 1, step))
            if len(terms) >= 2:
                ap_indices.append([t - 1 for t in terms])
    
    num_aps = len(ap_indices)
    
    # Build position-to-AP mapping
    pos_to_aps = [[] for _ in range(n)]
    for ap_id, idx in enumerate(ap_indices):
        for i in idx:
            pos_to_aps[i].append(ap_id)
    
    # Store AP indices as numpy arrays for fast summing
    ap_arrays = [np.array(idx, dtype=np.int32) for idx in ap_indices]
    
    # Build incidence matrix: 256 x num_aps, entry (pos, ap) = 1 if pos in ap
    # This is sparse, so we use a list of arrays approach
    # Actually, let's use a dense matrix for vectorized operations
    # num_aps can be large, so let's estimate: roughly sum over start,step of (n-start)/step
    # This is around n * log(n) * some constant, maybe ~2000-5000 APs
    # A 256 x 5000 int8 matrix is ~1.3MB, fine
    
    # Build the incidence matrix
    incidence = np.zeros((n, num_aps), dtype=np.int8)
    for ap_id, idx in enumerate(ap_indices):
        for pos in idx:
            incidence[pos, ap_id] = 1
    
    # Also build reverse: for each AP, which positions it contains
    # Already have ap_arrays
    
    def compute_ap_sums(signs):
        """Vectorized: signs is (n,), incidence is (n, num_aps) -> ap_sums is (num_aps,)"""
        return signs.astype(np.int32) @ incidence  # (num_aps,)
    
    def max_from_sums(ap_sums):
        return int(np.max(np.abs(ap_sums)))
    
    def build_hist(ap_sums):
        max_hist = n + 1
        hist = np.zeros(max_hist, dtype=np.int32)
        for s in ap_sums:
            hist[abs(s)] += 1
        return hist
    
    def get_max_from_hist(hist):
        for k in range(len(hist) - 1, -1, -1):
            if hist[k] > 0:
                return k
        return 0
    
    def thue_morse_signs(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs
    
    def local_search_sa(signs_init, max_iters, deadline, seed=42):
        """Simulated Annealing with fast vectorized evaluation."""
        signs = signs_init.copy().astype(np.int8)
        ap_sums = compute_ap_sums(signs)
        hist = build_hist(ap_sums)
        current_max = get_max_from_hist(hist)
        best_signs = signs.copy()
        best_max = current_max
        
        rng = random.Random(seed)
        temperature = 5.0
        cooling = 0.9998
        
        # Precompute for each position, the list of AP IDs it affects
        # And for faster delta computation, we need to know the contribution
        # When we flip position i, ap_sum[ap] changes by -2 * signs[i]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Choose a move: single flip or double flip
            if rng.random() < 0.1:
                # Double flip
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i == j:
                    j = (j + 1) % n
                
                old_si = signs[i]
                old_sj = signs[j]
                
                signs[i] = -old_si
                signs[j] = -old_sj
                
                # Affected APs
                affected_set = set(pos_to_aps[i]) | set(pos_to_aps[j])
                affected_list = list(affected_set)
                
                # Update ap_sums and hist
                for ap_id in affected_list:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    
                    delta = 0
                    if ap_id in pos_to_aps[i]:
                        delta -= 2 * old_si
                    if ap_id in pos_to_aps[j]:
                        delta -= 2 * old_sj
                    ap_sums[ap_id] += delta
                    
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                
                new_max = get_max_from_hist(hist)
                delta_obj = new_max - current_max
                
                if delta_obj <= 0 or (temperature > 0.05 and rng.random() < np.exp(-delta_obj / temperature)):
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                else:
                    # Revert
                    signs[i] = old_si
                    signs[j] = old_sj
                    for ap_id in affected_list:
                        old_abs = abs(ap_sums[ap_id])
                        hist[old_abs] -= 1
                        
                        delta = 0
                        if ap_id in pos_to_aps[i]:
                            delta += 2 * old_si
                        if ap_id in pos_to_aps[j]:
                            delta += 2 * old_sj
                        ap_sums[ap_id] += delta
                        
                        new_abs = abs(ap_sums[ap_id])
                        hist[new_abs] += 1
            else:
                # Single flip
                i = rng.randint(0, n - 1)
                old_sign = signs[i]
                
                signs[i] = -old_sign
                affected = pos_to_aps[i]
                
                for ap_id in affected:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    ap_sums[ap_id] -= 2 * old_sign
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                
                new_max = get_max_from_hist(hist)
                delta_obj = new_max - current_max
                
                if delta_obj <= 0 or (temperature > 0.05 and rng.random() < np.exp(-delta_obj / temperature)):
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_signs = signs.copy()
                else:
                    signs[i] = old_sign
                    for ap_id in affected:
                        old_abs = abs(ap_sums[ap_id])
                        hist[old_abs] -= 1
                        ap_sums[ap_id] += 2 * old_sign
                        new_abs = abs(ap_sums[ap_id])
                        hist[new_abs] += 1
            
            temperature *= cooling
        
        return best_signs, best_max
    
    def greedy_polish(signs, ap_sums, hist, current_max, deadline):
        """Greedy single-pass improvement."""
        signs = signs.copy()
        ap_sums = ap_sums.copy()
        hist = hist.copy()
        
        order = list(range(n))
        rng = random.Random(int(time.time() * 1000) % 100000)
        rng.shuffle(order)
        
        improved = True
        passes = 0
        while improved and time.time() < deadline - 2.0 and passes < 10:
            improved = False
            passes += 1
            for i in order:
                if time.time() > deadline - 2.0:
                    break
                old_sign = signs[i]
                signs[i] = -old_sign
                for ap_id in pos_to_aps[i]:
                    old_abs = abs(ap_sums[ap_id])
                    hist[old_abs] -= 1
                    ap_sums[ap_id] -= 2 * old_sign
                    new_abs = abs(ap_sums[ap_id])
                    hist[new_abs] += 1
                new_max = get_max_from_hist(hist)
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                else:
                    signs[i] = old_sign
                    for ap_id in pos_to_aps[i]:
                        old_abs = abs(ap_sums[ap_id])
                        hist[old_abs] -= 1
                        ap_sums[ap_id] += 2 * old_sign
                        new_abs = abs(ap_sums[ap_id])
                        hist[new_abs] += 1
        
        return signs, ap_sums, hist, current_max
    
    # Generate diverse initializations
    initial_signs_list = []
    
    # Thue-Morse and negation
    tm = thue_morse_signs(n)
    initial_signs_list.append(tm)
    initial_signs_list.append(-tm)
    
    # Random balanced sequences
    seeds = [42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618, 9999, 5555, 7777, 1111, 2222, 3333, 4444, 8888, 6666]
    for seed in seeds:
        rng = random.Random(seed)
        signs = np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8)
        initial_signs_list.append(signs)
    
    # Structured: alternating blocks
    for block_size in [2, 4, 8, 16, 32]:
        signs = np.array([1 if (i // block_size) % 2 == 0 else -1 for i in range(n)], dtype=np.int8)
        initial_signs_list.append(signs)
    
    # Hadamard-like: use Kronecker products
    # 2x2 Hadamard H2 = [[1,1],[1,-1]]
    # Build up to 256x256
    def hadamard_order(n):
        if n == 1:
            return np.array([1])
        H_half = hadamard_order(n // 2)
        return np.concatenate([np.concatenate([H_half, H_half], axis=1),
                               np.concatenate([H_half, -H_half], axis=1)])
    
    H = hadamard_order(256)
    # Take first row as signs
    initial_signs_list.append(H[0].astype(np.int8))
    initial_signs_list.append(H[1].astype(np.int8))
    initial_signs_list.append(H[2].astype(np.int8))
    initial_signs_list.append(H[3].astype(np.int8))
    
    # Run multi-restart SA
    best_overall = None
    best_overall_max = n + 1
    
    # Sort initializations by their initial max to prioritize good starts
    init_scores = []
    for s in initial_signs_list:
        ap_sums = compute_ap_sums(s)
        mx = max_from_sums(ap_sums)
        init_scores.append((mx, s))
    init_scores.sort(key=lambda x: x[0])
    
    # Allocate time per restart
    num_restarts = len(init_scores)
    time_per_restart = (deadline - start_time - 10.0) / max(1, num_restarts)
    
    for idx, (init_max, signs_init) in enumerate(init_scores):
        if time.time() > deadline - 10.0:
            break
        
        # More iterations for better initializations
        iters = max(1000, int(3000 * (1.0 + 0.5 * (num_restarts - idx) / num_restarts)))
        signs, mx = local_search_sa(signs_init, max_iters=iters, deadline=deadline, seed=idx * 137 + 7)
        
        if mx < best_overall_max:
            best_overall_max = mx
            best_overall = signs.copy()
            if best_overall_max <= 2:
                break
    
    # Final greedy polish
    if best_overall is not None:
        signs = best_overall.copy().astype(np.int8)
        ap_sums = compute_ap_sums(signs)
        hist = build_hist(ap_sums)
        current_max = get_max_from_hist(hist)
        
        signs, ap_sums, hist, current_max = greedy_polish(signs, ap_sums, hist, current_max, deadline)
        
        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall = signs.copy()
    
    if best_overall is None:
        rng = random.Random(42)
        return [rng.choice([-1, 1]) for _ in range(n)]
    
    final_signs = [int(s) for s in best_overall]
    assert len(final_signs) == n
    assert all(s in [-1, 1] for s in final_signs)
    
    return final_signs
# EVOLVE-BLOCK-END
