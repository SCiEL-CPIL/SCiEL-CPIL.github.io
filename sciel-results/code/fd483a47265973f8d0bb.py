"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

N = 256
DEADLINE = 115.0

def _count_mono4(c):
    """Count monochromatic 4-term APs in a 0/1 numpy array of length N (0-indexed)."""
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        c0 = c[0:max_a]
        c1 = c[d:d + max_a]
        c2 = c[2 * d:2 * d + max_a]
        c3 = c[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        count += np.count_nonzero(mask)
    return count

def _thue_morse():
    """Generate Thue-Morse sequence for 0..255."""
    c = np.zeros(N, dtype=np.int8)
    for i in range(N):
        c[i] = bin(i).count('1') % 2
    return c

def _precompute_ap_membership():
    """Precompute which APs contain each position.
    Returns list of numpy arrays: ap_arrays[i] = array of (a, d, k) where position i is in AP."""
    ap_list = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        for a in range(max_a):
            for k in range(4):
                pos = a + k * d
                ap_list[pos].append((a, d, k))
    ap_arrays = []
    for i in range(N):
        if ap_list[i]:
            arr = np.array(ap_list[i], dtype=np.int32)
        else:
            arr = np.empty((0, 3), dtype=np.int32)
        ap_arrays.append(arr)
    return ap_arrays

def _delta_flip_fast(c, i, ap_arrays):
    """Compute change in mono-4AP count if we flip position i, using precomputed membership."""
    ap_arr = ap_arrays[i]
    if len(ap_arr) == 0:
        return 0
    
    a_vals = ap_arr[:, 0]
    d_vals = ap_arr[:, 1]
    k_vals = ap_arr[:, 2]
    
    idx0 = a_vals
    idx1 = a_vals + d_vals
    idx2 = a_vals + 2 * d_vals
    idx3 = a_vals + 3 * d_vals
    
    v0 = c[idx0]
    v1 = c[idx1]
    v2 = c[idx2]
    v3 = c[idx3]
    
    was_mono = (v0 == v1) & (v0 == v2) & (v0 == v3)
    
    new_v = 1 - c[i]
    
    is_mono_after = np.zeros(len(ap_arr), dtype=bool)
    
    mask_k0 = (k_vals == 0)
    if np.any(mask_k0):
        is_mono_after[mask_k0] = (new_v == v1[mask_k0]) & (new_v == v2[mask_k0]) & (new_v == v3[mask_k0])
    
    mask_k1 = (k_vals == 1)
    if np.any(mask_k1):
        is_mono_after[mask_k1] = (v0[mask_k1] == new_v) & (new_v == v2[mask_k1]) & (new_v == v3[mask_k1])
    
    mask_k2 = (k_vals == 2)
    if np.any(mask_k2):
        is_mono_after[mask_k2] = (v0[mask_k2] == v1[mask_k2]) & (v1[mask_k2] == new_v) & (new_v == v3[mask_k2])
    
    mask_k3 = (k_vals == 3)
    if np.any(mask_k3):
        is_mono_after[mask_k3] = (v0[mask_k3] == v1[mask_k3]) & (v1[mask_k3] == v2[mask_k3]) & (v2[mask_k3] == new_v)
    
    delta = np.count_nonzero(~was_mono & is_mono_after) - np.count_nonzero(was_mono & ~is_mono_after)
    return int(delta)

def _local_search_delta(c, time_limit, ap_arrays=None):
    """Improve coloring via single-flip local search using delta computation."""
    if ap_arrays is None:
        ap_arrays = _precompute_ap_membership()
    start = time.time()
    current = _count_mono4(c)
    improved = True
    max_passes = 100
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        order = np.random.default_rng(passes * 7 + 1).permutation(N)
        for i in order:
            if time.time() - start > time_limit:
                return c, current
            delta = _delta_flip_fast(c, int(i), ap_arrays)
            if delta < 0:
                c[int(i)] = 1 - c[int(i)]
                current += delta
                improved = True
    return c, current

def _simulated_annealing(c, time_limit, ap_arrays=None, T_start=8.0, T_end=0.05, alpha=0.9995):
    """Simulated annealing to escape local minima."""
    if ap_arrays is None:
        ap_arrays = _precompute_ap_membership()
    start = time.time()
    current = _count_mono4(c)
    T = T_start
    rng = np.random.default_rng(456)
    best_c = c.copy()
    best_count = current
    
    while time.time() - start < time_limit:
        i = int(rng.integers(0, N))
        delta = _delta_flip_fast(c, i, ap_arrays)
        
        if delta < 0:
            c[i] = 1 - c[i]
            current += delta
            if current < best_count:
                best_count = current
                best_c = c.copy()
        else:
            if delta == 0:
                c[i] = 1 - c[i]
            else:
                if T > 0.01:
                    if -delta / T > np.log(rng.random()):
                        c[i] = 1 - c[i]
                        current += delta
        
        T *= alpha
        if T < T_end:
            T = T_end
    
    return best_c, best_count

def _crossover(p1, p2, rng):
    """Uniform crossover."""
    mask = rng.random(N) < 0.5
    child = np.where(mask, p1, p2)
    return child.copy()

def _mutate(c, rng, rate=0.05):
    """Bit-flip mutation."""
    flips = rng.random(N) < rate
    c[flips] = 1 - c[flips]
    return c

def construct():
    start_time = time.time()
    
    # Precompute AP membership for fast delta computation
    ap_arrays = _precompute_ap_membership()
    
    # Start with Thue-Morse
    c = _thue_morse()
    best_c = c.copy()
    best_count = _count_mono4(c)
    
    # Local search from Thue-Morse using fast delta
    time_remaining = DEADLINE - (time.time() - start_time)
    if time_remaining > 5:
        c, cnt = _local_search_delta(c, time_remaining - 5, ap_arrays)
        if cnt < best_count:
            best_count = cnt
            best_c = c.copy()
    
    # Simulated annealing from the locally optimized solution
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 15:
        sa_c, sa_cnt = _simulated_annealing(best_c.copy(), min(10, remaining - 10), ap_arrays)
        if sa_cnt < best_count:
            best_count = sa_cnt
            best_c = sa_c.copy()
    
    # Genetic Algorithm
    rng = np.random.default_rng(42)
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    if remaining > 20:
        pop_size = 16
        elite_size = 3
        mutation_rate = 0.05
        
        # Initialize population: best solution + random solutions
        population = [best_c.copy()]
        scores = [best_count]
        
        for _ in range(pop_size - 1):
            c_rand = rng.integers(0, 2, size=N).astype(np.int8)
            # Quick local search to improve random start
            time_left = max(0.3, (DEADLINE - (time.time() - start_time)) / (pop_size - 1))
            c_rand, cnt_rand = _local_search_delta(c_rand, min(2, time_left), ap_arrays)
            population.append(c_rand)
            scores.append(cnt_rand)
        
        # Check if we need to update best
        for i in range(pop_size):
            if scores[i] < best_count:
                best_count = scores[i]
                best_c = population[i].copy()
        
        # GA loop
        while time.time() - start_time < DEADLINE - 10:
            # Sort by score
            sorted_idx = np.argsort(scores)
            population = [population[i].copy() for i in sorted_idx]
            scores = [scores[i] for i in sorted_idx]
            
            # Elitism: keep best elite_size
            new_population = [population[i].copy() for i in range(elite_size)]
            new_scores = [scores[i] for i in range(elite_size)]
            
            # Generate offspring until population is full
            while len(new_population) < pop_size:
                if time.time() - start_time > DEADLINE - 10:
                    break
                
                # Tournament selection
                idxs = rng.integers(0, pop_size, size=3)
                best_idx = idxs[np.argmin([scores[i] for i in idxs])]
                p1 = population[best_idx].copy()
                
                idxs2 = rng.integers(0, pop_size, size=3)
                best_idx2 = idxs2[np.argmin([scores[i] for i in idxs2])]
                p2 = population[best_idx2].copy()
                
                # Crossover
                child = _crossover(p1, p2, rng)
                
                # Mutation
                child = _mutate(child, rng, mutation_rate)
                
                # Local search on child
                time_left_for_ls = min(1.0, (DEADLINE - (time.time() - start_time)) / 10)
                if time_left_for_ls > 0.1:
                    child, cnt_child = _local_search_delta(child, time_left_for_ls, ap_arrays)
                else:
                    cnt_child = _count_mono4(child)
                
                new_population.append(child)
                new_scores.append(cnt_child)
            
            population = new_population
            scores = new_scores
            
            # Update best
            min_score_idx = int(np.argmin(scores))
            if scores[min_score_idx] < best_count:
                best_count = scores[min_score_idx]
                best_c = population[min_score_idx].copy()
    
    return [int(x) for x in best_c]
# EVOLVE-BLOCK-END
