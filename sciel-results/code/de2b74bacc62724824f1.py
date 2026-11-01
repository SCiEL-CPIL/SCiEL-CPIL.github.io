"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math
from array import array

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Precompute neighbors as tuples for faster iteration
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = 2187
    
    neighbors = []
    for w in all_words:
        w_list = list(w)
        idxs = [word_to_idx[w]]
        for i in range(7):
            orig = w_list[i]
            for v in range(3):
                if v != orig:
                    w_list[i] = v
                    idxs.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig
        neighbors.append(tuple(idxs))
    
    # Precompute neighbor bitmask for fast overlap checking
    # Each word has 15 neighbors (itself + 14 at distance 1)
    # Use two 64-bit integers to represent up to 128 bits, but we need 2187 bits
    # Instead, use a set of frozensets for membership testing
    neighbor_sets = [frozenset(nb) for nb in neighbors]
    
    # For faster "losing" computation, precompute for each pair of words
    # how many neighbors are unique to the first but not in the second
    # This is too expensive (2187^2 = ~4.8M pairs), so we'll compute on the fly
    # but optimize the inner loop
    
    # Greedy initialization with multiple strategies
    def greedy_init(seed, strategy='max_gain'):
        random.seed(seed)
        counts = [0] * N
        covered = 0
        code = []
        code_set = set()
        uncovered = N
        
        if strategy == 'max_gain':
            # Standard greedy: pick word covering most uncovered words
            while uncovered > 0:
                best_idx = -1
                max_cov = 0
                for w_idx in range(N):
                    if w_idx in code_set:
                        continue
                    cov = 0
                    for nb in neighbors[w_idx]:
                        if counts[nb] == 0:
                            cov += 1
                    if cov > max_cov:
                        max_cov = cov
                        best_idx = w_idx
                if best_idx == -1:
                    break
                code.append(all_words[best_idx])
                code_set.add(best_idx)
                for nb in neighbors[best_idx]:
                    if counts[nb] == 0:
                        covered += 1
                        uncovered -= 1
                    counts[nb] += 1
        elif strategy == 'randomized':
            # Randomized greedy with temperature
            T = 1.0
            while uncovered > 0:
                # Sample several candidates and pick based on gain
                candidates = []
                for _ in range(50):
                    w_idx = random.randint(0, N - 1)
                    if w_idx in code_set:
                        continue
                    cov = 0
                    for nb in neighbors[w_idx]:
                        if counts[nb] == 0:
                            cov += 1
                    candidates.append((cov, w_idx))
                if not candidates:
                    break
                # Pick from top candidates with softmax-like selection
                max_cov = max(c for c, _ in candidates)
                # Filter to top 20% or those within 2 of max
                threshold = max(0, max_cov - 2)
                top = [(c, w) for c, w in candidates if c >= threshold]
                if not top:
                    top = candidates
                # Weighted random selection
                weights = [c + 0.1 for c, _ in top]
                total_w = sum(weights)
                r = random.random() * total_w
                cum = 0
                chosen = top[0][1]
                for (c, w), wt in zip(top, weights):
                    cum += wt
                    if r <= cum:
                        chosen = w
                        break
                code.append(all_words[chosen])
                code_set.add(chosen)
                for nb in neighbors[chosen]:
                    if counts[nb] == 0:
                        covered += 1
                        uncovered -= 1
                    counts[nb] += 1
                T *= 0.999
        elif strategy == 'pairwise':
            # Greedy with pairwise conflict consideration
            while uncovered > 0:
                best_idx = -1
                max_score = 0
                for w_idx in range(N):
                    if w_idx in code_set:
                        continue
                    cov = 0
                    for nb in neighbors[w_idx]:
                        if counts[nb] == 0:
                            cov += 1
                    # Penalize if this word's neighbors are already well-covered
                    overlap = 0
                    for nb in neighbors[w_idx]:
                        if counts[nb] > 0:
                            overlap += 1
                    score = cov * 2 - overlap * 0.5
                    if score > max_score:
                        max_score = score
                        best_idx = w_idx
                if best_idx == -1:
                    break
                code.append(all_words[best_idx])
                code_set.add(best_idx)
                for nb in neighbors[best_idx]:
                    if counts[nb] == 0:
                        covered += 1
                        uncovered -= 1
                    counts[nb] += 1
        
        return code, counts, covered, code_set
    
    def try_greedy_reduction(code, counts, covered, code_set, max_passes=10):
        """Try to remove redundant codewords in multiple passes."""
        size = len(code)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            indices = list(range(size))
            random.shuffle(indices)
            for idx in indices:
                if idx >= size:
                    continue
                w = code[idx]
                w_idx = word_to_idx[w]
                would_uncover = 0
                for nb in neighbors[w_idx]:
                    if counts[nb] == 1:
                        would_uncover += 1
                if would_uncover == 0:
                    code.pop(idx)
                    code_set.remove(w_idx)
                    for nb in neighbors[w_idx]:
                        counts[nb] -= 1
                    size -= 1
                    improved = True
        return code, counts, covered, code_set, size
    
    # Simulated annealing
    def simulated_annealing(init_code, init_counts, init_covered, init_code_set, start_time, time_limit, seed):
        random.seed(seed)
        code = list(init_code)
        counts = list(init_counts)
        covered = init_covered
        size = len(code)
        code_set = set(init_code_set)
        
        P = 10.0
        def cost(sz, cov):
            return sz + P * (N - cov)
        
        current_cost = cost(size, covered)
        best_code = list(code)
        best_size = size
        best_covered = covered
        best_cost = current_cost
        
        T = 5.0
        T_min = 0.01
        cooling = 0.99992
        
        all_indices = list(range(N))
        
        # Periodic greedy reduction
        last_reduction = time.time()
        reduction_interval = 1.0
        
        # Track how long since improvement
        last_improvement = time.time()
        no_improve_timeout = 8.0
        
        while time.time() - start_time < time_limit:
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            
            # If no improvement for a while, reset temperature
            if time.time() - last_improvement > no_improve_timeout:
                T = max(T, 2.0)
                last_improvement = time.time()
            
            # Periodic greedy reduction: try to remove redundant codewords
            if time.time() - last_reduction > reduction_interval:
                last_reduction = time.time()
                code, counts, covered, code_set, size = try_greedy_reduction(
                    code, counts, covered, code_set, max_passes=5
                )
                current_cost = cost(size, covered)
                if covered == N and size < best_size:
                    best_size = size
                    best_covered = covered
                    best_code = list(code)
                    best_cost = current_cost
                    last_improvement = time.time()
            
            move_type = random.random()
            
            if move_type < 0.60:
                # Swap
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx_remove = random.randint(0, size - 1)
                w_remove = code[idx_remove]
                idx_w_remove = word_to_idx[w_remove]
                
                # Sample multiple candidates for addition and pick the best
                best_idx_add = -1
                best_delta = 0
                nbs_remove = neighbors[idx_w_remove]
                nbs_remove_set = neighbor_sets[idx_w_remove]
                
                for _ in range(8):
                    idx_add = random.choice(all_indices)
                    if idx_add == idx_w_remove or idx_add in code_set:
                        continue
                    nbs_add = neighbors[idx_add]
                    nbs_add_set = neighbor_sets[idx_add]
                    
                    # Calculate delta coverage
                    losing = 0
                    for nb in nbs_remove:
                        if counts[nb] == 1 and nb not in nbs_add_set:
                            losing += 1
                    gaining = 0
                    for nb in nbs_add:
                        if counts[nb] == 0:
                            gaining += 1
                    
                    delta = gaining - losing
                    if delta > best_delta:
                        best_delta = delta
                        best_idx_add = idx_add
                
                if best_idx_add == -1:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_add = neighbors[best_idx_add]
                nbs_add_set = neighbor_sets[best_idx_add]
                
                # Recalculate with the best candidate
                losing = 0
                for nb in nbs_remove:
                    if counts[nb] == 1 and nb not in nbs_add_set:
                        losing += 1
                gaining = 0
                for nb in nbs_add:
                    if counts[nb] == 0:
                        gaining += 1
                
                delta = gaining - losing
                new_covered = covered + delta
                new_cost = cost(size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code[idx_remove] = all_words[best_idx_add]
                    code_set.remove(idx_w_remove)
                    code_set.add(best_idx_add)
                    for nb in nbs_remove: counts[nb] -= 1
                    for nb in nbs_add: counts[nb] += 1
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                        last_improvement = time.time()
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    
            elif move_type < 0.85:
                # Remove
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx = random.randint(0, size - 1)
                w = code[idx]
                w_idx = word_to_idx[w]
                
                delta_covered = 0
                for nb in neighbors[w_idx]:
                    if counts[nb] == 1:
                        delta_covered -= 1
                
                new_size = size - 1
                new_covered = covered + delta_covered
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code.pop(idx)
                    code_set.remove(w_idx)
                    for nb in neighbors[w_idx]:
                        counts[nb] -= 1
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                        last_improvement = time.time()
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    
            else:
                # Add (only if not fully covered, or to explore)
                if covered == N:
                    # Convert to swap
                    if size > 0:
                        idx_remove = random.randint(0, size - 1)
                        idx_w_remove = word_to_idx[code[idx_remove]]
                        idx_add = random.choice(all_indices)
                        if idx_add == idx_w_remove or idx_add in code_set:
                            T *= cooling
                            if T < T_min: T = T_min
                            continue
                        nbs_remove = neighbors[idx_w_remove]
                        nbs_add = neighbors[idx_add]
                        nbs_remove_set = neighbor_sets[idx_w_remove]
                        nbs_add_set = neighbor_sets[idx_add]
                        
                        losing = 0
                        for nb in nbs_remove:
                            if counts[nb] == 1 and nb not in nbs_add_set:
                                losing += 1
                        gaining = 0
                        for nb in nbs_add:
                            if counts[nb] == 0:
                                gaining += 1
                        
                        delta = gaining - losing
                        new_covered = covered + delta
                        new_cost = cost(size, new_covered)
                        delta_cost = new_cost - current_cost
                        
                        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                            code[idx_remove] = all_words[idx_add]
                            code_set.remove(idx_w_remove)
                            code_set.add(idx_add)
                            for nb in nbs_remove: counts[nb] -= 1
                            for nb in nbs_add: counts[nb] += 1
                            covered = new_covered
                            current_cost = new_cost
                            
                            if covered == N and size < best_size:
                                best_size = size
                                best_covered = covered
                                best_code = list(code)
                                best_cost = current_cost
                                last_improvement = time.time()
                        T *= cooling
                        if T < T_min: T = T_min
                    continue
                
                # Pick word with best marginal gain from several candidates
                best_idx_add = -1
                best_gain = 0
                for _ in range(15):
                    idx_add = random.choice(all_indices)
                    if idx_add in code_set:
                        continue
                    gain = 0
                    for nb in neighbors[idx_add]:
                        if counts[nb] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx_add = idx_add
                
                if best_idx_add == -1:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_add = neighbors[best_idx_add]
                delta_covered = 0
                for nb in nbs_add:
                    if counts[nb] == 0:
                        delta_covered += 1
                        
                new_size = size + 1
                new_covered = covered + delta_covered
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code.append(all_words[best_idx_add])
                    code_set.add(best_idx_add)
                    for nb in nbs_add:
                        counts[nb] += 1
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                        last_improvement = time.time()
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
        
        # Final greedy reduction
        code, counts, covered, code_set, size = try_greedy_reduction(
            code, counts, covered, code_set, max_passes=20
        )
        if covered == N and size < best_size:
            best_size = size
            best_covered = covered
            best_code = list(code)
            best_cost = cost(size, covered)
        
        return best_code, best_size
    
    start_time = time.time()
    time_limit = 115.0
    
    # Multi-restart with different strategies
    best_overall = None
    best_overall_size = float('inf')
    
    strategies = ['max_gain', 'randomized', 'pairwise', 'max_gain']
    num_restarts = len(strategies)
    
    for r in range(num_restarts):
        t0 = time.time()
        remaining = time_limit - (t0 - start_time)
        if remaining < 10:
            break
        strategy = strategies[r % len(strategies)]
        seed = 42 + r * 7
        init_code, init_counts, init_covered, init_code_set = greedy_init(seed, strategy)
        t1 = time.time()
        sa_remaining = time_limit - (t1 - start_time)
        if sa_remaining < 5:
            break
        best_code, best_size = simulated_annealing(
            init_code, init_counts, init_covered, init_code_set,
            start_time, time_limit, 100 + r * 13
        )
        if best_size < best_overall_size:
            best_overall_size = best_size
            best_overall = best_code
    
    if best_overall is None:
        # Fallback
        init_code, _, _, _ = greedy_init(42, 'max_gain')
        best_overall = init_code
    
    return [list(w) for w in best_overall]
# EVOLVE-BLOCK-END
