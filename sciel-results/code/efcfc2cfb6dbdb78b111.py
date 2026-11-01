"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    N = 3**7
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor masks and indices
    neighbor_masks = [0] * N
    neighbor_indices = [[] for _ in range(N)]
    
    for i, w in enumerate(all_words):
        mask = 1 << i
        inds = [i]
        w_list = list(w)
        for pos in range(n):
            orig = w_list[pos]
            for val in range(3):
                if val != orig:
                    w_list[pos] = val
                    nb_idx = word_to_idx[tuple(w_list)]
                    mask |= (1 << nb_idx)
                    inds.append(nb_idx)
            w_list[pos] = orig
        neighbor_masks[i] = mask
        neighbor_indices[i] = inds

    def greedy_init(seed):
        random.seed(seed)
        code = []
        code_set = set()
        covered_mask = 0
        uncovered_count = N
        
        indices = list(range(N))
        random.shuffle(indices)
        
        while uncovered_count > 0:
            best_idx = -1
            best_gain = 0
            
            for idx in indices:
                if idx in code_set:
                    continue
                gain = (neighbor_masks[idx] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            code.append(best_idx)
            code_set.add(best_idx)
            covered_mask |= neighbor_masks[best_idx]
            uncovered_count -= best_gain
            
        return code, covered_mask, len(code)

    def simulated_annealing(init_code, init_counts, init_covered, start_time, time_limit, seed):
        random.seed(seed)
        code = list(init_code)
        counts = list(init_counts)
        covered = init_covered
        size = len(code)
        code_set = set(code)
        
        P = 20.0
        def cost(sz, cov):
            return sz + P * (N - cov)
        
        current_cost = cost(size, covered)
        best_code = list(code)
        best_size = size
        best_covered = covered
        best_cost = current_cost
        
        T = 10.0
        T_min = 0.1
        cooling = 0.9995
        
        all_indices = list(range(N))
        
        while time.time() - start_time < time_limit:
            move_type = random.random()
            
            if move_type < 0.7:
                # Swap
                if size == 0:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                idx_remove = random.randint(0, size - 1)
                idx_add = random.choice(all_indices)
                
                if idx_add == code[idx_remove] or idx_add in code_set:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_remove = neighbor_indices[code[idx_remove]]
                nbs_add = neighbor_indices[idx_add]
                
                # Calculate delta
                delta = 0
                # Losses
                for nb in nbs_remove:
                    if counts[nb] == 1:
                        found = False
                        for nb2 in nbs_add:
                            if nb2 == nb:
                                found = True
                                break
                        if not found:
                            delta -= 1
                # Gains
                for nb in nbs_add:
                    if counts[nb] == 0:
                        delta += 1
                
                new_covered = covered + delta
                new_cost = cost(size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    old_val = code[idx_remove]
                    code[idx_remove] = idx_add
                    code_set.remove(old_val)
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
                w_idx = code[idx]
                nbs_remove = neighbor_indices[w_idx]
                
                delta = 0
                for nb in nbs_remove:
                    if counts[nb] == 1:
                        delta -= 1
                
                new_size = size - 1
                new_covered = covered + delta
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code.pop(idx)
                    code_set.remove(w_idx)
                    for nb in nbs_remove: counts[nb] -= 1
                    size = new_size
                    covered = new_covered
                    current_cost = new_cost
                    
                    if covered == N and size < best_size:
                        best_size = size
                        best_covered = covered
                        best_code = list(code)
                        best_cost = current_cost
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
                    
            else:
                # Add
                if covered == N:
                    # Try swap instead
                    if size > 0:
                        idx_remove = random.randint(0, size - 1)
                        idx_add = random.choice(all_indices)
                        if idx_add == code[idx_remove] or idx_add in code_set:
                            T *= cooling
                            if T < T_min: T = T_min
                            continue
                        nbs_remove = neighbor_indices[code[idx_remove]]
                        nbs_add = neighbor_indices[idx_add]
                        
                        delta = 0
                        for nb in nbs_remove:
                            if counts[nb] == 1:
                                found = False
                                for nb2 in nbs_add:
                                    if nb2 == nb:
                                        found = True
                                        break
                                if not found:
                                    delta -= 1
                        for nb in nbs_add:
                            if counts[nb] == 0:
                                delta += 1
                        
                        new_covered = covered + delta
                        new_cost = cost(size, new_covered)
                        delta_cost = new_cost - current_cost
                        
                        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                            old_val = code[idx_remove]
                            code[idx_remove] = idx_add
                            code_set.remove(old_val)
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
                        T *= cooling
                        if T < T_min: T = T_min
                    continue
                
                # Pick best add candidate
                best_idx_add = -1
                best_gain = 0
                for _ in range(10):
                    idx_add = random.choice(all_indices)
                    if idx_add in code_set:
                        continue
                    gain = 0
                    for nb in neighbor_indices[idx_add]:
                        if counts[nb] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx_add = idx_add
                
                if best_idx_add == -1:
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                
                nbs_add = neighbor_indices[best_idx_add]
                delta = 0
                for nb in nbs_add:
                    if counts[nb] == 0:
                        delta += 1
                        
                new_size = size + 1
                new_covered = covered + delta
                new_cost = cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    code.append(best_idx_add)
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
                    
                    T *= cooling
                    if T < T_min: T = T_min
                else:
                    T *= cooling
                    if T < T_min: T = T_min
        
        return best_code, best_size

    start_time = time.time()
    time_limit = 115.0
    
    best_overall = None
    best_overall_size = float('inf')
    
    # Restart 1
    code, covered_mask, size = greedy_init(42)
    counts = [0] * N
    covered = 0
    for idx in code:
        for nb in neighbor_indices[idx]:
            if counts[nb] == 0:
                covered += 1
            counts[nb] += 1
            
    best_code, best_size = simulated_annealing(
        code, counts, covered, start_time, time_limit, 100
    )
    
    if best_size < best_overall_size:
        best_overall_size = best_size
        best_overall = best_code
        
    # Restart 2
    if time.time() - start_time < 100:
        code2, covered_mask2, size2 = greedy_init(43)
        counts2 = [0] * N
        covered2 = 0
        for idx in code2:
            for nb in neighbor_indices[idx]:
                if counts2[nb] == 0:
                    covered2 += 1
                counts2[nb] += 1
                
        best_code2, best_size2 = simulated_annealing(
            code2, counts2, covered2, start_time, time_limit, 101
        )
        
        if best_size2 < best_overall_size:
            best_overall_size = best_size2
            best_overall = best_code2

    if best_overall is None:
        best_overall = code
        
    return [list(all_words[i]) for i in best_overall]
# EVOLVE-BLOCK-END
