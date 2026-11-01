"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

def construct(q, n, radius):
    # For the specific task: q=3, n=7, radius=1
    # We use a refined Simulated Annealing approach.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to direct sum for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Precomputation for q=3, n=7, r=1
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # neighbors[i] is a list of indices of words within distance 1 of all_words[i]
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
        neighbors.append(idxs)
        
    random.seed(42)
    
    # 1. Greedy Initialization: Max Coverage Greedy
    counts = [0] * 2187
    covered = 0
    code = []
    uncovered_set = set(range(2187))
    
    while uncovered_set:
        best_word_idx = -1
        max_cov = 0
        
        for w_idx in range(2187):
            cov = 0
            for nb in neighbors[w_idx]:
                if nb in uncovered_set:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_word_idx = w_idx
        
        if best_word_idx == -1:
            break
            
        code.append(all_words[best_word_idx])
        for nb in neighbors[best_word_idx]:
            if counts[nb] == 0:
                covered += 1
                uncovered_set.remove(nb)
            counts[nb] += 1
            
    # Start Simulated Annealing
    current_code = code
    current_counts = counts
    current_covered = covered
    current_size = len(current_code)
    
    P = 10.0  # Penalty for uncovered words
    
    def get_cost(size, covered):
        return size + P * (2187 - covered)
        
    current_cost = get_cost(current_size, current_covered)
    
    best_code = list(current_code)
    best_size = current_size
    best_covered = current_covered
    best_cost = current_cost
    
    start_time = time.time()
    time_limit = 115.0
    
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        move_type = random.random()
        
        if move_type < 0.4:
            # Swap: Remove one, Add one
            if current_size == 0:
                continue
            idx_remove = random.randint(0, current_size - 1)
            w_remove = current_code[idx_remove]
            idx_w_remove = word_to_idx[w_remove]
            
            idx_add = random.randint(0, 2186)
            
            if idx_add == idx_w_remove:
                continue
                
            w_add = all_words[idx_add]
            
            nbs_remove = neighbors[idx_w_remove]
            nbs_add = neighbors[idx_add]
            
            delta_covered = 0
            for nb in nbs_remove:
                if current_counts[nb] == 1:
                    delta_covered -= 1
            for nb in nbs_add:
                if current_counts[nb] == 0:
                    delta_covered += 1
            
            new_size = current_size
            new_covered = current_covered + delta_covered
            new_cost = get_cost(new_size, new_covered)
            delta_cost = new_cost - current_cost
            
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                current_code[idx_remove] = w_add
                for nb in nbs_remove: current_counts[nb] -= 1
                for nb in nbs_add: current_counts[nb] += 1
                current_covered = new_covered
                current_cost = new_cost
                
                if current_covered == 2187 and current_size < best_size:
                    best_size = current_size
                    best_covered = current_covered
                    best_code = list(current_code)
                    best_cost = current_cost
                    
                T *= cooling
                if T < T_min: T = T_min
                continue
            else:
                T *= cooling
                if T < T_min: T = T_min
                continue

        elif move_type < 0.8:
            # Remove
            if current_size == 0:
                continue
            idx = random.randint(0, current_size - 1)
            w = current_code[idx]
            w_idx = word_to_idx[w]
            
            delta_covered = 0
            for nb in neighbors[w_idx]:
                if current_counts[nb] == 1:
                    delta_covered -= 1
            
            new_size = current_size - 1
            new_covered = current_covered + delta_covered
            new_cost = get_cost(new_size, new_covered)
            delta_cost = new_cost - current_cost
            
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                current_code.pop(idx)
                for nb in neighbors[w_idx]:
                    current_counts[nb] -= 1
                current_size = new_size
                current_covered = new_covered
                current_cost = new_cost
                
                if current_covered == 2187 and current_size < best_size:
                    best_size = current_size
                    best_covered = current_covered
                    best_code = list(current_code)
                    best_cost = current_cost
                    
                T *= cooling
                if T < T_min: T = T_min
                continue
            else:
                T *= cooling
                if T < T_min: T = T_min
                continue

        else:
            # Add
            if current_covered == 2187:
                pass
            
            if current_covered < 2187:
                idx_add = random.randint(0, 2186)
                w_add = all_words[idx_add]
                
                nbs_add = neighbors[idx_add]
                delta_covered = 0
                for nb in nbs_add:
                    if current_counts[nb] == 0:
                        delta_covered += 1
                        
                new_size = current_size + 1
                new_covered = current_covered + delta_covered
                new_cost = get_cost(new_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    current_code.append(w_add)
                    for nb in nbs_add:
                        current_counts[nb] += 1
                    current_size = new_size
                    current_covered = new_covered
                    current_cost = new_cost
                    
                    if current_covered == 2187 and current_size < best_size:
                        best_size = current_size
                        best_covered = current_covered
                        best_code = list(current_code)
                        best_cost = current_cost
                        
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
            else:
                pass
            
            T *= cooling
            if T < T_min: T = T_min
            continue

    return [list(w) for w in best_code]
# EVOLVE-BLOCK-END
