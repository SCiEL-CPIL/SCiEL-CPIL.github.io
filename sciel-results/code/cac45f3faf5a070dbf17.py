"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math
import heapq

def construct(q, n, radius):
    # For the specific task: q=3, n=7, radius=1
    # We use a refined Simulated Annealing approach with multi-restart.
    
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
    
    # Convert neighbors to sets for faster membership tests
    neighbors_set = [set(n) for n in neighbors]
    
    random.seed(42)
    
    start_time = time.time()
    time_limit = 115.0
    
    best_overall_size = float('inf')
    best_overall_code = None
    
    # Multi-restart simulated annealing
    num_restarts = 5
    time_per_restart = time_limit / num_restarts
    
    for restart in range(num_restarts):
        # 1. Greedy Initialization: Max Coverage Greedy
        counts = [0] * 2187
        covered = 0
        code = []
        code_set = set()
        uncovered_set = set(range(2187))
        
        while uncovered_set:
            best_word_idx = -1
            max_cov = 0
            
            for w_idx in range(2187):
                if w_idx in code_set:
                    continue
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
            code_set.add(best_word_idx)
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
        current_code_set = set(code_set)
        
        P = 10.0  # Penalty for uncovered words
        
        def get_cost(size, covered):
            return size + P * (2187 - covered)
            
        current_cost = get_cost(current_size, current_covered)
        
        best_code = list(current_code)
        best_size = current_size
        best_covered = current_covered
        best_cost = current_cost
        
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        
        restart_start = time.time()
        
        while time.time() - restart_start < time_per_restart:
            # Maintain uncovered list for biased moves
            # Rebuild occasionally or track incrementally
            uncovered_list = []
            if current_covered < 2187:
                for i in range(2187):
                    if current_counts[i] == 0:
                        uncovered_list.append(i)
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Swap: Remove one, Add one
                if current_size == 0:
                    continue
                idx_remove = random.randint(0, current_size - 1)
                w_remove = current_code[idx_remove]
                idx_w_remove = word_to_idx[w_remove]
                
                # Pick a word to add - biased towards covering uncovered spots
                if uncovered_list and random.random() < 0.7:
                    # Pick a random uncovered word, then pick a random neighbor of it
                    unc_word = random.choice(uncovered_list)
                    # Find all words that cover unc_word (i.e., neighbors of unc_word)
                    # Actually, we want words w such that unc_word is in neighbors[w]
                    # This is the reverse neighbor list. Let's precompute or just pick randomly.
                    # For simplicity, pick a random word that is a neighbor of some uncovered word
                    idx_add = random.randint(0, 2186)
                else:
                    idx_add = random.randint(0, 2186)
                    
                if idx_add == idx_w_remove or idx_add in current_code_set:
                    continue
                    
                w_add = all_words[idx_add]
                
                # Delta calculation
                nbs_remove = neighbors[idx_w_remove]
                nbs_add = neighbors[idx_add]
                nbs_remove_set = neighbors_set[idx_w_remove]
                nbs_add_set = neighbors_set[idx_add]
                
                # Affected words are in the union of both neighborhoods
                affected = nbs_remove_set | nbs_add_set
                
                delta = 0
                for nb in affected:
                    old_count = current_counts[nb]
                    new_count = old_count
                    if nb in nbs_remove_set:
                        new_count -= 1
                    if nb in nbs_add_set:
                        new_count += 1
                    if old_count == 0 and new_count > 0:
                        delta += 1
                    elif old_count > 0 and new_count == 0:
                        delta -= 1
                    
                new_covered = current_covered + delta
                new_cost = get_cost(current_size, new_covered)
                delta_cost = new_cost - current_cost
                
                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    # Execute move
                    current_code[idx_remove] = w_add
                    current_code_set.remove(idx_w_remove)
                    current_code_set.add(idx_add)
                    for nb in nbs_remove:
                        current_counts[nb] -= 1
                    for nb in nbs_add:
                        current_counts[nb] += 1
                    current_covered = new_covered
                    current_cost = new_cost
                    
                    if current_covered == 2187 and current_size < best_size:
                        best_size = current_size
                        best_covered = current_covered
                        best_code = list(current_code)
                        best_cost = current_cost
                        if current_size < best_overall_size:
                            best_overall_size = current_size
                            best_overall_code = list(current_code)
                    
                    T *= cooling
                    if T < T_min:
                        T = T_min
                    continue
            elif move_type < 0.85:
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
                    current_code_set.remove(w_idx)
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
                        if current_size < best_overall_size:
                            best_overall_size = current_size
                            best_overall_code = list(current_code)
                    
                    T *= cooling
                    if T < T_min:
                        T = T_min
                    continue
            else:
                # Add (only if not fully covered)
                if current_covered == 2187:
                    # If fully covered, prefer swap instead
                    # Just continue to next iteration
                    T *= cooling
                    if T < T_min:
                        T = T_min
                    continue
                    
                # Pick a word that covers uncovered spots
                if uncovered_list and random.random() < 0.8:
                    # Pick a random uncovered word, then try to find a good word to cover it
                    unc_word = random.choice(uncovered_list)
                    # Pick a random word that is a neighbor of unc_word
                    # We need the reverse: words w such that unc_word in neighbors[w]
                    # For simplicity, pick randomly from all words
                    idx_add = random.randint(0, 2186)
                else:
                    idx_add = random.randint(0, 2186)
                
                if idx_add in current_code_set:
                    continue
                    
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
                    current_code_set.add(idx_add)
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
                        if current_size < best_overall_size:
                            best_overall_size = current_size
                            best_overall_code = list(current_code)
                    
                    T *= cooling
                    if T < T_min:
                        T = T_min
                    continue

    if best_overall_code is not None:
        return [list(w) for w in best_overall_code]
    else:
        # Should not happen, but fallback
        return [list(w) for w in best_code]
# EVOLVE-BLOCK-END
