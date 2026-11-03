import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4:
            return tetracode
        if n > 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return []
        
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = [[] for _ in range(729)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(6):
            for alt in range(3):
                if w[pos] == alt:
                    continue
                new_w = list(w)
                new_w[pos] = alt
                new_w = tuple(new_w)
                j = word_to_idx[new_w]
                neighbors[i].append(j)
                
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    best_size = len(initial_codewords)
    best_code = initial_codewords
    
    def find_swap_candidates(active, cov_counts, c):
        uncovered = []
        for w in neighbors[c]:
            if cov_counts[w] == 1:
                uncovered.append(w)
                
        if not uncovered:
            return []
            
        first_u = uncovered[0]
        # Candidates must be neighbors of first_u and not in active
        potential_candidates = [w_prime for w_prime in neighbors[first_u] if w_prime not in active]
        
        valid_candidates = []
        for w_prime in potential_candidates:
            ok = True
            for u in uncovered[1:]:
                # Check if w_prime is in neighbors[u]
                # Since neighbors[u] is a list, we can use 'in'
                if w_prime not in neighbors[u]:
                    ok = False
                    break
            if ok:
                valid_candidates.append(w_prime)
                
        return valid_candidates
        
    def evaluate_and_reduce(initial_set):
        active = set(initial_set)
        cov_counts = [0] * 729
        
        for c in active:
            for w in neighbors[c]:
                cov_counts[w] += 1
                
        improved = True
        swaps_since_removal = 0
        while improved and swaps_since_removal < 50:
            improved = False
            
            # 1. Try removals
            removable = []
            for c in active:
                is_rem = True
                for w in neighbors[c]:
                    if cov_counts[w] == 1:
                        is_rem = False
                        break
                if is_rem:
                    removable.append(c)
                    
            if removable:
                c_to_remove = random.choice(removable)
                active.remove(c_to_remove)
                for w in neighbors[c_to_remove]:
                    cov_counts[w] -= 1
                improved = True
                swaps_since_removal = 0
                continue
                
            # 2. Try swaps if no removal
            if active: # Ensure active is not empty
                c_to_swap = random.choice(list(active))
                candidates = find_swap_candidates(active, cov_counts, c_to_swap)
                
                if candidates:
                    w_new = random.choice(candidates)
                    
                    # Perform swap
                    for w in neighbors[c_to_swap]:
                        cov_counts[w] -= 1
                    active.remove(c_to_swap)
                    
                    active.add(w_new)
                    for w in neighbors[w_new]:
                        cov_counts[w] += 1
                    
                    improved = True
                    swaps_since_removal += 1
                    continue
                    
            # If no removal and no swap, we are stuck
            break
            
        return active
        
    restarts = 0
    while time.time() < deadline:
        current_active = evaluate_and_reduce(initial_codewords)
        
        if len(current_active) < best_size:
            best_size = len(current_active)
            best_code = [all_words[i] for i in current_active]
            
        restarts += 1
        
    return [list(c) for c in best_code]
