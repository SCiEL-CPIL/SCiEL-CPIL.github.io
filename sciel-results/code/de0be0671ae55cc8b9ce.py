import time
import numpy as np
from itertools import product
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q != 3 or n != 8 or radius != 1:
        # Fallback for general parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n-4)]
        else:
            return [list(w) for w in product(range(q), repeat=n)]

    total_words = q ** n  # 6561
    num_neighbors = 1 + n * (q - 1)  # 17

    # Precompute neighbors as a flat array
    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)
    pow3 = [3 ** (n - 1 - i) for i in range(n)]

    for i in range(total_words):
        temp = i
        word = [0] * n
        for j in range(n - 1, -1, -1):
            word[j] = temp % q
            temp //= q

        neighbors[i, 0] = i
        nb_idx = 1
        for pos in range(n):
            orig_val = word[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                delta = (new_val - orig_val) * pow3[pos]
                neighbors[i, nb_idx] = i + delta
                nb_idx += 1

    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for j in range(n - 1, -1, -1):
            w[j] = temp % q
            temp //= q
        return w

    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        uncovered = np.ones(total_words, dtype=bool)
        selected_indices = []
        
        # Greedy: pick word covering most uncovered
        # To allow randomization, we can add small noise or pick randomly among top-k
        # Here we stick to deterministic greedy for stability, but random restarts help.
        
        while time.time() < deadline - 20.0: # Reserve 20s for local search
            counts = uncovered[neighbors].sum(axis=1)
            best_idx = np.argmax(counts)
            best_count = counts[best_idx]
            
            if best_count == 0:
                break
                
            selected_indices.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            
            if not uncovered.any():
                break
                
        return selected_indices

    def local_search(initial_indices):
        # initial_indices: list of int
        if not initial_indices:
            return []
            
        # Use a set for O(1) lookup
        code_set = set(initial_indices)
        # We need to iterate over code_set, so we might need a list version or just iterate set
        # But we need to update it.
        
        # Coverage counts
        # Initialize counts
        counts = np.zeros(total_words, dtype=np.int32)
        for idx in code_set:
            counts[neighbors[idx]] += 1
            
        improved = True
        no_improvement_rounds = 0
        max_no_improvement = 100
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try to remove redundant codewords
            # A codeword c is removable if all its neighbors have count > 1
            # i.e., min(counts[neighbors[c]]) > 1
            
            removable = []
            for c in code_set:
                if np.min(counts[neighbors[c]]) > 1:
                    removable.append(c)
            
            if removable:
                # Remove one (or all? Removing one changes counts, so we should remove one and restart loop or process carefully)
                # Removing one at a time is safer for correctness of counts
                c_to_remove = random.choice(removable) if len(removable) > 1 else removable[0]
                code_set.remove(c_to_remove)
                counts[neighbors[c_to_remove]] -= 1
                improved = True
                no_improvement_rounds = 0
                continue # Restart loop to re-evaluate removals
            
            # 2. If no removals, try 1-swap
            # Find codeword c with smallest number of "unique" neighbors (count == 1)
            # Let unique_count(c) = sum(counts[neighbors[c]] == 1)
            
            best_c = None
            best_unique_count = float('inf')
            
            # To speed up, we can sample if code_set is large, but 400-500 is small enough to scan
            for c in code_set:
                uc = np.sum(counts[neighbors[c]] == 1)
                if uc < best_unique_count:
                    best_unique_count = uc
                    best_c = c
                    if uc == 1: # Can't do better than 1 (0 would be removable)
                        break
            
            if best_c is None:
                break
                
            # If best_unique_count is very high, swap is unlikely to help or findable
            # But let's try if best_unique_count <= 3 (heuristic limit for intersection search)
            if best_unique_count > 3:
                no_improvement_rounds += 1
                if no_improvement_rounds > max_no_improvement:
                    break
                continue
                
            # Get critical neighbors of best_c
            crit_neighbors = neighbors[best_c][counts[neighbors[best_c]] == 1]
            
            # Find candidate c' in intersection of N(w) for w in crit_neighbors
            # N(w) is neighbors[w]
            
            if len(crit_neighbors) == 1:
                w = crit_neighbors[0]
                candidates = neighbors[w]
            elif len(crit_neighbors) == 2:
                w1, w2 = crit_neighbors
                # Intersection of two arrays of size 17
                set1 = set(neighbors[w1])
                candidates = [x for x in neighbors[w2] if x in set1]
            elif len(crit_neighbors) == 3:
                w1, w2, w3 = crit_neighbors
                set1 = set(neighbors[w1])
                set2 = set(neighbors[w2])
                candidates = [x for x in neighbors[w3] if x in set1 and x in set2]
            else:
                candidates = []
                
            # Filter candidates: must not be in code_set
            valid_candidates = [c_new for c_new in candidates if c_new not in code_set]
            
            if valid_candidates:
                # Pick one (random or first)
                c_new = random.choice(valid_candidates)
                
                # Perform swap
                code_set.remove(best_c)
                code_set.add(c_new)
                
                # Update counts
                counts[neighbors[best_c]] -= 1
                counts[neighbors[c_new]] += 1
                
                improved = True
                no_improvement_rounds = 0
                continue # Restart loop to check for new removals
            
            # No valid swap found
            no_improvement_rounds += 1
            if no_improvement_rounds > max_no_improvement:
                break
                
        return list(code_set)

    # Main Strategy:
    # 1. Run greedy to get a baseline
    # 2. Run local search
    # 3. Try a few random restarts of greedy + local search if time permits
    
    best_code = []
    
    # First run: Standard Greedy
    init = greedy_construct(seed=42)
    refined = local_search(init)
    best_code = refined
    
    # Check time
    if time.time() < deadline - 30.0:
        # Try a few random restarts
        for seed in [1, 2, 3]:
            if time.time() > deadline - 15.0:
                break
            init = greedy_construct(seed=seed)
            # Quick greedy might be slightly different if we introduce randomness in tie-breaking
            # For now, just run local search on the new start
            refined = local_search(init)
            if len(refined) < len(best_code):
                best_code = refined
                
    # Final conversion to list of words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
        
    return result
