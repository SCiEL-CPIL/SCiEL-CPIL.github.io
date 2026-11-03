import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    # Leave a safety margin of 5 seconds for cleanup and return
    deadline = start_time + 115.0
    
    total_words = q ** n
    if total_words == 1:
        return [[0] * n]
    
    # Precompute neighbors
    # For q=3, n=8, radius=1: n_neighbors = 1 + 8 * 2 = 17
    n_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)
    
    # Compute powers for index calculation
    q_powers = [q ** i for i in range(n)]
    
    # Build neighbor table
    for w in range(total_words):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            digit = (w // q_powers[n - 1 - pos]) % q
            for v in range(q):
                if v == digit:
                    continue
                diff = (v - digit) * q_powers[n - 1 - pos]
                neighbors[w, idx] = w + diff
                idx += 1

    # Precompute neighbor sets as Python sets for fast membership testing
    neighbor_sets = [set(neighbors[w].tolist()) for w in range(total_words)]
    
    def greedy_init(seed_offset=0):
        rng = random.Random(seed_offset)
        covered = np.zeros(total_words, dtype=bool)
        gain = np.full(total_words, n_neighbors, dtype=np.int32)
        codewords = []
        remaining = total_words
        
        while remaining > 0:
            if time.time() > deadline - 30.0:
                break
            best_word = int(np.argmax(gain))
            best_gain = gain[best_word]
            
            if best_gain == 0:
                uncovered = np.where(~covered)[0]
                if len(uncovered) == 0:
                    break
                best_word = int(uncovered[0])
            
            codewords.append(best_word)
            
            for v in neighbors[best_word]:
                if not covered[v]:
                    covered[v] = True
                    remaining -= 1
                    for w in neighbors[v]:
                        gain[w] -= 1
        
        return codewords
    
    def local_search(cw, time_limit):
        cw = cw[:]
        best_cw = cw[:]
        best_size = len(cw)
        
        count = np.zeros(total_words, dtype=np.int32)
        in_code = np.zeros(total_words, dtype=bool)
        
        for ci in cw:
            in_code[ci] = True
            for u in neighbors[ci]:
                count[u] += 1
        
        # Precompute for each word, which codewords cover it (for faster swap evaluation)
        # Actually, let's just do iterative local search with vectorized checks
        
        while time.time() < time_limit:
            improved = False
            
            # Pass 1: Remove redundant codewords (vectorized min check)
            indices = list(range(len(cw)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(cw):
                    continue
                ci = cw[i]
                
                # Check if all neighbors have count > 1
                if np.min(count[neighbors[ci]]) > 1:
                    for u in neighbors[ci]:
                        count[u] -= 1
                    in_code[ci] = False
                    cw.pop(i)
                    improved = True
                    
                    if len(cw) < best_size:
                        best_size = len(cw)
                        best_cw = cw[:]
            
            # Pass 2: Swaps
            if not improved:
                indices = list(range(len(cw)))
                random.shuffle(indices)
                
                for i in indices:
                    if i >= len(cw):
                        continue
                    ci = cw[i]
                    
                    critical_mask = count[neighbors[ci]] == 1
                    critical = neighbors[ci][critical_mask]
                    
                    if len(critical) == 0:
                        for u in neighbors[ci]:
                            count[u] -= 1
                        in_code[ci] = False
                        cw.pop(i)
                        improved = True
                        break
                    
                    if len(critical) > 0:
                        first_crit = critical[0]
                        
                        if len(critical) == 1:
                            valid_cands = []
                            for cand in neighbor_sets[first_crit]:
                                if cand == ci or in_code[cand]:
                                    continue
                                valid_cands.append(cand)
                            
                            if valid_cands:
                                cand = random.choice(valid_cands)
                                for u in neighbors[ci]:
                                    count[u] -= 1
                                for u in neighbors[cand]:
                                    count[u] += 1
                                
                                in_code[ci] = False
                                in_code[cand] = True
                                cw[i] = cand
                                improved = True
                                break
                        else:
                            # Multiple critical: need candidate in intersection of all critical neighbors' neighbor sets
                            # Start with candidates from first critical
                            candidate_set = neighbor_sets[first_crit].copy()
                            for u in critical[1:]:
                                candidate_set &= neighbor_sets[u]
                                if not candidate_set:
                                    break
                            
                            if candidate_set:
                                valid_cands = [c for c in candidate_set if c != ci and not in_code[c]]
                                if valid_cands:
                                    cand = random.choice(valid_cands)
                                    for u in neighbors[ci]:
                                        count[u] -= 1
                                    for u in neighbors[cand]:
                                        count[u] += 1
                                    
                                    in_code[ci] = False
                                    in_code[cand] = True
                                    cw[i] = cand
                                    improved = True
                                    break
            
            if not improved:
                # Try a perturbation: randomly swap a few codewords to escape local minimum
                if len(cw) > 2 and random.random() < 0.3:
                    idx1, idx2 = random.sample(range(len(cw)), 2)
                    c1, c2 = cw[idx1], cw[idx2]
                    
                    # Check if swapping maintains coverage
                    # Remove c1, add c2, check all still covered
                    temp_count = count.copy()
                    for u in neighbors[c1]:
                        temp_count[u] -= 1
                    for u in neighbors[c2]:
                        temp_count[u] += 1
                    
                    if np.min(temp_count) >= 1:
                        # Perform the swap
                        for u in neighbors[c1]:
                            count[u] -= 1
                        for u in neighbors[c2]:
                            count[u] += 1
                        in_code[c1] = False
                        in_code[c2] = True
                        cw[idx1] = c2
                        improved = True
                        
                        if len(cw) < best_size:
                            best_size = len(cw)
                            best_cw = cw[:]
            
            # If we've been stuck for a while, try a more aggressive perturbation
            if not improved and random.random() < 0.1:
                # Remove a random codeword and re-add a random one that helps
                if len(cw) > 5:
                    idx = random.randint(0, len(cw) - 1)
                    c_remove = cw[idx]
                    
                    # Find a random word that is currently covered
                    covered_words = np.where(count > 0)[0]
                    if len(covered_words) > 0:
                        c_add = int(covered_words[random.randint(0, len(covered_words) - 1)])
                        
                        # Check if this swap maintains coverage
                        temp_count = count.copy()
                        for u in neighbors[c_remove]:
                            temp_count[u] -= 1
                        for u in neighbors[c_add]:
                            temp_count[u] += 1
                        
                        if np.min(temp_count) >= 1:
                            for u in neighbors[c_remove]:
                                count[u] -= 1
                            for u in neighbors[c_add]:
                                count[u] += 1
                            in_code[c_remove] = False
                            in_code[c_add] = True
                            cw[idx] = c_add
                            improved = True
        
        return best_cw
    
    # Phase 1: Greedy Set Cover
    codewords = greedy_init(42)
    
    # Phase 2: Local Search
    best_cw = local_search(codewords, deadline - 10.0)
    
    # Convert back to list of lists
    result = []
    for ci in best_cw:
        word = []
        val = ci
        for pos in range(n - 1, -1, -1):
            word.append(val % q)
            val //= q
        word.reverse()
        result.append(word)
    
    return result
