import itertools
import random
import time
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Specific case for the task: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)
    
    # Fallback for other parameters (simple product construction)
    if n <= 0:
        return [[]]
    
    # Generic greedy set cover for other cases
    all_words = list(itertools.product(range(q), repeat=n))
    space_size = q ** n
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = [[] for _ in range(space_size)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for j in range(n):
            for val in range(q):
                if val == w[j]:
                    continue
                new_w = list(w)
                new_w[j] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors[i].append(idx)
    
    # Greedy set cover
    covered = [False] * space_size
    code = []
    remaining = set(range(space_size))
    
    while remaining and time.time() < deadline - 5.0:
        best_idx = -1
        best_count = -1
        for idx in list(remaining):
            if not covered[idx]:
                count = sum(1 for w in neighbors[idx] if not covered[w])
                if count > best_count:
                    best_count = count
                    best_idx = idx
        if best_idx == -1:
            break
        code.append(all_words[best_idx])
        for w in neighbors[best_idx]:
            covered[w] = True
        remaining.discard(best_idx)
    
    return code

def _solve_q3_n6_r1(deadline):
    n = 6
    q = 3
    space_size = 729
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (ball of radius 1) for each word
    neighbors = [[] for _ in range(space_size)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for j in range(n):
            for val in range(q):
                if val == w[j]:
                    continue
                new_w = list(w)
                new_w[j] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors[i].append(idx)
    
    # Convert neighbors to sets for fast membership testing
    neighbor_sets = [set(nb) for nb in neighbors]
    
    # Initial code: Tetracode product (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_code.append(h + list(tail))
    initial_indices = [word_to_idx[tuple(c)] for c in initial_code]
    
    best_code = initial_code
    best_size = len(best_code)
    
    rng = random.Random(42)
    
    def get_coverage(indices):
        cover_count = [0] * space_size
        for idx in indices:
            for w_idx in neighbors[idx]:
                cover_count[w_idx] += 1
        return cover_count
    
    def local_search(indices_set, deadline):
        """Local search with 2-for-1 swaps and single removals."""
        current = set(indices_set)
        cover_count = get_coverage(current)
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # 1. Try to remove a single codeword
            for idx in list(current):
                removable = True
                for w in neighbors[idx]:
                    if cover_count[w] == 1:
                        removable = False
                        break
                if removable:
                    current.remove(idx)
                    for w in neighbors[idx]:
                        cover_count[w] -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try 2-for-1 swap
            indices_list = list(current)
            L = len(indices_list)
            found_swap = False
            
            for a in range(L):
                i = indices_list[a]
                u_i = [w for w in neighbors[i] if cover_count[w] == 1]
                
                for b in range(a + 1, L):
                    j = indices_list[b]
                    u_j = [w for w in neighbors[j] if cover_count[w] == 1]
                    u_union = u_i + u_j
                    
                    if not u_union:
                        continue
                    
                    # Find k not in current that covers u_union
                    for k in range(space_size):
                        if k in current:
                            continue
                        
                        covers = True
                        for w in u_union:
                            if w not in neighbor_sets[k]:
                                covers = False
                                break
                        
                        if covers:
                            current.remove(i)
                            current.remove(j)
                            current.add(k)
                            
                            for w in neighbors[i]:
                                cover_count[w] -= 1
                            for w in neighbors[j]:
                                cover_count[w] -= 1
                            for w in neighbors[k]:
                                cover_count[w] += 1
                            
                            found_swap = True
                            improved = True
                            break
                    
                    if found_swap:
                        break
                if found_swap:
                    break
        
        return current, cover_count
    
    def greedy_repair(indices_set, deadline):
        """Add codewords via weighted greedy until valid cover."""
        current = set(indices_set)
        cover_count = get_coverage(current)
        
        # Find uncovered words
        uncovered = [w for w in range(space_size) if cover_count[w] == 0]
        
        while uncovered and time.time() < deadline - 0.5:
            # Weighted greedy: pick codeword covering most uncovered words
            best_gain = -1
            best_candidate = -1
            
            for cand in range(space_size):
                if cand in current:
                    continue
                gain = 0
                for w in neighbors[cand]:
                    if cover_count[w] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = cand
            
            if best_gain <= 0:
                break
            
            current.add(best_candidate)
            for w in neighbors[best_candidate]:
                cover_count[w] += 1
            
            # Recompute uncovered
            uncovered = [w for w in range(space_size) if cover_count[w] == 0]
        
        return current, cover_count
    
    # Run initial local search
    if time.time() < deadline - 5.0:
        current_indices, cover_count = local_search(set(initial_indices), deadline)
        if len(current_indices) < best_size:
            best_size = len(current_indices)
            best_code = [all_words[i] for i in current_indices]
    
    # Iterated local search with perturbation
    for trial in range(50):
        if time.time() > deadline - 2.0:
            break
        
        # Start from current best
        start_set = set([all_words.index(tuple(c)) if tuple(c) in word_to_idx else word_to_idx[tuple(c)] for c in best_code])
        # Actually, let's just use the indices directly
        start_indices = list(best_code)
        start_set = set()
        for c in best_code:
            start_set.add(word_to_idx[tuple(c)])
        
        # Perturbation: remove k random codewords
        k = rng.randint(2, 5)
        perturbed = set(start_set)
        to_remove = rng.sample(list(perturbed), min(k, len(perturbed)))
        for idx in to_remove:
            perturbed.remove(idx)
        
        # Repair: add codewords via greedy until valid
        repaired, _ = greedy_repair(perturbed, deadline)
        
        # Local search on repaired
        optimized, _ = local_search(repaired, deadline)
        
        if len(optimized) < best_size:
            best_size = len(optimized)
            best_code = [all_words[i] for i in optimized]
    
    return best_code
