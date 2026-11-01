import itertools
import random
import time
import sys
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (evaluator kills at 150s)
    
    # Specific case for the task: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)
    
    # Fallback for other parameters
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
    
    # Generate all words and index mapping
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Build neighbor lists and numpy adjacency matrix
    neighbors = [[] for _ in range(space_size)]
    adj_matrix = np.zeros((space_size, space_size), dtype=np.int8)
    
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        adj_matrix[i, i] = 1
        for j in range(n):
            for val in range(q):
                if val == w[j]:
                    continue
                new_w = list(w)
                new_w[j] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors[i].append(idx)
                adj_matrix[i, idx] = 1
    
    # Precompute neighbor sets for fast membership
    neighbor_sets = [set(nb) for nb in neighbors]
    
    # Tetracode direct sum initialization (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            word = h + list(tail)
            initial_indices.append(word_to_idx[tuple(word)])
    
    best_code_indices = set(initial_indices)
    best_size = len(best_code_indices)
    
    rng = random.Random(42)
    
    def compute_coverage(indices):
        """Compute coverage count array using numpy."""
        cov = np.zeros(space_size, dtype=np.int32)
        for idx in indices:
            cov += adj_matrix[idx]
        return cov
    
    def local_search(indices_set, deadline):
        """Local search: single removals and 2-for-1 swaps."""
        current = set(indices_set)
        cover_count = compute_coverage(current)
        
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
                    cover_count -= adj_matrix[idx]
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
                    
                    # Union of uniquely covered words
                    u_union_set = set(u_i) | set(u_j)
                    if not u_union_set:
                        continue
                    
                    u_union_list = list(u_union_set)
                    
                    # Find candidate k not in current that covers all u_union
                    for k in range(space_size):
                        if k in current:
                            continue
                        
                        # Quick check: does k cover all words in u_union?
                        covers = True
                        for w in u_union_list:
                            if not adj_matrix[k, w]:
                                covers = False
                                break
                        
                        if covers:
                            # Perform the swap
                            current.remove(i)
                            current.remove(j)
                            current.add(k)
                            
                            cover_count -= adj_matrix[i]
                            cover_count -= adj_matrix[j]
                            cover_count += adj_matrix[k]
                            
                            found_swap = True
                            improved = True
                            break
                    
                    if found_swap:
                        break
                if found_swap:
                    break
        
        return current, cover_count
    
    def greedy_repair(indices_set, deadline):
        """Add codewords via greedy until valid cover."""
        current = set(indices_set)
        cover_count = compute_coverage(current)
        
        # Find uncovered words
        uncovered = np.where(cover_count == 0)[0]
        
        while len(uncovered) > 0 and time.time() < deadline - 0.5:
            # Weighted greedy: pick codeword covering most uncovered words
            # Use numpy for speed
            uncovered_set = set(uncovered.tolist())
            
            best_gain = 0
            best_candidate = -1
            
            for cand in range(space_size):
                if cand in current:
                    continue
                # Count how many uncovered words are neighbors of cand
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
            cover_count += adj_matrix[best_candidate]
            
            # Recompute uncovered
            uncovered = np.where(cover_count == 0)[0]
        
        return current, cover_count
    
    # Run initial local search
    if time.time() < deadline - 5.0:
        current_indices, _ = local_search(set(initial_indices), deadline)
        if len(current_indices) < best_size:
            best_size = len(current_indices)
            best_code_indices = current_indices
    
    # Iterated local search with perturbation
    trial = 0
    while time.time() < deadline - 2.0:
        trial += 1
        
        # Start from current best
        start_set = set(best_code_indices)
        
        # Perturbation: remove k random codewords
        k = rng.randint(2, 6)
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
            best_code_indices = optimized
        
        # If we've hit 73, try harder to find 72
        if best_size <= 73 and time.time() < deadline - 3.0:
            # More aggressive perturbation
            k2 = rng.randint(3, 8)
            perturbed2 = set(best_code_indices)
            to_remove2 = rng.sample(list(perturbed2), min(k2, len(perturbed2)))
            for idx in to_remove2:
                perturbed2.remove(idx)
            
            repaired2, _ = greedy_repair(perturbed2, deadline)
            optimized2, _ = local_search(repaired2, deadline)
            
            if len(optimized2) < best_size:
                best_size = len(optimized2)
                best_code_indices = optimized2
    
    # Convert back to list of words
    return [list(all_words[i]) for i in sorted(best_code_indices)]
