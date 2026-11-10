"""Maximize pairwise-noncoprime subset of [1,255] containing 255."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # Candidates: numbers in [1,255] with gcd(x, 255) > 1
    candidates = []
    for i in range(1, 256):
        if math.gcd(i, 255) > 1:
            candidates.append(i)
    
    n = len(candidates)
    
    # Precompute adjacency: two candidates are adjacent if gcd > 1
    # Use numpy for speed
    import numpy as np
    
    # Build adjacency matrix using numpy
    # For n~127, this is manageable
    adj_matrix = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj_matrix[i, j] = True
                adj_matrix[j, i] = True
    
    # Precompute adjacency as sets for quick lookup
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and adj_matrix[i, j]:
                adj[i].add(j)
    
    # Also precompute non-adjacency for quick rejection
    non_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and not adj_matrix[i, j]:
                non_adj[i].add(j)
    
    idx_255 = candidates.index(255)
    
    def can_add(vertex, clique_set):
        """Check if vertex can be added to clique_set."""
        # vertex must be adjacent to all members of clique_set
        # i.e., no member of clique_set is in non_adj[vertex]
        return len(clique_set & non_adj[vertex]) == 0
    
    def greedy_clique(initial_clique, order=None, seed=None):
        """Greedy clique expansion."""
        if seed is not None:
            random.seed(seed)
        
        clique = set(initial_clique)
        remaining = set(range(n)) - clique
        
        if order is None:
            # Default: by degree descending
            order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
        
        # Filter order to only remaining vertices
        for v in order:
            if v not in remaining:
                continue
            if can_add(v, clique):
                clique.add(v)
                remaining.discard(v)
        
        return clique
    
    def local_search(clique, max_iters=200):
        """Local search: try to add vertices, then try swaps."""
        clique = set(clique)
        best_size = len(clique)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try adding any compatible vertex
            for i in range(n):
                if i in clique:
                    continue
                if can_add(i, clique):
                    clique.add(i)
                    improved = True
                    best_size = len(clique)
            
            # Try swaps: remove one, add another
            if not improved:
                for remove_idx in list(clique):
                    if time.time() > deadline - 5:
                        break
                    temp_clique = set(clique)
                    temp_clique.discard(remove_idx)
                    for add_idx in range(n):
                        if add_idx in temp_clique:
                            continue
                        if can_add(add_idx, temp_clique):
                            new_size = len(temp_clique) + 1
                            if new_size > best_size:
                                clique = temp_clique | {add_idx}
                                best_size = new_size
                                improved = True
                                break
                    if improved:
                        break
        
        return clique, best_size
    
    # Strategy 1: Start with multiples of 3 (largest single-prime group)
    clique1 = set()
    for i, v in enumerate(candidates):
        if v % 3 == 0:
            clique1.add(i)
    if idx_255 not in clique1:
        clique1.add(idx_255)
    
    # Expand greedily
    clique1 = greedy_clique(clique1)
    # Local search
    clique1, size1 = local_search(clique1)
    
    # Strategy 2: Start with multiples of 5
    clique2 = set()
    for i, v in enumerate(candidates):
        if v % 5 == 0:
            clique2.add(i)
    if idx_255 not in clique2:
        clique2.add(idx_255)
    clique2 = greedy_clique(clique2)
    clique2, size2 = local_search(clique2)
    
    # Strategy 3: Start with multiples of 17
    clique3 = set()
    for i, v in enumerate(candidates):
        if v % 17 == 0:
            clique3.add(i)
    if idx_255 not in clique3:
        clique3.add(idx_255)
    clique3 = greedy_clique(clique3)
    clique3, size3 = local_search(clique3)
    
    # Strategy 4: Start with 255 only, then greedy by degree
    clique4 = {idx_255}
    # Order by degree
    order4 = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    clique4 = greedy_clique(clique4, order=order4)
    clique4, size4 = local_search(clique4)
    
    # Strategy 5-14: Randomized greedy with different seeds
    best_clique = max([clique1, clique2, clique3, clique4], key=len)
    best_size = len(best_clique)
    
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        random.seed(1000 + trial)
        
        # Random initial seed: pick a few random compatible vertices
        initial = {idx_255}
        
        # Add 3-5 random compatible vertices
        for _ in range(random.randint(3, 5)):
            candidates_to_add = [v for v in range(n) if v not in initial and can_add(v, initial)]
            if not candidates_to_add:
                break
            v = random.choice(candidates_to_add)
            initial.add(v)
        
        # Random order for greedy
        order = list(range(n))
        random.shuffle(order)
        # But weight by degree
        order.sort(key=lambda v: (len(adj[v]) * random.random(), -random.random()), reverse=True)
        
        clique_trial = greedy_clique(initial, order=order)
        clique_trial, size_trial = local_search(clique_trial, max_iters=100)
        
        if size_trial > best_size:
            best_clique = clique_trial
            best_size = size_trial
    
    # Strategy: Try a more systematic approach
    # For each prime factor of 255, try to build a clique around it
    # Then merge
    
    # Actually, let's try a different approach:
    # The maximum clique in this graph likely includes most multiples of 3
    # plus some multiples of 5 or 17 that are compatible
    
    # Let's try: start with all multiples of 3, then try to add multiples of 5
    # that are compatible with all current members
    
    clique5 = set()
    for i, v in enumerate(candidates):
        if v % 3 == 0:
            clique5.add(i)
    if idx_255 not in clique5:
        clique5.add(idx_255)
    
    # Try to add each multiple of 5 (not already in clique)
    for i, v in enumerate(candidates):
        if v % 5 == 0 and i not in clique5:
            if can_add(i, clique5):
                clique5.add(i)
    
    # Try to add each multiple of 17
    for i, v in enumerate(candidates):
        if v % 17 == 0 and i not in clique5:
            if can_add(i, clique5):
                clique5.add(i)
    
    # Now try to add any remaining candidate
    for i in range(n):
        if i not in clique5 and can_add(i, clique5):
            clique5.add(i)
    
    clique5, size5 = local_search(clique5)
    if size5 > best_size:
        best_clique = clique5
        best_size = size5
    
    # Try more local search iterations on the best clique
    best_clique, best_size = local_search(best_clique, max_iters=500)
    
    # Additional: try to find larger cliques by considering the structure
    # Numbers divisible by 3 form a clique of size 85
    # Numbers divisible by 5 form a clique of size 51
    # Numbers divisible by 17 form a clique of size 15
    
    # The intersection of multiples of 3 and 5 (i.e., multiples of 15) is size 17
    # These are in both cliques
    
    # Let's try a branch and bound for a small improvement
    # But given time constraints, let's just do more local search
    
    # Final: try adding any vertex that can be added
    for i in range(n):
        if i not in best_clique and can_add(i, best_clique):
            best_clique.add(i)
            best_size = len(best_clique)
    
    # Convert back to values
    result = [candidates[i] for i in best_clique]
    
    # Ensure 255 is present
    if 255 not in result:
        result.append(255)
    
    return sorted(result)
# EVOLVE-BLOCK-END
