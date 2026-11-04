import numpy as np
import time
import itertools
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Build the full vector pool:
    # 1) E8 minimal vectors (240): weight-4, supports from {1..8}
    # 2) D3 vectors (12): weight-2 in {9,10,11}
    # 3) Mixed vectors (96): weight-4 with 2 from {1..8}, 2 from {9,10,11}
    
    e8_vectors = []
    d3_vectors = []
    mixed_vectors = []
    
    # E8: 4 nonzero ±1 in positions 0-7
    for support in itertools.combinations(range(8), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(support, signs):
                v[pos] = s
            e8_vectors.append(v)
    
    # D3: weight-2 in positions 8,9,10 (0-indexed: 8,9,10)
    for support in itertools.combinations(range(8, 11), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(support, signs):
                v[pos] = s
            d3_vectors.append(v)
    
    # Mixed: 2 from {0..7}, 2 from {8,9,10}
    for s1 in itertools.combinations(range(8), 2):
        for s2 in itertools.combinations(range(8, 11), 2):
            support = tuple(sorted(s1 + s2))
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for pos, s in zip(support, signs):
                    v[pos] = s
                mixed_vectors.append(v)
    
    all_vectors = e8_vectors + d3_vectors + mixed_vectors
    n = len(all_vectors)
    
    # Precompute all pairs of vectors as a matrix for dot products
    vec_arr = np.array(all_vectors, dtype=np.int64)
    
    # Compute dot product matrix
    dot = vec_arr @ vec_arr.T
    
    # Norm squared of each vector
    norms = np.sum(vec_arr ** 2, axis=1)
    max_norm = np.max(norms)
    
    # Compatibility: min pairwise distance >= max norm
    # ||v - w||^2 = ||v||^2 + ||w||^2 - 2<v,w>
    # We need ||v-w||^2 >= max_norm for all pairs v != w
    # i.e., ||v||^2 + ||w||^2 - 2<v,w> >= max_norm
    
    # For a selected set, we need this for all pairs in the set.
    # Let's precompute the conflict: two vectors conflict if their distance^2 < max_norm
    
    # Build conflict matrix
    conflict = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            dist_sq = norms[i] + norms[j] - 2 * dot[i, j]
            if dist_sq < max_norm:
                conflict[i, j] = True
                conflict[j, i] = True
    
    # Now solve Maximum Independent Set on the conflict graph
    # Use branch and bound with greedy coloring bound
    
    # Greedy coloring for bound
    def greedy_coloring(conflict, remaining):
        """Return upper bound on MIS size using greedy coloring."""
        if not remaining:
            return 0
        
        colors = np.zeros(len(remaining), dtype=int)
        color_count = 0
        uncolored = set(remaining)
        
        while uncolored:
            color_count += 1
            # Greedy: pick a vertex with fewest neighbors in uncolored
            best_v = None
            min_deg = len(uncolored)
            for v in uncolored:
                deg = sum(1 for u in uncolored if u != v and conflict[v, u])
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v is None:
                break
            
            # Color this vertex and all its uncolored neighbors with this color
            to_remove = set()
            for u in uncolored:
                if u == best_v or conflict[best_v, u]:
                    to_remove.add(u)
            uncolored -= to_remove
        
        return color_count
    
    # Branch and bound for MIS
    best_set = []
    best_size = 0
    
    # Start with a greedy solution as initial bound
    greedy_set = []
    remaining = list(range(n))
    while remaining:
        # Pick vertex with minimum degree in remaining
        best_v = min(remaining, key=lambda v: sum(1 for u in remaining if u != v and conflict[v, u]))
        greedy_set.append(best_v)
        remaining = [u for u in remaining if u != best_v and not conflict[best_v, u]]
    
    best_set = greedy_set
    best_size = len(greedy_set)
    
    def branch_and_bound(conflict, n, initial_bound):
        """Find MIS using branch and bound."""
        global best_set, best_size, deadline, start_time
        
        best_set = list(initial_bound)
        best_size = len(initial_bound)
        
        def bb(selected, remaining, bound):
            nonlocal best_set, best_size
            nonlocal deadline, start_time
            
            if time.time() > deadline - 2.0:
                return
            
            if not remaining:
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_set = list(selected)
                return
            
            # Bound: if current selected + bound <= best_size, prune
            if len(selected) + bound <= best_size:
                return
            
            # Pick a vertex from remaining
            # Heuristic: pick vertex with maximum degree in remaining
            v = max(remaining, key=lambda x: sum(1 for u in remaining if u != x and conflict[x, u]))
            
            # Branch 1: include v
            selected.append(v)
            new_remaining = [u for u in remaining if u != v and not conflict[v, u]]
            new_bound = len(new_remaining)
            bb(selected, new_remaining, new_bound)
            selected.pop()
            
            # Branch 2: exclude v
            new_remaining2 = [u for u in remaining if u != v]
            bb(selected, new_remaining2, len(new_remaining2))
        
        # Initial bound
        bb([], list(range(n)), n)
    
    # The graph has 348 nodes. Branch and bound might be slow.
    # Let's use a different approach: iterative local search / simulated annealing
    
    def local_search(initial_set, max_iter=5000):
        """Improve MIS via 1-swap and 2-swap moves."""
        selected = set(initial_set)
        current_size = len(selected)
        
        for _ in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            
            # Try to add a vertex
            for v in range(n):
                if v in selected:
                    continue
                # Check if v is compatible with all selected
                compatible = True
                for u in selected:
                    if conflict[v, u]:
                        compatible = False
                        break
                if compatible:
                    selected.add(v)
                    current_size += 1
                    improved = True
                    break
            
            if not improved:
                # Try 1-swap: remove u, add v
                for u in list(selected):
                    for v in range(n):
                        if v in selected:
                            continue
                        # Check if v is compatible with (selected - {u})
                        compatible = True
                        for w in selected:
                            if w == u:
                                continue
                            if conflict[v, w]:
                                compatible = False
                                break
                        if compatible:
                            # Check if removing u and adding v increases size (it stays same size, so skip)
                            # Actually we want to increase size, so 1-swap doesn't help unless we can add more
                            pass
                break
            
            # Continue loop to try more additions
            continue
        
        return list(selected)
    
    # Run multiple restarts with different initial solutions
    best_solution = list(greedy_set)
    
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization
        rng = random.Random(restart * 12345 + 678)
        random_subset = rng.sample(range(n), min(50, n))
        
        # Greedy completion
        selected = []
        for v in random_subset:
            if all(not conflict[v, u] for u in selected):
                selected.append(v)
        
        # Local search
        result = local_search(selected, max_iter=2000)
        
        if len(result) > len(best_solution):
            best_solution = result
    
    # Also try with the full greedy set as starting point
    result = local_search(greedy_set, max_iter=3000)
    if len(result) > len(best_solution):
        best_solution = result
    
    # Build the result array
    result_vectors = vec_arr[best_solution]
    
    return result_vectors.astype(np.int64)
