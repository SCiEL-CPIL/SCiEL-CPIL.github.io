import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    # 1. Generate all vectors
    all_vectors = []
    support_map = {}
    
    idx = 0
    for comb in itertools.combinations(range(D), 4):
        support = comb
        support_map[support] = []
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(support, signs):
                v[k] = s
            all_vectors.append((idx, support, signs, v))
            support_map[support].append(idx)
            idx += 1
            
    N = len(all_vectors)
    
    # 2. Build Conflict Graph
    neighbors = [set() for _ in range(N)]
    
    supports = list(support_map.keys())
    sup_to_vecs = {s: support_map[s] for s in supports}
    
    # Precompute intersection info for support pairs to speed up neighbor finding
    # We only care about pairs with intersection size 3.
    
    # To avoid O(N^2) vector checks, we iterate over support pairs.
    for i_s, s1 in enumerate(supports):
        for i_t, s2 in enumerate(supports):
            if i_t <= i_s: continue
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                # Find indices in s1 and s2 corresponding to inter
                pos_in_s1 = [s1.index(x) for x in inter]
                pos_in_s2 = [s2.index(x) for x in inter]
                
                vecs_s1 = sup_to_vecs[s1]
                vecs_s2 = sup_to_vecs[s2]
                
                for idx1 in vecs_s1:
                    signs1 = all_vectors[idx1][2]
                    for idx2 in vecs_s2:
                        signs2 = all_vectors[idx2][2]
                        
                        # Check if signs match on the 3 positions
                        match = True
                        for p1, p2 in zip(pos_in_s1, pos_in_s2):
                            if signs1[p1] != signs2[p2]:
                                match = False
                                break
                        if match:
                            neighbors[idx1].add(idx2)
                            neighbors[idx2].add(idx1)

    # Convert neighbors to lists for faster iteration in some contexts, 
    # but sets are needed for membership testing. We'll keep sets.
    
    # 3. CSP Backtracking Search
    
    best_size = 0
    best_set = set()
    start_time = time.time()
    time_limit = 110.0
    
    # Greedy initialization to set a good lower bound
    greedy_selected = set()
    remaining = set(range(N))
    while remaining:
        # Pick node with min degree? Or just first?
        # Let's pick the one with lowest degree in remaining to leave more options?
        # Or highest? 
        # Standard greedy MIS: pick node, remove it and neighbors.
        # Heuristic: pick node with lowest degree in remaining graph (to minimize removals).
        # Calculating degree for all is expensive.
        # Let's just pick the first one for simplicity in init, 
        # or use a random restart for greedy?
        # Let's use a simple greedy: pick node with smallest index for determinism.
        v = min(remaining)
        greedy_selected.add(v)
        remaining -= neighbors[v]
        remaining.discard(v)
        
    best_size = len(greedy_selected)
    best_set = set(greedy_selected)
    
    # State for backtracking
    # candidates: list of indices
    # candidate_set: set of indices for O(1) lookup
    # selected: set of indices
    
    def solve(candidates, candidate_set, selected):
        nonlocal best_size, best_set, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        if len(selected) + len(candidates) <= best_size:
            return
            
        if not candidates:
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = set(selected)
            return
            
        # Heuristic: Most Constrained Variable
        # Pick v in candidates with max degree in the induced subgraph (candidates)
        max_deg = -1
        v = -1
        # To speed up, we can sample if candidates is large?
        # For now, iterate all.
        # deg(u) = len(neighbors[u] & candidate_set)
        for u in candidates:
            # Intersection size
            # Using set intersection
            deg = len(neighbors[u] & candidate_set)
            if deg > max_deg:
                max_deg = deg
                v = u
                
        if v == -1:
            return
            
        # Branch 1: Include v
        # New candidates: candidates \ {v} \ neighbors[v]
        removed = set()
        removed.add(v)
        # Find neighbors of v that are in candidate_set
        for n in neighbors[v]:
            if n in candidate_set:
                removed.add(n)
                
        # Update state
        new_candidate_set = candidate_set - removed
        new_candidates = [u for u in candidates if u not in removed]
        
        selected.add(v)
        solve(new_candidates, new_candidate_set, selected)
        selected.remove(v)
        
        # Branch 2: Exclude v
        # New candidates: candidates \ {v}
        new_candidate_set_2 = candidate_set - {v}
        new_candidates_2 = [u for u in candidates if u != v]
        
        solve(new_candidates_2, new_candidate_set_2, selected)

    # Initial state
    initial_candidates = list(range(N))
    initial_candidate_set = set(range(N))
    initial_selected = set()
    
    try:
        solve(initial_candidates, initial_candidate_set, initial_selected)
    except RecursionError:
        pass
        
    # Convert best_set to array
    if not best_set:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result_vectors = [all_vectors[i][3] for i in best_set]
    return np.array(result_vectors, dtype=np.int64)
