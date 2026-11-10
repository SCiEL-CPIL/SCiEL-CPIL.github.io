import numpy as np
import time
import random
from itertools import combinations
from collections import defaultdict

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580
    
    # Precompute all 4-subsets and their 3-subset IDs
    # Map 3-subsets to IDs
    three_subsets = list(combinations(range(d), 3))
    three_to_id = {s: i for i, s in enumerate(three_subsets)}
    
    four_subsets = list(combinations(range(d), 4))
    # For each 4-subset, store the IDs of its 4 three-subsets
    four_to_threes = []
    for s4 in four_subsets:
        ids = []
        for i in range(4):
            # Remove i-th element to get a 3-subset
            s3 = tuple(x for j, x in enumerate(s4) if j != i)
            ids.append(three_to_id[s3])
        four_to_threes.append(np.array(ids, dtype=np.int64))
        
    four_to_threes = np.array(four_to_threes) # Shape (330, 4)
    
    # Build conflict graph: two 4-subsets conflict if they share 3 or more elements
    # (i.e., their intersection has size >= 3)
    n_four = len(four_subsets)
    conflict = [set() for _ in range(n_four)]
    for i in range(n_four):
        for j in range(i + 1, n_four):
            # Check if they share at least 3 elements
            si = set(four_subsets[i])
            sj = set(four_subsets[j])
            if len(si & sj) >= 3:
                conflict[i].add(j)
                conflict[j].add(i)
    
    best_supports = []
    best_count = 0
    
    # Branch and bound for maximum independent set
    def branch_and_bound(available, current_set):
        nonlocal best_count, best_supports, deadline
        
        if time.time() > deadline - 20:
            return
        
        if not available:
            if len(current_set) > best_count:
                best_count = len(current_set)
                best_supports = list(current_set)
            return
        
        # Upper bound: current size + max possible from available
        # Use a simple bound: current + len(available)
        if len(current_set) + len(available) <= best_count:
            return
        
        # Choose vertex with highest degree in available subgraph
        # This helps prune more
        max_deg = -1
        chosen = -1
        for v in available:
            deg = len(conflict[v] & available)
            if deg > max_deg:
                max_deg = deg
                chosen = v
        
        if chosen == -1:
            # No vertices left
            if len(current_set) > best_count:
                best_count = len(current_set)
                best_supports = list(current_set)
            return
        
        # Branch 1: include chosen
        new_available = available - {chosen}
        # Remove all neighbors of chosen
        new_available -= conflict[chosen]
        current_set.append(chosen)
        branch_and_bound(new_available, current_set)
        current_set.pop()
        
        # Branch 2: exclude chosen
        new_available = available - {chosen}
        branch_and_bound(new_available, current_set)
    
    # Start with all vertices
    all_vertices = set(range(n_four))
    branch_and_bound(all_vertices, [])
    
    # If B&B didn't find anything good, fall back to greedy
    if best_count < 20:
        rng = np.random.default_rng(42)
        num_restarts = 0
        while time.time() < deadline - 10:
            num_restarts += 1
            order = rng.permutation(len(four_subsets))
            used = np.zeros(len(three_subsets), dtype=bool)
            selected = []
            
            for idx in order:
                ids = four_to_threes[idx]
                if not np.any(used[ids]):
                    used[ids] = True
                    selected.append(idx)
                    
            if len(selected) > best_count:
                best_count = len(selected)
                best_supports = selected
                
            if best_count >= 41:
                break
    
    # Construct points
    points = []
    for idx in best_supports:
        support = four_subsets[idx]
        # Generate all 16 sign combinations
        for signs in range(16):
            v = np.zeros(d, dtype=np.int64)
            for k, pos in enumerate(support):
                if (signs >> k) & 1:
                    v[pos] = 1
                else:
                    v[pos] = -1
            points.append(v)
            
    if len(points) < 2:
        # Fallback
        points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
        
    pts_array = np.array(points, dtype=np.int64)
    return np.around(pts_array).astype(np.int64)
