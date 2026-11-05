import random
import time
import math
from itertools import combinations
import heapq

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    diffs_per_row = k * (k + 1) // 2
    
    # Helper to compute max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # Helper to check validity of a set of rows
    def is_valid(rows):
        used_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        return False
                    used_diffs.add(d)
        return True

    # Initial baseline using pure greedy
    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    baseline_rows = pure_greedy()
    best_rows = [row[:] for row in baseline_rows]
    best_max = get_max_entry(best_rows)
    
    # Strategy:
    # 1. Generate a large pool of candidate rows (Golomb rulers).
    # 2. Compute differences for each candidate.
    # 3. Build a conflict graph where an edge exists if two rows share a difference.
    # 4. Use a Weighted Maximum Independent Set (MIS) heuristic to select n_rows rows
    #    that minimize the maximum scope. We can do this by iterating over potential
    #    max_scope limits and checking if an independent set of size n_rows exists
    #    using only rows with scope <= limit. To improve quality, we prioritize rows
    #    with smaller scopes.
    
    def generate_random_golomb_row(max_val, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
            
        row = [0]
        for _ in range(k):
            current_diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    current_diffs.add(row[j] - row[i])
            
            found = False
            # Randomize candidates
            candidates = list(range(row[-1] + 1, max_val + 1))
            rng.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - a for a in row}
                if not new_diffs & current_diffs:
                    row.append(x)
                    found = True
                    break
            
            if not found:
                return None
        
        return row

    # Phase 1: Generate a diverse pool of candidates
    # We generate candidates with varying scope limits to ensure diversity.
    # We want to find a solution with max scope < best_max.
    # Let's generate candidates with scope up to best_max - 1.
    # If we can't find a solution, we might need to relax, but we start tight.
    
    pool_size_target = 5000
    candidates = []
    candidate_diffs = []
    seen_rows = set()
    
    # Generate candidates
    # We'll use multiple seeds and varying max_val to get diversity
    # Start with max_val = best_max - 1
    # If that's too tight, we might not get enough candidates.
    # But if we can't find a solution with max_val < best_max, we can't improve.
    # So we only generate candidates with max_val < best_max.
    
    gen_max_val = best_max - 1
    
    if gen_max_val < k + 1:
         # Can't even fit the minimum span
         return best_rows

    # Generate pool
    # Use a mix of random seeds
    attempts = 0
    max_attempts = pool_size_target * 5
    while len(candidates) < pool_size_target and attempts < max_attempts and time.time() < deadline - 20:
        attempts += 1
        # Occasionally try a slightly larger max_val to ensure we have *some* candidates
        # if the tight limit is too restrictive? No, we want to improve best_max.
        # If we can't find candidates with scope < best_max, we are stuck.
        
        # Actually, if best_max is the greedy result, there are likely many rulers
        # with scope < best_max.
        
        row = generate_random_golomb_row(gen_max_val)
        if row:
            row_tuple = tuple(row)
            if row_tuple not in seen_rows:
                seen_rows.add(row_tuple)
                # Compute diffs
                diffs = set()
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        diffs.add(row[j] - row[i])
                candidates.append(row)
                candidate_diffs.append(diffs)
    
    if not candidates:
        return best_rows
    
    # Phase 2: Build Conflict Graph
    # For efficiency, we represent the conflict graph implicitly or use a heuristic.
    # With 5000 nodes, a full adjacency matrix is 25M entries, which is okay in memory
    # if sparse, but checking conflicts is O(N^2 * D) where D is diffs per row.
    # D is 36 for k=8. 5000^2 * 36 is ~900M ops, which is too slow in Python.
    
    # We need a faster MIS heuristic or a smarter graph construction.
    # Let's use a greedy weighted independent set approach on the fly.
    # Or, we can reduce the pool size to make pairwise checks feasible.
    # Let's try with a smaller pool, e.g., 1000 candidates.
    
    # Let's restart with a smaller pool if the above took too long or was too big.
    # Actually, let's just use a smaller pool size for the graph construction phase.
    pool_size = min(len(candidates), 2000)
    
    # If we generated more than 2000, we can just take the first 2000 or random sample.
    # Let's random sample to maintain diversity.
    if len(candidates) > pool_size:
        indices = random.sample(range(len(candidates)), pool_size)
        candidates = [candidates[i] for i in indices]
        candidate_diffs = [candidate_diffs[i] for i in indices]
    
    n_cand = len(candidates)
    
    # Build adjacency list
    # adj[i] = list of indices j such that candidates[i] and candidates[j] conflict
    adj = [[] for _ in range(n_cand)]
    
    # Optimize conflict checking
    # For each pair (i, j), check if candidate_diffs[i] & candidate_diffs[j] is non-empty.
    # We can iterate i < j.
    for i in range(n_cand):
        di = candidate_diffs[i]
        for j in range(i + 1, n_cand):
            dj = candidate_diffs[j]
            if di & dj:
                adj[i].append(j)
                adj[j].append(i)
    
    # Phase 3: Find Max Weight Independent Set of size n_rows
    # Weight: We want to minimize the max scope.
    # This is a constrained optimization.
    # Heuristic: 
    # 1. Sort candidates by scope (ascending).
    # 2. Try to pick n_rows non-conflicting candidates from this sorted list.
    #    This is equivalent to finding an independent set of size n_rows in the
    #    induced subgraph of candidates with scope <= current_limit, and we want
    #    the smallest current_limit.
    
    # Let's implement a greedy MIS with backtracking/repair for a fixed set of nodes.
    # But we want the *best* (min max scope).
    
    # Alternative: 
    # Iterate over potential max_scope values (from min_scope to best_max - 1).
    # For each max_scope, filter candidates with scope <= max_scope.
    # Check if an independent set of size n_rows exists in this filtered graph.
    # If yes, return it.
    
    # To speed up, we can sort unique scopes and binary search?
    # Or just iterate scopes from low to high.
    
    # Let's get unique scopes of our candidates
    scopes = [max(c) for c in candidates]
    unique_scopes = sorted(set(scopes))
    
    found_solution = None
    
    for limit in unique_scopes:
        if limit >= best_max:
            break
            
        # Filter candidates
        valid_indices = [i for i in range(n_cand) if max(candidates[i]) <= limit]
        
        if len(valid_indices) < n_rows:
            continue
            
        # Map valid indices to a smaller graph
        # We need to check if there's an independent set of size n_rows in G[valid_indices]
        
        # Greedy heuristic: 
        # Pick the node with the smallest degree (or smallest scope, tie-break by degree)
        # Remove it and its neighbors. Repeat.
        
        # To make it robust, we run this greedy multiple times with randomization.
        
        def greedy_mis(indices, limit, n_needed, time_limit=2.0):
            start_t = time.time()
            # Create a local adjacency list for the subgraph
            # To save memory/time, we can just use the global adj and filter.
            
            # State: set of available nodes
            available = set(indices)
            chosen = []
            
            while len(chosen) < n_needed:
                if time.time() > start_t + time_limit:
                    return None
                
                if not available:
                    return None
                
                # Score each available node:
                # We want to pick a node that allows us to pick more nodes later.
                # Heuristic: Pick node with minimum degree in the induced subgraph.
                # Degree of u = number of neighbors v in available.
                
                min_deg = float('inf')
                best_node = None
                best_deg = float('inf')
                
                # To add randomness, we can pick from the top K lowest degree nodes
                # Or just pick the best one.
                
                # Optimization: 
                # If n_needed is small, this is fast.
                
                for u in available:
                    deg = 0
                    for v in adj[u]:
                        if v in available:
                            deg += 1
                    if deg < best_deg:
                        best_deg = deg
                        best_node = u
                        # Early break if deg is 0?
                        if deg == 0:
                            break
                
                if best_node is None:
                    return None
                    
                chosen.append(best_node)
                # Remove best_node and its neighbors from available
                available.remove(best_node)
                for v in adj[best_node]:
                    if v in available:
                        available.remove(v)
                        
            return chosen
        
        # Run greedy MIS a few times to see if we can find a solution
        # We can also randomize the tie-breaking or the order of nodes.
        
        for _ in range(10):
            sol = greedy_mis(valid_indices, limit, n_rows)
            if sol:
                # Verify
                rows = [candidates[i] for i in sol]
                if is_valid(rows):
                    found_solution = rows
                    break
        if found_solution:
            break
            
    if found_solution:
        curr_max = get_max_entry(found_solution)
        if curr_max < best_max:
            best_max = curr_max
            best_rows = [row[:] for row in found_solution]
            
    # If we found a solution, we might be able to improve further by generating
    # more candidates with the new tighter limit?
    # But we are time-constrained.
    
    return best_rows
# EVOLVE-BLOCK-END
