import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Generate all candidate supports (4-element subsets)
    # There are C(11, 4) = 330 supports.
    supports = []
    for comb in combinations(range(D), 4):
        mask = 0
        for idx in comb:
            mask |= (1 << idx)
        supports.append(mask)
    
    n_supports = len(supports)
    
    # 2. Precompute intersection sizes between all supports
    # inter[i][j] = number of common indices
    # We need this for fast conflict checking during backtracking.
    # 330x330 is small enough to store in memory.
    inter = np.zeros((n_supports, n_supports), dtype=np.int8)
    for i in range(n_supports):
        for j in range(i + 1, n_supports):
            inter[i][j] = (supports[i] & supports[j]).bit_count()
            inter[j][i] = inter[i][j]
            
    # 3. Define the search space
    # Each support has 16 possible sign assignments (2^4).
    # Total potential vectors: 330 * 16 = 5280.
    # We want to select a subset of these vectors such that:
    # For any two selected vectors u, v: ||u - v||^2 >= ||u||^2
    # ||u||^2 = 4.
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 8 - 2 u.v
    # Condition: 8 - 2 u.v >= 4  =>  u.v <= 2.
    # Since u, v are distinct and non-zero, max u.v is 3 (if 3 overlap with same sign).
    # If u.v = 3, distance^2 = 2 < 4. Invalid.
    # If u.v = 2, distance^2 = 4 >= 4. Valid.
    # So we need u.v <= 2 for all pairs.
    
    # Backtracking search
    # We iterate through supports. For each support, we can choose:
    # - None of its sign variants
    # - One or more of its sign variants (up to 16, but usually 1 is best for density? 
    #   Actually, if we pick two vectors with the SAME support, their dot product is 4 (if same sign) 
    #   or less. If same support, u.v = sum(s1_k * s2_k).
    #   If we pick u and v with same support, max u.v is 4. 8-8=0 < 4. Invalid.
    #   So we can pick at most ONE vector per support?
    #   Wait. If u and v have the same support, they differ only in signs.
    #   u.v = sum(u_k v_k). Since u_k, v_k in {-1, 1}, u.v is even, in {-4, -2, 0, 2, 4}.
    #   We need u.v <= 2.
    #   So we CAN pick multiple vectors from the same support as long as their dot product is <= 2.
    #   Example: u=(1,1,1,1), v=(-1,1,1,1). u.v = -1+1+1+1 = 2. Valid.
    #   Example: u=(1,1,1,1), v=(1,-1,1,1). u.v = 1-1+1+1 = 2. Valid.
    #   Example: u=(1,1,1,1), v=(1,1,-1,1). u.v = 2. Valid.
    #   Example: u=(1,1,1,1), v=(1,1,1,-1). u.v = 2. Valid.
    #   So for a given support, we can pick a set of sign vectors that are pairwise "compatible" (dot <= 2).
    #   In fact, for a fixed support, the max size of such a set is 8 (all vectors with an even number of -1s? No).
    #   Let's check: If we pick all 8 vectors with sum of signs = 0? No, dot product depends on overlap.
    #   Actually, it's easier to just treat each (support, sign) pair as a unique candidate vector.
    #   Total candidates: 5280.
    #   We want the largest subset of these candidates where pairwise dot <= 2.
    
    # This is a Maximum Independent Set problem on a graph of 5280 nodes.
    # Edge exists if dot > 2.
    # Building this graph takes 5280^2 / 2 ~ 14M operations. Feasible.
    # Solving MIS on 5280 nodes is hard, but the graph structure is highly regular.
    
    # Alternative: Greedy + Local Search on the 5280-node graph.
    
    # 1. Generate all candidate vectors
    candidates = []
    cand_support_idx = []
    for s_idx, mask in enumerate(supports):
        inds = [i for i in range(D) if (mask >> i) & 1]
        for sign_mask in range(16):
            v = np.zeros(D, dtype=np.int64)
            for j, idx_val in enumerate(inds):
                if sign_mask & (1 << j):
                    v[idx_val] = 1
                else:
                    v[idx_val] = -1
            candidates.append(v)
            cand_support_idx.append(s_idx)
            
    N = len(candidates)
    candidates = np.array(candidates, dtype=np.int64)
    
    # 2. Build Conflict Graph
    # Conflict if dot > 2
    # We can compute dot products in batches to speed up.
    # But 5280x5280 matrix of ints is ~28MB. OK.
    # However, checking dot > 2 for all pairs is O(N^2 * D).
    # N=5280, D=11. 5280^2 * 11 ~ 300M ops. Might be slow in Python loop.
    # Use NumPy for dot product matrix?
    # M = candidates @ candidates.T
    # This is 5280x5280. 5280*5280*11 * 2 flops ~ 600M flops. NumPy handles this in <1s.
    
    if time.time() > deadline - 10:
        return np.array([]).reshape(0, D)
        
    dots = candidates @ candidates.T
    
    # Conflict if dots[i][j] > 2 and i != j
    # Note: diagonal is 4.
    # We want to select a subset S of indices such that for all i, j in S, i!=j, dots[i][j] <= 2.
    
    # Greedy MIS Heuristic:
    # 1. Compute degree (number of conflicts) for each node.
    # 2. Pick node with min degree.
    # 3. Add to solution.
    # 4. Remove node and its neighbors from consideration.
    
    # To make it faster, we can pre-sort nodes by degree?
    # Degree = sum(dots > 2, axis=1) - 1 (excluding self)
    # But as we remove nodes, degrees change.
    
    # Let's use a standard greedy MIS with restarts.
    
    def greedy_mis(deadline_local):
        # Create a copy of the conflict mask
        # conflict[i][j] = True if dots[i][j] > 2
        conflict = (dots > 2)
        # Zero out diagonal
        np.fill_diagonal(conflict, False)
        
        # Degrees
        degrees = conflict.sum(axis=1)
        
        # Nodes to consider
        alive = np.ones(N, dtype=bool)
        selected = []
        
        while alive.any():
            if time.time() > deadline_local:
                break
            
            # Find node with min degree among alive
            # To avoid O(N) scan every time, we can just do it for now. N=5280 is small.
            # But we do this ~5000 times? No, we remove many nodes each step.
            
            # Get alive indices
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
                
            # Get degrees of alive nodes
            alive_degrees = degrees[alive_indices]
            
            # Pick min
            min_idx_in_alive = np.argmin(alive_degrees)
            v = alive_indices[min_idx_in_alive]
            
            selected.append(v)
            
            # Remove v and its neighbors from alive
            # Neighbors are where conflict[v] is True
            # v itself is not in conflict[v] (diagonal is False)
            to_remove = np.where(conflict[v])[0]
            to_remove = np.append(to_remove, v)
            
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes?
            # This is expensive. O(N) per step.
            # Total O(N^2) ~ 28M ops. Feasible.
            
            # For each remaining alive node u, if u was a neighbor of v, its degree decreases by 1.
            # Actually, we just need the degree of the *alive* neighbors.
            # It's easier to recompute degrees for alive nodes if N is small, 
            # but N=5280 and steps~500? 
            # Let's just recompute degrees for the alive set.
            # Or simpler: don't update degrees, just use the initial degree as a heuristic?
            # Initial degree is a good proxy.
            
        return selected

    best_selected = []
    
    # Run a few restarts with different orderings or randomizations?
    # The greedy "min degree" is deterministic.
    # Let's try to improve by local search:
    # After greedy, try to add any node that doesn't conflict with selected.
    # This is O(N * |selected|).
    
    def local_search(selected_set, deadline_local):
        selected_set = set(selected_set)
        # Try to add nodes not in selected_set
        # A node u can be added if no v in selected_set has dots[u][v] > 2.
        # i.e., max(dots[u][selected_set]) <= 2.
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline_local:
                break
            
            # Compute max dot with selected for all nodes
            # dots is (N, N). selected is a list of indices.
            # max_dots = dots[:, list(selected_set)].max(axis=1)
            # This is O(N * |selected|).
            # If |selected| is ~1000, this is 5M ops. Fast.
            
            sel_list = list(selected_set)
            if not sel_list:
                # If empty, add all? No, they might conflict with each other.
                # But greedy handles that. This is for adding *more*.
                break
                
            max_dots = dots[:, sel_list].max(axis=1)
            
            # Candidates are nodes not in selected_set and max_dots <= 2
            candidates_to_add = []
            for i in range(N):
                if i not in selected_set and max_dots[i] <= 2:
                    candidates_to_add.append(i)
                    
            # If we can add multiple, which one?
            # Add the one that "blocks" the fewest other candidates?
            # Or just add them all? No, they might conflict with each other.
            # So we must pick one, add it, and restart.
            
            if candidates_to_add:
                # Heuristic: pick the one with min degree in the "candidate subgraph"?
                # Or just pick the first one?
                # Let's pick the one with min global degree to be consistent.
                # Or random?
                # Let's pick the one that allows the most future additions? Hard to know.
                # Simple: pick the one with min degree in the full graph.
                v = min(candidates_to_add, key=lambda x: degrees[x])
                
                selected_set.add(v)
                improved = True
                
        return selected_set

    # Run greedy
    selected = greedy_mis(deadline - 5)
    selected_set = set(selected)
    
    # Run local search
    selected_set = local_search(selected_set, deadline - 2)
    
    best_selected = list(selected_set)
    
    # Try one more restart with a slightly different heuristic?
    # E.g., pick node with max degree? (Usually bad for MIS)
    # Or random node?
    
    if time.time() < deadline - 5:
        # Randomized Greedy
        np.random.seed(42) # Fixed seed for reproducibility in this block? 
        # Actually, let's just shuffle the order of tie-breaking.
        
        selected2 = []
        alive = np.ones(N, dtype=bool)
        conflict = (dots > 2)
        np.fill_diagonal(conflict, False)
        
        while alive.any():
            if time.time() > deadline - 5:
                break
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            
            # Pick random from min degree?
            # Or just random?
            # Let's try picking a random node with low degree.
            # Get degrees
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            min_candidates = alive_indices[alive_degrees == min_deg]
            v = np.random.choice(min_candidates)
            
            selected2.append(v)
            to_remove = np.where(conflict[v])[0]
            to_remove = np.append(to_remove, v)
            alive[to_remove] = False
            
        selected2_set = set(selected2)
        selected2_set = local_search(selected2_set, deadline - 2)
        
        if len(selected2_set) > len(best_selected):
            best_selected = list(selected2_set)

    # Extract vectors
    if not best_selected:
        return np.array([]).reshape(0, D)
        
    result = candidates[best_selected]
    return result

# EVOLVE-BLOCK-END
