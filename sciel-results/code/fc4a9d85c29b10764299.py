    # It returns zeros.
    
    # Maybe my calculation of "complement" is wrong?
    # "Clockwise from s to t uses edges s, s+1, ..., t-1 (mod 10)"
    # If s=1, t=0.
    # CW: 1, 2, ..., 9. (Since 0 is target, we stop before 0? "t-1". 0-1 = -1 = 9 mod 10).
    # So CW edges: 1,2,3,4,5,6,7,8,9.
    # CCW edges: "all other". So edge 0.
    # Union: 0..9.
    
    # It really seems constant.
    
    # Is it possible the evaluator uses a different set of demands?
    # "Demands ... in this order: ..."
    # "Ring with nodes 0..9"
    
    # Let's assume the problem is NOT trivial and I'm missing a detail.
    # What if "counter-clockwise" means the shortest path in the counter-clockwise direction?
    # No, "uses all other ring edges" is very explicit.
    
    # Let's look at the inspiration 1 description again.
    # "uses numpy, random, time; defines _build, _loads_from_dir, _local_search, _anneal, ring_routing"
    # It's a local search / annealing approach.
    # If the objective is constant, local search should find the same value (or fail to improve, staying at 1.0).
    # Getting 0.88 implies it found a WORSE solution.
    # This can only happen if the objective function is NOT constant.
    
    # Let's reconsider the wording "counter-clockwise uses all other ring edges".
    # Could it be a translation artifact or a trick?
    # If it meant "uses the edges on the counter-clockwise path", and that path is the complement, then it is constant.
    # 
    # What if the "load" is not just sum of weights?
    # "Each demand adds its weight to the load of every edge on its route."
    
    # Let's try to interpret "all other ring edges" differently.
    # Maybe "other" refers to "other than the direct edge"? No.
    # 
    # Let's assume the standard SONET ring problem interpretation:
    # Choose direction to minimize max load.
    # In standard problems, the CW path is one way, CCW is the other.
    # They are complementary.
    # So each demand contributes to EVERY edge.
    # 
    # Wait! In a ring, if you route s->t CW, you use the CW arc.
    # If you route s->t CCW, you use the CCW arc.
    # The union of CW arc and CCW arc is the whole ring.
    # So every edge is used by every demand exactly once?
    # No. A demand is routed EITHER CW OR CCW.
    # If routed CW, it uses CW edges.
    # If routed CCW, it uses CCW edges.
    # 
    # Ah! I made a mistake in my constant load deduction.
    # A demand does NOT use both paths. It uses ONE path.
    # "An entry >= 0.5 routes the demand counter-clockwise, any other value clockwise."
    # So for demand d, it is assigned EITHER cw_edges(d) OR ccw_edges(d).
    # 
    # Load(e) = Sum of weights of demands d such that e is in chosen_path(d).
    # 
    # My previous deduction: "Since cw_mask and ccw_mask are complements... every edge carries every demand's weight."
    # This was WRONG.
    # The demand chooses ONE mask. It does not add weight to both.
    # 
    # So, Load(e) = Sum_{d: e in path(d)} w_d.
    # 
    # This is the standard problem. The load varies depending on routing.
    # 
    # Okay, so the problem is non-trivial.
    # 
    # Goal: Minimize max_e Load(e).
    # 
    # Algorithm:
    # 1. Exact search (DFS with pruning) for 15 demands. 2^15 = 32768 states.
    # 2. This is small enough to solve exactly with DFS + Branch and Bound.
    # 
    # Let's implement a robust DFS with pruning.
    # 
    # Steps:
    # 1. Pre-compute edge masks for CW and CCW for each demand.
    # 2. Sort demands by weight descending (heuristic: constrain largest first).
    # 3. DFS(k):
    #    - If k == D: update best_load, best_routing.
    #    - Pruning:
    #       - If current_max_load >= best_load, return.
    #       - Lower bound: For each edge e, min_possible_load_e = current_load[e] + sum of weights of remaining demands that MUST use e?
    #         No, remaining demands can choose either path.
    #         The minimum additional load on edge e from remaining demands is the sum of weights of remaining demands d where e is in BOTH cw(d) and ccw(d)?
    #         No, e is in exactly one of cw(d) or ccw(d).
    #         So for a remaining demand d, it will add w_d to edge e IF we choose the direction that includes e.
    #         The MINIMUM load we can add to edge e from demand d is 0 (if we choose the direction that doesn't use e).
    #         So the lower bound for edge e is just current_load[e].
    #         This pruning is weak.
    #         
    #         Better lower bound:
    #         Total load added to the ring by remaining demands is Sum(w_d for d >= k).
    #         This total load is distributed among 10 edges.
    #         The max load must be at least current_max_load.
    #         Also, the max load must be at least (Sum of all loads on edges) / 10?
    #         Sum of loads on all edges = Sum_{d<k} (w_d * 10) + Sum_{d>=k} (w_d * 10)?
    #         No. Each demand uses a path of length L_d.
    #         Sum of edge loads = Sum_{d} w_d * L_d.
    #         This is constant regardless of direction?
    #         Length of CW path + Length of CCW path = 10.
    #         So for each demand, it uses a path. The length varies.
    #         
    #         Actually, a stronger bound:
    #         Let S_rem = Sum_{d>=k} w_d.
    #         The remaining demands must be routed.
    #         The increase in max load is at least ...
    #         
    #         Simple bound:
    #         If current_max_load > best_load, prune.
    #         With D=15, this might be sufficient. 32768 leaves is tiny.
    # 
    # 4. To speed up, we can use bitsets.
    
    # Let's refine the DFS.
    
    # Data structures:
    # n = 10
    # D = 15
    # cw_masks: list of int (10-bit)
    # ccw_masks: list of int (10-bit)
    # weights: list of int
    
    # State:
    # current_loads: list of 10 ints
    # current_dirs: list of 15 ints (0/1)
    
    # Optimization:
    # Instead of list of ints for loads, use a tuple or array.
    # Since 2^15 is small, even Python loops might be fast enough.
    # 32768 * 15 * 10 operations ~ 5 million ops. Fast in Python.
    
    # Let's write the code.
    
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1),
        (0, 4, 5), (2, 8, 2), (5, 9, 3), (1, 4, 2),
        (6, 0, 4), (3, 8, 1), (7, 2, 3), (4, 9, 2),
        (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n = 10
    D = len(demands)
    
    cw_masks = []
    ccw_masks = []
    weights = []
    
    for s, t, w in demands:
        weights.append(w)
        mask_cw = 0
        if s <= t:
            for e in range(s, t):
                mask_cw |= (1 << e)
        else:
            for e in range(s, n):
                mask_cw |= (1 << e)
            for e in range(0, t):
                mask_cw |= (1 << e)
        
        mask_ccw = ((1 << n) - 1) ^ mask_cw
        
        cw_masks.append(mask_cw)
        ccw_masks.append(mask_ccw)
        
    # Sort by weight descending
    indices = np.argsort(-np.array(weights))
    ordered_cw = [cw_masks[i] for i in indices]
    ordered_ccw = [ccw_masks[i] for i in indices]
    ordered_weights = [weights[i] for i in indices]
    
    best_load = float('inf')
    best_dirs = [0] * D  # In ordered indices
    
    # Pre-compute suffix sums of weights for a simple bound?
    # Not strictly needed for 15 items.
    
    current_loads = [0] * n
    current_dirs = [0] * D
    
    def dfs(k):
        nonlocal best_load, best_dirs
        
        if k == D:
            max_l = max(current_loads)
            if max_l < best_load:
                best_load = max_l
                best_dirs = current_dirs[:]
            return
        
        # Pruning
        if max(current_loads) >= best_load:
            return
            
        # Branch 1: CW
        mask = ordered_cw[k]
        w = ordered_weights[k]
        
        # Apply
        for e in range(n):
            if (mask >> e) & 1:
                current_loads[e] += w
        
        current_dirs[k] = 0
        
        dfs(k + 1)
        
        # Undo
        for e in range(n):
            if (mask >> e) & 1:
                current_loads[e] -= w
        
        # Branch 2: CCW
        mask = ordered_ccw[k]
        
        # Apply
        for e in range(n):
            if (mask >> e) & 1:
                current_loads[e] += w
        
        current_dirs[k] = 1
        
        dfs(k + 1)
        
        # Undo
        for e in range(n):
            if (mask >> e) & 1:
                current_loads[e] -= w

    dfs(0)
    
    # Map back to original order
    final_routing = np.zeros(D)
    for i in range(D):
        orig_idx = indices[i]
        # best_dirs[i] is 0 (CW) or 1 (CCW)
        # Contract: >= 0.5 is CCW, else CW.
        # So 0 -> 0.0, 1 -> 1.0
        final_routing[orig_idx] = best_dirs[i]
        
    return final_routing

# EVOLVE-BLOCK-END
