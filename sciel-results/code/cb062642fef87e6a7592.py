"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int8)
    vectors_v1 = np.array(v1_candidates, dtype=np.int8)

    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # Compute conflict matrix: conflict if |dot| > 2
    dots = vectors_v4.astype(np.int32) @ vectors_v4.astype(np.int32).T
    conflicts = np.abs(dots) > 2
    np.fill_diagonal(conflicts, False)

    # Convert to bitsets for fast operations
    # Each vertex's neighbors as a set of indices
    neighbor_sets = [set(np.where(conflicts[i])[0]) for i in range(N4)]
    neighbor_counts = np.array([len(ns) for ns in neighbor_sets], dtype=np.int32)

    # Branch and bound maximum independent set
    # Use a coloring-based upper bound (greedy coloring of remaining graph)
    
    def greedy_coloring(candidates):
        """Greedy coloring of the subgraph induced by candidates.
        Returns number of colors used (upper bound on independence number)."""
        if len(candidates) == 0:
            return 0
        uncolored = list(candidates)
        colors = {}
        color_count = 0
        while uncolored:
            color_count += 1
            # Greedy: pick vertices with highest degree in uncolored subgraph first
            uncolored.sort(key=lambda v: -sum(1 for u in uncolored if u != v and u in neighbor_sets[v]))
            for v in uncolored[:]:
                # Check if v can get this color (no neighbor already has this color)
                can_color = True
                for u in neighbor_sets[v]:
                    if u in colors and colors[u] == color_count:
                        can_color = False
                        break
                if can_color:
                    colors[v] = color_count
                    uncolored.remove(v)
        return color_count

    def branch_and_bound(candidates, current_set, current_size, best_size, best_set, time_limit):
        """Branch and bound for maximum independent set."""
        if time.time() > time_limit:
            return current_size, current_set.copy()
        
        if len(candidates) == 0:
            if current_size > best_size:
                best_size = current_size
                best_set[:] = current_set
            return best_size, best_set
        
        # Upper bound via greedy coloring
        ub = greedy_coloring(candidates)
        if current_size + ub <= best_size:
            return best_size, best_set
        
        # Pick vertex with highest degree in candidate subgraph
        deg_in_cand = {v: sum(1 for u in candidates if u != v and u in neighbor_sets[v]) for v in candidates}
        v = max(candidates, key=lambda x: deg_in_cand[x])
        
        # Branch 1: include v
        new_candidates = [u for u in candidates if u != v and u not in neighbor_sets[v]]
        current_set[v] = True
        size1, _ = branch_and_bound(new_candidates, current_set, current_size + 1, 
                                      best_size, best_set, time_limit)
        current_set[v] = False
        
        # Branch 2: exclude v
        new_candidates2 = [u for u in candidates if u != v]
        size2, _ = branch_and_bound(new_candidates2, current_set, current_size,
                                      best_size, best_set, time_limit)
        
        return max(size1, size2), best_set

    # Use a heuristic initial solution: greedy min-degree
    def greedy_min_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        while np.any(alive) and time.time() < deadline - 10:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            # Pick vertex with min degree among alive
            min_deg = min(sum(1 for u in alive_indices if u in neighbor_sets[v]) for v in alive_indices)
            cands = [v for v in alive_indices if sum(1 for u in alive_indices if u in neighbor_sets[v]) == min_deg]
            chosen = cands[0]
            in_set[chosen] = True
            to_remove = {chosen} | neighbor_sets[chosen]
            for u in to_remove:
                alive[u] = False
        return in_set

    # Get initial solution
    init_set = greedy_min_degree()
    best_size = init_set.sum()
    best_set = init_set.copy()

    # Try branch and bound with time limit
    # Work on the complement: find max independent set
    # Start with all vertices
    all_vertices = list(range(N4))
    
    # Use a simpler approach: iterative improvement with local search
    # since full B&B on 5280 vertices is too slow
    
    # Better: use the structure. Group vectors by their position set.
    # There are C(11,4) = 330 position sets, each with 16 sign variants.
    
    # For each position set, we can pick at most a few vectors.
    # Two vectors with same positions: dot = 4 - 2*(sign differences).
    # dot > 2 iff sign differences < 1, i.e., 0 differences (same vector) or... 
    # dot = 4 - 2d > 2 => d < 1 => d = 0. So only identical vectors conflict within same position set.
    # Wait: dot = sum of products over shared positions. If same positions, all 4 shared.
    # dot = s1*t1 + s2*t2 + s3*t3 + s4*t4 where s,t in {±1}.
    # dot = 4 - 2*(number of sign disagreements).
    # dot > 2 iff 4 - 2d > 2 iff d < 1 iff d = 0.
    # So within same position set, only the exact same vector conflicts!
    
    # Between different position sets:
    # If they share k positions, dot = sum over shared positions of s_i * t_i.
    # For k=3: dot = 3 - 2d where d is sign disagreements among shared.
    #   dot > 2 iff d = 0.
    # For k=2: dot = 2 - 2d, max is 2, never > 2.
    # For k≤1: dot ≤ 1, never > 2.
    
    # So conflicts only between:
    # 1. Same vector (excluded by diagonal)
    # 2. Same position set, different sign pattern: NO conflict (dot ≤ 2)
    # 3. Different position sets sharing 3+ positions with specific sign patterns
    
    # This means: within each position set, we can take ALL 16 vectors!
    # Conflicts only arise between different position sets.
    
    # Let's verify: two vectors with same 4 positions but different signs.
    # e.g., (1,1,1,1) and (1,1,1,-1): dot = 1+1+1-1 = 2. Not > 2. OK.
    # e.g., (1,1,1,1) and (1,1,-1,-1): dot = 1+1-1-1 = 0. OK.
    # e.g., (1,1,1,1) and (1,-1,1,-1): dot = 1-1+1-1 = 0. OK.
    # Max dot with same positions is 4 (identical) or 2 (one sign flip). Never > 2 for distinct.
    
    # So the problem reduces to: for each of 330 position sets, we can take all 16 vectors,
    # but we need to check conflicts between different position sets.
    
    # Actually wait - we can take all 16 from a position set only if none of them conflict
    # with vectors from other position sets we also take.
    
    # Let me re-examine. The conflict graph has 5280 vertices. Within each group of 16 (same positions),
    # there are NO edges (since dot ≤ 2 for distinct vectors). Between groups, there are edges.
    
    # So the problem is: select a subset of position sets (and for each, which sign patterns)
    # such that no two selected vectors conflict.
    
    # Since within a group there are no conflicts, if we decide to use a position set,
    # we should use ALL 16 vectors (they don't conflict with each other).
    
    # So the problem becomes: find max number of position sets such that no vector from one
    # conflicts with any vector from another.
    
    # Two position sets P and Q conflict if there exist sign patterns s (for P) and t (for Q)
    # such that dot(v_s, v_t) > 2.
    
    # This happens when |P ∩ Q| = 3 and the signs agree on all 3 shared positions (dot = 3 > 2).
    
    # So two position sets are "compatible" if they don't share exactly 3 positions with matching signs possible.
    # Actually, they're incompatible if they share 3 positions (because we can choose signs to match).
    
    # Wait, we're choosing ALL sign patterns for each position set. So if P and Q share 3 positions,
    # there exist s and t with matching signs on those 3 positions, giving dot = 3 > 2.
    
    # Therefore: two position sets are compatible iff |P ∩ Q| ≠ 3.
    
    # So the problem reduces to: find the maximum number of 4-subsets of {0,...,10}
    # such that no two share exactly 3 elements.
    
    # This is a combinatorial design problem!
    
    # Let me solve this directly.
    
    # All 4-subsets of an 11-element set: C(11,4) = 330.
    # Two 4-subsets share exactly 3 elements iff their symmetric difference is 2 elements.
    
    # We want max subset of 4-subsets with no two sharing exactly 3 elements.
    
    # This is equivalent to: max code of length 11, weight 4, with minimum distance... 
    # Two weight-4 vectors have Hamming distance 2*(4 - |intersection|).
    # |intersection| = 3 => Hamming distance = 2.
    # |intersection| = 2 => Hamming distance = 4.
    # |intersection| = 1 => Hamming distance = 6.
    # |intersection| = 0 => Hamming distance = 8.
    
    # We want no two with Hamming distance 2, i.e., minimum Hamming distance ≥ 4.
    
    # This is A(11, 4, 4) - the maximum size of a binary code of length 11, minimum distance 4, weight 4.
    
    # By the Plotkin bound or other bounds... Let me just solve it computationally.
    
    # Generate all 4-subsets
    all_4subsets = list(itertools.combinations(range(D), 4))
    n_subsets = len(all_4subsets)
    
    # Build incompatibility graph: two subsets incompatible if they share exactly 3 elements
    incompat = np.zeros((n_subsets, n_subsets), dtype=bool)
    for i in range(n_subsets):
        for j in range(i+1, n_subsets):
            shared = len(set(all_4subsets[i]) & set(all_4subsets[j]))
            if shared == 3:
                incompat[i, j] = True
                incompat[j, i] = True
    
    # Now find max independent set in this graph of 330 vertices
    # This is much smaller!
    
    # Use branch and bound with coloring
    def max_independent_set_330(n, incompat, time_limit):
        """Max independent set on n-vertex graph."""
        best_size = 0
        best_set = np.zeros(n, dtype=bool)
        
        # Greedy initial solution
        alive = np.ones(n, dtype=bool)
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0:
                break
            # Min degree
            degs = [sum(incompat[v, u] for u in alive_idx if u != v) for v in alive_idx]
            min_d = min(degs)
            cands = [v for v, d in zip(alive_idx, degs) if d == min_d]
            v = cands[0]
            best_set[v] = True
            to_remove = np.where(incompat[v])[0]
            to_remove = np.append(to_remove, v)
            alive[to_remove] = False
        
        best_size = best_set.sum()
        
        # Branch and bound
        def bbs(candidates, current, size, best_sz, best_sol, tl):
            nonlocal best_size, best_set
            if time.time() > tl:
                return
            if len(candidates) == 0:
                if size > best_sz:
                    best_sz = size
                    best_sol[:] = current
                return
            
            # Upper bound: greedy coloring
            # Simple bound: len(candidates)
            if size + len(candidates) <= best_sz:
                return
            
            # Pick vertex with max degree in candidate subgraph
            cand_list = list(candidates)
            degs = {v: sum(1 for u in cand_list if u != v and incompat[v, u]) for v in cand_list}
            v = max(cand_list, key=lambda x: degs[x])
            
            # Branch 1: include v
            new_cand1 = [u for u in cand_list if u != v and not incompat[v, u]]
            current[v] = True
            bbs(new_cand1, current, size + 1, best_sz, best_sol, tl)
            current[v] = False
            
            # Branch 2: exclude v
            new_cand2 = [u for u in cand_list if u != v]
            bbs(new_cand2, current, size, best_sz, best_sol, tl)
        
        current = np.zeros(n, dtype=bool)
        bbs(list(range(n)), current, 0, best_size, best_set, time_limit)
        
        return best_size, best_set
    
    # Solve the 330-vertex problem
    bnb_deadline = deadline - 5
    n_compat, compat_set = max_independent_set_330(n_subsets, incompat, bnb_deadline)
    
    # For each compatible position set, take all 16 sign variants
    result_vectors = []
    for i in range(n_subsets):
        if compat_set[i]:
            positions = all_4subsets[i]
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for pos, s in zip(positions, signs):
                    v[pos] = s
                result_vectors.append(v)
    
    # Add weight-1 vectors
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            result_vectors.append(v)
    
    result = np.array(result_vectors, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
