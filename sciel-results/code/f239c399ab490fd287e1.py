"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0  # Safety margin before 90s kill
    
    n = 256
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    def is_diagonal_safe(a):
        return 2 * a not in squares

    # Candidates that are diagonal-safe
    all_candidates = [a for a in range(1, n + 1) if is_diagonal_safe(a)]
    M = len(all_candidates)
    
    # Build conflict graph using bitsets for speed
    # adj[i] is bitset of vertices j (j != i) such that all_candidates[i] + all_candidates[j] is a square
    adj = [0] * M
    for idx_i, a in enumerate(all_candidates):
        for idx_j in range(idx_i + 1, M):
            b = all_candidates[idx_j]
            if (a + b) in squares:
                adj[idx_i] |= (1 << idx_j)
                adj[idx_j] |= (1 << idx_i)
    
    # Precompute for fast degree checks
    # deg[i] = number of conflicts for vertex i
    deg = [adj[i].bit_count() for i in range(M)]
    
    def greedy_min_degree(candidates_indices):
        """Greedy minimum degree independent set."""
        remaining = set(candidates_indices)
        independent_set = set()
        while remaining:
            min_deg = float('inf')
            chosen = None
            for v in remaining:
                d = (adj[v] & sum((1 << x for x in remaining), 0)).bit_count()
                if d < min_deg:
                    min_deg = d
                    chosen = v
            independent_set.add(chosen)
            remaining.remove(chosen)
            # Remove neighbors of chosen
            to_remove = adj[chosen] & sum((1 << x for x in remaining), 0)
            while to_remove:
                low_bit = to_remove & -to_remove
                idx = low_bit.bit_length() - 1
                remaining.discard(idx)
                to_remove &= ~low_bit
        return independent_set

    def try_add_fast(s_bitset, v):
        """Check if vertex v can be added to set represented by bitset."""
        return (adj[v] & s_bitset) == 0

    def local_search_improve(s_set):
        """Local search: try additions and 1-swaps."""
        s = set(s_set)
        s_bitset = sum((1 << v for v in s), 0)
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any single node
            for v in range(M):
                if v not in s and try_add_fast(s_bitset, v):
                    s.add(v)
                    s_bitset |= (1 << v)
                    improved = True
            
            # 2. Try to remove one node to add >= 1 node
            for u in list(s):
                s.remove(u)
                s_bitset &= ~(1 << u)
                
                addable = []
                for v in range(M):
                    if v not in s and v != u:
                        if try_add_fast(s_bitset, v):
                            addable.append(v)
                
                if len(addable) >= 2:
                    for v in addable:
                        s.add(v)
                        s_bitset |= (1 << v)
                    improved = True
                    break
                elif len(addable) == 1:
                    # Swap: remove u, add v
                    s.add(addable[0])
                    s_bitset |= (1 << addable[0])
                    improved = True
                    break
                else:
                    s.add(u)
                    s_bitset |= (1 << u)
                    
        return s

    # Initial solution via greedy
    best_set = greedy_min_degree(range(M))
    best_size = len(best_set)
    
    # Local search improvement
    best_set = local_search_improve(best_set)
    best_size = len(best_set)
    
    # Try random restarts with local search
    random.seed(42)
    for trial in range(100):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        shuffled = list(range(M))
        random.shuffle(shuffled)
        current_set = set()
        current_bitset = 0
        for v in shuffled:
            if try_add_fast(current_bitset, v):
                current_set.add(v)
                current_bitset |= (1 << v)
        
        current_set = local_search_improve(current_set)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # Now use exact branch-and-bound (Tomita-style) to try to find larger solution
    # or verify optimality in remaining time
    best_clique_size = best_size
    best_clique_set = set(best_set)
    
    # Convert to complement graph for max clique
    # In complement, edge exists if NOT conflicting (i.e., can be together)
    # comp_adj[i] = bitset of j where i and j can be together (no conflict)
    comp_adj = [0] * M
    for i in range(M):
        # All vertices except i and its conflicts
        all_bits = (1 << M) - 1
        comp_adj[i] = all_bits & ~(adj[i] | (1 << i))
    
    def greedy_color(P):
        """Greedy coloring for upper bound. Returns number of colors."""
        if P == 0:
            return 0
        colors = 0
        U = P
        while U:
            colors += 1
            C = 0
            U_next = U
            while U_next:
                # Pick lowest index
                low_bit = U_next & -U_next
                v_idx = low_bit.bit_length() - 1
                # Check if v can take this color (not adjacent to any in C in comp graph)
                # In comp graph, adjacent means can be together
                # For coloring in max clique: same color = non-adjacent in processing graph
                # Processing graph is comp, so non-adjacent in comp = adjacent in original = conflict
                # Actually for Tomita: we color the candidate set P
                # Two vertices get same color if they are NOT adjacent in the graph being processed
                # We're finding max clique in comp graph, so "adjacent" in comp means can be together
                # Non-adjacent in comp means conflict in original
                # For greedy coloring: v gets color c if v is not adjacent to any vertex already colored c
                # In comp graph: v not adjacent to w in comp means v+w is a square (conflict)
                # So v can share color with C if v is NOT in comp_adj of any vertex in C
                # i.e., for all w in C, v is not in comp_adj[w], i.e., w is not in comp_adj[v]
                # i.e., (comp_adj[v] & C) == 0
                if (comp_adj[v_idx] & C) == 0:
                    C |= low_bit
                    U_next &= ~low_bit
                    # Remove neighbors of v in comp from U_next (they can't share this color)
                    U_next &= ~comp_adj[v_idx]
            U &= ~C
        return colors

    def branch_and_bound(P, Q_size, Q):
        """Branch and bound for max clique in comp graph."""
        nonlocal best_clique_size, best_clique_set, start_time
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if P == 0:
            if Q_size > best_clique_size:
                best_clique_size = Q_size
                best_clique_set = set(Q)
            return
        
        # Upper bound via greedy coloring
        k = greedy_color(P)
        if Q_size + k <= best_clique_size:
            return
        
        # Select vertex with highest degree in P
        max_deg = -1
        v_sel = -1
        temp_P = P
        while temp_P:
            low_bit = temp_P & -temp_P
            v_idx = low_bit.bit_length() - 1
            d = (comp_adj[v_idx] & P).bit_count()
            if d > max_deg:
                max_deg = d
                v_sel = v_idx
            temp_P &= ~low_bit
        
        if v_sel == -1:
            return
        
        v_bit = 1 << v_sel
        
        # Branch 1: Include v_sel
        new_P = P & comp_adj[v_sel]
        Q.add(v_sel)
        branch_and_bound(new_P, Q_size + 1, Q)
        Q.discard(v_sel)
        
        # Branch 2: Exclude v_sel
        new_P2 = P & ~v_bit
        branch_and_bound(new_P2, Q_size, Q)
    
    # Run branch and bound starting from best known
    if time.time() - start_time < TIME_LIMIT - 5:
        initial_P = (1 << M) - 1
        branch_and_bound(initial_P, 0, set())
    
    # Use the best found
    if best_clique_size > best_size:
        best_set = best_clique_set
        best_size = best_clique_size
    
    # Map indices back to values
    result = sorted([all_candidates[i] for i in best_set])
    
    # Final verification
    for i, a in enumerate(result):
        for b in result[i:]:
            s = a + b
            root = math.isqrt(s)
            if root * root == s:
                raise ValueError(f"Conflict: {a} + {b} = {s}")
                
    return result
# EVOLVE-BLOCK-END
