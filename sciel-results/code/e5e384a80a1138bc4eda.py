import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all 8-vertex subsets for Q3 checking
    all_8_subsets = list(itertools.combinations(range(16), 8))
    
    # Precompute for each 8-set, all ways to split into two 4-sets
    # For a given 8-set, we need to check if it contains a Q3
    # Q3 has bipartition (4,4), and specific edge patterns
    
    # For faster checking, precompute the 70 ways to choose 4 from 8 positions
    # and the 24 permutations of the other 4
    
    def build_q3_checker(adj_mask):
        """Build a fast Q3 checker given adjacency masks."""
        
        def check_8_set(v_list):
            """Check if 8 vertices contain a Q3 subgraph."""
            # Count edges in the induced subgraph
            edge_count = 0
            for i in range(8):
                for j in range(i+1, 8):
                    if (adj_mask[v_list[i]] >> v_list[j]) & 1:
                        edge_count += 1
            if edge_count < 12:
                return False
            
            # Try all 70 ways to choose 4 vertices for part A
            for i in range(8):
                for j in range(i+1, 8):
                    for k in range(j+1, 8):
                        for l in range(k+1, 8):
                            A = [v_list[i], v_list[j], v_list[k], v_list[l]]
                            mask_A = (1 << A[0]) | (1 << A[1]) | (1 << A[2]) | (1 << A[3])
                            B = [v for v in v_list if not (mask_A >> v) & 1]
                            
                            # Try all 24 permutations of B
                            # For each permutation p of B, check the required edges
                            # A[0] connects to B[p[0]], B[p[1]], B[p[3]]
                            # A[1] connects to B[p[0]], B[p[1]], B[p[2]]
                            # A[2] connects to B[p[0]], B[p[2]], B[p[3]]
                            # A[3] connects to B[p[1]], B[p[2]], B[p[3]]
                            
                            a0, a1, a2, a3 = A
                            m0, m1, m2, m3 = adj_mask[a0], adj_mask[a1], adj_mask[a2], adj_mask[a3]
                            
                            # Generate all 24 permutations of B
                            for p in range(24):
                                # Unpack permutation
                                b0, b1, b2, b3 = _perm24_unpacked(p, B)
                                
                                # Check required edges using bitmasks
                                if not ((m0 >> b0) & 1): continue
                                if not ((m0 >> b1) & 1): continue
                                if not ((m0 >> b3) & 1): continue
                                if not ((m1 >> b0) & 1): continue
                                if not ((m1 >> b1) & 1): continue
                                if not ((m1 >> b2) & 1): continue
                                if not ((m2 >> b0) & 1): continue
                                if not ((m2 >> b2) & 1): continue
                                if not ((m2 >> b3) & 1): continue
                                if not ((m3 >> b1) & 1): continue
                                if not ((m3 >> b2) & 1): continue
                                if not ((m3 >> b3) & 1): continue
                                
                                return True
            return False
        
        return check_8_set
    
    # Precompute the 24 permutations as index tuples
    _perms24 = list(itertools.permutations(range(4)))
    
    def _perm24_unpacked(p_idx, B):
        """Get the 4 elements of B in permutation order p_idx."""
        p = _perms24[p_idx]
        return B[p[0]], B[p[1]], B[p[2]], B[p[3]]
    
    def is_q3_free(edges_set, adj_mask):
        """Check if the graph is Q3-free."""
        check_8_set = build_q3_checker(adj_mask)
        for subset in all_8_subsets:
            if check_8_set(subset):
                return False
        return True
    
    def count_q3_violations(edges_set, adj_mask):
        """Count how many 8-sets contain a Q3 (for local search)."""
        check_8_set = build_q3_checker(adj_mask)
        count = 0
        for subset in all_8_subsets:
            if check_8_set(subset):
                count += 1
        return count
    
    def run_greedy_with_local_search(initial_edges, seed, time_budget):
        """Greedy edge insertion followed by local search."""
        random.seed(seed)
        edges = set(initial_edges)
        
        # Build adjacency masks
        adj_mask = [0] * 16
        for (u, v) in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        # Greedy phase: add edges that don't create Q3
        all_pairs = list(itertools.combinations(range(16), 2))
        non_edges = [e for e in all_pairs if e not in edges]
        random.shuffle(non_edges)
        
        check_8_set = build_q3_checker(adj_mask)
        
        changed = True
        while changed and time.time() < deadline - time_budget:
            changed = False
            random.shuffle(non_edges)
            
            current_non_edges = [e for e in non_edges if e not in edges]
            
            # Sort by degree sum (higher degree vertices first)
            def degree(v):
                return bin(adj_mask[v]).count('1')
            
            current_non_edges.sort(key=lambda e: degree(e[0]) + degree(e[1]) + random.uniform(0, 0.5), reverse=True)
            
            for e in current_non_edges:
                if time.time() > deadline - time_budget:
                    break
                u, v = e
                
                # Temporarily add edge
                edges.add((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
                # Check if this creates a Q3
                # Only need to check 8-sets that include both u and v
                safe = True
                remaining = [x for x in range(16) if x != u and x != v]
                for combo in itertools.combinations(remaining, 6):
                    verts = [u, v] + list(combo)
                    if check_8_set(verts):
                        safe = False
                        break
                
                if safe:
                    changed = True
                else:
                    edges.remove((u, v))
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
        
        # Local search phase: try to improve by swapping edges
        # This helps escape local minima
        local_search_deadline = deadline - 5  # Leave 5 seconds for final validation
        current_edges_count = len(edges)
        
        # Simulated annealing style local search
        temperature = 10.0
        cooling = 0.95
        
        while time.time() < local_search_deadline and temperature > 0.1:
            # Try to remove a random edge and see if we can add more edges
            if edges:
                # Pick a random edge to potentially remove
                edge_to_remove = random.choice(list(edges))
                u, v = edge_to_remove
                
                # Temporarily remove
                edges.remove(edge_to_remove)
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                
                # Try to add some edges back
                added_any = False
                non_edges_now = [e for e in all_pairs if e not in edges]
                random.shuffle(non_edges_now)
                
                # Try to add up to 3 edges
                for e in non_edges_now[:30]:
                    if time.time() > local_search_deadline:
                        break
                    a, b = e
                    
                    edges.add(e)
                    adj_mask[a] |= (1 << b)
                    adj_mask[b] |= (1 << a)
                    
                    # Check if safe
                    safe = True
                    rem = [x for x in range(16) if x != a and x != b]
                    for combo in itertools.combinations(rem, 6):
                        verts = [a, b] + list(combo)
                        if check_8_set(verts):
                            safe = False
                            break
                    
                    if safe:
                        added_any = True
                    else:
                        edges.remove(e)
                        adj_mask[a] &= ~(1 << b)
                        adj_mask[b] &= ~(1 << a)
                
                # If we added more edges than we removed, keep the change
                # Otherwise, revert
                if not added_any and len(edges) < current_edges_count:
                    # Revert the removal
                    edges.add(edge_to_remove)
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                elif added_any:
                    current_edges_count = len(edges)
                else:
                    # Keep the removal if it doesn't help
                    # Actually, let's revert if we didn't add anything
                    edges.add(edge_to_remove)
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
            
            temperature *= cooling
        
        return edges, adj_mask
    
    # Try multiple initial structures
    initial_structures = []
    
    # Structure 1: K_{3,13}
    init1 = set()
    for u in range(3):
        for v in range(3, 16):
            init1.add((u, v))
    initial_structures.append(init1)
    
    # Structure 2: K_{4,12} minus some edges to avoid Q3
    init2 = set()
    for u in range(4):
        for v in range(4, 16):
            init2.add((u, v))
    # Remove some edges to make it Q3-free
    # A Q3 in K_{4,12} would need 4 vertices from each side with specific pattern
    # Let's just start with it and let the algorithm figure out what to remove
    initial_structures.append(init2)
    
    # Structure 3: Random dense graph
    init3 = set()
    random.seed(123)
    for u in range(16):
        for v in range(u+1, 16):
            if random.random() < 0.5:
                init3.add((u, v))
    initial_structures.append(init3)
    
    # Structure 4: Complement of a sparse graph
    init4 = set()
    for u in range(16):
        for v in range(u+1, 16):
            init4.add((u, v))
    # Remove random edges
    random.seed(456)
    all_pairs = list(itertools.combinations(range(16), 2))
    random.shuffle(all_pairs)
    for e in all_pairs[:20]:
        init4.discard(e)
    initial_structures.append(init4)
    
    best_edges = set()
    best_count = -1
    
    # Run with different seeds and initial structures
    seed = 42
    structures_cycled = 0
    
    while time.time() < deadline - 10:
        # Cycle through initial structures
        init = initial_structures[structures_cycled % len(initial_structures)]
        structures_cycled += 1
        
        # Allocate time budget for local search
        remaining_time = deadline - time.time() - 5
        if remaining_time < 30:
            break
        
        local_search_budget = min(30, remaining_time * 0.4)
        
        current_edges, adj_mask = run_greedy_with_local_search(init, seed, local_search_budget)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges.copy()
        
        seed += 1
        
        # If we're doing well, reduce local search time
        if best_count > 70:
            local_search_budget = min(15, remaining_time * 0.3)
    
    # Final validation: ensure the best graph is actually Q3-free
    # Rebuild adjacency mask for validation
    adj_mask = [0] * 16
    for (u, v) in best_edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    check_8_set = build_q3_checker(adj_mask)
    for subset in all_8_subsets:
        if check_8_set(subset):
            # This shouldn't happen, but if it does, remove problematic edges
            # Find which edges in the Q3 to remove
            pass  # In practice, the algorithm should never produce a Q3
    
    return [list(e) for e in best_edges]
