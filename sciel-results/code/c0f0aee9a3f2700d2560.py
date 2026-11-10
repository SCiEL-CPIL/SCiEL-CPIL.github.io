"""Combined program: Fast bitmask C6 check + Greedy + Hierarchical Local Search."""
# EVOLVE-BLOCK-START
import time
import random

DEADLINE = 110.0  # Safety margin before 120s kill

def _check_c6_through_edge(adj, u, v, n=40):
    """
    Check if adding edge (u,v) creates a C6.
    A C6 containing edge (u,v) has the form u-v-x1-x2-x3-x4-u.
    This requires a path of length 4 from u to v in G - {u, v} 
    (specifically, u is connected to x4, v is connected to x1, 
    and x1-x2-x3-x4 is a path in the remaining graph).
    """
    # x1 in N(v) \ {u}
    x1_mask = adj[v] & ~(1 << u)
    while x1_mask:
        x1 = (x1_mask & (-x1_mask)).bit_length() - 1
        x1_mask &= x1_mask - 1
        
        # x4 in N(u) \ {v, x1}
        x4_mask = adj[u] & ~(1 << v) & ~(1 << x1)
        while x4_mask:
            x4 = (x4_mask & (-x4_mask)).bit_length() - 1
            x4_mask &= x4_mask - 1
            
            # x2 in N(x1) \ {v, x4, u}
            # x2 must be neighbor of x1.
            # x2 != v, x2 != x4, x2 != u
            x2_mask = adj[x1] & ~(1 << v) & ~(1 << x4) & ~(1 << u)
            while x2_mask:
                x2 = (x2_mask & (-x2_mask)).bit_length() - 1
                x2_mask &= x2_mask - 1
                
                # x3 in N(x4) \ {u, x2, v, x1}
                # x3 must be neighbor of x4.
                # x3 != u, x3 != x2, x3 != v, x3 != x1
                x3_mask = adj[x4] & ~(1 << u) & ~(1 << x2) & ~(1 << v) & ~(1 << x1)
                while x3_mask:
                    x3 = (x3_mask & (-x3_mask)).bit_length() - 1
                    x3_mask &= x3_mask - 1
                    
                    # Check if x2-x3 is an edge
                    if adj[x2] & (1 << x3):
                        return False  # C6 found
    return True  # No C6 through this edge

def _count_edges(adj_bits, n=40):
    count = 0
    for i in range(n):
        count += bin(adj_bits[i]).count('1')
    return count // 2

def _get_edges(adj_bits, n=40):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                edges.append([i, j])
    return edges

def _is_c6_free(adj_bits, n=40):
    """Full check: verify no C6 exists in the graph."""
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                if not _check_c6_through_edge(adj_bits, i, j, n):
                    return False
    return True

def _random_greedy(n=40, seed=None, max_time=1.0):
    """Randomized greedy: try edges in random order, add if C6-free."""
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    adj = [0] * n
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() - start_time > max_time:
            break
        # Tentatively add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not _check_c6_through_edge(adj, u, v, n):
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    return adj, _count_edges(adj, n)

def _local_search_1add_1swap_2swap(adj, n=40, max_time=10.0):
    """
    Local search using:
    1. 1-add: Add a non-edge if it doesn't create a C6.
    2. 1-swap: Remove 1 edge, add 1 non-edge (keeps edge count, changes structure).
    3. 2-swap: Remove 2 edges, add 2 non-edges (keeps edge count, changes structure).
    """
    start_time = time.time()
    current = _count_edges(adj, n)
    best_adj = adj[:]
    best_count = current
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        improved = False
        
        # 1. Try to add edges (1-add)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() - start_time > max_time:
                return best_adj, best_count
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _check_c6_through_edge(adj, u, v, n):
                current += 1
                improved = True
                if current > best_count:
                    best_count = current
                    best_adj = adj[:]
                no_improve_count = 0
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        if improved:
            continue
            
        # If no 1-add, try 1-swaps
        # We need to be careful: a 1-swap might allow a future 1-add.
        cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if len(cur_edges) == 0 or len(non_edges) == 0:
            break
            
        # Try a limited number of 1-swaps
        swaps_tried = 0
        max_swaps_to_try = 50
        found_1swap = False
        
        while swaps_tried < max_swaps_to_try and time.time() - start_time < max_time:
            swaps_tried += 1
            rem_u, rem_v = random.choice(cur_edges)
            add_u, add_v = random.choice(non_edges)
            
            # Ensure they are distinct
            if (rem_u, rem_v) == (add_u, add_v) or (rem_u, rem_v) == (add_v, add_u):
                continue
                
            # Apply removal
            adj[rem_u] &= ~(1 << rem_v)
            adj[rem_v] &= ~(1 << rem_u)
            
            # Apply addition
            adj[add_u] |= (1 << add_v)
            adj[add_v] |= (1 << add_u)
            
            # Check if C6-free
            # Only need to check the added edge? 
            # Removing edges cannot create C6. Adding edge (add_u, add_v) might.
            if _check_c6_through_edge(adj, add_u, add_v, n):
                # Valid swap
                found_1swap = True
                break
            else:
                # Revert
                adj[add_u] &= ~(1 << add_v)
                adj[add_v] &= ~(1 << add_u)
                adj[rem_u] |= (1 << rem_v)
                adj[rem_v] |= (1 << rem_u)
                
        if found_1swap:
            # We made a 1-swap. This doesn't increase edge count, but might open up for 1-add.
            # We don't update best_count, but we continue the loop to try 1-add again.
            continue
        
        # If no 1-swap found (or tried enough), try 2-swaps
        # Reset lists as they might be stale or we want fresh randomness
        cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if len(cur_edges) < 2 or len(non_edges) < 2:
            break
            
        swaps_tried = 0
        max_swaps_to_try = 100
        found_2swap = False
        
        while swaps_tried < max_swaps_to_try and time.time() - start_time < max_time:
            swaps_tried += 1
            # Pick 2 distinct edges to remove
            e1, e2 = random.sample(cur_edges, 2)
            # Pick 2 distinct non-edges to add
            a1, a2 = random.sample(non_edges, 2)
            
            # Check for overlaps (shouldn't happen as one is edge, one is non-edge, but just in case)
            # Actually, e1, e2 are edges, a1, a2 are non-edges. So no overlap.
            
            # Apply removals
            adj[e1[0]] &= ~(1 << e1[1])
            adj[e1[1]] &= ~(1 << e1[0])
            adj[e2[0]] &= ~(1 << e2[1])
            adj[e2[1]] &= ~(1 << e2[0])
            
            # Apply additions
            adj[a1[0]] |= (1 << a1[1])
            adj[a1[1]] |= (1 << a1[0])
            adj[a2[0]] |= (1 << a2[1])
            adj[a2[1]] |= (1 << a2[0])
            
            # Check if C6-free
            is_valid = True
            if not _check_c6_through_edge(adj, a1[0], a1[1], n):
                is_valid = False
            elif not _check_c6_through_edge(adj, a2[0], a2[1], n):
                is_valid = False
            
            if is_valid:
                found_2swap = True
                break
            else:
                # Revert
                adj[a2[0]] &= ~(1 << a2[1])
                adj[a2[1]] &= ~(1 << a2[0])
                adj[a1[0]] &= ~(1 << a1[1])
                adj[a1[1]] &= ~(1 << a1[0])
                adj[e2[0]] |= (1 << e2[1])
                adj[e2[1]] |= (1 << e2[0])
                adj[e1[0]] |= (1 << e1[1])
                adj[e1[1]] |= (1 << e1[0])
                
        if found_2swap:
            continue
            
        # If no moves found, we are stuck.
        # To avoid infinite loop, we can break or try a larger perturbation.
        # For now, break out of this local search instance.
        break
        
    return best_adj, best_count

def _known_good(n=40):
    """8 disjoint K5s = 100 edges, C6-free."""
    adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a+1, 5):
                adj[base+a] |= (1 << (base+b))
                adj[base+b] |= (1 << (base+a))
    return adj

def construct():
    start_time = time.time()
    n = 40
    best_adj = None
    best_edges = 0
    
    # Phase 1: Start from 8xK5
    adj = _known_good(n)
    best_adj = adj[:]
    best_edges = _count_edges(adj, n)
    
    # Phase 2: Local search from 8xK5
    if time.time() - start_time < DEADLINE - 5:
        adj_ls, count_ls = _local_search_1add_1swap_2swap(adj, n, max_time=min(10.0, DEADLINE - (time.time() - start_time) - 2))
        if count_ls > best_edges:
            best_edges = count_ls
            best_adj = adj_ls[:]
    
    # Phase 3: Multiple random greedy restarts + local search
    attempt = 0
    while time.time() - start_time < DEADLINE - 5:
        attempt += 1
        remaining = DEADLINE - (time.time() - start_time) - 2
        if remaining < 2:
            break
        
        # Short greedy
        mt_greedy = min(1.5, remaining / 4)
        adj_g, count_g = _random_greedy(n, seed=1000 + attempt, max_time=mt_greedy)
        
        if count_g > best_edges:
            best_edges = count_g
            best_adj = adj_g[:]
            
        # Apply local search on the greedy result
        # Give it a reasonable chunk of time
        mt_ls = min(8.0, remaining / 2)
        adj_l, count_l = _local_search_1add_1swap_2swap(adj_g, n, max_time=mt_ls)
        if count_l > best_edges:
            best_edges = count_l
            best_adj = adj_l[:]
            
        # If we hit a high score, do one more refinement on the best found so far
        if best_edges >= 110 and time.time() - start_time < DEADLINE - 10:
             # Refine the current best
             adj_ref, count_ref = _local_search_1add_1swap_2swap(best_adj, n, max_time=5.0)
             if count_ref > best_edges:
                 best_edges = count_ref
                 best_adj = adj_ref[:]

    # Final verification
    if best_adj is None:
        best_adj = _known_good(n)
    
    if not _is_c6_free(best_adj, n):
        best_adj = _known_good(n)
    
    return _get_edges(best_adj, n)
# EVOLVE-BLOCK-END
