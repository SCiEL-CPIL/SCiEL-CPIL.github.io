import random
import itertools
import time

# Precompute Q3 properties
q3_edges = []
q3_adj = [0]*8
for i in range(8):
    adj = 0
    for j in range(8):
        if i != j and bin(i ^ j).count('1') == 1:
            adj |= (1 << j)
    q3_adj[i] = adj

def contains_q3_fast(subset, adj_bits):
    """
    Fast check if induced subgraph on 8 vertices contains Q3.
    adj_bits[v] is a bitmask of neighbors of vertex v.
    subset is a tuple/list of 8 vertex ids.
    """
    # Map local indices 0-7 to global vertices
    # Build local adjacency bitmasks
    local_adj = [0]*8
    for i in range(8):
        vi = subset[i]
        for j in range(8):
            if i != j and (adj_bits[vi] >> subset[j]) & 1:
                local_adj[i] |= (1 << j)
    
    # Check if local_adj contains Q3 as subgraph
    # Q3 vertices 0-7, edges where popcount(i^j)==1
    # Use backtracking with pruning
    mapping = [-1]*8
    
    # Degree check: all must have degree >= 3 in local graph
    for i in range(8):
        if bin(local_adj[i]).count('1') < 3:
            return False
    
    # Edge count check
    total_deg = sum(bin(local_adj[i]).count('1') for i in range(8))
    if total_deg < 24:  # 12 edges * 2
        return False
    
    # Backtracking
    def backtrack(q3_v):
        if q3_v == 8:
            return True
        # Try each unmapped local vertex
        used_mask = 0
        for u in range(q3_v):
            used_mask |= (1 << mapping[u])
        
        for s_idx in range(8):
            if (used_mask >> s_idx) & 1:
                continue
            # Check compatibility with already mapped neighbors
            valid = True
            for u in range(q3_v):
                if (q3_adj[u] >> q3_v) & 1:
                    if not ((local_adj[mapping[u]] >> s_idx) & 1):
                        valid = False
                        break
            if valid:
                mapping[q3_v] = s_idx
                if backtrack(q3_v + 1):
                    return True
                mapping[q3_v] = -1
        return False
    
    for first in range(8):
        mapping[0] = first
        if backtrack(1):
            return True
        mapping[0] = -1
    
    return False

def creates_q3_with_edge(adj_bits, u, v):
    """
    Check if adding edge (u,v) to the graph represented by adj_bits creates a Q3.
    We only check 8-subsets containing u and v.
    """
    other_vertices = [x for x in range(16) if x != u and x != v]
    for rest in itertools.combinations(other_vertices, 6):
        subset = list(rest) + [u, v]
        if contains_q3_fast(subset, adj_bits):
            return True
    return False

def construct():
    n = 16
    total_edges = n * (n-1) // 2  # 120
    
    # Start with a random graph
    adj_bits = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.6:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    start_time = time.time()
    time_limit = 115
    
    # Phase 1: Remove edges to destroy all Q3s
    # Use a loop that finds a Q3 and removes a random edge from it
    while time.time() - start_time < time_limit - 30:
        # Find a Q3
        found = False
        # Try random subsets first
        for _ in range(100):
            subset = random.sample(range(n), 8)
            if contains_q3_fast(subset, adj_bits):
                found = True
                # Remove a random edge in this subset
                edges_in_q3 = []
                for i in range(8):
                    for j in range(i+1, 8):
                        if (adj_bits[subset[i]] >> subset[j]) & 1:
                            edges_in_q3.append((subset[i], subset[j]))
                if edges_in_q3:
                    u, v = random.choice(edges_in_q3)
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                break
        
        if found:
            continue
            
        # If random didn't find one, do a full scan
        for subset in itertools.combinations(range(n), 8):
            if contains_q3_fast(subset, adj_bits):
                found = True
                edges_in_q3 = []
                for i in range(8):
                    for j in range(i+1, 8):
                        if (adj_bits[subset[i]] >> subset[j]) & 1:
                            edges_in_q3.append((subset[i], subset[j]))
                if edges_in_q3:
                    u, v = random.choice(edges_in_q3)
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                break
        
        if not found:
            break
    
    current_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
    best_edges = current_edges
    best_adj_bits = adj_bits[:]
    
    # Phase 2: Local Search with Swaps
    # We try to improve the edge count by swapping edges or adding edges.
    # We maintain a list of current edges and missing edges.
    
    # Precompute combinations of 6 other vertices for fast check
    # This is static, so we can compute it once? No, u and v change.
    # But the list of "other vertices" is just all except u, v.
    
    while time.time() - start_time < time_limit - 5:
        # Get current edges and missing edges
        current_edges_list = []
        missing_edges_list = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_bits[i] >> j) & 1:
                    current_edges_list.append((i, j))
                else:
                    missing_edges_list.append((i, j))
        
        if not missing_edges_list:
            break
            
        # Try random swaps
        improved = False
        for _ in range(100): # Try 100 swaps per iteration
            if time.time() - start_time > time_limit - 5:
                break
                
            # Pick a random missing edge to add
            u_add, v_add = random.choice(missing_edges_list)
            
            # Check if adding this edge creates a Q3
            # We need to temporarily add it for the check?
            # creates_q3_with_edge assumes the edge is NOT present and checks if adding it creates a Q3.
            # Wait, the function `contains_q3_fast` checks the *current* state.
            # So we must add the edge to adj_bits, check, then remove if bad.
            
            adj_bits[u_add] |= (1 << v_add)
            adj_bits[v_add] |= (1 << u_add)
            
            is_valid = True
            # Check only subsets containing u_add and v_add
            other_vertices = [x for x in range(n) if x != u_add and x != v_add]
            for rest in itertools.combinations(other_vertices, 6):
                subset = list(rest) + [u_add, v_add]
                if contains_q3_fast(subset, adj_bits):
                    is_valid = False
                    break
            
            if is_valid:
                # Accept addition
                current_edges += 1
                improved = True
                # Update lists? For simplicity, we'll rebuild lists at the start of the outer loop.
                # But we can just break to the outer loop to rebuild.
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj_bits = adj_bits[:]
                break # Restart outer loop to update edge lists
            else:
                # Revert addition
                adj_bits[u_add] &= ~(1 << v_add)
                adj_bits[v_add] &= ~(1 << u_add)
                
                # If adding directly failed, try a swap: remove a random existing edge, then add u_add-v_add
                if current_edges_list:
                    u_rem, v_rem = random.choice(current_edges_list)
                    # Remove u_rem-v_rem
                    adj_bits[u_rem] &= ~(1 << v_rem)
                    adj_bits[v_rem] &= ~(1 << u_rem)
                    
                    # Now try to add u_add-v_add again
                    # Note: u_add, v_add are still missing in adj_bits
                    adj_bits[u_add] |= (1 << v_add)
                    adj_bits[v_add] |= (1 << u_add)
                    
                    is_valid_swap = True
                    # Check for Q3
                    # Any new Q3 must contain the new edge (u_add, v_add)
                    # OR it could be a Q3 that was previously "blocked" by the removal?
                    # No, removing an edge cannot create a new Q3.
                    # So we only need to check if the new edge (u_add, v_add) is part of a Q3.
                    for rest in itertools.combinations(other_vertices, 6):
                        subset = list(rest) + [u_add, v_add]
                        if contains_q3_fast(subset, adj_bits):
                            is_valid_swap = False
                            break
                    
                    if is_valid_swap:
                        # Accept swap
                        # Edge count remains same
                        # But we might have moved to a better position?
                        # For now, we just accept valid swaps to explore the space.
                        # We don't track "improvement" in edge count for swaps, 
                        # but we might enable future additions.
                        pass
                    else:
                        # Revert swap
                        adj_bits[u_add] &= ~(1 << v_add)
                        adj_bits[v_add] &= ~(1 << u_add)
                        adj_bits[u_rem] |= (1 << v_rem)
                        adj_bits[v_rem] |= (1 << u_rem)
        
        # If we made an addition, the edge lists are stale. 
        # The outer loop will rebuild them.
        # If we didn't make an addition, we might want to try more swaps or exit.
        # To avoid infinite loops of non-improving swaps, we can limit iterations.
        # But since we break on addition, and rebuild lists, it's okay.
        # If no addition was made in this batch of 100 swaps, we might be stuck.
        # We can try 2-swaps (remove 2, add 2) or just exit if stuck for a while.
        
        # Simple heuristic: if no improvement in last 1000 swaps, try a random restart of the local search phase?
        # Or just continue.
        
    # Use the best solution found
    adj_bits = best_adj_bits
    
    # Final verification and cleanup
    if time.time() - start_time < time_limit - 1:
        while time.time() - start_time < time_limit - 0.5:
            # Find any Q3
            found = False
            for _ in range(50):
                subset = random.sample(range(n), 8)
                if contains_q3_fast(subset, adj_bits):
                    found = True
                    edges_in_q3 = []
                    for i in range(8):
                        for j in range(i+1, 8):
                            if (adj_bits[subset[i]] >> subset[j]) & 1:
                                edges_in_q3.append((subset[i], subset[j]))
                    if edges_in_q3:
                        u, v = random.choice(edges_in_q3)
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                    break
            
            if found:
                continue
                
            for subset in itertools.combinations(range(n), 8):
                if contains_q3_fast(subset, adj_bits):
                    found = True
                    edges_in_q3 = []
                    for i in range(8):
                        for j in range(i+1, 8):
                            if (adj_bits[subset[i]] >> subset[j]) & 1:
                                edges_in_q3.append((subset[i], subset[j]))
                    if edges_in_q3:
                        u, v = random.choice(edges_in_q3)
                        adj_bits[u] &= ~(1 << v)
                        adj_bits[v] &= ~(1 << u)
                    break
            
            if not found:
                break

    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
