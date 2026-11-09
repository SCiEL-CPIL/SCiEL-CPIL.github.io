import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Helper to check time
    def time_left():
        return deadline - time.time()

    # Bitmask helpers
    def get_bit(mask, i):
        return (mask >> i) & 1

    def set_bit(mask, i, val):
        if val:
            return mask | (1 << i)
        else:
            return mask & ~(1 << i)

    def find_k4(adj, n):
        # Returns (i, j, k, l) if K4 found, else None
        # adj[i] is bitmask of neighbors of i
        for i in range(n):
            # Only look at neighbors j > i to avoid duplicates
            j_mask = adj[i] & ~((1 << (i + 1)) - 1) # neighbors with index > i
            while j_mask:
                lsb = j_mask & -j_mask
                j = lsb.bit_length() - 1
                # Check common neighbors of i and j
                common = adj[i] & adj[j]
                # Exclude i and j themselves
                common &= ~(1 << i)
                common &= ~(1 << j)
                
                # We need to find an edge (k, l) within 'common'
                # Iterate through bits in common
                k_mask = common
                while k_mask:
                    lsb_k = k_mask & -k_mask
                    k = lsb_k.bit_length() - 1
                    # Check if k is connected to any other node in common
                    # neighbors of k that are in common
                    k_neighbors_in_common = adj[k] & common
                    if k_neighbors_in_common:
                        # Find one such neighbor l
                        lsb_l = k_neighbors_in_common & -k_neighbors_in_common
                        l = lsb_l.bit_length() - 1
                        return (i, j, k, l)
                    k_mask &= k_mask - 1 # remove lsb_k
                j_mask &= j_mask - 1 # remove lsb_j
        return None

    def find_ind6(adj, n):
        # Independent set of size 6 in G is K6 in ~G
        # Construct complement adjacency bitmasks
        # comp_adj[i] has bits for j where NOT (i==j or adj[i] has bit j)
        # Note: adj[i] does not have bit i set.
        
        # Optimization: Precompute complement masks? 
        # For n <= 40, we can just compute on the fly or maintain them.
        # Let's compute on the fly for simplicity, it's O(n) per call.
        
        # Actually, maintaining comp_adj is better if we do many checks.
        # But we are doing repair, so adj changes.
        
        # Let's write a specific finder for Ind6 using bitmasks on the fly.
        # An independent set {v1..v6} means for all a!=b, vi and vj are NOT connected.
        
        # This is equivalent to finding a K6 in the complement graph.
        # Let's build comp_adj list.
        comp_adj = [0] * n
        for i in range(n):
            # All bits 1 to n-1, except i, except adj[i]
            mask_all = (1 << n) - 1
            mask_all &= ~(1 << i)
            comp_adj[i] = mask_all & ~adj[i]
        
        # Now find K6 in comp_adj
        # We can use a similar logic to find_k4 but for size 6.
        # This might be slow if done naively.
        # Let's use a recursive backtracking with bitmasks for K6.
        
        def find_clique_in_comp(size):
            # Returns a list of vertices forming a clique of 'size' in comp_adj
            # Or None
            # We'll use a simple recursive approach with pruning
            # Since n is small (<=40), and we are looking for size 6,
            # we can iterate combinations? No, 40C6 is large.
            # Bitmask recursion is better.
            
            # We'll maintain a list of candidates.
            # Start with all vertices.
            # This is essentially the Bron-Kerbosch algorithm for maximal cliques,
            # but we just need *any* clique of size 6.
            
            # A faster heuristic: 
            # Iterate i, j, k, l, m, p? No.
            
            # Let's use the property: 
            # For each i, consider neighbors in comp.
            # For each pair (i, j) in comp, consider common neighbors in comp.
            # For each triple (i, j, k) in comp, consider common neighbors in comp.
            # ...
            
            # Let's implement a generic clique finder for small sizes using bitmasks.
            # def find_clique(candidates, size):
            #   if size == 1: return [candidates & -candidates]
            #   for each v in candidates:
            #      new_candidates = candidates & comp_adj[v] & (mask > v)
            #      res = find_clique(new_candidates, size-1)
            #      if res: return [v] + res
            #   return None
            
            # This can be exponential but for n=40, size=6, it's usually fast if the graph is dense/sparse appropriately.
            # However, in the context of R(4,6), the graph is likely "balanced", so complement is also balanced.
            
            # To make it fast, we can order vertices by degree in comp?
            # Let's just use the recursive bitmask method.
            
            def _find_clique(candidates, size, start_idx):
                if size == 1:
                    # Return the first bit
                    lsb = candidates & -candidates
                    idx = lsb.bit_length() - 1
                    return [idx]
                
                # Iterate through bits in candidates
                # Optimization: only consider bits with index > start_idx to avoid duplicates
                mask = candidates & ~((1 << (start_idx + 1)) - 1)
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    # New candidates must be neighbors of v in comp, and > v
                    new_cands = candidates & comp_adj[v] & ~((1 << (v + 1)) - 1)
                    res = _find_clique(new_cands, size - 1, v)
                    if res:
                        return [v] + res
                    mask &= mask - 1
                return None
            
            return _find_clique((1 << n) - 1, size, -1)
            
        return find_clique_in_comp(6)

    def is_valid(adj, n):
        if find_k4(adj, n):
            return False
        if find_ind6(adj, n):
            return False
        return True

    # Initial best solution: Paley 17
    n_best = 17
    adj_best = [0] * 17
    squares = {x * x % 17 for x in range(1, 17)}
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj_best[i] |= (1 << j)
                adj_best[j] |= (1 << i)
                
    # Verify initial
    if not is_valid(adj_best, 17):
        # Fallback to empty graph? No, empty has Ind17. 
        # Paley 17 is known to be valid.
        pass

    # Try to extend to larger n
    # Strategy: Conflict-Driven Repair
    for n in range(18, 36):
        if time_left() < 5.0:
            break
            
        # Try multiple restarts for this n
        for restart in range(50):
            if time_left() < 2.0:
                break
            
            # Initialize adj
            # Start from previous best if possible? 
            # Or random?
            # Let's try random with density ~0.5
            # Or better: Start from a valid graph of size n-1 and add a vertex.
            # But maintaining validity is hard.
            # Let's use Random Init + Repair.
            
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Repair Loop
            # We want to eliminate K4s and Ind6s.
            # We'll iterate, finding a violation and fixing it.
            
            steps = 0
            max_steps = 5000
            while steps < max_steps and time_left() > 1.0:
                k4 = find_k4(adj, n)
                if k4:
                    # Remove a random edge from the K4
                    i, j, k, l = k4
                    edges = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
                    u, v = random.choice(edges)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    steps += 1
                    continue
                    
                ind6 = find_ind6(adj, n)
                if ind6:
                    # Add a random edge within the Ind6 to break it
                    verts = ind6
                    # Pick two distinct vertices from the set
                    a, b = random.sample(verts, 2)
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    steps += 1
                    continue
                    
                # No violations found
                # Check validity one more time to be sure
                if is_valid(adj, n):
                    n_best = n
                    adj_best = adj[:]
                    # Try to go to next n
                    break
            else:
                # Failed to find valid graph for this restart
                continue
            
            # If we found a valid graph for n, break restart loop and try n+1
            if n_best == n:
                break
                
    # Return best found
    edges = []
    for i in range(n_best):
        for j in range(i + 1, n_best):
            if (adj_best[i] >> j) & 1:
                edges.append([i, j])
                
    return {'n': n_best, 'edges': edges}

if __name__ == "__main__":
    # Test
    res = construct()
    print(f"Found graph with n={res['n']}")
