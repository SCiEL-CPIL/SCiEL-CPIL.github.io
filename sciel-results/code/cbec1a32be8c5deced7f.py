import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Build conflict graph: edge (u,v) exists if there's a w such that
    # gcd(u,v) == gcd(v,w) == gcd(w,u)
    # Using bitmasks for fast operations
    # Element i corresponds to bit (i-1) in the mask
    
    # First find all bad triples
    bad_pairs = [0] * (N+1)  # bad_pairs[i] is a bitmask of elements that form a bad triple with i and some third element
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcd_table[a][b]
            for c in range(b+1, N+1):
                g_bc = gcd_table[b][c]
                if g_ab == g_bc:
                    g_ca = gcd_table[c][a]
                    if g_ab == g_ca:
                        # Bad triple (a,b,c)
                        # a conflicts with b and c
                        # b conflicts with a and c
                        # c conflicts with a and b
                        bad_pairs[a] |= (1 << (b-1)) | (1 << (c-1))
                        bad_pairs[b] |= (1 << (a-1)) | (1 << (c-1))
                        bad_pairs[c] |= (1 << (a-1)) | (1 << (b-1))
    
    # Now we need: two elements u,v are in conflict if they appear together in a bad triple
    # i.e., there exists w such that {u,v,w} is a bad triple
    # This means: v is in bad_pairs[u] AND there exists w such that u is in bad_pairs[w] and v is in bad_pairs[w]
    # Actually simpler: u and v are adjacent in the conflict graph if there exists w != u,v 
    # such that gcd(u,v)==gcd(v,w)==gcd(w,u)
    # Which is equivalent to: w is in the intersection of bad_pairs[u] and bad_pairs[v] (excluding u and v themselves)
    
    # Let's build the adjacency as bitmasks
    adj = [0] * (N+1)
    for u in range(1, N+1):
        for v in range(u+1, N+1):
            # Check if there exists w != u, v such that (u,v,w) is a bad triple
            # w must be in bad_pairs[u] and bad_pairs[v]
            # w != u, v
            common = bad_pairs[u] & bad_pairs[v]
            # Remove u and v from consideration
            common &= ~(1 << (u-1))
            common &= ~(1 << (v-1))
            if common != 0:
                adj[u] |= (1 << (v-1))
                adj[v] |= (1 << (u-1))
    
    # Now find maximum independent set using branch and bound
    # An independent set has no two elements connected by an edge in adj
    
    # Convert to 0-indexed for easier bitmask operations
    # Element i (1-indexed) -> bit (i-1)
    
    # Branch and bound for maximum independent set
    # We use a recursive approach with pruning
    
    best_set_mask = 0
    best_size = 0
    node_count = 0
    max_nodes = 5000000  # Limit to avoid timeout
    
    def count_bits(x):
        return bin(x).count('1')
    
    def get_clique_upper_bound(remaining_mask):
        """Upper bound on independent set size using greedy coloring of the complement graph.
        In the complement, an independent set becomes a clique.
        We color the vertices of the complement restricted to remaining_mask.
        The number of colors is an upper bound on the clique size.
        """
        if remaining_mask == 0:
            return 0
        
        # Greedy coloring: repeatedly pick a vertex, assign a color, remove it and its non-neighbors in complement
        # In complement graph: u and v are adjacent if NOT adjacent in original
        # So in complement, u is adjacent to v iff (adj[u] & (1<<(v-1))) == 0 and u != v
        
        colors = 0
        uncolored = remaining_mask
        
        while uncolored != 0:
            colors += 1
            # Pick a vertex from uncolored
            # Use a simple heuristic: pick the one with most neighbors in uncolored in complement
            # For speed, just pick the lowest bit
            v_bit = uncolored & (-uncolored)
            v = v_bit.bit_length()  # 1-indexed position
            
            # In complement, v is adjacent to all w in uncolored where w is NOT adjacent to v in original
            # And w != v
            # So neighbors in complement = uncolored & ~(adj[v] | (1 << (v-1)))
            comp_neighbors = uncolored & ~(adj[v] | (1 << (v-1)))
            
            # Remove v and its complement-neighbors from uncolored
            uncolored &= ~(comp_neighbors | v_bit)
        
        return colors
    
    def branch_and_bound(remaining_mask, current_size, current_mask):
        nonlocal best_set_mask, best_size, node_count, max_nodes
        
        if time.time() > deadline - 2.0:
            return
        
        node_count += 1
        if node_count > max_nodes:
            return
        
        if remaining_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
        
        # Upper bound: current_size + number of remaining vertices
        # Better: current_size + clique upper bound on complement of remaining
        remaining_count = count_bits(remaining_mask)
        if current_size + remaining_count <= best_size:
            return
        
        # Greedy coloring upper bound (more expensive but tighter)
        # Only use when remaining is small enough to be worth it
        if remaining_count <= 30:
            ub = get_clique_upper_bound(remaining_mask)
            if current_size + ub <= best_size:
                return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in remaining subgraph
        # For simplicity, pick the lowest set bit
        v_bit = remaining_mask & (-remaining_mask)
        v = v_bit.bit_length()  # 1-indexed
        
        # Option 1: include v
        # Can include v if no neighbor of v is in current_mask
        # Actually, we need to check against all previously included vertices
        # But we're building the set incrementally, so current_mask contains included vertices
        # v can be added if adj[v] & current_mask == 0
        if (adj[v] & current_mask) == 0:
            new_mask = current_mask | v_bit
            # Remaining vertices that are not neighbors of v
            new_remaining = remaining_mask & ~v_bit & ~adj[v]
            branch_and_bound(new_remaining, current_size + 1, new_mask)
        
        # Option 2: exclude v
        branch_and_bound(remaining_mask & ~v_bit, current_size, current_mask)
    
    # Initial call
    all_mask = (1 << N) - 1
    branch_and_bound(all_mask, 0, 0)
    
    if best_size == 0:
        # Fallback: just return {1}
        return [1]
    
    # Convert mask to list
    result = []
    for i in range(1, N+1):
        if best_set_mask & (1 << (i-1)):
            result.append(i)
    
    # Verify the result
    def is_valid_set(S):
        S_set = set(S)
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                for k in range(j+1, len(S)):
                    a, b, c = S[i], S[j], S[k]
                    if gcd(a,b) == gcd(b,c) == gcd(c,a):
                        return False
        return True
    
    if not is_valid_set(result):
        # Fallback to a simple valid set
        result = [1]
    
    return result
