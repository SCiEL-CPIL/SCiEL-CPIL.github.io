import time
import random
import sys

def construct():
    n = 40
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edge_count = 0
    best_adj = [0] * n  # Bitset representation
    
    # Helper to check if adding (u, v) creates a C6
    # Path u-a-b-c-v
    # a in N(u), c in N(v)
    # b in N(a) and b in N(c)
    # b not in {u, v, a, c}
    # a != c
    def creates_c6(adj, u, v):
        Nu = adj[u]
        Nv = adj[v]
        
        # Iterate through neighbors of u
        temp_Nu = Nu
        while temp_Nu:
            a_bit = temp_Nu & -temp_Nu
            temp_Nu -= a_bit
            a = a_bit.bit_length() - 1
            
            if a == v:
                continue # a is v, but v is not in N(u) initially if edge not added, 
                         # but in this context we are checking BEFORE adding.
                         # So v is not in Nu.
            
            # Check if there is a c in Nv such that a and c share a neighbor b
            # b must be in N(a) & N(c)
            # and b not in {u, v, a, c}
            
            # Get N(a)
            Na = adj[a]
            
            # We need to check all c in Nv
            temp_Nv = Nv
            while temp_Nv:
                c_bit = temp_Nv & -temp_Nv
                temp_Nv -= c_bit
                c = c_bit.bit_length() - 1
                
                if c == a:
                    continue # a != c required for simple path u-a-...-c-v?
                             # If a==c, path is u-a-b-a-v. Not a simple cycle.
                             # So a!=c is required.
                
                # Check common neighbors of a and c
                common = Na & adj[c]
                
                # Mask out u, v, a, c from common
                # u is not in Na (since a in Nu, u-a edge exists, but u is not neighbor of a? 
                # Wait, a is neighbor of u. So u IS in N(a). 
                # We need b to be distinct from u, v, a, c.
                
                mask = (1 << u) | (1 << v) | (1 << a) | (1 << c)
                valid_common = common & ~mask
                
                if valid_common:
                    return True
        
        return False

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2

    # Main Loop: Randomized Greedy Restarts
    while time.time() < deadline:
        # Randomize edge order
        random.shuffle(all_edges)
        
        current_adj = [0] * n
        current_edges = 0
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            
            # Check if edge already exists
            if current_adj[u] & (1 << v):
                continue
                
            # Check if adding creates C6
            if not creates_c6(current_adj, u, v):
                add_edge(current_adj, u, v)
                current_edges += 1
        
        if current_edges > best_edge_count:
            best_edge_count = current_edges
            best_adj = current_adj[:]
            
        # If we found a very good graph, we might want to do local search on it
        # But for now, let's rely on restarts.
        # If we have time, we can try to improve best_adj with swaps.
        
        # Optional: Local Search on the current best found in this iteration
        # This is risky if it takes too long. 
        # Let's stick to fast restarts.

    # Convert best_adj back to list of edges
    edges = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            v_bit = mask & -mask
            mask -= v_bit
            v = v_bit.bit_length() - 1
            if u < v:
                edges.append([u, v])
                
    return edges

