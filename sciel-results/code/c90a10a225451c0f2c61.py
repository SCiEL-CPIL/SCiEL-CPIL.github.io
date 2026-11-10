import itertools
import random
import time

def construct():
    n = 18
    # Initialize with a random K4-free graph or Turan graph
    # We will use a random start and repair, then local search
    
    def is_k4_free(adj):
        # adj is a list of n ints (bitmasks)
        # Check all 4-tuples
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] >> j) & 1:
                    continue
                # i, j are connected
                common = adj[i] & adj[j]
                # Check if common has any edge
                # Iterate bits in common
                c = common
                while c:
                    lsb = c & -c
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    # Check if k is connected to any other in common
                    # Exclude k itself? adj[k] doesn't have k.
                    # So if (adj[k] & common) != 0, there is an edge.
                    # But we must ensure the other vertex is in common and not k.
                    # Since k is in common, and adj[k] doesn't have k, 
                    # any bit in (adj[k] & common) is a neighbor of k in common.
                    if (adj[k] & common) != 0:
                        return False
        return True

    def max_triangle_free_size(adj):
        # Returns the size of the largest triangle-free induced subgraph
        # Try sizes from n down to 1
        for s in range(n, 0, -1):
            # Check if any subset of size s is triangle-free
            # Use combinations for simplicity, n=18 is small
            # Optimization: We can stop early if found
            for comb in itertools.combinations(range(n), s):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                # Check if mask is triangle-free
                is_tf = True
                # For each u in comb
                for u in comb:
                    neighbors_in_subset = adj[u] & mask
                    # Check if neighbors_in_subset has any internal edge
                    nn = neighbors_in_subset
                    while nn:
                        lsb = nn & -nn
                        v = lsb.bit_length() - 1
                        nn ^= lsb
                        # If v is connected to anyone else in neighbors_in_subset
                        # adj[v] & neighbors_in_subset should be 0
                        if (adj[v] & neighbors_in_subset) != 0:
                            is_tf = False
                            break
                    if not is_tf:
                        break
                if is_tf:
                    return s
        return 0

    def get_score(adj):
        return max_triangle_free_size(adj)

    # Initialize
    # Start with Turan graph K6,6,6
    adj = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    current_score = get_score(adj)
    
    start_time = time.time()
    timeout = 110 # Leave some time for final output
    
    # Local Search
    # We will try to flip edges
    while time.time() - start_time < timeout:
        # Pick a random pair
        u, v = random.sample(range(n), 2)
        
        edge_exists = (adj[u] >> v) & 1
        
        # Create a copy of adj for the move
        new_adj = adj[:]
        
        if edge_exists:
            # Remove edge
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
            
            # Removing edges preserves K4-freeness
            new_score = get_score(new_adj)
            
            if new_score < current_score:
                adj = new_adj
                current_score = new_score
        else:
            # Add edge
            # Check K4-freeness first
            common = adj[u] & adj[v]
            valid = True
            c = common
            while c:
                lsb = c & -c
                k = lsb.bit_length() - 1
                c ^= lsb
                if (adj[k] & common) != 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            new_score = get_score(new_adj)
            
            if new_score < current_score:
                adj = new_adj
                current_score = new_score
                
    # Convert adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
