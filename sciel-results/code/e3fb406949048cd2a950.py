import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 512
    limit = 512
    
    # Precompute LCM conflicts
    # neighbors[i] is a bitmask of vertices j such that lcm(i, j) > limit
    # We only need j > i for symmetry, but for bitset operations, 
    # it's easier to have full adjacency masks for the subgraph.
    # However, storing 512 big integers is fine.
    
    # Initialize adjacency masks
    # adj[i] will be a bitmask where bit j is 1 if i and j conflict (lcm > limit)
    adj = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # Calculate LCM
            g = gcd(i, j)
            lcm_val = (i // g) * j
            if lcm_val > limit:
                # Conflict exists
                # Set bit j in adj[i] and bit i in adj[j]
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Initial greedy solution to get a lower bound
    # Sort vertices by degree in descending order? 
    # Or just iterate 1..N. 
    # A common heuristic for MIS is to pick the vertex with the smallest degree 
    # in the *remaining* graph, but for an initial bound, a simple greedy 
    # on the full graph is okay. Let's try greedy by descending index 
    # or by ascending degree.
    # Actually, for "lcm <= limit", numbers with many divisors might be 
    # "safe" with many others? No, if lcm is small, they are safe.
    # Conflicts are when lcm is large.
    # Let's just do a simple greedy: 
    # candidate = all vertices
    # selected = []
    # while candidate:
    #    pick v from candidate that has minimum conflicts with *other candidates*?
    #    Or just pick the first one, remove it and its neighbors.
    
    # Let's use a simple greedy: iterate 1 to N, add if no conflict with already added.
    # This is fast and gives a valid set.
    greedy_set = []
    greedy_mask = 0
    for i in range(1, N + 1):
        # Check if i conflicts with any in greedy_set
        # i conflicts with j in greedy_set if bit i is set in adj[j]
        # We can just check if (greedy_mask & adj[i]) != 0
        if (greedy_mask & adj[i]) == 0:
            greedy_set.append(i)
            greedy_mask |= (1 << i)
            
    best_size = len(greedy_set)
    best_set = greedy_set
    
    # Branch and Bound
    # State: current_size, available_mask
    # We want to maximize current_size + popcount(available_mask)
    
    import sys
    sys.setrecursionlimit(10000)
    
    def popcount(x):
        return bin(x).count('1')
    
    def search(current_size, available_mask, current_set_list):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set_list)
            return
        
        # Upper bound: current_size + number of bits in available_mask
        upper_bound = current_size + popcount(available_mask)
        if upper_bound <= best_size:
            return
            
        # Choose a vertex to branch on
        # Heuristic: pick the vertex in available_mask with the highest degree
        # within the available subgraph.
        # This requires calculating degrees for all available vertices, which is O(N^2) per node.
        # For N=512, this might be slow if the tree is deep.
        # Alternative: just pick the lowest set bit. Simple but potentially less effective pruning.
        # Let's try picking the vertex with the most neighbors in the available set.
        
        # To speed up, we can just pick the first available vertex.
        # Or, we can precompute degrees? No, degrees change.
        # Let's stick to picking the lowest available vertex for now, 
        # but maybe sort by degree initially?
        # Actually, a good heuristic is to pick the vertex that has the most conflicts 
        # with other available vertices.
        
        # Let's find the vertex v in available_mask that maximizes popcount(adj[v] & available_mask)
        # This is O(N * cost_of_popcount) per node.
        # If the tree is small, this is fine.
        
        best_v = -1
        max_deg = -1
        m = available_mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            # Calculate degree of v in available subgraph
            deg = popcount(adj[v] & available_mask)
            if deg > max_deg:
                max_deg = deg
                best_v = v
            m ^= lsb
        
        if best_v == -1:
            return
            
        # Branch 1: Include best_v
        # New available: remove best_v and its neighbors
        new_mask_include = available_mask & ~(adj[best_v] | (1 << best_v))
        current_set_list.append(best_v)
        search(current_size + 1, new_mask_include, current_set_list)
        current_set_list.pop()
        
        # Branch 2: Exclude best_v
        # New available: remove best_v
        new_mask_exclude = available_mask & ~(1 << best_v)
        search(current_size, new_mask_exclude, current_set_list)
        
    # Start search
    # Initial available mask is all vertices
    all_mask = (1 << (N + 1)) - 2  # Bits 1..N set
    search(0, all_mask, [])
    
    return best_set

# EVOLVE-BLOCK-START
# The function construct is defined above. 
# We just need to ensure it's the entry point.
# EVOLVE-BLOCK-END
