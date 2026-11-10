import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 20
    # adj[u] is a bitmask of neighbors of u
    adj = [0] * n
    # cc[u][v] is the number of common neighbors of u and v
    cc = [[0] * n for _ in range(n)]
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(u, v):
        if (adj[u] >> v) & 1:
            return False
        if cc[u][v] > 2:
            return False
        # Check neighbors of v
        mask_v = adj[v]
        # Iterate through set bits in mask_v
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            if cc[u][w] > 1:
                return False
            mask_v ^= lsb
        # Check neighbors of u
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if cc[v][w] > 1:
                return False
            mask_u ^= lsb
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update cc
        mask_v = adj[v] & ~(1 << u) # Neighbors of v excluding u (u not in N(v) before)
        # Actually adj[v] was updated, so we need old neighbors.
        # But we can just iterate current adj[v] and skip u?
        # No, u is now in adj[v]. We need to update cc[u][w] for w in OLD N(v).
        # OLD N(v) = adj[v] ^ (1 << u).
        old_mask_v = adj[v] ^ (1 << u)
        m = old_mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[u][w] += 1
            cc[w][u] += 1
            m ^= lsb
            
        old_mask_u = adj[u] ^ (1 << v)
        m = old_mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[v][w] += 1
            cc[w][v] += 1
            m ^= lsb

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Update cc
        # OLD N(v) included u. So we subtract 1 for w in OLD N(v) \ {u}
        # But wait, when we added, we incremented for w in OLD N(v) (before u was added).
        # So we decrement for the same set.
        # The set is adj[v] ^ (1 << u) AFTER removal?
        # After removal, adj[v] does not contain u.
        # So adj[v] IS the set of neighbors excluding u.
        m = adj[v]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[u][w] -= 1
            cc[w][u] -= 1
            m ^= lsb
            
        m = adj[u]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            cc[v][w] -= 1
            cc[w][v] -= 1
            m ^= lsb

    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Initialize with a random graph or greedy
    # Let's try greedy first to get a good start
    random.seed(42)
    order = list(range(n))
    random.shuffle(order)
    
    # Try to add edges in random order
    # We can do multiple passes or just one
    for _ in range(10):
        random.shuffle(order)
        # Create list of all possible edges
        possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(possible)
        for u, v in possible:
            if can_add(u, v):
                add_edge(u, v)
                
    best_edges = get_edges()
    best_count = len(best_edges)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9999
    
    while time.time() < deadline:
        # Pick a random move
        # 50% add, 50% remove?
        # Or pick two random vertices and toggle?
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        if (adj[u] >> v) & 1:
            # Edge exists, try to remove
            remove_edge(u, v)
            # Try to add a random other edge?
            # Or just accept removal if it doesn't hurt?
            # Removal always reduces edge count, so only accept if we can add more later.
            # For simplicity, let's do swaps or just add/remove with SA logic.
            # If we remove, we lose 1 edge. We should only do this if we expect to gain more.
            # Let's try to add a random edge immediately after removal?
            # Or just revert?
            # Better: If we remove, we accept it with probability exp(-1/T).
            # But we need to check if the state is valid (it is).
            # Let's try to add a random edge to compensate.
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 != v2 and not ((adj[u2] >> v2) & 1):
                if can_add(u2, v2):
                    add_edge(u2, v2)
                    # Net change 0 or 1?
                    # We removed 1, added 1. Net 0.
                    # This is a swap.
                else:
                    # Failed to add, revert removal
                    add_edge(u, v)
            else:
                # Failed to find edge to add, revert removal
                add_edge(u, v)
        else:
            # Edge does not exist, try to add
            if can_add(u, v):
                add_edge(u, v)
                # Accept
            else:
                # Reject
                pass
                
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Final greedy pass
    possible = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(possible)
    for u, v in possible:
        if can_add(u, v):
            add_edge(u, v)

    edges = get_edges()
    if len(edges) > best_count:
        best_edges = edges
        best_count = len(edges)
        
    return best_edges
