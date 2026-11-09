import time
import random

def construct():
    n = 20
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add(u, v):
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
            
        # Check neighbors of u
        # For each w in N(u), we need common(w, v) < 2
        # common(w, v) is popcount(adj[w] & adj[v])
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if popcount(adj[w] & adj[v]) >= 2:
                return False
            mask_u ^= lsb
            
        # Check neighbors of v
        # For each x in N(v), we need common(u, x) < 2
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            x = lsb.bit_length() - 1
            if popcount(adj[u] & adj[x]) >= 2:
                return False
            mask_v ^= lsb
            
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def count_edges():
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2

    start_time = time.time()
    deadline = start_time + 115.0 # 5s buffer
    
    best_adj = [0] * n
    best_count = 0
    
    # We will run multiple random restarts or a single long local search
    # Given n=20, a single local search with edge removals should suffice.
    
    # Initialize with empty graph
    # Or random? Empty is safer to start, then add.
    
    # To improve speed, we can pre-compute common neighbors?
    # No, bitmask AND is very fast.
    
    # Strategy:
    # 1. Greedily add edges in random order until no more can be added.
    # 2. If stuck, remove a random edge and continue.
    # 3. Keep track of best.
    
    # To avoid getting stuck in a very poor local optimum, we can restart occasionally.
    # But n=20 is small. Let's try one long run first.
    
    # Shuffle vertices to add randomness
    vertices = list(range(n))
    random.shuffle(vertices)
    
    # Initial greedy phase
    # Try to add all possible edges
    # We can just iterate through all pairs and try to add them
    # But order matters. Random order is good.
    
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(pairs)
    
    for u, v in pairs:
        if can_add(u, v):
            add_edge(u, v)
            
    current_count = count_edges()
    if current_count > best_count:
        best_count = current_count
        best_adj = adj[:]
        
    # Local search phase
    # While time allows
    while time.time() < deadline:
        # Try to add a random missing edge
        # Or iterate through all missing edges?
        # Iterating through all missing edges is O(n^2) checks. n=20, 190 pairs. Very fast.
        # So we can just do a full pass of "try to add any missing edge".
        
        added = False
        # Shuffle pairs for randomness in which one gets added if multiple are available
        random.shuffle(pairs)
        
        for u, v in pairs:
            if not ((adj[u] >> v) & 1): # if edge missing
                if can_add(u, v):
                    add_edge(u, v)
                    added = True
                    # Continue trying to add more? Yes, loop through all pairs again or just continue?
                    # Just continuing is fine, but a full pass is better to catch all.
                    # Actually, if we add an edge, it might block others, but might also enable nothing.
                    # It only restricts. So if we can add, we should add.
                    # But adding one might make another invalid?
                    # No, adding an edge only increases common neighbor counts.
                    # So if (a,b) is valid, and we add (c,d), (a,b) might become invalid?
                    # Yes. So we should re-check?
                    # But we are just trying to find *any* valid edge to add.
                    # If we add it, we might lose the opportunity to add others.
                    # This is standard greedy.
                    pass
                    
        current_count = count_edges()
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        if not added:
            # Stuck. Remove a random edge to escape.
            # Pick a random existing edge
            existing_edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        existing_edges.append((i, j))
                    mask ^= lsb
                    
            if existing_edges:
                u, v = random.choice(existing_edges)
                remove_edge(u, v)
            else:
                # Empty graph, reset?
                # Just break or restart
                break
                
        # Check time
        if time.time() > deadline:
            break
            
    # Return the best graph found
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges
