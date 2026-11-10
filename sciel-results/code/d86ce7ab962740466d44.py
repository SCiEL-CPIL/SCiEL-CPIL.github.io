import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 20
    adj = [0] * n
    # cc[u][v] is the number of common neighbors of u and v
    cc = [[0] * n for _ in range(n)]
    
    def can_add(u, v):
        if (adj[u] >> v) & 1:
            return False
        if cc[u][v] > 2:
            return False
        # Check neighbors of v: if w is neighbor of v, and cc[u][w] > 1, 
        # adding v to N(u) makes cc[u][w] increase by 1 if v is neighbor of w?
        # Wait. cc[u][w] = |N(u) \cap N(w)|.
        # If we add (u, v), N(u) becomes N(u) U {v}.
        # New cc[u][w] = |(N(u) U {v}) \cap N(w)| = |N(u) \cap N(w)| + (1 if v in N(w) else 0).
        # So if v in N(w) and cc[u][w] == 2, new cc is 3. Forbidden.
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            if cc[u][w] > 1: # If cc is 2, adding v (which is in N(w)) makes it 3. If cc is 1, adding v makes it 2 (ok).
                # Wait, if cc[u][w] == 2, and v is in N(w), then new cc is 3.
                # If cc[u][w] == 1, and v is in N(w), new cc is 2. OK.
                # So condition is cc[u][w] >= 2? No.
                # If cc[u][w] == 2, new is 3. Bad.
                # If cc[u][w] == 1, new is 2. Good.
                # So if cc[u][w] >= 2, it's bad ONLY IF v is in N(w).
                # The loop iterates w in N(v). So v is in N(w) is true.
                # So if cc[u][w] >= 2, return False.
                # My previous code used > 1. Which is >= 2. Correct.
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
        # For each w in N(v) (excluding u, but u is not in N(v) yet), cc[u][w] += 1
        m = adj[v]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u:
                cc[u][w] += 1
                cc[w][u] += 1
            m ^= lsb
            
        # For each w in N(u) (excluding v), cc[v][w] += 1
        m = adj[u]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v:
                cc[v][w] += 1
                cc[w][v] += 1
            m ^= lsb

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Update cc
        m = adj[v]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u:
                cc[u][w] -= 1
                cc[w][u] -= 1
            m ^= lsb
            
        m = adj[u]
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v:
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

    def edge_count():
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2

    def greedy_fill():
        possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(possible)
        for u, v in possible:
            if can_add(u, v):
                add_edge(u, v)

    def get_blocking_edges(u, v):
        """
        Returns a list of edges that, if removed, might allow (u, v) to be added.
        """
        blockers = []
        # Case 1: cc[u][v] > 2
        if cc[u][v] > 2:
            # Common neighbors are the problem
            common = adj[u] & adj[v]
            m = common
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                # Removing (u, w) or (v, w) reduces common neighbors
                blockers.append((u, w))
                blockers.append((v, w))
                m ^= lsb
        
        # Case 2: Conflict with neighbor w of v
        # If w in N(v) and cc[u][w] > 1, then adding v to N(u) makes cc[u][w] >= 3.
        # We need to remove an edge such that cc[u][w] decreases or v is removed from N(w).
        # Removing (w, v) removes v from N(w).
        # Removing (u, x) where x in N(u) \cap N(w) decreases cc[u][w].
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            if cc[u][w] > 1:
                # Remove (w, v)
                blockers.append((w, v))
                # Remove one of the common neighbors of u and w
                common_uw = adj[u] & adj[w]
                m = common_uw
                while m:
                    lsb2 = m & -m
                    x = lsb2.bit_length() - 1
                    blockers.append((u, x))
                    m ^= lsb2
            mask_v ^= lsb
            
        # Case 3: Conflict with neighbor w of u
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if cc[v][w] > 1:
                blockers.append((w, u))
                common_vw = adj[v] & adj[w]
                m = common_vw
                while m:
                    lsb2 = m & -m
                    x = lsb2.bit_length() - 1
                    blockers.append((v, x))
                    m ^= lsb2
            mask_u ^= lsb
            
        return blockers

    best_edges = []
    best_count = 0
    
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        # Reset
        adj = [0] * n
        cc = [[0] * n for _ in range(n)]
        
        # Greedy start
        greedy_fill()
        
        count = edge_count()
        if count > best_count:
            best_count = count
            best_edges = get_edges()
            
        # Local Search
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < deadline - 2:
            # Pick random non-edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            if (adj[u] >> v) & 1:
                continue
                
            if can_add(u, v):
                add_edge(u, v)
                count += 1
                if count > best_count:
                    best_count = count
                    best_edges = get_edges()
            else:
                # Try to remove a blocking edge
                blockers = get_blocking_edges(u, v)
                if blockers:
                    # Pick a random blocker
                    # To ensure we don't pick an edge that doesn't exist
                    valid_blockers = [b for b in blockers if (adj[b[0]] >> b[1]) & 1]
                    if valid_blockers:
                        a, b = random.choice(valid_blockers)
                        # Remove (a, b)
                        remove_edge(a, b)
                        count -= 1
                        
                        # Try to add (u, v)
                        if can_add(u, v):
                            add_edge(u, v)
                            count += 1
                            if count > best_count:
                                best_count = count
                                best_edges = get_edges()
                        else:
                            # Revert
                            remove_edge(u, v) if False else None # u,v not added
                            add_edge(a, b)
                            count += 1 # back to original count
                            
            T *= cooling
            if T < min_T:
                T = min_T
                
    # Final greedy pass on best solution
    adj = [0] * n
    cc = [[0] * n for _ in range(n)]
    for u, v in best_edges:
        add_edge(u, v)
        
    improved = True
    while improved:
        improved = False
        for u in range(n):
            for v in range(u+1, n):
                if not ((adj[u] >> v) & 1) and can_add(u, v):
                    add_edge(u, v)
                    improved = True
                    
    final_edges = get_edges()
    if len(final_edges) > best_count:
        best_edges = final_edges
        
    return best_edges
