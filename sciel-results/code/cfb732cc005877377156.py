import time
import random
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = 115  # Safety margin: 5s buffer before 120s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Bitset adjacency
    # adj[i] is an integer where bit j is set if (i,j) is an edge
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def edge_count():
        return sum(bin(a).count("1") for a in adj) // 2

    def can_add(u, v):
        """
        Returns True if adding edge (u,v) does not create a C6.
        """
        # C6: u-v-x-y-z-w-u
        # x in N(v), w in N(u)
        # y in N(x), z in N(w)
        # y~z
        # Distinctness: u,v,x,y,z,w distinct.
        
        Nv = adj[v]
        Nu = adj[u]
        
        # Iterate x in N(v)
        # We can iterate through set bits of Nv
        x_mask = Nv
        while x_mask:
            x = (x_mask & -x_mask).bit_length() - 1
            x_mask &= x_mask - 1
            
            if x == u: continue
            
            w_mask = Nu
            while w_mask:
                w = (w_mask & -w_mask).bit_length() - 1
                w_mask &= w_mask - 1
                
                if w == v or w == x: continue
                
                # We need y in N(x) and z in N(w) such that y~z
                # y != u, v, x, w
                # z != u, v, x, w
                # y != z
                
                # S_x = N(x) \ {u, v, w}
                S_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                # S_w = N(w) \ {u, v, x}
                S_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                
                # Check if there is an edge between S_x and S_w
                # Iterate y in S_x
                y_mask = S_x
                while y_mask:
                    y = (y_mask & -y_mask).bit_length() - 1
                    y_mask &= y_mask - 1
                    
                    # Check if y has any neighbor in S_w
                    # Also ensure y != z is handled by S_w excluding x, but we need to ensure z != y
                    # If y is in S_w, then (y,y) is not an edge, but y~y is false.
                    # The check `adj[y] & S_w` checks if any neighbor of y is in S_w.
                    # If y is in S_w, then y is a candidate for z. But y!=z is required.
                    # So we should exclude y from S_w?
                    # Actually, if y is in S_w, then z could be y? No, z is in N(w).
                    # If y is in N(w) and y is in N(x), then y is a candidate for both?
                    # No, y is a vertex. z is a vertex. y != z.
                    # So if we pick y from S_x, we need to check if there is z in S_w \ {y} such that y~z.
                    # So check `adj[y] & (S_w & ~(1 << y))`
                    
                    if adj[y] & (S_w & ~(1 << y)):
                        return False
        return True

    def get_edges():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edges.append([i, j])
        return edges

    random.seed(42)
    
    best_edges = 0
    best_adj = [0] * n
    
    # Phase 1: Random Greedy
    # Try several random orderings
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for trial in range(50):
        if time_left() < 10:
            break
            
        adj_try = [0] * n
        ec = 0
        
        # Random shuffle
        shuffled = all_pairs[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if time_left() < 5:
                break
            
            if can_add(u, v):
                adj_try[u] |= (1 << v)
                adj_try[v] |= (1 << u)
                ec += 1
        
        if ec > best_edges:
            best_edges = ec
            best_adj = adj_try[:]

    # If we didn't find anything good, use a simple structure (e.g., union of cliques)
    # But greedy usually finds something.
    if best_edges == 0:
        # Fallback: 8 disjoint K5s
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5*k+a
                    v = 5*k+b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_edges = 8 * 10 # K5 has 10 edges

    # Phase 2: Local Search
    adj = best_adj[:]
    current_edges = best_edges
    
    # Precompute list of all non-edges for faster sampling?
    # Or just sample from all pairs and check if edge exists.
    # Since n=40, total pairs 780.
    # If graph has ~100 edges, 680 non-edges.
    # Probability of picking non-edge is ~87%.
    
    no_improve_count = 0
    last_improve_time = time.time()
    
    while time_left() > 2:
        # Try to add an edge
        added = False
        
        # Try a few random non-edges
        # To speed up, we can try up to 20 random candidates
        for _ in range(20):
            if time_left() < 2:
                break
            u, v = random.choice(all_pairs)
            if adj[u] & (1 << v):
                continue # Already an edge
            
            if can_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                added = True
                no_improve_count = 0
                last_improve_time = time.time()
                break
        
        if not added:
            # Try to remove an edge to escape local maximum
            # Only remove if we've been stuck for a while
            if no_improve_count > 50:
                # Remove a random edge
                # Pick random u, then random v in adj[u]
                u = random.randint(0, n-1)
                mask = adj[u]
                if mask:
                    v = (mask & -mask).bit_length() - 1
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_edges -= 1
                    no_improve_count = 0
                else:
                    # Find any edge
                    found = False
                    for i in range(n):
                        if adj[i]:
                            u = i
                            v = (adj[i] & -adj[i]).bit_length() - 1
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            current_edges -= 1
                            found = True
                            break
                    if not found:
                        break
            
            no_improve_count += 1

    # Update best if improved
    if current_edges > best_edges:
        best_edges = current_edges
        best_adj = adj[:]

    return get_edges()
