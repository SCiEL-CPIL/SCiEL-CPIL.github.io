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
    adj = [0] * n
    
    def can_add(u, v):
        """
        Returns True if adding edge (u,v) does not create a C6.
        Optimized: iterate over the smaller neighbor set in the outer loop.
        C6: u-v-x-y-z-w-u, all distinct.
        """
        Nv = adj[v]
        Nu = adj[u]
        
        # To minimize work, iterate x over the smaller set
        # Case 1: deg(v) <= deg(u): x in N(v), w in N(u)
        # Case 2: deg(u) < deg(v): swap roles, x in N(u), w in N(v)
        
        if Nv.bit_count() <= Nu.bit_count():
            # x in N(v), w in N(u)
            x_mask = Nv
            while x_mask:
                x = (x_mask & -x_mask).bit_length() - 1
                x_mask &= x_mask - 1
                if x == u:
                    continue
                w_mask = Nu
                while w_mask:
                    w = (w_mask & -w_mask).bit_length() - 1
                    w_mask &= w_mask - 1
                    if w == v or w == x:
                        continue
                    # Need y in N(x)\{u,v,w}, z in N(w)\{u,v,x}, y~z, y!=z
                    S_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    S_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                    y_mask = S_x
                    while y_mask:
                        y = (y_mask & -y_mask).bit_length() - 1
                        y_mask &= y_mask - 1
                        if adj[y] & (S_w & ~(1 << y)):
                            return False
        else:
            # deg(u) < deg(v): x in N(u), w in N(v)
            # C6: u-v-w-z-y-x-u (renamed)
            # x in N(u), w in N(v), y in N(x), z in N(w), y~z
            x_mask = Nu
            while x_mask:
                x = (x_mask & -x_mask).bit_length() - 1
                x_mask &= x_mask - 1
                if x == v:
                    continue
                w_mask = Nv
                while w_mask:
                    w = (w_mask & -w_mask).bit_length() - 1
                    w_mask &= w_mask - 1
                    if w == u or w == x:
                        continue
                    # y in N(x)\{u,v,w}, z in N(w)\{u,v,x}, y~z, y!=z
                    S_x = adj[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                    S_w = adj[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                    y_mask = S_x
                    while y_mask:
                        y = (y_mask & -y_mask).bit_length() - 1
                        y_mask &= y_mask - 1
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
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Phase 1: Random Greedy with multiple trials
    for trial in range(100):
        if time_left() < 10:
            break
            
        adj_try = [0] * n
        ec = 0
        
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

    # Fallback
    if best_edges == 0:
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5*k+a
                    v = 5*k+b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_edges = 80

    # Phase 2: Local Search
    adj = best_adj[:]
    current_edges = best_edges
    
    no_improve_count = 0
    
    while time_left() > 2:
        added = False
        
        # Try to add a random non-edge
        for _ in range(50):
            if time_left() < 2:
                break
            u, v = random.choice(all_pairs)
            if adj[u] & (1 << v):
                continue
            
            if can_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                added = True
                no_improve_count = 0
                break
        
        if not added:
            # Escape local maximum by removing an edge
            if no_improve_count > 30:
                # Find a random edge to remove
                # Collect all edges for random selection
                edge_list = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        j = (mask & -mask).bit_length() - 1
                        mask &= mask - 1
                        if i < j:
                            edge_list.append((i, j))
                
                if edge_list:
                    u, v = random.choice(edge_list)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    current_edges -= 1
                    no_improve_count = 0
                else:
                    break
            
            no_improve_count += 1

    if current_edges > best_edges:
        best_edges = current_edges
        best_adj = adj[:]

    return get_edges()
