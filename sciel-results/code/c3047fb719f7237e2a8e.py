import time
import random
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = 115.0  # 5s buffer for 120s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Bitset adjacency
    # adj[i] is an integer where bit j is set if (i,j) is an edge
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
    num_pairs = len(all_pairs)
    
    # Phase 1: Random Greedy with multiple trials
    # We run many trials because greedy is fast and can find good structures.
    trials = 0
    while time_left() > 10:
        trials += 1
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
    # Start from the best greedy solution
    adj = best_adj[:]
    current_edges = best_edges
    best_local_edges = best_edges
    best_local_adj = best_adj[:]
    
    # Precompute list of edges for removal/swap
    def get_edge_list(adj_state):
        edges = []
        for i in range(n):
            mask = adj_state[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edges.append((i, j))
        return edges

    no_improve_count = 0
    
    while time_left() > 2:
        improved = False
        
        # 1. Try to add a random non-edge
        # We try a batch of random non-edges. If any can be added, we do it.
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
                improved = True
                no_improve_count = 0
                if current_edges > best_local_edges:
                    best_local_edges = current_edges
                    best_local_adj = adj[:]
                break
        
        if improved:
            continue

        # 2. If no edge added, try to escape local maximum
        # Strategy: 1-for-2 move (remove 1, add 2)
        # If that fails, try 1-for-1 swap (remove 1, add 1) to shuffle
        # We prefer 1-for-2 because it increases edge count.
        
        edge_list = get_edge_list(adj)
        if not edge_list:
            break
            
        # Try a few candidate removals for 1-for-2
        success = False
        for _ in range(10):
            if time_left() < 2:
                break
            u_rem, v_rem = random.choice(edge_list)
            
            # Temporarily remove edge
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            
            added_count = 0
            added_edges = []
            
            # Try to add two edges
            for _ in range(100):
                u_add, v_add = random.choice(all_pairs)
                if adj[u_add] & (1 << v_add):
                    continue
                if (u_add, v_add) in added_edges:
                    continue
                if can_add(u_add, v_add):
                    adj[u_add] |= (1 << v_add)
                    adj[v_add] |= (1 << u_add)
                    added_edges.append((u_add, v_add))
                    added_count += 1
                    if added_count == 2:
                        break
            
            if added_count == 2:
                # Net +1 edge
                current_edges += 1
                improved = True
                no_improve_count = 0
                if current_edges > best_local_edges:
                    best_local_edges = current_edges
                    best_local_adj = adj[:]
                success = True
                break
            elif added_count == 1:
                # Net 0 edges, but structure changed. Accept with low probability?
                # For now, let's only accept if we are stuck for a long time (annealing)
                # To keep it simple and fast, we reject unless we allow swaps.
                # Let's undo for now.
                pass
            else:
                # Undo additions
                for ua, va in added_edges:
                    adj[ua] &= ~(1 << va)
                    adj[va] &= ~(1 << ua)
            
            # Undo removal if we didn't succeed in adding 2
            if not success:
                adj[u_rem] |= (1 << v_rem)
                adj[v_rem] |= (1 << u_rem)
                
        if success:
            continue
            
        # 3. If 1-for-2 failed, try 1-for-1 swap to shuffle
        # This allows us to move to a different local maximum
        if no_improve_count > 10:
            # Try to find a valid swap
            for _ in range(20):
                if time_left() < 2:
                    break
                u_rem, v_rem = random.choice(edge_list)
                
                # Temporarily remove edge
                adj[u_rem] &= ~(1 << v_rem)
                adj[v_rem] &= ~(1 << u_rem)
                
                added_edge = None
                # Try to add one edge
                for _ in range(50):
                    u_add, v_add = random.choice(all_pairs)
                    if adj[u_add] & (1 << v_add):
                        continue
                    if can_add(u_add, v_add):
                        adj[u_add] |= (1 << v_add)
                        adj[v_add] |= (1 << u_add)
                        added_edge = (u_add, v_add)
                        break
                
                if added_edge:
                    # Swap accepted. Edge count unchanged.
                    no_improve_count = 0
                    break
                else:
                    # Undo removal
                    adj[u_rem] |= (1 << v_rem)
                    adj[v_rem] |= (1 << u_rem)
        
        no_improve_count += 1

    if best_local_edges > best_edges:
        best_edges = best_local_edges
        best_adj = best_local_adj

    return get_edges()
