import time
import random
import numpy as np

def construct():
    deadline = time.time() + 145
    n = 40
    m = 401
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set = set(all_edges)
    
    # Bitmask representation for adjacency
    # adj[i] is an integer where bit j is 1 if edge (i,j) exists
    
    def compute_max_book(adj):
        max_book = 0
        for i in range(n):
            mask_i = adj[i]
            # Only consider j > i
            mask_j = mask_i & ~((1 << (i+1)) - 1)
            while mask_j:
                lsb = mask_j & -mask_j
                j = lsb.bit_length() - 1
                common = (adj[i] & adj[j]).bit_count()
                if common > max_book:
                    max_book = common
                mask_j ^= lsb
        return max_book

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            mask &= ~((1 << (i+1)) - 1)
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                edges.append([i, j])
                mask ^= lsb
        return edges

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += adj[i].bit_count()
        return total // 2

    # Local search with 2-opt swaps
    def local_search(adj, time_limit):
        start_time = time.time()
        current_score = compute_max_book(adj)
        
        # Precompute degrees for quick access
        degrees = [adj[i].bit_count() for i in range(n)]
        
        while time.time() < time_limit:
            # Find the edge with the maximum book size
            max_book_edge = None
            max_book_val = -1
            for i in range(n):
                mask_i = adj[i]
                mask_j = mask_i & ~((1 << (i+1)) - 1)
                while mask_j:
                    lsb = mask_j & -mask_j
                    j = lsb.bit_length() - 1
                    common = (adj[i] & adj[j]).bit_count()
                    if common > max_book_val:
                        max_book_val = common
                        max_book_edge = (i, j)
                    mask_j ^= lsb
            
            if max_book_val <= 7:
                break
                
            u, v = max_book_edge
            
            # Try to find a swap that reduces the max book
            # We remove (u,v) and add (x,y)
            # To keep the graph "balanced", we prefer x,y with lower degrees
            
            improved = False
            # Try several random swaps
            for _ in range(50):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                
                # (x,y) must not be an edge
                if (adj[x] >> y) & 1:
                    continue
                
                # (x,y) must not be (u,v)
                if (x == u and y == v) or (x == v and y == u):
                    continue
                
                # Perform swap
                old_x = adj[x]
                old_y = adj[y]
                old_u = adj[u]
                old_v = adj[v]
                
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                adj[x] |= (1 << y)
                adj[y] |= (1 << x)
                
                new_score = compute_max_book(adj)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    # Revert
                    adj[u] = old_u
                    adj[v] = old_v
                    adj[x] = old_x
                    adj[y] = old_y
            
            if not improved:
                # Try a more systematic approach:
                # Look for non-edges (x,y) where x is a neighbor of u and y is a neighbor of v
                # This might break the triangle structure
                found_better = False
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if (adj[u] >> x) & 1:  # x is neighbor of u
                        for y in range(n):
                            if y == u or y == v or y == x:
                                continue
                            if (adj[v] >> y) & 1:  # y is neighbor of v
                                if (adj[x] >> y) & 1:  # (x,y) already exists
                                    continue
                                
                                # Check if this swap helps
                                old_x = adj[x]
                                old_y = adj[y]
                                old_u = adj[u]
                                old_v = adj[v]
                                
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                adj[x] |= (1 << y)
                                adj[y] |= (1 << x)
                                
                                new_score = compute_max_book(adj)
                                
                                if new_score < current_score:
                                    current_score = new_score
                                    found_better = True
                                    improved = True
                                    break
                                else:
                                    adj[u] = old_u
                                    adj[v] = old_v
                                    adj[x] = old_x
                                    adj[y] = old_y
                        if improved:
                            break
                    if improved:
                        break
                
                if not improved:
                    # Stuck, try a random perturbation
                    # Remove a random edge and add a random non-edge
                    for _ in range(10):
                        # Pick a random edge to remove
                        i = random.randint(0, n-1)
                        mask_i = adj[i]
                        mask_j = mask_i & ~((1 << (i+1)) - 1)
                        if not mask_j:
                            continue
                        lsb = mask_j & -mask_j
                        j = lsb.bit_length() - 1
                        
                        # Pick a random non-edge to add
                        x = random.randint(0, n-1)
                        y = random.randint(0, n-1)
                        if x == y:
                            continue
                        if x > y:
                            x, y = y, x
                        if (adj[x] >> y) & 1:
                            continue
                        
                        old_i = adj[i]
                        old_j = adj[j]
                        old_x = adj[x]
                        old_y = adj[y]
                        
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                        adj[x] |= (1 << y)
                        adj[y] |= (1 << x)
                        
                        new_score = compute_max_book(adj)
                        
                        if new_score <= current_score + 1:  # Allow small increases
                            current_score = new_score
                            improved = True
                            break
                        else:
                            adj[i] = old_i
                            adj[j] = old_j
                            adj[x] = old_x
                            adj[y] = old_y
                    
                    if not improved:
                        break

        return current_score, adj

    best_score = float('inf')
    best_adj = None

    # Multiple initializations
    for seed in range(200):
        if time.time() > deadline - 5:
            break
            
        random.seed(seed)
        
        # Different initialization strategies
        init_type = seed % 4
        
        if init_type == 0:
            # Random graph with 401 edges
            adj = [0] * n
            edges = random.sample(all_edges, m)
            for u, v in edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        elif init_type == 1:
            # K_{20,20} + 1 edge
            adj = [0] * n
            for i in range(20):
                for j in range(20, 40):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            # Add one edge within a part
            u = random.randint(0, 19)
            v = (u + 1) % 20
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        elif init_type == 2:
            # Random regular-like graph
            # Start with a random permutation and create a circulant-like structure
            perm = list(range(n))
            random.shuffle(perm)
            adj = [0] * n
            # Connect each vertex to its 10 nearest neighbors in the permuted order
            for i in range(n):
                for k in range(1, 11):
                    j1 = perm[(perm.index(i) + k) % n]
                    j2 = perm[(perm.index(i) - k) % n]
                    adj[i] |= (1 << j1)
                    adj[i] |= (1 << j2)
            # This gives 20-regular, so 400 edges. Add 1 more.
            # Find a non-edge to add
            for _ in range(1000):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x != y and not ((adj[x] >> y) & 1):
                    adj[x] |= (1 << y)
                    adj[y] |= (1 << x)
                    break
                    
        else:
            # Start with a 20-regular circulant graph
            adj = [0] * n
            for i in range(n):
                for k in range(1, 11):
                    j1 = (i + k) % n
                    j2 = (i - k) % n
                    adj[i] |= (1 << j1)
                    adj[i] |= (1 << j2)
            # This is 20-regular, 400 edges. Add 1 more.
            # Find a non-edge
            for _ in range(1000):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x != y and not ((adj[x] >> y) & 1):
                    adj[x] |= (1 << y)
                    adj[y] |= (1 << x)
                    break
        
        # Local search
        score, adj = local_search(adj, deadline - 2)
        
        if score < best_score:
            best_score = score
            best_adj = adj.copy()
            
        if best_score <= 7:
            break

    if best_adj is None:
        # Fallback: K_{20,20} + 1 edge
        adj = [0] * n
        for i in range(20):
            for j in range(20, 40):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        best_adj = adj

    return get_edge_list(best_adj)
