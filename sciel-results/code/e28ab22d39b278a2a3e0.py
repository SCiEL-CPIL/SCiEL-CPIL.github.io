import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit

    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

    def can_add_edge(adj, u, v):
        # adj is list of int bitmasks
        if adj[u] & (1 << v):
            return False
        
        # 1. Check common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) > 2:
            return False
        
        # 2. Check neighbors of v
        # For each w in N(v), adding (u,v) makes v a common neighbor of u and w.
        # So we need |N(u) & N(w)| <= 1
        mask_v = adj[v]
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != u:
                common_uw = adj[u] & adj[w]
                if popcount(common_uw) > 1:
                    return False
            m -= lsb
            
        # 3. Check neighbors of u
        # For each w in N(u), adding (u,v) makes u a common neighbor of v and w.
        # So we need |N(v) & N(w)| <= 1
        mask_u = adj[u]
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if w != v:
                common_vw = adj[v] & adj[w]
                if popcount(common_vw) > 1:
                    return False
            m -= lsb
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2

    def get_edge_list(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                m -= lsb
        return edges

    def local_search(adj, max_steps=1000):
        # Greedy fill first
        modified = True
        while modified:
            modified = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] & (1 << v)):
                        if can_add_edge(adj, u, v):
                            add_edge(adj, u, v)
                            modified = True
        
        # Hill climbing with random restarts within the local search
        # If stuck, remove a random edge and try to fill again
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Check if we can add any edge
            added = False
            # Randomize order of vertices to avoid bias
            verts = list(range(n))
            random.shuffle(verts)
            
            for u in verts:
                for v in verts:
                    if u >= v:
                        continue
                    if not (adj[u] & (1 << v)):
                        if can_add_edge(adj, u, v):
                            add_edge(adj, u, v)
                            added = True
                            break
                if added:
                    break
            
            if not added:
                # Stuck. Remove a random edge to allow new configurations
                # Collect all edges
                edges_to_remove = []
                for u in range(n):
                    mask = adj[u]
                    m = mask
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges_to_remove.append((u, v))
                        m -= lsb
                
                if edges_to_remove:
                    u, v = random.choice(edges_to_remove)
                    remove_edge(adj, u, v)
                    # Try to fill again after removal
                    modified = True
                    while modified:
                        modified = False
                        for a in range(n):
                            for b in range(a+1, n):
                                if not (adj[a] & (1 << b)):
                                    if can_add_edge(adj, a, b):
                                        add_edge(adj, a, b)
                                        modified = True
                else:
                    # No edges at all? Should not happen unless n=0
                    break

    best_edges = 0
    best_adj = None
    best_edge_list = []
    
    # Run multiple random restarts
    num_restarts = 2000 # High number, but limited by time
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        adj = [0] * n
        # Start with a sparse random graph
        # Density around 0.2 - 0.3
        density = 0.2 + 0.1 * (r % 10) / 10.0
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < density:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Perform local search
        local_search(adj)
        
        # Calculate edges
        e_count = count_edges(adj)
        
        if e_count > best_edges:
            best_edges = e_count
            best_adj = adj[:]
            best_edge_list = get_edge_list(adj)
            
            # If we reach a high number of edges, we can reduce the number of restarts or tighten the search
            # ex(20, K2,3) is likely around 100-110.
            if best_edges >= 105:
                # Tighten the search by increasing max_steps for subsequent runs?
                # Or just break if we hit a known high bound?
                # Let's just continue but maybe reduce restarts if time is low.
                pass

    # If no graph found (should not happen), return a valid small graph
    if best_adj is None:
        edges = []
        # Empty graph is valid
        return edges
        
    return best_edge_list
