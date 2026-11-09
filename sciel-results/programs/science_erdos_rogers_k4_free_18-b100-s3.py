import random
import time
import itertools

def is_k4_free(adj):
    n = len(adj)
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] & (1 << b)): continue
            for c in range(b+1, n):
                if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                if adj[a] & adj[b] & adj[c]:
                    return False
    return True

def get_largest_triangle_free(adj):
    n = len(adj)
    # Check from n down to 1
    # Optimization: If we know L is at most K, check K+1
    # But we don't know K initially.
    # We can binary search or just go down.
    # For speed, we can check specific sizes.
    
    # Precompute common neighbors for edges?
    # adj[u] & adj[v]
    
    for k in range(n, 0, -1):
        # Iterate combinations of size k
        # This is slow for large k.
        # But n=18.
        # We can stop early if we find one.
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            is_tf = True
            # Check if any triangle in subset
            # For each u in subset
            for u in subset:
                # Neighbors of u in subset
                nu = adj[u] & mask
                # For each neighbor v of u in subset
                # We just need to check if there is a triangle u-v-w
                # This is equivalent to:
                # For each edge (u, v) in induced subgraph,
                # is there a w in subset adjacent to both?
                
                # Faster:
                # Iterate edges in subset?
                # Or just:
                # If (adj[u] & adj[v] & mask) != 0 for any v in subset adjacent to u?
                # Actually, if we iterate u, and then v in subset with u < v:
                #   if edge(u, v):
                #      if adj[u] & adj[v] & mask: triangle found.
                
                # To avoid duplicates, only check u < v.
                # But we need to check all edges.
                # Let's just check all pairs u, v in subset with u < v.
                for v in subset:
                    if u >= v: continue
                    if adj[u] & (1 << v):
                        if adj[u] & adj[v] & mask:
                            is_tf = False
                            break
                if not is_tf:
                    break
            if is_tf:
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 18
    best_l = 18
    best_graph = None
    
    # Helper to check K4 free and return violation if any
    def find_k4(adj):
        for a in range(n):
            for b in range(a+1, n):
                if not (adj[a] & (1 << b)): continue
                for c in range(b+1, n):
                    if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)): continue
                    common = adj[a] & adj[b] & adj[c]
                    if common:
                        return (a, b, c, (common & -common).bit_length() - 1)
        return None

    def gen_random_k4_free():
        adj = [0] * n
        # Start with random p=0.6
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.6:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair
        for _ in range(50):
            viol = find_k4(adj)
            if not viol:
                return adj
            a, b, c, d = viol
            # Remove edge (a, d)
            adj[a] &= ~(1 << d)
            adj[d] &= ~(1 << a)
        return adj

    # Initial graph
    adj = gen_random_k4_free()
    l = get_largest_triangle_free(adj)
    best_l = l
    best_graph = adj[:]
    
    while time.time() < deadline:
        # Local search: try flipping edges
        # Pick random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        
        # Check if edge exists
        has_edge = bool(adj[u] & (1 << v))
        
        # Try flipping
        new_adj = adj[:]
        if has_edge:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
        # Check K4 free
        if not is_k4_free(new_adj):
            continue
            
        # Calculate new L
        # Optimization: If we are trying to beat best_l, 
        # we only need to check if L < best_l.
        # get_largest_triangle_free returns the max.
        # We can modify it to check if there is a TF subset of size >= best_l.
        # If yes, L >= best_l.
        # If no, L < best_l.
        
        # Let's just calculate full L for now, it's fast enough for n=18
        new_l = get_largest_triangle_free(new_adj)
        
        if new_l < best_l:
            best_l = new_l
            best_graph = new_adj
            adj = new_adj
        elif new_l == best_l:
            # Accept with some probability to escape local optima?
            # Or just accept to explore
            if random.random() < 0.5:
                adj = new_adj
        # else reject

    # Convert to edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_graph[i] & (1 << j):
                edges.append([i, j])
    return edges
