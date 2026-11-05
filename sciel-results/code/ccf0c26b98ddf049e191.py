import random
import time
import itertools

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s
    n = 20
    # Adjacency as set of frozensets or bitmasks
    # Use integer bitmask for neighbors
    # Start with complete graph
    adj = [(1 << n) - 1] * n
    for i in range(n):
        adj[i] &= ~(1 << i)  # no self-loop
    
    def common_neighbors_count(triple):
        a, b, c = triple
        mask = adj[a] & adj[b] & adj[c]
        return bin(mask).count('1')
    
    def is_valid():
        for triple in itertools.combinations(range(n), 3):
            if common_neighbors_count(triple) >= 3:
                return False
        return True
    
    def edge_count():
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2
    
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def add_edge(u, v):
        if u != v and not (adj[u] & (1 << v)):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False
    
    # Phase 1: Remove edges to make K3,3-free
    # Find violating triples and remove an edge
    removed = 0
    max_remove_iter = 500
    for _ in range(max_remove_iter):
        if time.time() > deadline - 30:
            break
        found_violation = False
        for triple in itertools.combinations(range(n), 3):
            a, b, c = triple
            mask = adj[a] & adj[b] & adj[c]
            cnt = bin(mask).count('1')
            if cnt >= 3:
                # Find edges to remove: pick a common neighbor d
                # Remove edge (a,d) or (b,d) or (c,d)
                # Choose the one that appears in most other violations
                d_list = [i for i in range(n) if mask & (1 << i)]
                # Pick random d from common neighbors
                d = random.choice(d_list)
                # Remove edge (a, d)
                remove_edge(a, d)
                found_violation = True
                removed += 1
                break
        if not found_violation:
            break
    
    # Phase 2: Local search - try to add edges
    best_edges = edge_count()
    for iteration in range(2000):
        if time.time() > deadline - 10:
            break
        # Try adding a random missing edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if adj[u] & (1 << v):
            continue
        # Check if adding (u,v) creates a K3,3
        # For any triple containing u and some pair, check common neighbors
        # Actually, adding (u,v) only affects triples where u or v is a member
        # and the common neighbor set changes
        valid = True
        # Check triples that include u
        for x, y in itertools.combinations(range(n), 2):
            if x == u or y == u or x == v or y == v:
                continue
            triple = (u, x, y)
            # After adding (u,v), common neighbors of u,x,y include v if v is neighbor of x and y
            # Actually we need to check: does adding (u,v) cause any triple to have >=3 common neighbors?
            # The only new common neighbor that could appear is v for triples containing u, 
            # or u for triples containing v
            # For triple (u, x, y): common neighbors are N(u)∩N(x)∩N(y)
            # Before: adj[u] & adj[x] & adj[y]
            # After: (adj[u] | (1<<v)) & adj[x] & adj[y] = old | ((1<<v) & adj[x] & adj[y])
            # So v becomes a common neighbor iff v is neighbor of both x and y
            # We need to ensure total count stays < 3
            old_cnt = bin(adj[u] & adj[x] & adj[y]).count('1')
            # v is new common neighbor iff adj[x] & (1<<v) and adj[y] & (1<<v)
            if (adj[x] & (1 << v)) and (adj[y] & (1 << v)):
                new_cnt = old_cnt + 1
            else:
                new_cnt = old_cnt
            if new_cnt >= 3:
                valid = False
                break
            if not valid:
                break
        if not valid:
            continue
        # Also check triples containing v (symmetric)
        for x, y in itertools.combinations(range(n), 2):
            if x == u or y == u or x == v or y == v:
                continue
            old_cnt = bin(adj[v] & adj[x] & adj[y]).count('1')
            if (adj[x] & (1 << u)) and (adj[y] & (1 << u)):
                new_cnt = old_cnt + 1
            else:
                new_cnt = old_cnt
            if new_cnt >= 3:
                valid = False
                break
        if valid:
            add_edge(u, v)
            if edge_count() > best_edges:
                best_edges = edge_count()
    
    # Return edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges
