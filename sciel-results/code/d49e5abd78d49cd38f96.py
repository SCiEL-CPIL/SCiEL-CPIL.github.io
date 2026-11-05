import random
import time

def construct():
    n = 20
    best_edges = 0
    best_graph = None
    
    def popcount(x):
        return bin(x).count('1')
        
    def can_add_edge(adj, u, v):
        # Check if adding edge (u, v) creates a K3,3
        # This happens if there is a triple T such that |N(T) U {v}| >= 3
        # where u in T, v not in T, and v is connected to the other two in T.
        
        # Case 1: Triple contains u. T = {u, a, b}
        # v must be neighbor of a and b.
        # If so, v becomes a common neighbor of {u, a, b}.
        # If current common neighbors of {u, a, b} >= 2, then adding v makes it >= 3.
        neighbors_v = [x for x in range(n) if (adj[v] >> x) & 1 and x != u]
        for a in neighbors_v:
            if (adj[u] >> a) & 1:
                for b in neighbors_v:
                    if b == a: continue
                    if (adj[u] >> b) & 1:
                        common = adj[u] & adj[a] & adj[b]
                        if popcount(common) >= 2:
                            return False
                            
        # Case 2: Triple contains v. T = {v, a, b}
        # u must be neighbor of a and b.
        neighbors_u = [x for x in range(n) if (adj[u] >> x) & 1 and x != v]
        for a in neighbors_u:
            if (adj[v] >> a) & 1:
                for b in neighbors_u:
                    if b == a: continue
                    if (adj[v] >> b) & 1:
                        common = adj[v] & adj[a] & adj[b]
                        if popcount(common) >= 2:
                            return False
                            
        return True

    start_time = time.time()
    limit = 110.0
    
    run_count = 0
    while time.time() - start_time < limit:
        run_count += 1
        # Alternate between dense start and random start
        if run_count % 2 == 0:
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        else:
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.7:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        # Removal Phase
        while True:
            edge_scores = {}
            violation_found = False
            
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        common = adj[i] & adj[j] & adj[k]
                        count = popcount(common)
                        if count >= 3:
                            violation_found = True
                            neighbors = [x for x in range(n) if (common >> x) & 1]
                            weight = (count - 1) * (count - 2) // 2 # C(count-1, 2)
                            for a in (i, j, k):
                                for b in neighbors:
                                    if a == b: continue
                                    e = (min(a, b), max(a, b))
                                    if e in edge_scores:
                                        edge_scores[e] += weight
                                    else:
                                        edge_scores[e] = weight
            
            if not violation_found:
                break
            
            items = list(edge_scores.items())
            if not items:
                break
            items.sort(key=lambda x: x[1], reverse=True)
            # Pick from top 5 to introduce randomness
            k = min(5, len(items))
            max_edge = random.choice(items[:k])[0]
            
            u, v = max_edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
        # Fill Phase
        changed = True
        while changed:
            changed = False
            # Randomize order of edges to try
            edges_to_try = [(u, v) for u in range(n) for v in range(u+1, n) if not ((adj[u] >> v) & 1)]
            random.shuffle(edges_to_try)
            
            for u, v in edges_to_try:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
        
        edge_count = sum(popcount(x) for x in adj) // 2
        if edge_count > best_edges:
            best_edges = edge_count
            best_graph = [[u, v] for u in range(n) for v in range(n) if u < v and (adj[u] >> v) & 1]
            
    if best_graph is None:
        return []
        
    return best_graph
