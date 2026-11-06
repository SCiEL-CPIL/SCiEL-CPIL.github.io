import random

def _is_k4_free(n, adj):
    """Check if graph with adjacency sets adj on n vertices is K4-free."""
    for a in range(n):
        for b in adj[a]:
            if b <= a: 
                continue
            for c in adj[b]:
                if c <= b: 
                    continue
                if c in adj[a]:
                    # a,b,c form triangle
                    common = adj[a] & adj[b] & adj[c]
                    for d in common:
                        if d != a and d != b and d != c:
                            return False
    return True

def _max_triangle_free_induced(n, adj):
    """Exact branch and bound for largest triangle-free induced subgraph."""
    best = [0]
    
    def dfs(start, current_set, current_size):
        if current_size + (n - start) <= best[0]:
            return
        if current_size > best[0]:
            best[0] = current_size
        
        for v in range(start, n):
            ok = True
            for u in current_set:
                if v in adj[u]:
                    common = adj[u] & adj[v]
                    if common & current_set:
                        ok = False
                        break
            if not ok:
                continue
            
            current_set.add(v)
            dfs(v + 1, current_set, current_size + 1)
            current_set.remove(v)
    
    dfs(0, set(), 0)
    return best[0]

def _local_search(n, adj, steps=1000):
    current_score = _max_triangle_free_induced(n, adj)
    best_score = current_score
    best_adj = [s.copy() for s in adj]
    
    for _ in range(steps):
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
            
        # Flip edge
        if v in adj[u]:
            adj[u].remove(v)
            adj[v].remove(u)
        else:
            adj[u].add(v)
            adj[v].add(u)
            
        if not _is_k4_free(n, adj):
            # Revert
            if v in adj[u]:
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                adj[u].add(v)
                adj[v].add(u)
            continue
            
        new_score = _max_triangle_free_induced(n, adj)
        
        if new_score < best_score:
            best_score = new_score
            best_adj = [s.copy() for s in adj]
            if best_score <= 5:
                break
        else:
            # Revert
            if v in adj[u]:
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                adj[u].add(v)
                adj[v].add(u)
                
    return best_adj, best_score

def construct():
    n = 18
    random.seed(42)
    
    best_adj = None
    best_score = n + 1
    
    # Try a few initial configurations
    # 1. Complete tripartite
    adj1 = [set() for _ in range(n)]
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj1[a].add(b)
                    adj1[b].add(a)
                    
    adj1, score1 = _local_search(n, adj1, steps=2000)
    if score1 < best_score:
        best_score = score1
        best_adj = adj1
        
    # 2. Random K4-free
    adj2 = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj2[i].add(j)
                adj2[j].add(i)
    # Make K4-free
    changed = True
    while changed:
        changed = False
        for a in range(n):
            for b in adj[a]:
                if b <= a: continue
                for c in adj[b]:
                    if c <= b: continue
                    if c in adj[a]:
                        common = adj[a] & adj[b] & adj[c]
                        for d in common:
                            if d != a and d != b and d != c:
                                # Remove random edge from K4
                                edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                                e = random.choice(edges)
                                adj2[e[0]].remove(e[1])
                                adj2[e[1]].remove(e[0])
                                changed = True
                                break
                        if changed: break
                    if changed: break
                if changed: break
            if changed: break
            
    adj2, score2 = _local_search(n, adj2, steps=2000)
    if score2 < best_score:
        best_score = score2
        best_adj = adj2

    edges = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
                
    return edges
