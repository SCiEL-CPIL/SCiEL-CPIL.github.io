"""Construct a K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph."""
import random
import math
import itertools

def _is_k4_free(n, adj):
    """Check if graph with adjacency matrix adj on n vertices is K4-free."""
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]:
                continue
            for c in range(b+1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
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
            for i in range(len(current_set)):
                u1 = current_set[i]
                if not adj[u1][v]:
                    continue
                for j in range(i+1, len(current_set)):
                    u2 = current_set[j]
                    if adj[u1][u2] and adj[u2][v]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            current_set.append(v)
            dfs(v + 1, current_set, current_size + 1)
            current_set.pop()
    
    dfs(0, [], 0)
    return best[0]

def _estimate_max_triangle_free(n, adj):
    """Heuristic estimate for largest triangle-free induced subgraph size."""
    # Greedy: Start with all vertices, remove vertices from triangles
    # to maximize the remaining set size.
    S = set(range(n))
    # Precompute triangles? Or just check on the fly.
    # To speed up, we can maintain a list of triangles in S.
    # But n=18 is small.
    
    # Simple greedy:
    # While there is a triangle in S:
    #   Find one.
    #   Remove the vertex in the triangle with the minimum degree in G[S].
    # This tends to keep high-degree vertices which might form more triangles,
    # but we are removing them to make it triangle-free.
    # Actually, we want to KEEP a large set. So we want to remove the "least useful" vertex.
    # A vertex with low degree in S is "less connected", so removing it might be cheaper?
    # Or a vertex with high degree is "more likely to form triangles", so removing it breaks more triangles?
    # Let's try removing the vertex with the HIGHEST degree in the triangle-induced subgraph?
    # No, let's just remove the one with the minimum degree in S.
    
    while True:
        # Find a triangle in S
        found = False
        triangle = None
        verts = list(S)
        # Optimization: only check triangles among vertices
        for i in range(len(verts)):
            u = verts[i]
            for j in range(i+1, len(verts)):
                v = verts[j]
                if not adj[u][v]:
                    continue
                for k in range(j+1, len(verts)):
                    w = verts[k]
                    if adj[u][w] and adj[v][w]:
                        found = True
                        triangle = (u, v, w)
                        break
                if found:
                    break
            if found:
                break
        
        if not found:
            break
        
        u, v, w = triangle
        # Calculate degrees in S
        degs = {}
        for x in [u, v, w]:
            deg = 0
            for y in S:
                if x != y and adj[x][y]:
                    deg += 1
            degs[x] = deg
        
        # Remove the one with min degree (or max? let's try min)
        # If we remove min degree, we keep the highly connected ones, which might form more triangles later?
        # Actually, if we keep highly connected ones, we might have to remove more later.
        # Let's try removing the one with MAX degree to break the most triangles at once?
        # Let's try both in the search? No, let's stick to one heuristic.
        # Usually, to find a large independent set, you remove high degree vertices.
        # Here, "independent" means "triangle-free".
        # Let's remove the vertex with the maximum degree in S.
        max_deg_vertex = max([u, v, w], key=lambda x: degs[x])
        S.remove(max_deg_vertex)
        
    return len(S)

def construct():
    """Construct a K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph."""
    n = 18
    random.seed(42)
    
    # Initial candidates: Complete multipartite graphs
    candidates = []
    
    # 1. Complete tripartite K6,6,6
    adj1 = [[False] * n for _ in range(n)]
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj1[a][b] = adj1[b][a] = True
    candidates.append(adj1)
    
    # 2. Complete 4-partite with sizes 5,4,4,5
    adj2 = [[False] * n for _ in range(n)]
    sizes = [5, 4, 4, 5]
    start = 0
    parts2 = []
    for s in sizes:
        parts2.append(list(range(start, start+s)))
        start += s
    for i in range(4):
        for j in range(i+1, 4):
            for a in parts2[i]:
                for b in parts2[j]:
                    adj2[a][b] = adj2[b][a] = True
    candidates.append(adj2)
    
    # 3. Complete 5-partite with sizes 4,4,4,3,3
    adj3 = [[False] * n for _ in range(n)]
    sizes3 = [4, 4, 4, 3, 3]
    start = 0
    parts3 = []
    for s in sizes3:
        parts3.append(list(range(start, start+s)))
        start += s
    for i in range(5):
        for j in range(i+1, 5):
            for a in parts3[i]:
                for b in parts3[j]:
                    adj3[a][b] = adj3[b][a] = True
    candidates.append(adj3)
    
    # 4. Complete 6-partite with sizes 3,3,3,3,3,3
    adj4 = [[False] * n for _ in range(n)]
    sizes4 = [3, 3, 3, 3, 3, 3]
    start = 0
    parts4 = []
    for s in sizes4:
        parts4.append(list(range(start, start+s)))
        start += s
    for i in range(6):
        for j in range(i+1, 6):
            for a in parts4[i]:
                for b in parts4[j]:
                    adj4[a][b] = adj4[b][a] = True
    candidates.append(adj4)
    
    # Select the best initial candidate using the exact score (since n is small, it's fast enough for 4 candidates)
    best_adj = None
    best_score = n + 1
    for adj in candidates:
        if not _is_k4_free(n, adj):
            continue
        score = _max_triangle_free_induced(n, adj)
        if score < best_score:
            best_score = score
            best_adj = [row[:] for row in adj]
    
    if best_adj is None:
        best_adj = [row[:] for row in candidates[0]]
        best_score = _max_triangle_free_induced(n, best_adj)
    
    # Local Search / Simulated Annealing
    # We will try to improve best_score.
    # We use the heuristic _estimate_max_triangle_free for fast evaluation during search.
    # We periodically verify with the exact score.
    
    T = 5.0 # Initial temperature
    cooling = 0.99
    min_T = 0.1
    max_iters = 5000
    exact_check_interval = 500
    
    current_adj = best_adj
    current_heur_score = _estimate_max_triangle_free(n, current_adj)
    
    for it in range(max_iters):
        # Pick a random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        # Flip edge
        old_val = current_adj[u][v]
        current_adj[u][v] = not old_val
        current_adj[v][u] = not old_val
        
        # Check K4-free
        if not _is_k4_free(n, current_adj):
            # Revert
            current_adj[u][v] = old_val
            current_adj[v][u] = old_val
            continue
        
        # Calculate new heuristic score
        new_heur_score = _estimate_max_triangle_free(n, current_adj)
        
        # Acceptance
        if new_heur_score <= current_heur_score:
            # Accept
            current_heur_score = new_heur_score
            # Periodically check exact score
            if it % exact_check_interval == 0:
                exact_score = _max_triangle_free_induced(n, current_adj)
                if exact_score < best_score:
                    best_score = exact_score
                    best_adj = [row[:] for row in current_adj]
        else:
            # Reject or accept with probability
            delta = new_heur_score - current_heur_score
            prob = math.exp(-delta / T)
            if random.random() < prob:
                current_heur_score = new_heur_score
                if it % exact_check_interval == 0:
                    exact_score = _max_triangle_free_induced(n, current_adj)
                    if exact_score < best_score:
                        best_score = exact_score
                        best_adj = [row[:] for row in current_adj]
            else:
                # Revert
                current_adj[u][v] = old_val
                current_adj[v][u] = old_val
        
        T *= cooling
        if T < min_T:
            T = min_T
            
    # Final exact check on best_adj
    final_score = _max_triangle_free_induced(n, best_adj)
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return edges
