# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _count_k222_via_triangles(adj, n):
    """
    Counts the number of K2,2,2 subgraphs in the graph defined by adj.
    This is equivalent to counting induced matchings of size 3 in the complement?
    No.
    K2,2,2 in G means:
    Vertices {a,b,c,d,e,f} partitioned into {a,b}, {c,d}, {e,f}.
    All cross edges are in G.
    This means {a,b}, {c,d}, {e,f} are NON-edges in G.
    And there are NO edges in G between these pairs.
    So, in the complement graph \bar{G}, {a,b}, {c,d}, {e,f} are edges.
    And there are NO edges in \bar{G} between these pairs.
    So, K2,2,2 in G <=> Induced Matching of size 3 in \bar{G}.
    
    So we count induced matchings of size 3 in the complement graph.
    """
    if n < 6:
        return 0
    
    # Build complement adjacency
    # adj is boolean n x n
    comp = ~adj
    np.fill_diagonal(comp, False)
    comp = (comp | comp.T)
    np.fill_diagonal(comp, False)
    
    # Get list of edges in comp
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if comp[i, j]:
                edges.append((i, j))
                
    m = len(edges)
    if m < 3:
        return 0
        
    # Build compatibility graph
    # Node i in compat_graph corresponds to edge i in edges
    # Edge (i, j) in compat_graph if edges[i] and edges[j] are disjoint and have no cross-edges in comp
    # Cross-edges in comp means: if (u,v) and (x,y) are the edges, 
    # comp[u,x], comp[u,y], comp[v,x], comp[v,y] must all be False.
    
    # For n=30, m ~ 100-400.
    # Compat matrix is m x m boolean.
    
    # Optimization: Precompute neighbor sets?
    # For each edge (u,v), let N(u,v) be the set of vertices w such that comp[u,w] and comp[v,w] are True?
    # No, we need NO cross edges.
    # So w must NOT be connected to u or v in comp.
    
    # Let's just do the O(m^2) check.
    compat = np.zeros((m, m), dtype=bool)
    
    for i in range(m):
        u, v = edges[i]
        for j in range(i + 1, m):
            x, y = edges[j]
            # Check disjoint
            if x == u or x == v or y == u or y == v:
                continue
            # Check no cross edges in comp
            # comp[u,x] == False, comp[u,y] == False, comp[v,x] == False, comp[v,y] == False
            if not (comp[u, x] or comp[u, y] or comp[v, x] or comp[v, y]):
                compat[i, j] = True
                compat[j, i] = True
                
    # Count triangles in compat graph
    # trace(compat @ compat @ compat) / 6
    if m > 0:
        try:
            # Use int8 to save memory? bool matmul might be slow or unsupported directly for trace?
            # Convert to int
            c_int = compat.astype(np.int8)
            # Matrix multiplication
            c2 = c_int @ c_int
            # trace(c2 @ c_int)
            # Or just sum of diagonal of c2 @ c_int
            c3 = c2 @ c_int
            triangles = np.trace(c3) / 6
            return int(triangles)
        except Exception:
            return 0
    return 0

def _violation_score(adj, n):
    """
    Total number of monochromatic K2,2,2.
    """
    # Color 0 is adj, Color 1 is ~adj
    v0 = _count_k222_via_triangles(adj, n)
    comp = ~adj
    np.fill_diagonal(comp, False)
    comp = (comp | comp.T)
    np.fill_diagonal(comp, False)
    v1 = _count_k222_via_triangles(comp, n)
    return v0 + v1

def _is_valid_fast(adj, n):
    if _violation_score(adj, n) == 0:
        return True
    return False

def _paley_coloring(p):
    if p % 4 != 1:
        return None
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            colors.append(0 if diff in squares else 1)
    return colors

def _colors_to_adj(colors, n):
    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            adj[i, j] = (colors[idx] == 0)
            adj[j, i] = (colors[idx] == 0)
            idx += 1
    return adj

def _adj_to_colors(adj, n):
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if adj[i, j] else 1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_n = 1
    best_colors = []
    
    # 1. Get a valid base coloring (n=29)
    start_adj = None
    for p in [29, 17, 13, 5]:
        if p % 4 != 1:
            continue
        cols = _paley_coloring(p)
        if cols is None:
            continue
        adj_test = _colors_to_adj(cols, p)
        if _is_valid_fast(adj_test, p):
            start_adj = adj_test
            best_n = p
            best_colors = cols
            break
            
    if start_adj is None:
        return {'n': 1, 'colors': []}
        
    # 2. Try to find n=30 using Simulated Annealing on violation score
    target_n = 30
    # Initialize adj for n=30
    # Copy base, add random edges for new vertex
    adj_curr = np.zeros((target_n, target_n), dtype=bool)
    adj_curr[:best_n, :best_n] = start_adj
    
    # Randomize new edges
    for i in range(best_n):
        for j in range(best_n, target_n):
            val = random.random() < 0.5
            adj_curr[i, j] = val
            adj_curr[j, i] = val
    for i in range(best_n, target_n):
        for j in range(i + 1, target_n):
            val = random.random() < 0.5
            adj_curr[i, j] = val
            adj_curr[j, i] = val
            
    current_score = _violation_score(adj_curr, target_n)
    
    if current_score == 0:
        return {'n': target_n, 'colors': _adj_to_colors(adj_curr, target_n)}
        
    # Simulated Annealing
    T = 1.0
    best_score_sa = current_score
    best_adj_sa = adj_curr.copy()
    
    for step in range(5000):
        if time.time() > deadline:
            break
            
        # Pick random edge to flip
        i = random.randint(0, target_n - 1)
        j = random.randint(0, target_n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        old_val = adj_curr[i, j]
        adj_curr[i, j] = not old_val
        adj_curr[j, i] = not old_val
        
        new_score = _violation_score(adj_curr, target_n)
        
        if new_score < current_score or (new_score == current_score and random.random() < 0.5):
            current_score = new_score
            if current_score < best_score_sa:
                best_score_sa = current_score
                best_adj_sa = adj_curr.copy()
                if current_score == 0:
                    return {'n': target_n, 'colors': _adj_to_colors(adj_curr, target_n)}
        else:
            # Metropolis criterion
            delta = new_score - current_score
            if delta > 0:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    adj_curr[i, j] = old_val
                    adj_curr[j, i] = old_val
            else:
                current_score = new_score
                
        # Cool down
        T *= 0.99
        
    # If we found a valid one in SA, return it
    if best_score_sa == 0:
        return {'n': target_n, 'colors': _adj_to_colors(best_adj_sa, target_n)}
        
    # 3. Fallback: Try a few more random restarts for n=30
    for restart in range(10):
        if time.time() > deadline:
            break
        adj_rand = np.random.rand(target_n, target_n) > 0.5
        np.fill_diagonal(adj_rand, False)
        adj_rand = (adj_rand | adj_rand.T)
        np.fill_diagonal(adj_rand, False)
        
        # Quick local search
        curr = adj_rand.copy()
        score = _violation_score(curr, target_n)
        for _ in range(100):
            if score == 0:
                return {'n': target_n, 'colors': _adj_to_colors(curr, target_n)}
            if time.time() > deadline:
                break
            i = random.randint(0, target_n - 1)
            j = random.randint(0, target_n - 1)
            if i == j: continue
            if i > j: i, j = j, i
            old = curr[i, j]
            curr[i, j] = not old
            curr[j, i] = not old
            new_s = _violation_score(curr, target_n)
            if new_s < score:
                score = new_s
            else:
                curr[i, j] = old
                curr[j, i] = old
                
    # 4. Return best valid found (likely 29)
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
