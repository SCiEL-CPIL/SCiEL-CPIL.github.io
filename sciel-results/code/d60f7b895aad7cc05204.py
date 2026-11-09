import time
def construct():
    t0 = time.time()
    deadline = t0 + 110.0

    q = 5
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx, ny, nz = 1, (y * inv_x) % 5, (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx, ny, nz = 0, 1, (z * inv_y) % 5
                else:
                    nx, ny, nz = 0, 0, 1
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    best_result = None
    best_total_c4 = float('inf')
    
    forms = [
        (1, 1, -1),
        (1, 1, 1),
        (1, -1, 1),
        (1, -1, -1),
        (-1, 1, 1),
        (-1, 1, -1),
        (-1, -1, 1),
        (-1, -1, -1),
    ]
    
    for (a, b, c) in forms:
        if time.time() > deadline - 30:
            break
        
        def make_adjacent(a, b, c):
            def adjacent(i, j):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                val = (a*x1*x2 + b*y1*y2 + c*z1*z2) % 5
                return val == 0
            return adjacent
        
        adjacent = make_adjacent(a, b, c)
        
        edges_core = []
        for i in range(31):
            for j in range(i+1, 31):
                if adjacent(i, j):
                    edges_core.append([i, j])
        
        if len(edges_core) != 90:
            continue
        
        adj = [set() for _ in range(31)]
        for u, v in edges_core:
            adj[u].add(v)
            adj[v].add(u)
        
        best_edge = None
        best_add = float('inf')
        
        for u in range(31):
            if time.time() > deadline - 25:
                break
            for v in range(u+1, 31):
                if v in adj[u]:
                    continue
                add = 0
                for a2 in adj[u]:
                    for b2 in adj[v]:
                        if b2 in adj[a2]:
                            add += 1
                if add < best_add:
                    best_add = add
                    best_edge = [u, v]
        
        if best_edge is None:
            continue
        
        total_c4 = best_add
        
        if total_c4 < best_total_c4:
            best_total_c4 = total_c4
            best_result = edges_core + [best_edge]
    
    if best_result is None:
        return _greedy_construct()
    
    final_edges = best_result
    adj = [set() for _ in range(31)]
    for u, v in final_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    edge_set = set(tuple(sorted(e)) for e in final_edges)
    
    def count_total_c4(adj_local):
        total = 0
        for u in range(31):
            for v in range(u+1, 31):
                cn = len(adj_local[u] & adj_local[v])
                total += cn * (cn - 1) // 2
        return total // 4
    
    current_c4 = count_total_c4(adj)
    
    # Improved Local Search: Best Improvement
    # We will iterate until no improvement or time limit.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        best_delta = 0 # We want to minimize C4s, so delta < 0 is good.
        best_move = None
        
        # Try all possible swaps: remove (u,v), add (a,b)
        for idx in range(len(final_edges)):
            if time.time() > deadline - 5:
                break
            u, v = final_edges[idx]
            
            # Pre-compute C4s containing (u,v)
            # C4s containing (u,v) are paths u-w-x-v where (w,x) is an edge.
            # Note: (u,v) is an edge.
            # Path: u-w-x-v. Edges (u,w), (w,x), (x,v).
            # w in N(u), x in N(v), w != x, (w,x) in E.
            # Also w != v, x != u (implied by distinctness in simple graph C4).
            
            c4_with_uv = 0
            for w in adj[u]:
                if w == v: continue
                for x in adj[v]:
                    if x == u or x == w: continue
                    if x in adj[w]:
                        c4_with_uv += 1
            
            # Try adding (a,b)
            for a in range(31):
                if time.time() > deadline - 5:
                    break
                for b in range(a+1, 31):
                    if b in adj[a]:
                        continue
                    if (a, b) in edge_set or (b, a) in edge_set:
                        continue
                    
                    # Count C4s created by (a,b)
                    # Path a-w-x-b. (a,w), (w,x), (x,b).
                    # w in N(a), x in N(b), w != x, (w,x) in E.
                    c4_new = 0
                    for w in adj[a]:
                        if w == b: continue
                        for x in adj[b]:
                            if x == a or x == w: continue
                            if x in adj[w]:
                                # Check if this C4 relies on (u,v)
                                # If it does, it won't exist after removing (u,v).
                                # So we should NOT count it in c4_new for the delta.
                                # The C4 is a-w-x-b-a.
                                # Edges: (a,b) [new], (a,w), (w,x), (x,b).
                                # Does it contain (u,v)?
                                # (u,v) must be one of (a,w), (w,x), (x,b).
                                relies_on_uv = False
                                if (a == u and w == v) or (a == v and w == u):
                                    relies_on_uv = True
                                elif (w == u and x == v) or (w == v and x == u):
                                    relies_on_uv = True
                                elif (x == u and b == v) or (x == v and b == u):
                                    relies_on_uv = True
                                    
                                if not relies_on_uv:
                                    c4_new += 1
                    
                    delta = c4_new - c4_with_uv
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (idx, a, b)
        
        if best_move is not None:
            idx, a, b = best_move
            u, v = final_edges[idx]
            
            # Apply swap
            adj[u].remove(v)
            adj[v].remove(u)
            edge_set.discard((u, v))
            edge_set.discard((v, u))
            
            final_edges[idx] = [a, b]
            adj[a].add(b)
            adj[b].add(a)
            edge_set.add(tuple(sorted([a, b])))
            
            current_c4 += best_delta
            improved = True
    
    # Final validation
    assert len(final_edges) == 91
    assert len(set(tuple(sorted(e)) for e in final_edges)) == 91
    
    return final_edges

def _greedy_construct():
    # Keep fallback
    import random
    random.seed(42)
    edges = []
    adj = [set() for _ in range(31)]
    remaining = []
    for i in range(31):
        for j in range(i+1, 31):
            remaining.append([i, j])
    random.shuffle(remaining)
    for _ in range(91):
        best_e = None
        best_add = float('inf')
        for e in remaining:
            u, v = e
            if v in adj[u]:
                continue
            add = 0
            for a in adj[u]:
                for b in adj[v]:
                    if b in adj[a]:
                        add += 1
            if add < best_add:
                best_add = add
                best_e = e
            if add == 0:
                break
        if best_e is None:
            break
        u, v = best_e
        edges.append(best_e)
        adj[u].add(v)
        adj[v].add(u)
        remaining.remove(best_e)
    while len(edges) < 91 and remaining:
        for e in remaining:
            u, v = e
            if v not in adj[u]:
                edges.append(e)
                adj[u].add(v)
                adj[v].add(u)
                remaining.remove(e)
                break
    return edges[:91]
