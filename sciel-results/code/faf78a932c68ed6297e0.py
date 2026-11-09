import numpy as np
import time
import random

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
    
    assert len(points) == 31
    
    def adjacent(i, j):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[j]
        val = (x1*x2 + y1*y2 - z1*z2) % 5
        return val == 0
    
    edges_core = []
    for i in range(31):
        for j in range(i+1, 31):
            if adjacent(i, j):
                edges_core.append([i, j])
    
    if len(edges_core) != 90:
        return _greedy_construct()
    
    # Initial best
    best_edges = edges_core[:]
    best_score = float('inf')
    
    def get_best_edge_and_score(edge_list):
        adj = [set() for _ in range(31)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        
        best_edge = None
        best_add = float('inf')
        
        for u in range(31):
            for v in range(u+1, 31):
                if v in adj[u]:
                    continue
                add = 0
                for a in adj[u]:
                    for b in adj[v]:
                        if b in adj[a]:
                            add += 1
                if add < best_add:
                    best_add = add
                    best_edge = [u, v]
        
        if best_edge is None:
            return None, float('inf')
        
        # Count total C4s in the final graph
        # We can just count the added ones if we assume base is C4-free?
        # No, base might not be C4-free after swaps.
        # So we should count total C4s.
        # But counting total C4s is O(n^4) or O(n^3).
        # With 31 vertices, it's fast.
        
        final_edges = edge_list + [best_edge]
        return count_c4(final_edges), best_edge

    initial_score, initial_edge = get_best_edge_and_score(edges_core)
    if initial_score < best_score:
        best_score = initial_score
        best_edges = edges_core + [initial_edge]
    
    # Local Search
    random.seed(42)
    current_edges = edges_core[:]
    
    # Precompute all non-edges for speed?
    # There are 31*30/2 - 90 = 465 - 90 = 375 non-edges.
    
    while time.time() < deadline - 10:
        # Pick a random edge to remove
        e1_idx = random.randint(0, len(current_edges) - 1)
        e1 = current_edges[e1_idx]
        
        # Pick a random non-edge to add
        # Generate random u, v
        u = random.randint(0, 30)
        v = random.randint(0, 30)
        if u == v: continue
        if u > v: u, v = v, u
        e2 = [u, v]
        
        # Check if e2 is already in current_edges
        # Simple check for small list
        is_in = False
        for e in current_edges:
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                is_in = True
                break
        if is_in:
            continue
            
        # Perform swap
        current_edges[e1_idx] = e2
        
        new_score, new_edge = get_best_edge_and_score(current_edges)
        
        if new_score < best_score:
            best_score = new_score
            best_edges = current_edges + [new_edge]
        else:
            # Revert
            current_edges[e1_idx] = e1
            
    return best_edges

def count_c4(edge_list):
    adj = [set() for _ in range(31)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    total = 0
    for u in range(31):
        for v in range(u+1, 31):
            cn = len(adj[u] & adj[v])
            total += cn * (cn - 1) // 2
    return total // 4

def _greedy_construct():
    # Fallback
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
    while len(edges) < 91:
        for e in remaining:
            u, v = e
            if v not in adj[u]:
                edges.append(e)
                adj[u].add(v)
                adj[v].add(u)
                remaining.remove(e)
                break
        if not remaining:
            break
    return edges[:91]
