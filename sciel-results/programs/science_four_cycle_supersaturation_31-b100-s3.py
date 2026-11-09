"""Construct 91 edges on 31 vertices minimizing 4-cycles.
Uses polarity graph of PG(2,5) for a C4-free 90-edge core,
then adds one edge to minimize total C4s.
"""
import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 110.0
    
    n = 31
    q = 5
    
    # Build points of PG(2,5)
    # Each point is [x:y:z] normalized so first nonzero coord is 1
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: first nonzero coordinate becomes 1
                if x != 0:
                    inv_x = pow(x, q - 2, q)  # Fermat's little theorem
                    pt = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q - 2, q)
                    pt = (0, 1, (z * inv_y) % q)
                else:
                    pt = (0, 0, 1)
                if pt not in seen:
                    seen.add(pt)
                    points.append(pt)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Polarity: p=(x1,y1,z1) perp q=(x2,y2,z2) iff x1*x2 + y1*y2 - z1*z2 = 0 mod 5
    # This is the standard polarity from the conic x^2 + y^2 = z^2
    def is_adjacent(i, j):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[j]
        return (x1 * x2 + y1 * y2 - z1 * z2) % q == 0
    
    # Build core edge list
    core_edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if is_adjacent(i, j):
                core_edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    # If core doesn't have exactly 90 edges, use greedy fallback
    if len(core_edges) != 90:
        return _greedy_construct()
    
    # Count C4s in core (should be 0 or very small)
    def count_c4(adj_sets):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                cn = len(adj_sets[i] & adj_sets[j])
                if cn >= 2:
                    total += cn * (cn - 1) // 2
        return total // 2
    
    core_c4s = count_c4(adj)
    
    # Find best edge to add: minimize new C4s created
    # Adding edge (u,v) creates C4s = number of common neighbors of u and v that form paths
    # Actually: new C4s = number of pairs (a,b) where a~u, b~v, a~b, and a!=v, b!=u
    # This equals the number of edges between N(u) and N(v) minus any already counted
    
    best_edge = None
    best_new_c4s = float('inf')
    
    for u in range(n):
        if time.time() > deadline - 10:
            break
        for v in range(u + 1, n):
            if v in adj[u]:
                continue
            # Count new C4s: paths of length 3 from u to v
            # = |{ (a,b) : a in N(u), b in N(v), a~b, a!=v, b!=u }|
            # Since u-v is not an edge, a!=v and b!=u are automatic if we're careful
            # Actually a in N(u) means a~u, and we need a~b where b in N(v)
            new_c4s = 0
            for a in adj[u]:
                for b in adj[v]:
                    if b in adj[a]:
                        new_c4s += 1
            if new_c4s < best_new_c4s:
                best_new_c4s = new_c4s
                best_edge = [u, v]
    
    if best_edge is None:
        # Fallback: pick any non-edge
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    best_edge = [u, v]
                    break
            if best_edge:
                break
    
    final_edges = core_edges + [best_edge]
    
    # Verify
    assert len(final_edges) == 91
    edge_set = set(tuple(sorted(e)) for e in final_edges)
    assert len(edge_set) == 91
    
    return final_edges

def _greedy_construct():
    """Greedy construction: add edges one by one, each time choosing the edge
    that adds the fewest new C4s."""
    import random
    random.seed(42)
    
    n = 31
    target = 91
    
    adj = [set() for _ in range(n)]
    edges = []
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    
    def count_new_c4s(u, v):
        """Count C4s created by adding edge (u,v)."""
        cnt = 0
        for a in adj[u]:
            for b in adj[v]:
                if b in adj[a]:
                    cnt += 1
        return cnt
    
    remaining = list(all_edges)
    
    for _ in range(target):
        if time.time() > deadline - 5:
            break
        best_e = None
        best_cnt = float('inf')
        for e in remaining:
            u, v = e
            if v in adj[u]:
                continue
            cnt = count_new_c4s(u, v)
            if cnt < best_cnt:
                best_cnt = cnt
                best_e = e
            if cnt == 0:
                break
        if best_e is None:
            break
        u, v = best_e
        edges.append([u, v])
        adj[u].add(v)
        adj[v].add(u)
        remaining.remove(best_e)
    
    # If we didn't reach target, add arbitrary remaining edges
    while len(edges) < target and remaining:
        e = remaining.pop()
        u, v = e
        if v not in adj[u]:
            edges.append([u, v])
            adj[u].add(v)
            adj[v].add(u)
    
    return edges[:target]

# Module-level deadline for greedy fallback
import time as _time
deadline = _time.time() + 110.0
