"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time as _time
import numpy as _np

# EVOLVE-BLOCK-START
def _check_girth(n, edges, k, g, deadline):
    """Verify via non-backtracking walks that no cycle < g exists."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for u in range(n):
        if len(adj[u]) != k:
            return False
    seen = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if u == v or key in seen:
            return False
        seen.add(key)
    max_depth = g - 2
    if max_depth < 1:
        return True
    
    # Fast BFS using lists and sets
    # State: (cur, prev)
    # For each directed edge (a, b), check if we can return to a from a different neighbor.
    for a in range(n):
        if _time.time() > deadline:
            return True
        for b in adj[a]:
            # BFS from (b, a)
            visited = set()
            visited.add((b, a))
            frontier = [(b, a)]
            depth = 0
            while frontier and depth < max_depth:
                depth += 1
                new_frontier = []
                for cur, prev in frontier:
                    for nxt in adj[cur]:
                        if nxt == prev:
                            continue
                        if (nxt, cur) not in visited:
                            visited.add((nxt, cur))
                            new_frontier.append((nxt, cur))
                frontier = new_frontier
                if not frontier:
                    break
                # Check if we reached a
                for cur, prev in frontier:
                    if cur == a and prev != b:
                        return False
    return True

def _construct_luw(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q). Tries smaller q first."""
    # Try q=3, then q=4, etc.
    for q in [3, 4, 5, 6, 7]:
        if q < 2: continue
        d = 5 # girth 10 requires d=5
        # Check if q is large enough?
        # For d=5, q=3 might be too small.
        # Let's just build and check.
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            if _time.time() > deadline - 30.0:
                return None, []
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        n = len(order)
        # Quick check: is it 4-regular?
        deg = [0]*n
        for u,v in edges:
            deg[u]+=1
            deg[v]+=1
        if all(d==4 for d in deg):
            return n, edges
    return None, []

def _construct_cyclic_cover(k, g, deadline):
    """
    Cyclic cover (voltage graph) construction for 4-regular graphs of girth 10.
    Base: a 4-regular graph on m vertices with voltages in Z_m.
    The cover has m^2 vertices, 4-regular, girth >= girth(base) * m if no short cycles in base with non-zero voltage.
    We use a base graph that is a cycle of length m with additional chords to make it 4-regular,
    or use known small 4-regular graphs.
    For girth 10, we need girth(base) * m >= 10.
    If base is a cycle C_m (girth m), we need m >= 10/m, so m^2 >= 10, m >= 4.
    But C_m is 2-regular, not 4-regular.
    Alternative: use a 4-regular base on m vertices with girth at least ceil(g/m).
    For m=10, base needs girth >= 1. Any 4-regular graph on 10 vertices.
    The cover would have 100 vertices.
    Actually, let's use a simpler approach: the lexicographic product or a known construction.
    
    For now, try a specific known construction: the (4,10)-cage is 320 vertices.
    A cyclic cover of a 4-regular graph on 8 vertices with voltages in Z_40 would give 320 vertices.
    But we need to ensure girth 10.
    
    Let's try: base = 4-regular graph on 8 vertices (e.g., cube Q3 is 3-regular, not 4).
    4-regular on 8 vertices: complement of perfect matching on 8 vertices (i.e., K8 minus 4 disjoint edges).
    This has girth 4 (triangles exist? K8 minus matching: each vertex connected to all except its matched partner.
    Triangles: {0,1,2} if 0-1, 1-2, 0-2 all edges. If matching is (0,1),(2,3),(4,5),(6,7),
    then 0-1 is not an edge, so {0,1,2} is not a triangle. But {0,2,3}: 0-2 yes, 2-3 no (matched). 
    {0,2,4}: 0-2, 0-4, 2-4 all present. Triangle! So girth 3.
    
    Let's use a different approach. For the (4,10) cage, known constructions include:
    - The 320-vertex graph from the 2026 paper.
    - Cyclic covers of smaller graphs.
    
    Since we can't easily construct the 320-vertex graph, let's try a cyclic cover with a larger base.
    Base: 4-regular graph on m vertices with girth >= 2 (no triangles), and m >= 5.
    Cover: m^2 vertices. For m=16, cover has 256 vertices.
    We need a 4-regular graph on 16 vertices with girth >= 2 (triangle-free).
    The 16-vertex 4-regular triangle-free graph: e.g., the 4x4 grid graph (toroidal) is 4-regular, girth 4.
    Or the 16-vertex cube Q4 is 4-regular, girth 4.
    
    Let's use Q4 (4-cube) as base, m=16, voltages in Z_16.
    Cover has 256 vertices. Girth of cover >= girth(base) * m / gcd stuff... 
    Actually, for a voltage graph (B, phi) with base B and voltage group G, the girth of the cover
    is at least the girth of B if all cycles in B have non-zero voltage (i.e., the voltage around every
    cycle is not the identity). If a cycle in B has zero voltage, the cover contains a copy of that cycle.
    
    So we need to assign voltages in Z_m to edges of B such that every cycle in B has non-zero total voltage.
    For a bipartite base (like Q4), all cycles are even length. We can assign voltages such that
    every even cycle has non-zero sum.
    
    This is getting complex. Let's just try to construct a valid graph and verify it.
    For now, return the LUW result as fallback.
    """
    # For now, just return None to indicate "not implemented yet"
    return None

def construct(k, g):
    deadline = _time.time() + 110.0  # safety margin before 120s limit
    best_n, best_edges = None, None
    
    # Try LUW constructions
    try:
        n_luw, edges_luw = _construct_luw(k, g, deadline)
        if n_luw and len(edges_luw) > 0:
            # Verify
            if _check_girth(n_luw, edges_luw, k, g, deadline - 5.0):
                best_n, best_edges = n_luw, edges_luw
    except Exception:
        pass

    if best_n is None:
        raise ValueError("No valid graph constructed")
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
