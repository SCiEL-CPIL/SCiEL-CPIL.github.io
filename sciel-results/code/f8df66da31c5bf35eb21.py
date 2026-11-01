"""Construct a 4-regular graph of girth 10 via Lazebnik-Ustimenko-Woldar CD(d, q)."""
import time
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START

def _build_luw(q, d, deadline, max_vertices=None):
    """Build the connected component of the zero point in CD(d, q).
    Returns (n, edges) or None if the deadline is exceeded / size cap hit."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
    index = {start: 0}
    order = [start]
    edges = []
    for vertex in order:
        if time.time() > deadline:
            return None
        if max_vertices is not None and len(order) > max_vertices:
            return None
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                if max_vertices is not None and len(order) > max_vertices:
                    return None
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _verify_girth_numpy(n, edges, k, g, deadline):
    """Vectorized non-backtracking girth check using NumPy.
    Returns True iff no cycle shorter than g exists.
    
    Strategy: For each step L from 1 to g-1, track the set of directed edges
    reachable by non-backtracking walks of length L. A cycle of length L is
    detected when a directed edge (u,v) at step L has u equal to the start
    vertex of the walk. We do this from every start vertex.
    
    More efficient: use the non-backtracking matrix. For a graph with m directed
    edges, the non-backtracking matrix B is m x m where B[(u,v),(v,w)] = 1 if
    w != u. A cycle of length L exists iff B^L has a positive diagonal entry.
    We check B^1, B^2, ..., B^(g-1) for diagonal entries.
    """
    if time.time() > deadline:
        return False

    # Build directed edge list
    # Each undirected edge (u,v) gives directed edges (u,v) and (v,u)
    m = 2 * len(edges)
    # Map directed edge to index
    # directed_edges[i] = (src, dst)
    directed_src = np.zeros(m, dtype=np.int32)
    directed_dst = np.zeros(m, dtype=np.int32)
    edge_to_index = {}
    for idx, (u, v) in enumerate(edges):
        directed_src[2*idx] = u
        directed_dst[2*idx] = v
        directed_src[2*idx+1] = v
        directed_dst[2*idx+1] = u
        edge_to_index[(u, v)] = 2*idx
        edge_to_index[(v, u)] = 2*idx+1

    # Build the non-backtracking matrix as sparse: for each directed edge (u,v),
    # the next directed edges are (v,w) for all w != u.
    # We'll represent B as a list of (row, col) pairs for sparse matvec.
    # Actually, for matvec: B @ x where x is a vector of length m.
    # (B @ x)[i] = sum over j: B[i,j] * x[j]
    # B[i,j] = 1 if directed_edges[j] = (u,v) and directed_edges[i] = (v,w) with w != u
    # i.e., dst[j] == src[i] and dst[i] != src[j]
    
    # For efficiency, build adjacency: for each vertex v, list of directed edges starting at v
    # and list of directed edges ending at v
    from collections import defaultdict
    out_edges = defaultdict(list)  # vertex -> list of directed edge indices starting at vertex
    in_edges = defaultdict(list)   # vertex -> list of directed edge indices ending at vertex
    for i in range(m):
        out_edges[directed_src[i]].append(i)
        in_edges[directed_dst[i]].append(i)

    # Non-backtracking matvec: given vector x (length m), compute y = B @ x
    # y[i] where directed edge i is (u, v): y[i] = sum over j where (src[j], dst[j]) = (v, w), w != u of x[j]
    # = sum over j in out_edges[v] where dst[j] != u of x[j]
    # But we also need to track which start vertex each walk came from.
    
    # Alternative approach: for each start vertex s, do the non-backtracking walk
    # and check if we return to s. But that's n separate computations.
    
    # Better: use the fact that a cycle of length L < g exists iff there's a
    # non-backtracking closed walk of length L. We can track for each directed
    # edge (u,v) the set of start vertices that can reach it via non-backtracking
    # walks of a given length. But that's O(n*m) memory.
    
    # Most practical: for each start vertex s, do a BFS of non-backtracking walks
    # and check if s is reached. Use NumPy to speed up the inner loop.
    
    # Actually, let's use a smarter approach: 
    # For each directed edge (a, b), we want to know if there's a non-backtracking
    # walk from (b, a) back to (a, c) for some c != b, of length < g-1.
    # This is equivalent to: in the non-backtracking graph, is there a path from
    # directed edge (b,a) to any directed edge (a,c) with c!=b, of length < g-1?
    
    # Let's just do the per-vertex BFS but optimize the inner loop with numpy.
    # For each start vertex s:
    #   state: set of (cur, prev) pairs, with distance
    #   At each step, for each (cur, prev), extend to (nxt, cur) for nxt != prev
    #   If we reach (s, x) with x != start_neighbor, we found a cycle.
    
    # Given n=512, k=4, g=10: per vertex, max states = 4*3^8 = 26244
    # Total: 512 * 26244 = 13.4M state visits. In Python, this is ~5-10s.
    # With numpy vectorization per step, we can do much better.
    
    # Let me try a different vectorization: for each step L, maintain a boolean
    # array of length m (directed edges) indicating which directed edges are
    # reachable from start s via non-backtracking walks of length L.
    # But we need to do this for each s separately...
    
    # Alternative: matrix power approach.
    # Build the non-backtracking matrix B (sparse, m x m).
    # A cycle of length L exists iff trace(B^L) > 0.
    # We can compute B, B^2, ..., B^(g-1) and check diagonals.
    # B is m x m = 1024 x 1024. Sparse matvec is fast.
    
    # Build B as a sparse structure for matvec.
    # B[i, j] = 1 if directed_edges[j] = (u, v) and directed_edges[i] = (v, w) with w != u
    # For matvec: (B @ x)[i] = sum_{j: dst[j]=src[i], dst[i]!=src[j]} x[j]
    
    # Precompute for each directed edge i = (u, v):
    #   the set of j such that dst[j] == u (i.e., edges ending at u = src[i])
    #   and dst[i] = v != src[j]
    # So: j in in_edges[src[i]] and dst[i] != src[j]
    
    # For speed, precompute for each i the list of valid j's.
    # But that's O(m*k) = O(1024*4) = 4096 entries. Fine.
    
    # Actually, let's just build B as a dense numpy array since m=1024 is small.
    B = np.zeros((m, m), dtype=np.int8)
    for j in range(m):
        u_j = directed_src[j]
        v_j = directed_dst[j]
        # B[i, j] = 1 if src[i] == v_j and dst[i] != u_j
        for i in out_edges[v_j]:
            if directed_dst[i] != u_j:
                B[i, j] = 1
    
    if time.time() > deadline:
        return False
    
    # Now check B^L for L = 1, 2, ..., g-1
    # A cycle of length L exists iff diag(B^L) has a positive entry.
    # Compute B^L iteratively.
    Bp = B.copy()  # B^1
    for L in range(1, g):
        # Check diagonal
        if np.any(np.diag(Bp) > 0):
            return False
        if L == g - 1:
            break
        Bp = Bp @ B
        if time.time() > deadline:
            return False
    
    return True


def _verify(n, edges, k, g, deadline):
    """Return True iff the graph is simple, k-regular, and has girth >= g."""
    if time.time() > deadline:
        return False
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        if len(adj[u]) != k:
            return False

    return _verify_girth_numpy(n, edges, k, g, deadline)


def construct(k, g):
    start_time = time.time()
    hard_deadline = start_time + 118.0

    q = k
    base = g - 5
    if base % 2 == 0:
        base += 1
    base = max(3, base)

    def est(d):
        return 2 * q * (q ** (d - 1))

    candidates = []
    for d in range(base, base + 8):
        if d % 2 == 1:
            candidates.append(d)
    candidates.sort(key=lambda d: est(d))

    best = None
    for d in candidates:
        if time.time() > hard_deadline - 4.0:
            break
        build_deadline = min(hard_deadline - 6.0, time.time() + 30.0)
        res = _build_luw(q, d, build_deadline)
        if res is None:
            continue
        n, edges = res
        if n % 2 != 0:
            continue
        ok = _verify(n, edges, k, g, hard_deadline - 2.0)
        if ok:
            if best is None or n < best[0]:
                best = (n, edges)
            break

    if best is not None:
        return best

    # Fallback
    d = base
    res = _build_luw(q, d, hard_deadline - 1.0)
    if res is not None:
        return res

    for d in range(base, base + 4):
        if d % 2 == 1:
            res = _build_luw(q, d, hard_deadline - 0.5)
            if res is not None:
                return res
    raise ValueError("Failed to construct a valid graph within the time budget")

# EVOLVE-BLOCK-END
