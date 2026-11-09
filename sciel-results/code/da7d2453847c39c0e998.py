import time
import random
import sys

_START_TIME = time.time()
DEADLINE = 110.0  # seconds, safety margin before 120s

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _is_k4_free(adj, n=18):
    """Check no K4 in graph with adjacency bitmasks adj (list of ints)."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            # Check if common contains an edge (i.e., two vertices in common are adjacent)
            c = common
            while c:
                lsb = c & -c
                x = lsb.bit_length() - 1
                c -= lsb
                if (adj[x] & common) != 0:
                    return False
    return True

def _max_triangle_free_induced(adj, n=18):
    """Exact branch-and-bound for largest triangle-free induced subgraph using bitmasks."""
    # A set S is triangle-free if no three vertices in S are mutually adjacent.
    # We want max |S| such that the induced subgraph on S has no triangle.
    
    best = [0]
    
    # Precompute adjacency for fast access
    adj_mask = adj
    
    def _has_triangle_with(S_mask, v):
        # Check if adding v to S creates a triangle
        # A triangle is formed if there exist u, w in S such that u-v, v-w, u-w are all edges
        # i.e., v is adjacent to two vertices u, w in S that are also adjacent to each other
        neighbors_in_S = adj_mask[v] & S_mask
        # Check if neighbors_in_S has any edge
        nn = neighbors_in_S
        while nn:
            lsb = nn & -nn
            u = lsb.bit_length() - 1
            nn -= lsb
            if (adj_mask[u] & neighbors_in_S) != 0:
                return True
        return False

    def _bb(S_mask, R_mask):
        if _elapsed() > DEADLINE:
            return
        # Upper bound: current size + remaining
        if bin(S_mask).count('1') + bin(R_mask).count('1') <= best[0]:
            return
        if not R_mask:
            sz = bin(S_mask).count('1')
            if sz > best[0]:
                best[0] = sz
            return
        
        # Pick a vertex from R_mask
        lsb = R_mask & -R_mask
        v = lsb.bit_length() - 1
        rest = R_mask ^ lsb
        
        # Option 1: exclude v
        _bb(S_mask, rest)
        
        # Option 2: include v
        if not _has_triangle_with(S_mask, v):
            _bb(S_mask | lsb, rest)
    
    _bb(0, (1 << n) - 1)
    return best[0]

def _make_adj_from_edges(edges, n=18):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _edges_from_adj(adj, n=18):
    return [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

def _eval(adj, n=18):
    """Return (k4_free, max_tf). If not K4-free, max_tf = -1 (invalid)."""
    if not _is_k4_free(adj, n):
        return False, -1
    mt = _max_triangle_free_induced(adj, n)
    return True, mt

def construct():
    n = 18
    # Start from K6,6,6 (3-partite)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    current_adj = adj[:]
    current_score = None
    if _time_left() > 5:
        ok, sc = _eval(current_adj, n)
        current_score = sc if ok else -1
    
    best_adj = current_adj[:]
    best_score = current_score if current_score is not None and current_score > 0 else 12
    if current_score is None or current_score < 0:
        best_score = 12
    
    # Local search: try to improve by flipping edges
    # We want to minimize the max triangle-free induced subgraph size.
    # Strategy: try to add intra-part edges (to create triangles that break up large TF sets)
    # and remove some inter-part edges (to reduce bipartite TF sets), while staying K4-free.
    
    random.seed(42)
    
    def _try_flip(adj_mat, u, v, current_best):
        """Try flipping edge (u,v). Return (new_adj, new_score) if better, else (None, None)."""
        if _elapsed() > DEADLINE:
            return None, None
        new_adj = adj_mat[:]
        new_adj[u] ^= (1 << v)
        new_adj[v] ^= (1 << u)
        ok, sc = _eval(new_adj, n)
        if ok and sc < current_best:
            return new_adj, sc
        return None, None
    
    # Phase 1: Hill climbing on edge flips
    improved = True
    while improved and _time_left() > 10:
        improved = False
        # Try all possible edge flips
        for u in range(n):
            for v in range(u + 1, n):
                if _time_left() < 10:
                    break
                new_adj, sc = _try_flip(current_adj, u, v, best_score)
                if new_adj is not None:
                    current_adj = new_adj
                    best_score = sc
                    best_adj = current_adj[:]
                    improved = True
        if improved:
            continue
        # If no improvement, try random restarts with small perturbations
        if _time_left() < 20:
            break
        # Perturb: flip k random edges
        k = random.randint(3, 8)
        perturbed = current_adj[:]
        for _ in range(k):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                perturbed[u] ^= (1 << v)
                perturbed[v] ^= (1 << u)
        # Evaluate perturbed
        ok, sc = _eval(perturbed, n)
        if ok and sc < best_score:
            current_adj = perturbed
            best_score = sc
            best_adj = perturbed[:]
            improved = True
        elif ok:
            # Accept if comparable, to escape local optima
            if sc <= best_score + 1 and random.random() < 0.3:
                current_adj = perturbed
                improved = True
    
    # Final: ensure we have a valid K4-free graph
    ok, sc = _eval(best_adj, n)
    if not ok:
        # Fallback to seed
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    return _edges_from_adj(best_adj, n)
