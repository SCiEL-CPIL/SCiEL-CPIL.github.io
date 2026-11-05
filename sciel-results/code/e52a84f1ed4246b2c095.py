"""Search-based construction for R(C4,K11) > n.

Strategy:
  * Polarity graphs of projective planes PG(2,q) are provably C4-free
    (two points share at most one neighbour), so they are strong structural
    seeds.  For some q their independence number is < 11, certifying
    R(C4,K11) > n.
  * Verification uses fast integer bitmasks (no set intersections in hot
    loops) so the 150 s budget is spent on search, not checking.
"""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

# ---------- finite geometry helpers ----------

def _inv_mod(a, q):
    """Multiplicative inverse of a mod q (q prime).  a in 1..q-1."""
    return pow(a, q - 2, q)


def _projective_points_fast(q):
    """All points of PG(2,q) as triples (x,y,z) with first non-zero coord = 1."""
    pts = []
    # Precompute inverses
    invs = [0] + [pow(a, q - 2, q) for a in range(1, q)]
    
    # Type 1: x != 0, normalize to x=1
    for x in range(1, q):
        ix = invs[x]
        for y in range(q):
            for z in range(q):
                pts.append((1, (y * ix) % q, (z * ix) % q))
    
    # Type 2: x=0, y != 0, normalize to y=1
    for y in range(1, q):
        iy = invs[y]
        for z in range(q):
            pts.append((0, 1, (z * iy) % q))
    
    # Type 3: x=0, y=0, z != 0, normalize to z=1
    pts.append((0, 0, 1))
    
    return pts


def _polarity_graph_fast(q):
    """C4-free polarity graph of PG(2,q): p~q iff dot(p,q) == 0 (mod q).
    
    Uses vectorized numpy operations for speed.
    """
    pts = _projective_points_fast(q)
    n = len(pts)
    pts_arr = np.array(pts, dtype=np.int64)
    
    # Compute all pairwise dot products
    # dots[i,j] = dot(pts[i], pts[j]) mod q
    # Use matrix multiplication: dots = pts_arr @ pts_arr.T
    dots = (pts_arr @ pts_arr.T) % q
    
    # Build adjacency: adj[i] has bit j set if dots[i,j] == 0 and i != j
    adj = [0] * n
    for i in range(n):
        # Find all j where dots[i,j] == 0 and j != i
        mask = (dots[i] == 0) & (np.arange(n) != i)
        adj_bits = 0
        for j in range(n):
            if mask[j]:
                adj_bits |= (1 << j)
        adj[i] = adj_bits
    
    return n, adj


# ---------- fast verification (bitmask based) ----------

def _is_c4_free(n, adj):
    """No two vertices may share two or more neighbours."""
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] & adj[j]) & ((adj[i] & adj[j]) - 1):  # >= 2 bits set
                return False
    return True


def _greedy_alpha(n, adj):
    """Greedy (min-degree) lower bound on the independence number."""
    rem = (1 << n) - 1
    count = 0
    while rem:
        best_v = -1
        best_deg = n
        r = rem
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            d = (adj[v] & rem).bit_count()
            if d < best_deg:
                best_deg = d
                best_v = v
                if d == 0:
                    break
            r ^= lb
        count += 1
        rem &= ~(adj[best_v] | (1 << best_v))
    return count


def _has_independent_11(n, adj, limit=11):
    """Exact: is there an independent set of size >= limit?  (bounded DFS)
    
    Uses branch and bound with good vertex ordering.
    """
    if n < limit:
        return False

    # Precompute degrees for better ordering
    degs = [adj[v].bit_count() for v in range(n)]
    
    def dfs(rem, k):
        if k >= limit:
            return True
        if rem.bit_count() < limit - k:
            return False
        
        # Pick vertex with max degree in remaining set (for pruning)
        best_v = -1
        best_deg = -1
        r = rem
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            d = (adj[v] & rem).bit_count()
            if d > best_deg:
                best_deg = d
                best_v = v
            r ^= lb
        
        if best_v < 0:
            return False
        
        # Branch 1: include best_v (remove it and its neighbors)
        new_rem = rem & ~(adj[best_v] | (1 << best_v))
        if dfs(new_rem, k + 1):
            return True
        
        # Branch 2: exclude best_v
        new_rem = rem & ~(1 << best_v)
        if dfs(new_rem, k):
            return True
        
        return False

    return dfs((1 << n) - 1, 0)


# ---------- greedy extension (try to grow a valid graph) ----------

def _can_add(adj, n, cand_mask):
    """Can a new vertex n with neighbour-set cand_mask keep the graph C4-free?"""
    # (1) every existing vertex u must have <= 1 neighbour in cand
    for u in range(n):
        c = adj[u] & cand_mask
        if c & (c - 1):  # >= 2
            return False
    # (2) no two vertices in cand may share a neighbour
    r = cand_mask
    while r:
        lb = r & (-r)
        v = lb.bit_length() - 1
        if adj[v] & (cand_mask ^ lb):
            return False
        r ^= lb
    return True


def _grow_from_seed(n0, adj0, deadline, seed):
    """Greedily add vertices to a C4-free seed while keeping alpha < 11."""
    rng = random.Random(seed)
    n = n0
    adj = adj0[:]
    
    # Precompute current degrees
    degs = [adj[v].bit_count() for v in range(n)]
    
    while n < 43 and time.time() < deadline:
        added = False
        # Try multiple random candidates
        for attempt in range(500):
            if n == 0:
                cand = 0
            else:
                # Try different target densities
                density = rng.uniform(0.2, 0.5)
                target = max(1, int(n * density))
                target = min(target, n)
                order = list(range(n))
                rng.shuffle(order)
                cand = 0
                for v in order[:target]:
                    cand |= (1 << v)
            
            if _can_add(adj, n, cand):
                # Use greedy alpha bound: sum of 1/(deg+1)
                cw = 0.0
                for u in range(n):
                    nd = degs[u] + (1 if (cand >> u) & 1 else 0)
                    cw += 1.0 / (nd + 1.0)
                cw += 1.0 / (cand.bit_count() + 1.0)
                if cw <= 10.2:
                    # Accept the vertex
                    adj.append(cand)
                    for u in range(n):
                        if (cand >> u) & 1:
                            adj[u] |= (1 << n)
                    degs[u] += 1 if (cand >> u) & 1 else 0
                    degs.append(cand.bit_count())
                    n += 1
                    added = True
                    break
        
        if not added:
            # Try a different strategy: add vertices with low degree requirements
            for _ in range(100):
                if n == 0:
                    break
                # Try adding a vertex connected to a small random subset
                target = rng.randint(1, max(1, n // 5))
                order = list(range(n))
                rng.shuffle(order)
                cand = 0
                for v in order[:target]:
                    cand |= (1 << v)
                
                if _can_add(adj, n, cand):
                    cw = 0.0
                    for u in range(n):
                        nd = degs[u] + (1 if (cand >> u) & 1 else 0)
                        cw += 1.0 / (nd + 1.0)
                    cw += 1.0 / (cand.bit_count() + 1.0)
                    if cw <= 10.2:
                        adj.append(cand)
                        for u in range(n):
                            if (cand >> u) & 1:
                                adj[u] |= (1 << n)
                        degs[u] += 1 if (cand >> u) & 1 else 0
                        degs.append(cand.bit_count())
                        n += 1
                        added = True
                        break
            
            if not added:
                break
    return n, adj


# ---------- main entry point ----------

def construct():
    deadline = time.time() + 140.0  # safety margin before the 150 s kill

    best_n = 0
    best_adj = None

    def consider(n, adj):
        nonlocal best_n, best_adj
        if n <= best_n or n < 1:
            return
        if not _is_c4_free(n, adj):
            return
        if _has_independent_11(n, adj):
            return
        best_n = n
        best_adj = adj[:]

    # Strategy 1: polarity graphs (provably C4-free)
    # Try smaller q first for faster generation
    for q in [5, 7, 11, 13, 17, 19, 23]:
        if time.time() > deadline - 10:
            break
        try:
            n, adj = _polarity_graph_fast(q)
            if not _is_c4_free(n, adj):
                continue
            if _has_independent_11(n, adj):
                continue
            consider(n, adj)
            # try to extend the valid polarity graph
            if time.time() < deadline - 8:
                ng, adjg = _grow_from_seed(n, adj, deadline - 4, seed=q * 1000 + 1)
                consider(ng, adjg)
        except Exception:
            continue

    # Strategy 2: greedy from scratch (multiple seeds)
    for seed in range(30):
        if time.time() > deadline - 6:
            break
        try:
            ng, adjg = _grow_from_seed(0, [], deadline - 4, seed=10000 + seed)
            consider(ng, adjg)
        except Exception:
            continue
    
    # Strategy 3: Try to extend from known good constructions
    # Start from the 10-triangle fallback and try to grow
    if best_n <= 30:
        n = 30
        adj = [0] * n
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            adj[a] |= (1 << b) | (1 << c)
            adj[b] |= (1 << a) | (1 << c)
            adj[c] |= (1 << a) | (1 << b)
        
        if time.time() < deadline - 8:
            ng, adjg = _grow_from_seed(n, adj, deadline - 4, seed=99999)
            consider(ng, adjg)

    # Fallback: ten disjoint triangles (alpha = 10, C4-free, n = 30)
    if best_n < 30:
        n = 30
        adj = [0] * n
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            adj[a] |= (1 << b) | (1 << c)
            adj[b] |= (1 << a) | (1 << c)
            adj[c] |= (1 << a) | (1 << b)
        best_n = n
        best_adj = adj

    edges = []
    for i in range(best_n):
        m = best_adj[i]
        while m:
            lb = m & (-m)
            j = lb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= lb

    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
