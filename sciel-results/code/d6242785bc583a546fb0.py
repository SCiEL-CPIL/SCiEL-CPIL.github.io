import time
import random

_START = time.time()
_DEADLINE = 145.0
_N = 18
_MAXV = 1 << 18
_MASK_ALL = (1 << 18) - 1


def _elapsed():
    return time.time() - _START


def _bitcount(x):
    return bin(x).count("1")


def _popcount_list(bits):
    # bits is a list of 18 ints; count total set bits
    c = 0
    for b in bits:
        c += bin(b).count("1")
    return c


def _random_k4free_adj():
    """Generate a random K4-free graph via random 3-partition + random edge removals."""
    random.shuffle  # ensure RNG is seeded per call by caller
    parts = [list(range(_N))]
    random.shuffle(parts[0])
    # split into 3 roughly equal parts
    p0 = parts[0][:6]
    p1 = parts[0][6:12]
    p2 = parts[0][12:18]
    part_of = [-1] * _N
    for i, v in enumerate(p0):
        part_of[v] = 0
    for i, v in enumerate(p1):
        part_of[v] = 1
    for i, v in enumerate(p2):
        part_of[v] = 2
    adj = [0] * _N
    for u in range(_N):
        for v in range(u + 1, _N):
            if part_of[u] != part_of[v] and random.random() < 0.85:
                adj[u] |= 1 << v
                adj[v] |= 1 << u
    return adj


def _is_k4_free(adj):
    for u in range(_N):
        for v in range(u + 1, _N):
            if not (adj[u] >> v) & 1:
                continue
            common = adj[u] & adj[v]
            if _bitcount(common) >= 2:
                return False
    return True


def _build_triangle_adj(adj):
    """Triangle graph: t_adj[u] has bit v if u,v are in a common triangle."""
    t_adj = [0] * _N
    for u in range(_N):
        for v in range(u + 1, _N):
            if not (adj[u] >> v) & 1:
                continue
            common = adj[u] & adj[v]
            if _bitcount(common) >= 1:
                t_adj[u] |= 1 << v
                t_adj[v] |= 1 << u
    return t_adj


def _max_independent_set_size(t_adj, deadline):
    """Upper bound on alpha via greedy coloring branch-and-bound on complement."""
    # We want max independent set in graph G (triangle graph).
    # Equivalently, max clique in complement graph.
    # Use a simple branch-and-bound.
    best = [0]

    def _coloring_bound(cand_mask):
        """Greedy coloring of cand_mask in triangle graph -> lower bound on chromatic number of complement = upper bound on clique size in complement = upper bound on independent set size in triangle graph."""
        # Actually for independent set in G, we need clique in complement.
        # Simpler: use greedy upper bound via coloring G[cand].
        # Chromatic number of G[cand] >= size of any clique in G[cand], but we want independent set in G[cand].
        # Independent set in G = clique in complement of G.
        # Let's just do a simple recursive branch-and-bound for max independent set.
        pass
    # Simpler approach: recursive branch and bound
    def _bnb(remaining, current_size, chosen_mask):
        if _elapsed() > deadline:
            return
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        # Upper bound: current_size + popcount(remaining)
        if current_size + _bitcount(remaining) <= best[0]:
            return
        # Pick a vertex from remaining
        v = (remaining & -remaining).bit_length() - 1
        # Branch: include v
        _bnb(remaining & ~t_adj[v] & ~(1 << v), current_size + 1, chosen_mask | (1 << v))
        # Branch: exclude v
        _bnb(remaining & ~(1 << v), current_size, chosen_mask)

    _bnb(_MASK_ALL, 0, 0)
    return best[0]


def _max_independent_set_size_fast(t_adj):
    """Faster max independent set via greedy + branch and bound with coloring bound."""
    # Use a simple greedy coloring upper bound on the complement
    # For now, use a randomized greedy approach multiple times
    best = 0
    for _ in range(200):
        # Greedy: repeatedly pick vertex with minimum degree in remaining triangle graph
        remaining = _MASK_ALL
        count = 0
        while remaining:
            # Find vertex with fewest neighbors in remaining
            min_deg = 999
            min_v = -1
            for v in range(_N):
                if not (remaining >> v) & 1:
                    continue
                deg = _bitcount(t_adj[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            if min_v < 0:
                break
            count += 1
            remaining &= ~t_adj[min_v]
            remaining &= ~(1 << min_v)
        if count > best:
            best = count
    return best


def _try_improve(adj, deadline):
    """Local search: try to reduce max triangle-free induced subgraph by removing edges."""
    t_adj = _build_triangle_adj(adj)
    cur_score = _max_independent_set_size_fast(t_adj)
    
    # Try removing random edges (within a part of the implied partition) to break triangles
    for _ in range(5000):
        if _elapsed() > deadline:
            break
        # Try removing a random edge
        u = random.randrange(_N)
        v = random.randrange(_N)
        if u == v or not (adj[u] >> v) & 1:
            continue
        # Remove edge
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        if not _is_k4_free(adj):
            # Restore and skip
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            continue
        new_t_adj = _build_triangle_adj(adj)
        new_score = _max_independent_set_size_fast(new_t_adj)
        if new_score < cur_score:
            cur_score = new_score
            t_adj = new_t_adj
        else:
            # Restore
            adj[u] |= 1 << v
            adj[v] |= 1 << u
    return cur_score, t_adj


def _turan_graph_adj():
    """Return adjacency list for T(18,3)."""
    adj = [0] * _N
    for u in range(_N):
        pu = u // 6
        for v in range(_N):
            pv = v // 6
            if u != v and pu != pv:
                adj[u] |= 1 << v
    return adj


def construct():
    deadline = _START + _DEADLINE
    rng = random.Random(42)
    random.seed(42)
    
    # Start with Turan graph
    adj = _turan_graph_adj()
    t_adj = _build_triangle_adj(adj)
    best_score = _max_independent_set_size_fast(t_adj)
    best_adj = adj[:]
    
    # Try local search from Turan
    if _elapsed() < deadline - 30:
        s, _ = _try_improve(adj[:], deadline - 30)
        if s < best_score:
            best_score = s
            best_adj = adj[:]
    
    # Try random K4-free graphs
    for attempt in range(30):
        if _elapsed() > deadline - 10:
            break
        adj_try = _random_k4free_adj()
        if not _is_k4_free(adj_try):
            continue
        t_adj_try = _build_triangle_adj(adj_try)
        score_try = _max_independent_set_size_fast(t_adj_try)
        if score_try < best_score:
            best_score = score_try
            best_adj = adj_try[:]
        # Local search
        if _elapsed() < deadline - 10:
            s2, _ = _try_improve(adj_try[:], min(deadline - 5, _START + _elapsed() + 5))
            if s2 < best_score:
                best_score = s2
                best_adj = adj_try[:]
    
    # Build edge list from best_adj
    edges = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
