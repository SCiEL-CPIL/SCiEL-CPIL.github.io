import random
import time

_N = 18
# adjacency bitmask per vertex
_adj = [0] * _N
_edges = []          # list of (u, v) with u < v
_EDGE_SET = set()


def _add_edge(u, v):
    if u == v:
        return
    if u > v:
        u, v = v, u
    if (u, v) in _EDGE_SET:
        return
    _adj[u] |= 1 << v
    _adj[v] |= 1 << u
    _EDGE_SET.add((u, v))
    _edges.append((u, v))


def _remove_edge(u, v):
    if u > v:
        u, v = v, u
    if (u, v) not in _EDGE_SET:
        return
    _adj[u] &= ~(1 << v)
    _adj[v] &= ~(1 << u)
    _EDGE_SET.remove((u, v))
    _edges.remove((u, v))


def _is_k4_free():
    # A K4 exists iff there is an edge (u,v) whose common-neighborhood contains an edge.
    for (u, v) in _edges:
        common = _adj[u] & _adj[v]
        # check if any two vertices in common are adjacent
        # iterate over set bits of common
        m = common
        while m:
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            rest = m & ~(1 << a)
            if rest & _adj[a]:
                return False
            m = rest
    return True


def _max_triangle_free_induced_size():
    # Exact branch & bound: max size of subset S with no triangle in the induced subgraph.
    # Use recursive search with pruning.
    # State: available candidate set (bitmask), current size.
    # A triangle in induced subgraph on S means there exist u,v,w in S all pairwise adjacent.
    # We want to maximize |S| such that induced subgraph is triangle-free.

    full = (1 << _N) - 1

    # Precompute for pruning: for a set S, if it contains a triangle we must exclude at least one vertex.
    # We do a max independent set style search on a 3-uniform hypergraph of triangles.
    # Build list of triangles in current graph.
    triangles = []
    for (u, v) in _edges:
        common = _adj[u] & _adj[v]
        m = common
        while m:
            lsb = m & (-m)
            w = lsb.bit_length() - 1
            triangles.append((u, v, w))
            m = rest = m & ~(1 << w)

    # We need max S with no triangle fully contained in S.
    # This is the maximum size of a set hitting the complement: i.e., S is an independent set in the 3-uniform hypergraph? No:
    # We want S such that no triangle is fully in S. Equivalently, the set of removed vertices R = V \ S is a vertex cover of the triangle hypergraph.
    # So we want to MINIMIZE |R| such that every triangle has at least one vertex in R.
    # Then answer = _N - min_vertex_cover_size.

    if not triangles:
        return _N

    # Branch and bound for minimum vertex cover of 3-uniform hypergraph (18 vertices).
    # Use recursive: pick a triangle, branch on removing one of its 3 vertices.
    # Prune with lower bound: greedy packing of disjoint triangles.
    # Also upper bound on R: current best.

    best = [_N]  # best cover size found (we minimize); start with all vertices
    # initial upper bound: take all vertices of any triangle? use greedy
    def _greedy_cover():
        cover = 0
        remaining = set(range(_N))
        tris = [set(t) for t in triangles]
        while tris:
            # pick a triangle with minimum remaining vertices? just pick first
            t = tris.pop()
            # choose a vertex in t (prefer one in most triangles for efficiency)
            # count occurrences among remaining tris
            best_v = None
            best_cnt = -1
            for v in t:
                if v not in remaining:
                    continue
                cnt = 0
                for tt in tris:
                    if v in tt:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = v
            if best_v is None:
                # all vertices already covered? shouldn't happen
                continue
            cover += 1
            remaining.discard(best_v)
            tris = [tt for tt in tris if best_v not in tt]
        return cover

    best[0] = min(best[0], _greedy_cover())

    # Precompute for each vertex, the set of triangle indices it hits? For 18 vertices and modest triangles, do direct.
    # Represent triangles as list of (u,v,w).
    # Recursive search:
    # state: covered_mask of vertices already in cover (bitmask), current cover size.
    # We need to ensure every triangle has at least one vertex in cover.
    # Equivalently, the set of vertices NOT in cover must contain no full triangle.
    # Let free = ~cover_mask. We need free to be triangle-free.

    # We'll search by adding vertices to cover. Prune: if current cover size >= best[0], stop.
    # Lower bound: number of disjoint triangles in free (each needs at least one more cover vertex).

    def _disjoint_triangles_in_free(free_mask):
        # greedy count of disjoint triangles within free_mask
        cnt = 0
        used = 0
        for (u, v, w) in triangles:
            if (free_mask >> u) & 1 and (free_mask >> v) & 1 and (free_mask >> w) & 1:
                if not (used & (1 << u)) and not (used & (1 << v)) and not (used & (1 << w)):
                    cnt += 1
                    used |= (1 << u) | (1 << v) | (1 << w)
                    if cnt >= best[0]:
                        return cnt
        return cnt

    def _search(cover_mask, cover_size, free_mask):
        if cover_size >= best[0]:
            return
        # if free_mask has no triangle, we have a valid cover
        # check: does free_mask contain any triangle?
        has_tri = False
        for (u, v, w) in triangles:
            if (free_mask >> u) & 1 and (free_mask >> v) & 1 and (free_mask >> w) & 1:
                has_tri = True
                break
        if not has_tri:
            if cover_size < best[0]:
                best[0] = cover_size
            return
        # lower bound: disjoint triangles in free_mask
        lb = _disjoint_triangles_in_free(free_mask)
        if cover_size + lb >= best[0]:
            return
        # pick a triangle in free_mask to branch on
        # choose triangle with minimum "options"? all 3 vertices are in free.
        # pick first triangle in free
        branch_tri = None
        for (u, v, w) in triangles:
            if (free_mask >> u) & 1 and (free_mask >> v) & 1 and (free_mask >> w) & 1:
                branch_tri = (u, v, w)
                break
        # order vertices by degree in triangles within free (more constrained first)
        verts = [branch_tri[0], branch_tri[1], branch_tri[2]]
        def _deg_in_free(x):
            c = 0
            for (a, b, c2) in triangles:
                if x in (a, b, c2) and (free_mask >> a) & 1 and (free_mask >> b) & 1 and (free_mask >> c2) & 1:
                    c += 1
            return c
        verts.sort(key=lambda x: -_deg_in_free(x))
        for v in verts:
            new_cover = cover_mask | (1 << v)
            new_free = free_mask & ~(1 << v)
            _search(new_cover, cover_size + 1, new_free)

    _search(0, 0, (1 << _N) - 1)
    return _N - best[0]


def _score():
    if not _is_k4_free():
        return 10**9
    return _max_triangle_free_induced_size()


def _random_greedy_k4_free(rng):
    # Build a random K4-free graph by adding edges in random order, skipping if it creates a K4.
    _adj[:] = [0] * _N
    _edges.clear()
    _EDGE_SET.clear()
    all_pairs = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    rng.shuffle(all_pairs)
    for (u, v) in all_pairs:
        # tentatively add
        _add_edge(u, v)
        if not _is_k4_free():
            _remove_edge(u, v)
    return


def _try_add_edge(u, v):
    """Try to add edge (u,v). Returns True if added successfully (K4-free maintained)."""
    _add_edge(u, v)
    if _is_k4_free():
        return True
    _remove_edge(u, v)
    return False


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s (and 150s kill)
    rng = random.Random(12345)

    best_edges = None
    best_score = 10**9

    # Initial: complete tripartite K6,6,6
    _adj[:] = [0] * _N
    _edges.clear()
    _EDGE_SET.clear()
    for i in range(3):
        for j in range(i + 1, 3):
            for a in range(i * 6, i * 6 + 6):
                for b in range(j * 6, j * 6 + 6):
                    _add_edge(a, b)
    s = _score()
    if s < best_score:
        best_score = s
        best_edges = list(_edges)

    # Multiple random restarts with local search
    iters = 0
    while time.time() < deadline:
        iters += 1
        # random restart
        _adj[:] = [0] * _N
        _edges.clear()
        _EDGE_SET.clear()
        _random_greedy_k4_free(rng)
        s = _score()
        if s < best_score:
            best_score = s
            best_edges = list(_edges)
        # Also try starting from the best known graph and perturbing
        if iters % 3 == 0 and best_edges is not None and time.time() < deadline:
            # Load best edges, then randomly remove some and re-optimize
            _adj[:] = [0] * _N
            _edges.clear()
            _EDGE_SET.clear()
            for (u, v) in best_edges:
                _add_edge(u, v)
            # Randomly remove 3-8 edges to create a perturbation
            if _edges:
                n_remove = min(len(_edges), rng.randint(3, 8))
                to_remove = rng.sample(_edges, n_remove)
                for (u, v) in to_remove:
                    _remove_edge(u, v)
            s = _score()
            if s < best_score:
                best_score = s
                best_edges = list(_edges)
        # Local search: iteratively try to add missing edges that improve the score
        if time.time() < deadline:
            s = _score()
            all_pairs = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
            # Multiple passes over all pairs
            for pass_num in range(50):
                if time.time() > deadline:
                    break
                rng.shuffle(all_pairs)
                any_improved = False
                for (u, v) in all_pairs:
                    if time.time() > deadline:
                        break
                    if (u, v) in _EDGE_SET:
                        continue
                    if _try_add_edge(u, v):
                        ns = _score()
                        if ns < s:
                            s = ns
                            any_improved = True
                            if s < best_score:
                                best_score = s
                                best_edges = list(_edges)
                        else:
                            # Edge was added but didn't improve; remove it to keep graph sparse
                            _remove_edge(u, v)
                if not any_improved:
                    break

    if best_edges is None:
        # fallback to K6,6,6
        best_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in range(i * 6, i * 6 + 6) for b in range(j * 6, j * 6 + 6)]

    return [[u, v] for (u, v) in best_edges]
