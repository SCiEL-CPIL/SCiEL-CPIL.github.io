"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
import sys

_DEADLINE = None
_START = None

def _elapsed():
    return time.time() - _START

def _deadline_ok(margin=5.0):
    return _elapsed() < (_DEADLINE - margin)

def _init_time():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 145.0  # safety margin before 150s kill

def construct():
    _init_time()
    n = 18
    best_edges = None
    best_score = 999
    rng = random.Random(12345)

    # adjacency as set of frozensets or bitmask
    # Represent graph as list of bitmask (int) of neighbors
    # K4-free check and max triangle-free (max clique in complement)

    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            m = adj[i]
            j = i + 1
            while j < n:
                if (m >> j) & 1:
                    edges.append([i, j])
                j += 1
        return edges

    def edges_to_adj(edges):
        adj = [0] * n
        for u, v in edges:
            if u == v:
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_k4_free(adj):
        # check no 4-clique
        for a in range(n):
            na = adj[a]
            # iterate b > a in na
            b = a + 1
            while b < n:
                if (na >> b) & 1:
                    nb = na & adj[b]
                    c = b + 1
                    while c < n:
                        if (nb >> c) & 1:
                            nc = nb & adj[c]
                            d = c + 1
                            while d < n:
                                if (nc >> d) & 1:
                                    return False
                                d += 1
                        c += 1
                b += 1
        return True

    def max_clique_size(comp_adj):
        # comp_adj: adjacency of complement graph (bitmask per vertex)
        # max clique in comp_adj = max triangle-free induced in G
        # Use branch and bound with coloring (Tomita)
        best = 0
        # order vertices by degree in comp
        order = sorted(range(n), key=lambda v: bin(comp_adj[v]).count('1'))
        # remap
        remap = {v: i for i, v in enumerate(order)}
        cadj = [0] * n
        for v in range(n):
            m = 0
            for u in range(n):
                if (comp_adj[v] >> u) & 1:
                    m |= (1 << remap[u])
            cadj[remap[v]] = m
        # greedy coloring on the candidate set for upper bound
        # We'll implement a simple branch and bound with a coloring bound.

        def color_bound(cand_mask):
            # greedy coloring of the subgraph induced by cand_mask in cadj
            # returns number of colors
            colors = 0
            uncolored = cand_mask
            while uncolored:
                colors += 1
                color_class = 0
                uncolored_copy = uncolored
                while uncolored_copy:
                    # pick a vertex
                    v = (uncolored_copy & -uncolored_copy).bit_length() - 1
                    # can we add v to current color class? i.e. v not adjacent to any in color_class
                    if cadj[v] & color_class == 0:
                        color_class |= (1 << v)
                        uncolored &= ~(1 << v)
                    uncolored_copy &= ~(1 << v)
            return colors

        def expand(cand_mask, clique_size):
            nonlocal best
            if not cand_mask:
                if clique_size > best:
                    best = clique_size
                return
            # bound
            cb = color_bound(cand_mask)
            if clique_size + cb <= best:
                return
            # pick pivot? simple: iterate vertices
            # order by degree within cand
            cm = cand_mask
            while cm:
                v = (cm & -cm).bit_length() - 1
                cm &= cm - 1
                new_clique = clique_size + 1
                if new_clique + color_bound(cand_mask & cadj[v]) <= best:
                    continue
                expand(cand_mask & cadj[v], new_clique)
                cand_mask &= ~(1 << v)

        expand((1 << n) - 1, 0)
        return best

    def max_triangle_free(adj):
        # complement
        comp = [0] * n
        for i in range(n):
            full = ((1 << n) - 1) ^ (1 << i)
            comp[i] = full & ~adj[i]
        return max_clique_size(comp)

    def score_of(adj):
        if not is_k4_free(adj):
            return 999
        return max_triangle_free(adj)

    # Initial constructions
    candidates = []

    # 1. Complete tripartite K6,6,6 (seed)
    adj = [0] * n
    for i in range(6):
        for j in range(6, 12):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
        for j in range(12, 18):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
    for i in range(6, 12):
        for j in range(12, 18):
            adj[i] |= 1 << j
            adj[j] |= 1 << i
    candidates.append(adj)

    # 2. Blow-up of C5: partition 18 into 5 parts as equal as possible
    # C5 is triangle-free and K4-free. Blow-up: complete between adjacent parts, none within or non-adjacent.
    # This is K4-free if C5 is K4-free (yes). Triangle-free induced? A triangle in blow-up needs 3 mutually adjacent parts, which C5 has none. So any triangle-free... actually the whole graph is triangle-free? No: blow-up of C5 has triangles? C5 has no triangles, blow-up: a triangle needs 3 vertices in 3 parts pairwise adjacent in C5. C5 has no triangle, so blow-up is triangle-free. So max triangle-free = 18. Bad.
    # Instead use a K4-free graph with triangles, e.g. blow-up of the 5-cycle with chords? 
    # Better: use the icosahedron-like or the complement of a triangle-free graph.
    # Let's use a random K4-free graph and local search.

    # 3. Random K4-free: start empty, add random edges, reject if creates K4.
    for trial in range(3):
        adj = [0] * n
        # random edges
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_pairs)
        for u, v in all_pairs:
            # check if adding (u,v) creates a K4
            # K4 containing u,v: need a,b with a,b in N(u)∩N(v) and a-b edge, and a,b both >? just check
            common = adj[u] & adj[v]
            # for each pair a,b in common with a-b edge, adding u-v makes K4 {u,v,a,b}
            ok = True
            cm = common
            while cm:
                a = (cm & -cm).bit_length() - 1
                cm &= cm - 1
                cm2 = common & adj[a]
                while cm2:
                    b = (cm2 & -cm2).bit_length() - 1
                    cm2 &= cm2 - 1
                    if b > a:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                adj[u] |= 1 << v
                adj[v] |= 1 << u
        candidates.append(adj)

    # 4. Blow-up of the Grötzsch graph? Too complex. Use a known small K4-free graph with triangles.
    # Use the 5-wheel? W5 (5-cycle + center) is K4-free? Center connected to all 5 cycle vertices; cycle has no triangle; center + two adjacent cycle vertices = triangle; K4? center + 3 cycle vertices: need 3 cycle vertices mutually adjacent - no. So K4-free. Blow up.
    # Let's just do local search from the random and seed candidates.

    def local_search(adj, rng, max_iter=2000):
        nonlocal best_edges, best_score
        cur = [a for a in adj]
        cur_score = score_of(cur)
        if cur_score < best_score:
            best_score = cur_score
            best_edges = adj_to_edges(cur)
        for _ in range(max_iter):
            if not _deadline_ok():
                break
            # random move: add or remove an edge
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            u, v = rng.choice(all_pairs)
            has = (cur[u] >> v) & 1
            if has:
                # remove
                cur[u] &= ~(1 << v)
                cur[v] &= ~(1 << u)
            else:
                # add, check K4
                common = cur[u] & cur[v]
                ok = True
                cm = common
                while cm:
                    a = (cm & -cm).bit_length() - 1
                    cm &= cm - 1
                    cm2 = common & cur[a]
                    while cm2:
                        b = (cm2 & -cm2).bit_length() - 1
                        cm2 &= cm2 - 1
                        if b > a:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                cur[u] |= 1 << v
                cur[v] |= 1 << u
            new_score = score_of(cur)
            if new_score <= cur_score:
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_edges = adj_to_edges(cur)
                    if best_score <= 8:
                        return
            else:
                # revert
                if has:
                    cur[u] |= 1 << v
                    cur[v] |= 1 << u
                else:
                    cur[u] &= ~(1 << v)
                    cur[v] &= ~(1 << u)
        return

    # Evaluate candidates and run local search
    for adj in candidates:
        if not _deadline_ok():
            break
        local_search(adj, rng)
        if best_score <= 8:
            break

    # Additional random restarts
    restarts = 0
    while _deadline_ok() and restarts < 30:
        restarts += 1
        adj = [0] * n
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_pairs)
        for u, v in all_pairs:
            common = adj[u] & adj[v]
            ok = True
            cm = common
            while cm:
                a = (cm & -cm).bit_length() - 1
                cm &= cm - 1
                cm2 = common & adj[a]
                while cm2:
                    b = (cm2 & -cm2).bit_length() - 1
                    cm2 &= cm2 - 1
                    if b > a:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                adj[u] |= 1 << v
                adj[v] |= 1 << u
        local_search(adj, rng)
        if best_score <= 8:
            break

    if best_edges is None:
        # fallback: seed
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        best_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]

    return best_edges
