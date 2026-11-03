"""Cage solver: fast girth verifier + greedy random regular + targeted local search + LUW fallback."""
import time
import threading
import random
import itertools
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

_DEADLINE = 120.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


# ---------------- exact girth via non-backtracking walks (vectorized) ----------
def _girth_np(adj_lists, n, g):
    """Return the exact girth (shortest cycle length) of a simple graph, or
    g if no cycle shorter than g exists. Uses non-backtracking BFS from all
    vertices simultaneously, vectorized with NumPy."""
    deg = np.array([len(a) for a in adj_lists], dtype=np.int32)
    m = int(deg.sum())
    pos = np.zeros(m, dtype=np.int32)
    for u in range(n):
        pos[u * (k := deg[u]): (u + 1) * k] = np.arange(k)
    # neighbor lookup table: nbr[u][p] = adj[u][p]
    nbr = np.zeros((n, k), dtype=np.int32)
    for u in range(n):
        nbr[u, :] = np.array(adj_lists[u], dtype=np.int32)

    # state (v, p)  ->  index v*k + p
    # prev state: the (u, q) we came from, where u = nbr[v][p]
    # We want to detect when a non-backtracking walk returns to its origin.

    # BFS layers. Layer 1 = all directed edges (v, p) with v = nbr[u][p].
    # Represent a state by its index. We store for each state the origin vertex
    # (u) and the distance (number of edges so far).

    # For efficiency, do layered BFS with NumPy.
    # current: array of state indices
    # origin: array of origin vertex for each state
    # dist: scalar (all same within a layer)

    # Initialize layer 1: for each directed edge (u -> v) where v = nbr[u][p]
    # state index = v*k + p, origin = u, dist = 1
    u_idx = np.repeat(np.arange(n, dtype=np.int32), deg)
    p_idx = pos
    v_idx = nbr[u_idx, p_idx]
    init_state = v_idx * k + p_idx
    init_origin = u_idx

    # prev_state for state (v, p) is (u, q) where u = nbr[v][p] and q = position
    # of v in adj[u]. We need pos[u][v].
    # Build pos lookup: pos_lookup[u][v] = position of v in adj[u], or -1.
    pos_lookup = np.full((n, n), -1, dtype=np.int32)
    for u in range(n):
        for p, v in enumerate(adj_lists[u]):
            pos_lookup[u, v] = p

    # visited: for each state, store the origin vertex that first reached it,
    # and the distance. We'll use arrays.
    visited_origin = np.full(n * k, -1, dtype=np.int32)
    visited_dist = np.zeros(n * k, dtype=np.int32)

    # Mark initial layer
    visited_origin[init_state] = init_origin
    visited_dist[init_state] = 1

    cur_state = init_state
    cur_origin = init_origin
    dist = 1

    while len(cur_state) > 0:
        # Extend: from state (v, p), origin u, go to (w, r) where w != v's parent.
        # w = nbr[v][r], r in 0..k-1, r != p (non-backtracking)
        # new state index = w*k + r
        # new origin = u (same origin)
        # Check: if w == u -> cycle of length dist+1.
        # If state (w, r) already visited with a DIFFERENT origin -> cycle.
        #   cycle length = dist + 1 + visited_dist[(w,r)] ... actually need to
        #   be careful. Standard: if we reach (w, r) from (v, p) and (w, r) was
        #   first reached from some state, the cycle length = dist + 1 +
        #   visited_dist[(w, r)] only if same origin... Let me use the clean
        #   formulation: a non-backtracking walk from origin u: at each step
        #   the state is (current, prev). A cycle forms when current == origin
        #   (and prev != the neighbor we started from, but non-backtracking
        #   handles that). So we only need to check w == origin.

        v_cur = cur_state // k
        p_cur = cur_state % k

        # all next neighbors
        r_all = np.tile(np.arange(k, dtype=np.int32), len(v_cur))
        w_all = nbr[v_cur, r_all]
        # mask out r == p_cur (backtrack)
        mask = (r_all != p_cur)
        w_all = w_all[mask]
        r_all = r_all[mask]
        o_all = cur_origin[mask]

        # check w == origin
        hit = (w_all == o_all)
        if hit.any():
            return dist + 1

        new_state = w_all * k + r_all
        new_origin = o_all

        # check visited
        old_origin = visited_origin[new_state]
        is_new = (old_origin == -1)

        if not is_new.all():
            # some states were visited. If visited with a different origin,
            # that means two non-backtracking walks from different origins met
            # at the same state -> forms a cycle. Cycle length:
            #   dist + 1 + visited_dist[new_state] (the two walks join).
            # Actually the cycle is: origin_a -> ... -> (w,r) <- ... <- origin_b.
            # length = dist + 1 (from origin_a to w) + visited_dist (from w back
            # to origin_b) ... but the two paths meet at (w, r). The cycle
            # length = dist + 1 + visited_dist[new_state].
            diff = is_new == False
            cand = (dist + 1 + visited_dist[new_state[diff]])
            if cand.min() < g:
                return int(cand.min())
            # For same-origin revisit, that's just the same walk, ignore.
            # Actually if old_origin == new_origin it's a cycle too but we'd have
            # caught w==origin. So diff-origin is the only new cycle info.

        # proceed with only new states
        new_state = new_state[is_new]
        new_origin = new_origin[is_new]

        if len(new_state) == 0:
            break

        visited_origin[new_state] = new_origin
        visited_dist[new_state] = dist + 1
        cur_state = new_state
        cur_origin = new_origin
        dist += 1

        if dist >= g:
            break

    return g


def _nonback_girth(adj, n):
    """Exact girth via non-backtracking walks from every vertex."""
    if n == 0:
        return 10 ** 9
    try:
        return _girth_np(adj, n, 10 ** 9)
    except Exception:
        # fallback to pure-python
        prev = {}
        q = deque()
        for u in range(n):
            for p in adj[u]:
                key = (p, u)
                prev[key] = (u, 1)
                q.append(key)
        while q:
            (v, p), (u, dist) = q.popleft()
            for w in adj[v]:
                if w == p:
                    continue
                key = (w, v)
                if w == u:
                    return dist + 1
                if key in prev:
                    continue
                prev[key] = (v, dist + 1)
                q.append(key)
        return 10 ** 9


def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if any(len(adj[i]) != k for i in range(n)):
        return False
    for i in range(n):
        if len(set(adj[i])) != len(adj[i]):
            return False
        for j in adj[i]:
            if i <= j:
                return False
    return _nonback_girth(adj, n) >= g


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj


# ---------------- Moore bound feasibility --------------------------------------
def _moore(k, g):
    R = (g - 1) // 2
    return 1 + k * sum((k - 1) ** i for i in range(R))


def _min_n_feasible(k, g):
    """Smallest n for which a k-regular girth-g graph could exist (parity +
    Moore bound)."""
    moore = _moore(k, g)
    n = moore
    while (n * k) % 2 != 0 or n < k + 1:
        n += 1
    return n


# ---------------- construction 1: LUW (Lazebnik-Ustimenko-Woldar) ------------
def _luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# ---------------- construction 2: Greedy random regular graph ------------------
def _random_regular(n, k, rng):
    """Generate a random simple k-regular graph on n vertices using a greedy
    configuration model with re-pairing on collisions. Returns adj list or None."""
    if n * k % 2 != 0:
        return None
    if n < k + 1:
        return None
    for _ in range(50):
        adj = [set() for _ in range(n)]
        stubs = [i for i in range(n) for _ in range(k)]
        rng.shuffle(stubs)
        ok = True
        # greedy pairing: repeatedly take first stub, pair with a compatible one
        # Use a multiset of remaining stubs per vertex.
        remaining = {i: k for i in range(n)}
        # build adjacency incrementally
        # We'll do: pick a vertex with remaining>0, connect to a vertex with
        # remaining>0 that is not itself and not already adjacent.
        # To keep it fast, maintain a list of vertices with remaining>0.
        active = list(range(n))
        rng.shuffle(active)
        # Process: while there is an active vertex, connect it to another.
        # Simpler: iterate over stubs but resolve collisions.
        # Use the standard "swap until simple" approach.
        # Build a stub list, pair, then fix multi-edges/loops by swapping.
        stubs = [i for i in range(n) for _ in range(k)]
        rng.shuffle(stubs)
        edges = []
        for i in range(0, len(stubs), 2):
            edges.append((stubs[i], stubs[i + 1]))
        # Now fix: remove loops and multi-edges via swaps.
        for _ in range(200):
            # check for loops
            loop = None
            multi = None
            seen = set()
            for (a, b) in edges:
                if a == b:
                    loop = (a, b)
                    break
                key = (min(a, b), max(a, b))
                if key in seen:
                    multi = (a, b)
                    break
                seen.add(key)
            if loop is None and multi is None:
                ok = True
                break
            # fix by swapping endpoints
            if loop is not None:
                idx = edges.index(loop)
                a, b = loop
                # find another edge (c, d) with c != a, d != a, c != d
                for j in range(len(edges)):
                    if j == idx:
                        continue
                    c, d = edges[j]
                    if c == a or d == a or c == d:
                        continue
                    edges[idx] = (a, c)
                    edges[j] = (d, b) if False else (d, a) if False else (d, b)
                    # careful: we want to re-pair. Swap: (a,b)->(a,c), (c,d)->(d,b)? 
                    # but b==a. So (a,a) and (c,d) -> (a,c) and (d,a). 
                    edges[j] = (d, a)
                    break
            else:
                idx = edges.index(multi)
                a, b = multi
                for j in range(len(edges)):
                    if j == idx:
                        continue
                    c, d = edges[j]
                    if c == a or c == b or d == a or d == b:
                        continue
                    key = (min(c, d), max(c, d))
                    if key in seen:
                        continue
                    # swap b with d: (a,b)->(a,d), (c,d)->(c,b)
                    edges[idx] = (a, d)
                    edges[j] = (c, b)
                    break
        if not ok:
            continue
        # build adj
        adj = [set() for _ in range(n)]
        for (a, b) in edges:
            if a == b or b in adj[a]:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if not ok:
            continue
        if all(len(adj[i]) == k for i in range(n)):
            return [sorted(adj[i]) for i in range(n)]
    return None


# ---------------- construction 3: Random search + targeted local search --------
def _find_short_cycle(adj, n):
    """Return a list of vertices forming the shortest cycle, or None."""
    # BFS from each vertex to find shortest cycle (for moderate n)
    best = None
    for s in range(n):
        # BFS tree from s
        parent = {s: None}
        dist = {s: 0}
        q = deque([s])
        found = None
        while q and found is None:
            u = q.popleft()
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    dist[v] = dist[u] + 1
                    q.append(v)
                elif v != parent[u] and parent.get(v) != u:
                    # back edge u-v, cycle length = dist[u] + dist[v] + 1
                    L = dist[u] + dist[v] + 1
                    if best is None or L < len(best):
                        # reconstruct cycle
                        cyc = [u]
                        x = u
                        while x != v:
                            x = parent[x]
                            cyc.append(x)
                        # now go from v back to s via parent, then to u
                        # Actually cycle is u -> parent... but we need the path
                        # s->...->u and s->...->v. The cycle = path(u) + path(v) reversed.
                        # Simpler: cycle = [u, parent[u], ...] until LCA with v's path.
                        # For correctness, just record length; we'll rebuild a
                        # generic cycle below.
                        cyc = _reconstruct_cycle(adj, n, u, v, parent, dist, s)
                        if cyc is not None:
                            found = cyc
                            best = cyc
    return best


def _reconstruct_cycle(adj, n, u, v, parent, dist, s):
    # path from s to u and s to v
    path_u = []
    x = u
    while x is not None:
        path_u.append(x)
        x = parent[x]
    path_u.reverse()  # s -> ... -> u
    path_v = []
    x = v
    while x is not None:
        path_v.append(x)
        x = parent[x]
    path_v.reverse()  # s -> ... -> v
    # find LCA
    set_u = set(path_u)
    lca = None
    for node in path_v:
        if node in set_u:
            lca = node
            break
    if lca is None:
        return None
    # cycle = path_u from lca to u, then edge u-v, then path_v from v back to lca
    i_u = path_u.index(lca)
    i_v = path_v.index(lca)
    seg_u = path_u[i_u:]
    seg_v = path_v[i_v:i_v + (len(path_v) - i_v)][::-1]  # v -> ... -> lca
    cycle = seg_u + seg_v[:-1]
    return cycle


def _local_search(adj, n, k, g, rng, max_steps=2000):
    """Targeted local search: repeatedly find shortest cycle and break it via
    2-opt swaps that preserve regularity."""
    adj = [set(a) for a in adj]
    for step in range(max_steps):
        if _elapsed() > _DEADLINE - 8.0:
            return None
        girth = _nonback_girth([sorted(a) for a in adj], n)
        if girth >= g:
            return [sorted(a) for a in adj]
        # find a short cycle to break
        cyc = _find_short_cycle([sorted(a) for a in adj], n)
        if cyc is None or len(cyc) >= g:
            return [sorted(a) for a in adj]
        # pick an edge on the cycle and try to swap it out
        # edge (u, v) in cycle; find (w, x) not incident to u,v such that
        # (u,w) and (v,x) are non-edges, then swap.
        m = len(cyc)
        for _ in range(30):
            eidx = rng.randrange(m)
            u = cyc[eidx]
            v = cyc[(eidx + 1) % m]
            if v not in adj[u]:
                continue
            # find w, x
            for _ in range(40):
                w = rng.randrange(n)
                if w in (u, v):
                    continue
                if w in adj[u] or w in adj[v]:
                    continue
                if not adj[w]:
                    continue
                x = rng.choice(list(adj[w]))
                if x in (u, v, w):
                    continue
                if x in adj[v]:
                    continue
                # perform swap (u,v)+(w,x) -> (u,w)+(v,x)
                adj[u].remove(v)
                adj[v].remove(u)
                adj[w].remove(x)
                adj[x].remove(w)
                adj[u].add(w)
                adj[w].add(u)
                adj[v].add(x)
                adj[x].add(v)
                break
    girth = _nonback_girth([sorted(a) for a in adj], n)
    if girth >= g:
        return [sorted(a) for a in adj]
    return None


def _random_search(k, g, seed=0):
    rng = random.Random(seed)
    n0 = _min_n_feasible(k, g)
    for n in range(n0, n0 + 60):
        if (n * k) % 2 != 0:
            continue
        if n < k + 1:
            continue
        for attempt in range(40):
            if _elapsed() > _DEADLINE - 8.0:
                return None
            adj = _random_regular(n, k, rng)
            if adj is None:
                continue
            if _nonback_girth(adj, n) >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
            # local search
            res = _local_search(adj, n, k, g, rng, max_steps=600)
            if res is not None:
                edges = [(i, j) for i in range(n) for j in res[i] if i < j]
                return n, edges
    return None


# ---------------- construction 4: Non-abelian group Cayley graph ---------------
def _cayley_nonabelian(k, g):
    """Try Cayley graph over non-abelian group with k involutions as generators.
    For k=5, try groups of order 120 (A5), 168 (PSL(2,7)), 660 (PGL(2,11)), etc.
    Return (n, edges) if girth >= g, else None."""
    if k != 5:
        return None
    
    # Try A5 (order 60) - but we need 5 involutions, A5 has 15 involutions
    # A5 Cayley graph with 5 involutions gives 60 vertices
    # But girth of A5 Cayley graphs is typically small
    
    # Try S5 (order 120) - has many involutions
    # Actually, let's try a systematic approach:
    # Use a group of order n where n is even, with 5 involutions as generators
    
    # For now, try a few known small non-abelian groups
    # S4 (order 24): too small for girth 9
    # A5 (order 60): try it
    # S5 (order 120): try it
    # A6 (order 360): try it
    
    # Let's try A5 first with various sets of 5 involutions
    # A5 elements in cycle notation:
    # Identity: ()
    # 15 involutions: (ab)(cd) type
    # 20 elements of order 3: (abc)
    # 24 elements of order 5: (abcde)
    
    # This is getting complex. Let me try a different approach:
    # Use a voltage graph construction over a 5-regular base graph
    
    # Base graph: Petersen graph (10 vertices, 3-regular) won't work for 5-regular
    # Base graph: 5-regular graph on 10 vertices (complete graph K5 blown up?)
    
    # Actually, let's try a simpler approach: random 5-regular graphs with
    # improved local search that uses a better cycle-breaking heuristic
    
    return None


# ---------------- construction 5: Improved random search with better local search
def _improved_random_search(k, g, seed=0, max_n_offset=80):
    """Random regular graph generation with improved local search.
    Uses a smarter cycle-breaking strategy: when finding a short cycle,
    try multiple swap candidates and pick the one that maximizes girth."""
    rng = random.Random(seed)
    n0 = _min_n_feasible(k, g)
    
    for n in range(n0, n0 + max_n_offset):
        if (n * k) % 2 != 0:
            continue
        if n < k + 1:
            continue
        
        for attempt in range(60):
            if _elapsed() > _DEADLINE - 10.0:
                return None
            
            adj = _random_regular(n, k, rng)
            if adj is None:
                continue
            
            # Quick girth check
            girth = _nonback_girth(adj, n)
            if girth >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
            
            # Improved local search
            adj2 = [set(a) for a in adj]
            improved = _local_search_improved(adj2, n, k, g, rng, max_steps=1500)
            if improved is not None:
                edges = [(i, j) for i in range(n) for j in improved[i] if i < j]
                return n, edges
    
    return None


def _local_search_improved(adj, n, k, g, rng, max_steps=2000):
    """Improved local search: find shortest cycle, try multiple edge swaps,
    and pick the one that gives the best girth improvement."""
    adj = [set(a) for a in adj]
    
    for step in range(max_steps):
        if _elapsed() > _DEADLINE - 10.0:
            return None
        
        girth = _nonback_girth([sorted(a) for a in adj], n)
        if girth >= g:
            return [sorted(a) for a in adj]
        
        # Find a short cycle
        cyc = _find_short_cycle([sorted(a) for a in adj], n)
        if cyc is None or len(cyc) >= g:
            return [sorted(a) for a in adj]
        
        m = len(cyc)
        best_swap = None
        best_girth = girth - 1  # must improve
        
        # Try multiple random swaps
        for _ in range(50):
            eidx = rng.randrange(m)
            u = cyc[eidx]
            v = cyc[(eidx + 1) % m]
            if v not in adj[u]:
                continue
            
            # Find compatible w, x
            candidates = []
            for _ in range(20):
                w = rng.randrange(n)
                if w in (u, v) or w in adj[u] or w in adj[v]:
                    continue
                if not adj[w]:
                    continue
                x = rng.choice(list(adj[w]))
                if x in (u, v, w) or x in adj[v]:
                    continue
                candidates.append((w, x))
            
            for (w, x) in candidates[:5]:  # try up to 5 candidates
                # Perform swap on a copy
                adj_copy = [set(a) for a in adj]
                adj_copy[u].remove(v)
                adj_copy[v].remove(u)
                adj_copy[w].remove(x)
                adj_copy[x].remove(w)
                adj_copy[u].add(w)
                adj_copy[w].add(u)
                adj_copy[v].add(x)
                adj_copy[x].add(v)
                
                girth_new = _nonback_girth([sorted(a) for a in adj_copy], n)
                if girth_new > best_girth:
                    best_girth = girth_new
                    best_swap = (u, v, w, x)
                
                if girth_new >= g:
                    # Found a valid graph
                    return [sorted(a) for a in adj_copy]
        
        if best_swap is not None:
            u, v, w, x = best_swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            adj[u].add(w)
            adj[w].add(u)
            adj[v].add(x)
            adj[x].add(v)
    
    girth = _nonback_girth([sorted(a) for a in adj], n)
    if girth >= g:
        return [sorted(a) for a in adj]
    return None


# ---------------- construction 6: Circulant search (small n) -------------------
def _circulant_search(k, g):
    n0 = _min_n_feasible(k, g)
    max_n = min(n0 + 40, 220)
    if k % 2 == 0:
        half = k // 2
        for n in range(n0, max_n + 1):
            if (n * k) % 2 != 0:
                continue
            if n < k + 1:
                continue
            for S in itertools.combinations(range(1, n // 2 + 1), half):
                S = list(S)
                if any((n - s) % n in S for s in S):
                    continue
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S:
                        adj[i].add((i + s) % n)
                        adj[i].add((i - s) % n)
                if all(len(adj[i]) == k for i in range(n)):
                    adj = [sorted(adj[i]) for i in range(n)]
                    if _nonback_girth(adj, n) >= g:
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return n, edges
    else:
        m = (k - 1) // 2
        for n in range(n0, max_n + 1, 2):
            if n < k + 1:
                continue
            for S_vals in itertools.combinations(range(1, n // 2), m):
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S_vals:
                        adj[i].add((i + s) % n)
                        adj[i].add((i - s) % n)
                    adj[i].add((i + n // 2) % n)
                if all(len(adj[i]) == k for i in range(n)):
                    adj = [sorted(adj[i]) for i in range(n)]
                    if _nonback_girth(adj, n) >= g:
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return n, edges
    return None


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    lock = threading.Lock()

    jobs = [
        ("luw", lambda: _luw(k, g)),
        ("circ", lambda: _circulant_search(k, g)),
        ("cayley", lambda: _cayley_nonabelian(k, g)),
        ("rand0", lambda: _improved_random_search(k, g, seed=0)),
        ("rand1", lambda: _improved_random_search(k, g, seed=1)),
        ("rand2", lambda: _improved_random_search(k, g, seed=2)),
        ("rand3", lambda: _improved_random_search(k, g, seed=3)),
        ("rand4", lambda: _improved_random_search(k, g, seed=4, max_n_offset=100)),
        ("rand5", lambda: _improved_random_search(k, g, seed=5, max_n_offset=100)),
    ]

    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder, lock, k, g), daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 5.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.05)

    best = holder.get("best")
    if best is None:
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best


def _worker(name, fn, holder, lock, k, g):
    try:
        res = fn()
    except Exception:
        res = None
    if res is None:
        return
    n, edges = res
    if _elapsed() >= _DEADLINE:
        return
    try:
        if n > 0 and _is_valid(_build_adj(edges, n), n, k, g):
            with lock:
                cur = holder.get("best")
                if cur is None or n < cur[0]:
                    holder["best"] = (n, edges)
    except Exception:
        pass
