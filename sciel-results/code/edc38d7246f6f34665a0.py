import time
import random
from itertools import combinations

_N = 16
_TIME_LIMIT = 115.0


def build_adj(edges, n=_N):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj


def is_triangle_free(adj, n=_N):
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            common = adj[v] & adj[u]
            if common & ~(1 << u) & ~(1 << v):
                return False
    return True


def is_not_3_colorable(adj, n=_N):
    """Exact backtracking 3-colouring. Returns True iff the graph is NOT 3-colourable."""
    memo = set()
    node_count = [0]

    def dfs(uncolored_mask, colors):
        if uncolored_mask == 0:
            return False
        node_count[0] += 1
        if node_count[0] > 800000:
            return False
        key = (uncolored_mask, tuple(colors))
        if key in memo:
            return True
        # MRV heuristic: pick vertex with most colored neighbors
        best_v = -1
        best_score = -1
        m = uncolored_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            score = 0
            nb = adj[v]
            while nb:
                lsb2 = nb & -nb
                w = lsb2.bit_length() - 1
                nb ^= lsb2
                if colors[w] != -1:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        v = best_v
        forbidden = 0
        nb = adj[v]
        while nb:
            lsb2 = nb & -nb
            w = lsb2.bit_length() - 1
            nb ^= lsb2
            if colors[w] != -1:
                forbidden |= 1 << colors[w]
        color_order = sorted(range(3), key=lambda c: (forbidden & (1 << c)) != 0)
        ok = True
        for c in color_order:
            if not (forbidden & (1 << c)):
                colors[v] = c
                if not dfs(uncolored_mask & ~(1 << v), colors):
                    ok = False
                    break
                colors[v] = -1
        if ok:
            memo.add(key)
        return ok

    return dfs((1 << n) - 1, [-1] * n)


def greedy_3_colorable(adj, n=_N, max_trials=8):
    for _ in range(max_trials):
        order = list(range(n))
        random.shuffle(order)
        colors = [-1] * n
        for node in order:
            used = 0
            nb = adj[node]
            while nb:
                lsb = nb & -nb
                w = lsb.bit_length() - 1
                nb ^= lsb
                if colors[w] != -1:
                    used |= 1 << colors[w]
            placed = False
            for c in range(3):
                if not (used & (1 << c)):
                    colors[node] = c
                    placed = True
                    break
            if not placed:
                break
        else:
            return True
    return False


def is_not_3_colorable_fast(adj, n=_N):
    if greedy_3_colorable(adj, n, max_trials=6):
        return False
    return is_not_3_colorable(adj, n)


def build_groetzsch_edges():
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    for u, v in [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]:
        edges.append([5 + u, v])
        edges.append([5 + v, u])
    for u in range(5):
        edges.append([10, 5 + u])
    return edges


def build_clebsch_edges():
    """Clebsch graph: complement of 4x4 rook graph. 16 vertices, 40 edges."""
    edges = []
    for a in range(16):
        i1, j1 = a // 4, a % 4
        for b in range(a + 1, 16):
            i2, j2 = b // 4, b % 4
            if i1 != i2 and j1 != j2:
                edges.append([a, b])
    return edges


def generate_blowup_distributions(num_extra, num_original):
    dists = []

    def rec(rem, idx, current):
        if idx == num_original - 1:
            current.append(rem)
            dists.append(tuple(current))
            current.pop()
            return
        for i in range(rem + 1):
            current.append(i)
            rec(rem - i, idx + 1, current)
            current.pop()

    rec(num_extra, 0, [])
    return dists


def _saturate_and_polish(adj, edges, start_time, deadline):
    """Add all triangle-free edges that keep non-3-colorability, iteratively."""
    n = _N
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Collect all candidate non-edges that are triangle-free
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        candidates.append((u, v))
        random.shuffle(candidates)
        for u, v in candidates:
            if time.time() > deadline:
                break
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            edges.append([u, v])
            if is_not_3_colorable_fast(adj, n):
                improved = True
            else:
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                edges.pop()
    return edges, len(edges)


def _clebsch_polish(start_time):
    """Start from Clebsch graph and polish."""
    n = _N
    edges = build_clebsch_edges()
    adj = build_adj(edges, n)

    if not is_triangle_free(adj, n):
        return None, 0
    if not is_not_3_colorable_fast(adj, n):
        return None, 0

    deadline = start_time + _TIME_LIMIT
    edges, count = _saturate_and_polish(adj, edges, start_time, deadline)
    if is_not_3_colorable_fast(adj, n):
        return edges, count
    return None, 0


def _groetzsch_blowup(dist, start_time):
    """Blow up the Grötzsch graph according to distribution `dist`."""
    n = _N
    num_original = 11
    g_edges = build_groetzsch_edges()
    g_adj = build_adj(g_edges, num_original)

    sizes = [1 + d for d in dist]
    orig_map = [0] * n
    idx = 0
    for i in range(num_original):
        for _ in range(sizes[i]):
            orig_map[idx] = i
            idx += 1

    verts = [[] for _ in range(num_original)]
    for v in range(n):
        verts[orig_map[v]].append(v)

    adj = [0] * n
    edges = []

    for i in range(num_original):
        for j in range(i + 1, num_original):
            if g_adj[i] & (1 << j):
                for u in verts[i]:
                    for v in verts[j]:
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        edges.append([u, v])

    # Saturate
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    edges.append([u, v])

    if not is_triangle_free(adj, n):
        return None, 0

    deadline = start_time + _TIME_LIMIT
    edges, count = _saturate_and_polish(adj, edges, start_time, deadline)
    if is_not_3_colorable_fast(adj, n):
        return edges, count
    return None, 0


def _mycielski_groetzsch_delete(start_time):
    """
    Mycielski of Grötzsch (21 vertices), then delete 5 vertices to get 16.
    Mycielski preserves triangle-freeness and increases chromatic number.
    """
    n_orig = 11
    n_myc = 2 * n_orig + 1  # 23 vertices for Mycielski of 11-vertex graph
    # Actually Mycielski of G with n vertices has 2n+1 vertices
    # Grötzsch has 11 vertices, so Mycielski has 23 vertices

    g_edges = build_groetzsch_edges()
    g_adj = build_adj(g_edges, n_orig)

    # Build Mycielski graph
    myc_adj = [0] * n_myc
    myc_edges = []

    # Original edges
    for u, v in g_edges:
        myc_adj[u] |= 1 << v
        myc_adj[v] |= 1 << u
        myc_edges.append([u, v])

    # Copy vertices: n_orig + i is copy of i
    # Copy vertex n_orig+i connects to neighbors of i in original
    for i in range(n_orig):
        nb = g_adj[i]
        while nb:
            lsb = nb & -nb
            w = lsb.bit_length() - 1
            nb ^= lsb
            myc_adj[n_orig + i] |= 1 << w
            myc_adj[w] |= 1 << (n_orig + i)
            myc_edges.append([n_orig + i, w])

    # Root vertex connects to all copy vertices
    for i in range(n_orig):
        myc_adj[n_myc - 1] |= 1 << (n_orig + i)
        myc_adj[n_orig + i] |= 1 << (n_myc - 1)
        myc_edges.append([n_myc - 1, n_orig + i])

    # Now we have a 23-vertex triangle-free graph with chromatic number 5
    # We need to delete 7 vertices to get 16
    # Try all combinations of 7 vertices to delete (C(23,7) = 245157)
    # This is too many, so use a smarter approach:
    # Try to find a 16-vertex induced subgraph that is not 3-colorable
    # and has maximum edges

    deadline = start_time + _TIME_LIMIT
    best_edges = None
    best_count = 0

    # Strategy: try random subsets of 16 vertices
    all_vertices = list(range(n_myc))
    num_trials = 5000
    for _ in range(num_trials):
        if time.time() > deadline - 10:
            break
        subset = random.sample(all_vertices, 16)
        subset_set = set(subset)

        # Build induced subgraph
        sub_adj = [0] * 16
        sub_edges = []
        for i, u in enumerate(subset):
            for j, v in enumerate(subset):
                if i < j and (myc_adj[u] & (1 << v)):
                    sub_adj[i] |= 1 << j
                    sub_adj[j] |= 1 << i
                    sub_edges.append([subset[i], subset[j]])

        # Check triangle-free and non-3-colorable
        if is_triangle_free(sub_adj, 16) and is_not_3_colorable_fast(sub_adj, 16):
            if len(sub_edges) > best_count:
                best_count = len(sub_edges)
                best_edges = list(sub_edges)

    if best_edges is not None:
        # Remap to 0..15
        # The sub_edges already use 0..15 indices
        adj = build_adj(best_edges, 16)
        deadline2 = start_time + _TIME_LIMIT
        best_edges, best_count = _saturate_and_polish(adj, best_edges, start_time, deadline2)
        if is_not_3_colorable_fast(adj, 16):
            return best_edges, best_count

    return None, 0


def _random_maximal_tf(start_time):
    """Generate random maximal triangle-free graphs and check 4-chromatic."""
    n = _N
    all_edges = list(combinations(range(n), 2))
    deadline = start_time + _TIME_LIMIT
    best_edges = None
    best_count = 0

    num_trials = 30000
    for _ in range(num_trials):
        if time.time() > deadline - 5:
            break
        random.shuffle(all_edges)
        adj = [0] * n
        edges = []
        for u, v in all_edges:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                edges.append([u, v])

        if is_not_3_colorable_fast(adj, n):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)

    if best_edges is not None:
        adj = build_adj(best_edges, n)
        best_edges, best_count = _saturate_and_polish(adj, best_edges, start_time, deadline)
        if is_not_3_colorable_fast(adj, n):
            return best_edges, best_count

    return None, 0


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT

    best_edges = None
    best_count = 0

    # Strategy 1: Clebsch polish (strong base: 40 edges, 5-chromatic)
    clebsch_edges, clebsch_count = _clebsch_polish(start_time)
    if clebsch_edges is not None and clebsch_count > best_count:
        best_count = clebsch_count
        best_edges = clebsch_edges

    # Strategy 2: Grötzsch blow-ups
    if time.time() < deadline - 10:
        num_extra = _N - 11
        dists = generate_blowup_distributions(num_extra, 11)
        random.shuffle(dists)

        # Sort by expected edge count (heuristic: more balanced = more edges)
        # Actually, just try them all within time budget
        for dist in dists:
            if time.time() > deadline - 5:
                break
            edges, cnt = _groetzsch_blowup(dist, start_time)
            if edges is not None and cnt > best_count:
                best_count = cnt
                best_edges = edges

    # Strategy 3: Mycielski of Grötzsch, delete vertices
    if time.time() < deadline - 15:
        myc_edges, myc_count = _mycielski_groetzsch_delete(start_time)
        if myc_edges is not None and myc_count > best_count:
            best_count = myc_count
            best_edges = myc_edges

    # Strategy 4: Random maximal triangle-free
    if time.time() < deadline - 10:
        rand_edges, rand_count = _random_maximal_tf(start_time)
        if rand_edges is not None and rand_count > best_count:
            best_count = rand_count
            best_edges = rand_edges

    # Final polish on the best graph found
    if best_edges is not None and time.time() < deadline - 2:
        adj = build_adj(best_edges, _N)
        cur = list(best_edges)
        best_edges, best_count = _saturate_and_polish(adj, cur, start_time, deadline)

    # Fallback: return Grötzsch graph if nothing better found
    if best_edges is None:
        best_edges = build_groetzsch_edges()

    return best_edges
