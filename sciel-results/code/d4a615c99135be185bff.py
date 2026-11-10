"""
Reconstructs phylogenetic topology using BIONJ (variance-corrected Neighbor
Joining) on Jukes-Cantor corrected distances, followed by a quartet local-search
polish. Vectorized with NumPy for speed so that the spare time budget can fund
topological refinement. An explicit deadline guard returns the best result found
before the limit.
"""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START

_DEADLINE = 250.0          # seconds wall-clock; evaluator kills at 270s
_SAFETY = 10.0             # safety margin before the hard limit


def _jc_distance_vectorized(hamming_matrix, n_sites):
    """Jukes-Cantor corrected distance matrix from a Hamming matrix."""
    p = hamming_matrix / float(n_sites)
    p_safe = np.clip(p, 0.0, 0.749)
    d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p_safe)
    np.fill_diagonal(d, 0.0)
    return d


def _hamming_matrix(seq_matrix):
    """Pairwise Hamming distance matrix from an (n, L) int8 code matrix."""
    # n x n x L boolean compare, then sum over sites
    return np.sum(seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :],
                  axis=2).astype(np.float64)


def _bionj_step(D, n):
    """
    BIONJ variance-corrected neighbor-joining criterion.
    Q[i,j] = (n-2)*d_ij - r_i - r_j + (r_i^2 + r_j^2) / ((n-2)*d_ij)
    Returns (i, j) of the minimum-Q pair (i != j).
    """
    r = D.sum(axis=1)
    denom = (n - 2) * D
    np.fill_diagonal(denom, np.inf)          # avoid self division
    term = (r[:, None] ** 2 + r[None, :] ** 2) / denom
    Q = (n - 2) * D - r[:, None] - r[None, :] + term
    np.fill_diagonal(Q, np.inf)
    idx = np.unravel_index(np.argmin(Q), Q.shape)
    i, j = idx
    if i > j:
        i, j = j, i
    return i, j


def _bionj(D, labels):
    """
    BIONJ on a dense distance matrix. Returns a Newick string (no semicolon).
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    D = D.astype(np.float64).copy()
    names = list(labels)

    while len(names) > 3:
        m = len(names)
        i, j = _bionj_step(D, m)

        new_name = f"({names[i]},{names[j]})"
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])

        keep = [k for k in range(m) if k != i and k != j]
        new_m = m - 1
        new_D = np.zeros((new_m, new_m), dtype=np.float64)
        for a, ia in enumerate(keep):
            for b, ib in enumerate(keep):
                new_D[a, b] = D[ia, ib]
        nn = new_m - 1
        for a, ia in enumerate(keep):
            new_D[a, nn] = new_dist[ia]
            new_D[nn, a] = new_dist[ia]
        new_D[nn, nn] = 0.0

        names = [names[k] for k in keep] + [new_name]
        D = new_D

    return f"({names[0]},{names[1]},{names[2]})"


def _encode(sequences, labels):
    """Encode sequences into an (n, L) int8 matrix (A=0,C=1,G=2,T=3)."""
    n = len(labels)
    L = len(sequences[labels[0]])
    code = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, c in enumerate(s):
            if c == 'C':
                code[i, j] = 1
            elif c == 'G':
                code[i, j] = 2
            elif c == 'T':
                code[i, j] = 3
    return code


# ---------------------------------------------------------------------------
# Quartet local-search polish
# ---------------------------------------------------------------------------

def _build_unrooted(code, D, labels):
    """
    Build an unrooted tree as a list of edges (u, v). Leaf nodes 0..n-1,
    internal nodes n.. . Returns (edges, n, code, D).
    """
    n = len(labels)
    if n == 3:
        return [(0, 1), (1, 2), (2, 0)], n, code, D

    D = D.astype(np.float64).copy()
    names = list(range(n))
    edges = []
    next_id = n
    while len(names) > 3:
        m = len(names)
        i, j = _bionj_step(D, m)
        nid = next_id
        next_id += 1
        edges.append((names[i], nid))
        edges.append((names[j], nid))
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])
        keep = [k for k in range(m) if k != i and k != j]
        new_m = m - 1
        new_D = np.zeros((new_m, new_m), dtype=np.float64)
        for a, ia in enumerate(keep):
            for b, ib in enumerate(keep):
                new_D[a, b] = D[ia, ib]
        nn = new_m - 1
        for a, ia in enumerate(keep):
            new_D[a, nn] = new_dist[ia]
            new_D[nn, a] = new_dist[ia]
        names = keep + [nid]
        D = new_D
    a, b, c = names
    edges.append((a, b))
    edges.append((b, c))
    edges.append((c, a))
    return edges, n, code, D


def _build_adjacency(edges, n_nodes):
    adj = {v: [] for v in range(n_nodes)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _edge_set(edges):
    return frozenset(frozenset(e) for e in edges)


def _is_leaf(v, n):
    return v < n


def _quartet_split(edges, a, b, c, d):
    """
    Return the pair of leaves that are adjacent in the induced quartet,
    i.e. the grouping ((a,b),(c,d)) -> (ab, cd). Returns a frozenset of size 2
    (the two leaves on one side of the quartet split).
    """
    # For each leaf x in {a,b,c,d}, find its neighbor on the path toward the
    # rest. The two leaves whose paths first meet define a side.
    # Simpler: for each of the 3 possible splits, check which is consistent.
    # We determine the split by removing each candidate internal edge and seeing
    # which partition of {a,b,c,d} it induces.
    for u, v in edges:
        # temporarily remove edge (u,v); find component containing a
        adj = _build_adjacency([e for e in edges if not (frozenset(e) == frozenset((u, v)))],
                               max(edges))
        # BFS from a
        seen = {a}
        stack = [a]
        while stack:
            x = stack.pop()
            for y in adj.get(x, []):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        side = seen & {a, b, c, d}
        if len(side) == 2:
            return frozenset(side)
    return frozenset({a, b})


def _quartet_score(code, D, a, b, c, d, split_ab):
    """
    Score of a particular quartet grouping (lower is better).
    split_ab is a frozenset of the two leaves on one side.
    We use the least-squares distance implied by the quartet topology:
    for ((x,y),(z,w)) the implied distances are:
      d_xy = l1 + l2, d_zw = l3 + l4, d_xz = l1+l3, d_xw = l1+l4,
      d_yz = l2+l3, d_yw = l2+l4.
    We solve for branch lengths by least squares and return the residual.
    """
    if split_ab == frozenset({a, b}) or split_ab == frozenset({c, d}):
        x, y, z, w = a, b, c, d
    elif split_ab == frozenset({a, c}) or split_ab == frozenset({b, d}):
        x, y, z, w = a, c, b, d
    else:
        x, y, z, w = a, d, b, c

    # equations:  l1+l2 = d_xy ; l3+l4 = d_zw ; l1+l3 = d_xz ; l1+l4 = d_xw
    #            l2+l3 = d_yz ; l2+l4 = d_yw
    dxy = D[x, y]; dzw = D[z, w]; dxz = D[x, z]; dxw = D[x, w]
    dyz = D[y, z]; dyw = D[y, w]
    # solve: l1 = (dxy + dxz + dxw - dzw - dyz - dyw)/2 ... use standard
    l1 = 0.5 * (dxy + dxz + dxw - dzw - dyz - dyw)
    l2 = dxy - l1
    l3 = dxz - l1
    l4 = dzw - l3
    # residual
    res = 0.0
    res += (l1 + l2 - dxy) ** 2
    res += (l3 + l4 - dzw) ** 2
    res += (l1 + l3 - dxz) ** 2
    res += (l1 + l4 - dxw) ** 2
    res += (l2 + l3 - dyz) ** 2
    res += (l2 + l4 - dyw) ** 2
    return res


def _quartet_polish(edges, n, code, D, deadline, safety):
    """
    Local search: repeatedly try to flip a quartet (rotate around an internal
    edge) if it lowers the quartet least-squares score. Stop on deadline.
    Returns the (possibly improved) edge list.
    """
    start = time.time()
    edges = list(edges)
    n_nodes = max(edges) + 1

    # Precompute current quartet scores for all leaf quartets to have a baseline.
    def all_leaf_quartet_sums():
        total = 0.0
        leaves = list(range(n))
        for idx in range(len(leaves)):
            a = leaves[idx]
            for jdx in range(idx + 1, len(leaves)):
                b = leaves[jdx]
                for kdx in range(jdx + 1, len(leaves)):
                    c = leaves[kdx]
                    for ldx in range(kdx + 1, len(leaves)):
                        d = leaves[ldx]
                        split = _quartet_split(edges, a, b, c, d)
                        total += _quartet_score(code, D, a, b, c, d, split)
        return total

    # Only run polish for small trees where quartet enumeration is affordable.
    if n > 14:
        return edges

    improved = True
    while improved and (time.time() - start) < (deadline - safety):
        improved = False
        # For each internal edge, try the 3 possible quartet groupings on a
        # representative quartet spanning that edge and rotate if better.
        for u, v in list(edges):
            if _is_leaf(u, n) or _is_leaf(v, n):
                continue
            # find two leaves on each side of edge (u,v)
            adj = _build_adjacency([e for e in edges if not (frozenset(e) == frozenset((u, v)))],
                                   n_nodes)
            def side_leaves(start_node):
                seen = {start_node}
                stack = [start_node]
                while stack:
                    x = stack.pop()
                    for y in adj.get(x, []):
                        if y not in seen:
                            seen.add(y)
                            stack.append(y)
                return [x for x in seen if x < n]
            side_u = side_leaves(u)
            side_v = side_leaves(v)
            if len(side_u) < 1 or len(side_v) < 1:
                continue
            # pick up to 2 leaves from each side to form a quartet
            if len(side_u) >= 1 and len(side_v) >= 1:
                a = side_u[0]
                c = side_v[0]
                # need b (another from side_u if possible) and d (another from side_v)
                b = side_u[1] if len(side_u) >= 2 else a
                d = side_v[1] if len(side_v) >= 2 else c
                if len({a, b, c, d}) < 4:
                    continue
                cur_split = _quartet_split(edges, a, b, c, d)
                cur_score = _quartet_score(code, D, a, b, c, d, cur_split)
                # The alternative split is the "other" pairing
                # all three pairings:
                pairings = [frozenset({a, b}), frozenset({a, c}), frozenset({a, d})]
                # current pairing corresponds to one of these (or its complement)
                best_alt = None
                best_alt_score = cur_score - 1e-9
                for p in pairings:
                    if p == cur_split or p == frozenset({a, b, c, d} - set(cur_split)):
                        continue
                    s = _quartet_score(code, D, a, b, c, d, p)
                    if s < best_alt_score:
                        best_alt_score = s
                        best_alt = p
                if best_alt is not None:
                    # Perform a 1-1 rotation around edge (u,v) to realize best_alt.
                    new_edges = _rotate(edges, u, v, a, b, c, d, best_alt, n, n_nodes)
                    if new_edges is not None:
                        edges = new_edges
                        improved = True
                        if (time.time() - start) >= (deadline - safety):
                            return edges
    return edges


def _rotate(edges, u, v, a, b, c, d, target_split, n, n_nodes):
    """
    Attempt a 1-1 rotation (subtree swap) around internal edge (u,v) so that the
    quartet (a,b,c,d) realizes target_split. Returns new edge list or None.
    This is a best-effort topological edit; if it fails, return None.
    """
    # Identify the two subtrees attached to edge (u,v).
    # We'll try a simple approach: find the neighbor of u other than v and the
    # neighbor of v other than u, then swap which side 'u' and 'v' connect to.
    edge_set = {frozenset(e): e for e in edges}
    # neighbors
    def neighbors(x):
        nb = []
        for e in edges:
            if e[0] == x:
                nb.append(e[1])
            elif e[1] == x:
                nb.append(e[0])
        return nb
    nu = neighbors(u)
    nv = neighbors(v)
    # u should connect to v and one other (internal degree 3)
    if v not in nu or len(nu) != 3:
        return None
    if u not in nv or len(nv) != 3:
        return None
    other_u = [x for x in nu if x != v][0]
    other_v = [x for x in nv if x != u][0]
    # Swap: connect u-other_v and v-other_u instead of u-other_u and v-other_v.
    new_edges = []
    for e in edges:
        fe = frozenset(e)
        if fe == frozenset((u, other_u)):
            continue
        if fe == frozenset((v, other_v)):
            continue
        new_edges.append(e)
    new_edges.append((u, other_v))
    new_edges.append((v, other_u))
    return new_edges


def _edges_to_newick(edges, n, labels):
    """Convert an unrooted edge list to a Newick string over labels."""
    n_nodes = max(edges) + 1
    adj = {v: [] for v in range(n_nodes)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Pick an arbitrary leaf as root side; do a DFS producing a rooted Newick.
    root = 0
    parent = {root: None}
    stack = [root]
    order = []
    while stack:
        x = stack.pop()
        order.append(x)
        for y in adj[x]:
            if y != parent[x]:
                parent[y] = x
                stack.append(y)

    def build(x):
        if x < n:
            return labels[x]
        children = [y for y in adj[x] if y != parent[x]]
        if len(children) == 1:
            return f"({build(children[0])})"
        return f"({','.join(build(c) for c in children)})"

    return build(root) + ";"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    start_time = time.time()
    deadline = start_time + _DEADLINE

    # 1. Encode sequences and compute JC-corrected distances (vectorized).
    code = _encode(sequences, labels)
    hamming = _hamming_matrix(code)
    D = _jc_distance_vectorized(hamming, len(sequences[labels[0]]))

    # 2. BIONJ reconstruction.
    if time.time() > deadline - _SAFETY:
        return _edges_to_newick([(0, 1), (1, 2), (2, 0)] + [(i, 0) for i in range(3, n)], n, labels)
    edges, _, _, _ = _build_unrooted(code, D, labels)
    if not edges:
        return _bionj(D, labels) + ";"

    # 3. Quartet local-search polish, only if spare budget permits.
    if time.time() < deadline - _SAFETY - 1.0:
        edges = _quartet_polish(edges, n, code, D, deadline, _SAFETY)

    # 4. Emit Newick.
    try:
        return _edges_to_newick(edges, n, labels)
    except Exception:
        return _bionj(D, labels) + ";"

# EVOLVE-BLOCK-END
