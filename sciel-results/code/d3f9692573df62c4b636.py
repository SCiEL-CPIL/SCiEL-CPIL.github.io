"""Split-based (parsimony-scored) unrooted topology recovery for JC-simulated DNA."""
import math
import time

_START = time.time()
_DEADLINE = _START + 255.0  # safety margin under 270s wall-clock kill

_LETTERS = (0, 1, 2, 3)  # A C G T


def _time_ok():
    return time.time() < _DEADLINE


def _encode(sequences):
    labels = sorted(sequences, key=lambda n: int(n[1:]))
    n = len(labels)
    m = len(sequences[labels[0]])
    idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    S = []
    for lab in labels:
        s = sequences[lab]
        S.append([idx[c] for c in s])
    return labels, n, m, S


def _fitch_split_cost(S, A, n, m):
    """Minimum parsimony (Fitch) cost of the split A | (all \\ A)."""
    a = len(A)
    b = n - a
    if a <= 0 or b <= 0:
        return 0
    total = 0
    for j in range(m):
        top = S[0][j]
        # Fitch down-pass
        sets = [set()] * n
        sets[0] = {top}
        for i in range(1, n):
            s = S[i][j]
            sets[i] = {s}
        # post-order: process from last to first
        for i in range(n - 1, 0, -1):
            if i in A:
                child = sets[i]
                parent = sets[i - 1]
                inter = parent & child
                if inter:
                    new = inter
                else:
                    new = parent | child
                    total += 1
                sets[i - 1] = new
    return total


def _parsimony_split_scores(S, n, m, labels):
    """Score all non-trivial splits of size 2..n-2 by parsimony cost.
    Returns dict mapping frozenset(A) (with min(A)<min(B) canonical) -> cost."""
    # Precompute per-site character arrays for speed via lists
    scores = {}
    # We enumerate all subsets of size 2..n//2 (canonical: smaller side or tie-break)
    # Use bitmask enumeration for n up to ~16; for larger n use pairs + larger subsets heuristically
    if n <= 16:
        full = (1 << n) - 1
        # precompute site columns as lists of ints
        cols = [tuple(S[i][j] for i in range(n)) for j in range(m)]
        for mask in range(1, full):
            c = bin(mask).count('1')
            if c <= 1 or c >= n - 1:
                continue
            # canonical: ensure the "A" side is the lexicographically smaller (min element)
            # We represent split by the subset containing the smallest leaf? Instead:
            # canonical key = frozenset of the side with smaller min, else by size then min
            A = [i for i in range(n) if mask & (1 << i)]
            B = [i for i in range(n) if not (mask & (1 << i))]
            # canonical: choose side with smaller min element; tie -> smaller size
            if min(A) < min(B) or (min(A) == min(B) and len(A) <= len(B)):
                key = frozenset(A)
            else:
                key = frozenset(B)
            if key in scores:
                continue
            # compute fitch cost
            a = len(key)
            total = 0
            for col in cols:
                # Fitch
                top = col[0]
                sets = [set()] * n
                sets[0] = {top}
                for i in range(1, n):
                    sets[i] = {col[i]}
                for i in range(n - 1, 0, -1):
                    child = sets[i]
                    parent = sets[i - 1]
                    inter = parent & child
                    if inter:
                        new = inter
                    else:
                        new = parent | child
                        total += 1
                    sets[i - 1] = new
            scores[key] = total
        return scores
    else:
        # For larger n, only enumerate pair-splits and a sample of larger splits
        scores = {}
        cols = [tuple(S[i][j] for i in range(n)) for j in range(m)]
        # pair splits
        for i in range(n):
            for j in range(i + 1, n):
                key = frozenset([i, j])
                if key in scores:
                    continue
                total = 0
                for col in cols:
                    top = col[0]
                    sets = [set()] * n
                    sets[0] = {top}
                    for ii in range(1, n):
                        sets[ii] = {col[ii]}
                    for ii in range(n - 1, 0, -1):
                        child = sets[ii]
                        parent = sets[ii - 1]
                        inter = parent & child
                        if inter:
                            new = inter
                        else:
                            new = parent | child
                            total += 1
                        sets[ii - 1] = new
                scores[key] = total
        return scores


def _build_tree_from_splits(n, splits):
    """Build an unrooted tree from a compatible set of splits using a
    recursive display-tree construction. Returns list of (left, right, weight).
    Fallback: star tree if incompatible/empty."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1, 1)]
    if not splits:
        # star tree
        edges = []
        center = 10**6
        for i in range(n):
            edges.append((center, i, 1))
        return edges

    # Find a non-trivial split to recurse on. Prefer one that is "balanced"
    # and present in the set.
    # Represent splits as frozensets of leaf indices.
    all_leaves = frozenset(range(n))
    # Choose split with size closest to n/2
    best = None
    best_diff = 10**9
    for s in splits:
        a = len(s)
        if a <= 1 or a >= n - 1:
            continue
        diff = abs(a - n / 2.0)
        if diff < best_diff:
            best_diff = diff
            best = s
    if best is None:
        # star
        center = 10**6
        return [(center, i, 1) for i in range(n)]

    A = best
    B = all_leaves - A
    # Recurse on each side; if a side has <=2 leaves it's a star/edge.
    def build(leafset, local_splits):
        leaves = sorted(leafset)
        k = len(leaves)
        if k <= 1:
            return leaves[0] if k == 1 else None, []
        if k == 2:
            return None, [(leaves[0], leaves[1], 1)]
        # find a split that partitions this leafset non-trivially
        sub = None
        subdiff = 10**9
        for s in local_splits:
            sA = s & leafset
            sB = leafset - sA
            a = len(sA)
            if a <= 1 or a >= k - 1:
                continue
            diff = abs(a - k / 2.0)
            if diff < subdiff:
                subdiff = diff
                sub = sA
        if sub is None:
            # star on this leafset
            center = 10**6 + (leaves[0] * 31 + leaves[-1])
            edges = [(center, lv, 1) for lv in leaves]
            return center, edges
        A2 = sub
        B2 = leafset - A2
        nodeA, edgesA = build(A2, local_splits)
        nodeB, edgesB = build(B2, local_splits)
        # connect the two sides with a single edge
        if nodeA is None:
            # A2 was a 2-leaf edge; pick one endpoint as the node
            # find the edge in edgesA
            e = edgesA[0]
            nodeA = e[0]
        if nodeB is None:
            e = edgesB[0]
            nodeB = e[0]
        new_edges = edgesA + edgesB + [(nodeA, nodeB, 1)]
        return nodeA, new_edges

    root, edges = build(all_leaves, list(splits))
    return edges


def _edges_to_newick(edges, n):
    """Convert edge list (with possible internal node ids >= n) to Newick over 0..n-1."""
    # Build adjacency
    from collections import defaultdict
    adj = defaultdict(list)
    nodes = set()
    for (u, v, w) in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
        nodes.add(u)
        nodes.add(v)
    if not nodes:
        return ""
    # Find a leaf to root from
    start = None
    for i in range(n):
        if i in nodes:
            start = i
            break
    if start is None:
        start = next(iter(nodes))

    # DFS to build newick; internal nodes are those with id >= n or degree != 1
    # We'll do a recursive construction by picking an arbitrary leaf and growing.
    def is_leaf(x):
        return x < n

    # Build tree by rooting at start (a leaf). Do a DFS.
    def dfs(u, parent):
        # collect children
        children = [v for (v, w) in adj[u] if v != parent]
        if is_leaf(u) and parent is not None:
            return str(u)
        if not children:
            # leaf (shouldn't happen for start if n>1)
            return str(u)
        parts = []
        for v in children:
            sub = dfs(v, u)
            parts.append(sub)
        return "(" + ",".join(parts) + ")"

    result = dfs(start, None)
    return result + ";"


def _nj_fallback(labels, n, m, S):
    """Neighbour-joining fallback on JC-corrected distances."""
    idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # compute distance matrix
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            diff = 0
            for k in range(m):
                if S[i][k] != S[j][k]:
                    diff += 1
            p = diff / m
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i][j] = d
            D[j][i] = d
    # NJ
    active = list(range(n))
    tree = {i: str(i) for i in range(n)}
    dist = D
    counter = n
    while len(active) > 3:
        nn = len(active)
        r = {a: sum(dist[a][b] for b in active if b != a) for a in active}
        best = None
        for ii in range(len(active)):
            a = active[ii]
            for jj in range(ii + 1, len(active)):
                b = active[jj]
                crit = (nn - 2) * dist[a][b] - r[a] - r[b]
                if best is None or crit < best[0]:
                    best = (crit, a, b)
        _, a, b = best
        new = counter
        counter += 1
        tree[new] = "(" + tree[a] + "," + tree[b] + ")"
        rest = [x for x in active if x not in (a, b)]
        # add new row/col
        for x in rest:
            d = 0.5 * (dist[a][x] + dist[b][x] - dist[a][b])
            if d < 0:
                d = 0.0
            dist[new][x] = d
            dist[x][new] = d
        active = rest + [new]
    # finish with the last 3
    if len(active) == 3:
        a, b, c = active
        tree[new2] = None
        # make a trifurcation
        joined = "(" + tree[a] + "," + tree[b] + "," + tree[c] + ")"
        return joined + ";"
    return ""


def reconstruct(sequences):
    labels, n, m, S = _encode(sequences)

    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        if n == 2:
            return "(" + labels[0] + "," + labels[1] + ");"
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"

    # Try split-based approach if we have time and n is manageable
    if n <= 16 and _time_ok():
        scores = _parsimony_split_scores(S, n, m, labels)
        if scores:
            # Determine a threshold: keep splits with cost below a quantile
            costs = sorted(scores.values())
            # keep splits with cost <= median (heuristic)
            median = costs[len(costs) // 2]
            good = [s for s, c in scores.items() if c <= median]
            if good:
                # Filter for compatibility: greedily keep compatible splits
                # A set of splits is compatible if no two cross.
                # Two splits A|B and C|D cross if each side of one intersects both sides of the other.
                all_leaves = frozenset(range(n))
                def crosses(s1, s2):
                    s1b = all_leaves - s1
                    s2b = all_leaves - s2
                    return (s1 & s2) and (s1 & s2b) and (s1b & s2) and (s1b & s2b)
                # sort by cost ascending, greedily add
                good_sorted = sorted(good, key=lambda s: scores[s])
                chosen = []
                for s in good_sorted:
                    ok = True
                    for t in chosen:
                        if crosses(s, t):
                            ok = False
                            break
                    if ok:
                        chosen.append(s)
                edges = _build_tree_from_splits(n, chosen)
                newick = _edges_to_newick(edges, n)
                # Validate: every taxon exactly once
                if newick and all(lab in newick for lab in labels):
                    return newick

    # Fallback: NJ
    try:
        res = _nj_fallback(labels, n, m, S)
        if res:
            return res
    except Exception:
        pass
    # ultimate fallback: star
    return "(" + ",".join(labels) + ");"
