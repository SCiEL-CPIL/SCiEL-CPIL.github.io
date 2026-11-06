"""Phylogenetic topology recovery: vectorized NJ + small-n quartet polish."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = None


def _elapsed():
    return 0.0 if _DEADLINE is None else (time.time() - _DEADLINE[0])


def _deadline_exceeded(margin=5.0):
    if _DEADLINE is None:
        return False
    return (time.time() - _DEADLINE[0]) > (_DEADLINE[1] - margin)


def _jc_from_p(p):
    """Jukes-Cantor corrected distance from a proportion-difference (scalar)."""
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Pairwise JC-corrected distances via vectorized comparison (chunked)."""
    n = len(labels)
    L = len(sequences[labels[0]])
    # Encode each sequence as int8 array
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        row = mat[i]
        for j, c in enumerate(s):
            row[j] = enc[c]

    dist = np.zeros((n, n), dtype=np.float64)
    chunk = 16
    for i in range(0, n, chunk):
        i_end = min(i + chunk, n)
        # (chunk, n, L) boolean
        diff = (mat[i:i_end, None, :] != mat[None, :, :])
        counts = diff.sum(axis=2)  # (chunk, n)
        p = counts.astype(np.float64) / L
        p_safe = np.clip(p, 0.0, 0.7499)
        jc = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
        jc[p >= 0.7499] = 3.0
        dist[i:i_end, :] = jc
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_newick(dist, labels):
    """Neighbor Joining -> Newick (unrooted, branch lengths omitted)."""
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"

    nodes = list(range(n))
    tree = {i: labels[i] for i in range(n)}
    D = dist.copy()
    next_id = n

    while len(nodes) > 3:
        m = len(nodes)
        # Row sums restricted to current nodes
        row_sum = np.zeros(m, dtype=np.float64)
        for idx, i in enumerate(nodes):
            row_sum[idx] = D[i, nodes].sum()  # includes self (0)

        best_crit = float('inf')
        best_a = best_b = None
        for ii in range(m):
            i = nodes[ii]
            for jj in range(ii + 1, m):
                j = nodes[jj]
                crit = (m - 2) * D[i, j] - row_sum[ii] - row_sum[jj]
                if crit < best_crit:
                    best_crit = crit
                    best_a, best_b = i, j

        a, b = best_a, best_b
        if a is None or b is None:
            return "(" + ",".join(tree[x] for x in nodes) + ")"

        new_id = next_id
        next_id += 1
        tree[new_id] = f"({tree[a]},{tree[b]})"

        for x in nodes:
            if x == a or x == b:
                continue
            D[new_id, x] = D[x, new_id] = 0.5 * (D[a, x] + D[b, x] - D[a, b])

        nodes = [x for x in nodes if x != a and x != b]
        nodes.append(new_id)

    a, b, c = nodes
    return f"({tree[a]},{tree[b]},{tree[c]})"


def _quartet_parity_score(dist, labels):
    """
    For small n: enumerate all unrooted topologies is heavy; instead we use a
    quartet-consistency heuristic to *polish* the NJ tree. We build the NJ tree,
    then try local topological moves (NNI-like) and keep the one that best
    matches the observed quartet majority (using distances as a proxy).

    Returns a Newick string.
    """
    # We implement a lightweight quartet-based local search over the NJ tree
    # by working on its split set and applying split-exchange moves.
    # For n<=7 the space is small enough that we can enumerate candidate
    # topologies via a recursive split construction and score each by the
    # number of quartets it agrees with the distance-matrix quartet majority.

    n = len(labels)
    if n < 4:
        return _nj_newick(dist, labels)

    # Precompute the "preferred" quartet for each 4-subset using distances.
    # For 4 taxa a,b,c,d the three splits are ab|cd, ac|bd, ad|bc.
    # Under JC, the split whose two within-pair distances sum smallest is
    # preferred (the two closest pairs).
    def preferred_quartet_split(i, j, k, l):
        # candidates: (i,j)|(k,l) -> d[i,j]+d[k,l]; (i,k)|(j,l); (i,l)|(j,k)
        s1 = dist[i, j] + dist[k, l]
        s2 = dist[i, k] + dist[j, l]
        s3 = dist[i, l] + dist[j, k]
        m = min(s1, s2, s3)
        if m == s1:
            return frozenset((i, j))
        if m == s2:
            return frozenset((i, k))
        return frozenset((i, l))

    # All 4-subsets and their preferred splits
    from itertools import combinations
    quarts = []
    for quad in combinations(range(n), 4):
        i, j, k, l = quad
        quarts.append((quad, preferred_quartet_split(i, j, k, l)))

    # Build an initial tree from NJ and extract its non-trivial splits.
    # We parse the NJ newick into a set of splits over indices.
    # Simpler: recompute NJ splits directly.
    def _nj_splits(dist, labels):
        n = len(labels)
        D = dist.copy()
        nodes = list(range(n))
        splits = []
        next_id = n
        cluster = {i: frozenset((i,)) for i in range(n)}
        while len(nodes) > 3:
            m = len(nodes)
            row_sum = {i: D[i, nodes].sum() for i in nodes}
            best_crit = float('inf')
            best_a = best_b = None
            for ii, i in enumerate(nodes):
                for jj in range(ii + 1, m):
                    j = nodes[jj]
                    crit = (m - 2) * D[i, j] - row_sum[i] - row_sum[j]
                    if crit < best_crit:
                        best_crit = crit
                        best_a, best_b = i, j
            a, b = best_a, best_b
            new_id = next_id
            next_id += 1
            cluster[new_id] = cluster[a] | cluster[b]
            sp = cluster[a]
            if 1 < len(sp) < n - 1:
                splits.append(sp)
            for x in nodes:
                if x != a and x != b:
                    D[new_id, x] = D[x, new_id] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            nodes = [x for x in nodes if x != a and x != b]
            nodes.append(new_id)
        return splits

    def _score(splits):
        # Count quartets where the tree's restriction agrees with preferred.
        sp_set = set(splits)
        agree = 0
        for quad, pref in quarts:
            i, j, k, l = quad
            # Determine the tree's split for this quartet by checking which
            # of the three candidate 2-subsets is present in sp_set (or its
            # complement within the quartet).
            for cand in (frozenset((i, j)), frozenset((i, k)), frozenset((i, l))):
                # a split restricts to cand if cand == S ∩ quad or cand == complement
                if cand in sp_set:
                    agree += 1
                    break
                # also check if cand is the complement of some split within quad
                # (i.e., the other two)
                comp = frozenset(quad) - cand
                if comp in sp_set:
                    agree += 1
                    break
        return agree

    # Local search: start from NJ splits, try adding/removing/swapping splits
    # to increase quartet agreement. Keep it bounded.
    cur_splits = _nj_splits(dist, labels)
    cur_score = _score(cur_splits)
    best_splits = list(cur_splits)
    best_score = cur_score

    # Candidate split pool: all 2-subsets (and complements) for n<=7
    all_cands = []
    for r in range(2, n // 2 + 1):
        for combo in combinations(range(n), r):
            fs = frozenset(combo)
            all_cands.append(fs)

    improved = True
    guard = 0
    while improved and guard < 60 and not _deadline_exceeded(margin=8.0):
        improved = False
        guard += 1
        for cand in all_cands:
            # try replacing one existing split with cand
            for idx in range(len(cur_splits)):
                new_splits = list(cur_splits)
                new_splits[idx] = cand
                sc = _score(new_splits)
                if sc > best_score and not _deadline_exceeded(margin=8.0):
                    best_score = sc
                    best_splits = new_splits
                    cur_splits = new_splits
                    improved = True
                    break
            if _deadline_exceeded(margin=8.0):
                break

    # Convert best splits to a Newick via recursive construction.
    return _splits_to_newick(best_splits, labels)


def _splits_to_newick(splits, labels):
    """Reconstruct a Newick tree from a set of non-trivial splits (index sets)."""
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"

    sp_set = set(splits)
    # Canonicalize: also store complements so we can test membership both ways.
    def in_split(S):
        S = frozenset(S)
        if S in sp_set:
            return True
        return (frozenset(range(n)) - S) in sp_set

    # Recursive: find a split that separates the current node set into two
    # non-empty parts; recurse on each.
    def build(node_set):
        node_set = frozenset(node_set)
        if len(node_set) == 1:
            i = next(iter(node_set))
            return labels[i]
        if len(node_set) == 2:
            a, b = sorted(node_set)
            return f"({labels[a]},{labels[b]})"
        # find a split that cuts node_set into two parts both within node_set
        found = False
        for sp in sp_set:
            left = sp & node_set
            right = node_set - sp
            if 1 <= len(left) < len(node_set) and 1 <= len(right) < len(node_set):
                l = build(left)
                r = build(right)
                return f"({l},{r})"
            found = True
        # No valid split: star
        return "(" + ",".join(labels[i] for i in sorted(node_set)) + ")"

    return build(frozenset(range(n)))


def reconstruct(sequences):
    global _DEADLINE
    _DEADLINE = [time.time(), 250.0]  # safety margin before 270s kill

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 1:
        return "(" + ",".join(labels) + ");"

    dist = _compute_distance_matrix(sequences, labels)

    # Small-n: attempt quartet polish (bounded, guarded). Fall back to NJ on
    # any timeout or failure.
    if n <= 7:
        try:
            if not _deadline_exceeded(margin=20.0):
                result = _quartet_parity_score(dist, labels)
                if result:
                    return result + ";"
        except Exception:
            pass

    result = _nj_newick(dist, labels)
    return result + ";"
# EVOLVE-BLOCK-END
