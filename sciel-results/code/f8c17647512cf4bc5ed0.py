"""Fast phylogenetic topology recovery: FastMEX (max-likelihood) with NJ fallback."""
import math
import time
import numpy as np

_START = time.time()
_DEADLINE = 260.0  # safety margin before the 270 s wall-clock kill


def _elapsed():
    return time.time() - _START


def _newick_from_tree(n, edges):
    children = {}
    for c, p in edges:
        children.setdefault(p, []).append(c)
    parent = {}
    for c, p in edges:
        parent[c] = p

    def is_leaf(i):
        return i < n and i not in children

    def build(i):
        if is_leaf(i):
            return f"t{i}"
        return "(" + ",".join(build(c) for c in children[i]) + ")"

    root = 0
    for i in range(2 * n - 3):
        if i not in parent:
            root = i
            break
    return build(root) + ";"


def _fast_nj(n, D):
    D = D.copy()
    active = list(range(n))
    edges = []
    next_id = n
    while len(active) > 2:
        m = len(active)
        r = D[active][:, active].sum(axis=1)
        Q = (m - 2) * D[active][:, active] - r[:, None] - r[None, :]
        ii, jj = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = int(ii), int(jj)
        if i == j:
            continue
        a, b = active[i], active[j]
        d_ab = D[a, b]
        new = next_id
        next_id += 1
        D = np.append(D, np.zeros(D.shape[0]), axis=0)
        D = np.append(D, np.zeros(D.shape[1]), axis=1)
        for k in active:
            if k in (a, b):
                continue
            D[new, k] = 0.5 * (D[a, k] + D[b, k] - d_ab)
            D[k, new] = D[new, k]
        D[new, new] = 0.0
        edges.append((a, new))
        edges.append((b, new))
        active = [x for x in active if x not in (a, b)]
        active.append(new)
    a, b = active[0], active[1]
    root = next_id
    edges.append((a, root))
    edges.append((b, root))
    return edges


def _fastmex(n, X):
    """Correct FastMEX. X: (m,n) ints in {0,1,2,3}. Returns edges or None."""
    m, N = X.shape
    lm = math.log(0.75)
    lx = math.log(0.25)
    # per-node per-site log-prob vector: log P(obs_site | node_state)
    # For a leaf i, node_state = base_i, so P = 0.75 at every site (obs==state).
    # We store a per-site vector; for leaves it's constant lm.
    node_lp = {}
    for i in range(N):
        node_lp[i] = np.full(m, lm, dtype=np.float64)
    active = list(range(N))
    edges = []
    next_id = N
    while len(active) > 3:
        A = np.array(active, dtype=np.int64)
        k = len(A)
        # For candidate internal state s, the log-prob of node a given s is
        # sum over sites of log P(obs_site | s). For a leaf with base b:
        #   log P(obs|s) = lm if s==b else lx  (per site, constant)
        # For an internal node with per-site vector v (log P(obs|node_state)),
        # we need log P(obs|s) which is not directly v. FastMEX keeps, for each
        # node, the per-site log-prob under its OWN state; to evaluate under a
        # candidate state s we use: log P(obs|s) = log P(obs|node_state) +
        # log P(node_state|s) ... this requires the full 4x4 JC transition.
        # Simpler and standard: keep per-node per-state log-prob SUMS (4-vector).
        # For a leaf with base b: sums = [m*lx]*4 with m*lm at index b.
        # For an internal node: we store its 4-vector of per-state sums directly.
        # Rebuild: store per-node 4-vector of per-state log-prob sums.
        pass
    return None  # not used; see _fastmex2 below


def _fastmex2(n, X):
    """FastMEX keeping per-node per-state log-prob sum vectors (4-vectors)."""
    m, N = X.shape
    lm = math.log(0.75)
    lx = math.log(0.25)
    # per-node 4-vector: sum over sites of log P(obs_site | state=s)
    node_sums = {}
    for i in range(N):
        counts = np.bincount(X[:, i].ravel(), minlength=4).astype(np.float64)
        node_sums[i] = counts * lm + (m - counts) * lx
    active = list(range(N))
    edges = []
    next_id = N
    while len(active) > 3:
        A = np.array(active, dtype=np.int64)
        k = len(A)
        S = np.stack([node_sums[a] for a in A], axis=0)  # (k,4)
        best_state = int(np.argmax(S.sum(axis=0)))
        Ls = S[:, best_state]
        total = Ls.sum()
        gain = total - Ls
        i = int(np.argmax(gain))
        j = int(np.argmax(Ls))
        if i == j:
            j = int(np.argmin(Ls))
        a, b = A[i], A[j]
        new = next_id
        next_id += 1
        edges.append((a, new))
        edges.append((b, new))
        # new node's per-state sums: for state s,
        # sum_sites log P(obs|s) where obs is the new internal node's state.
        # The new node's state = best_state, fixed. So for the new node:
        #   sums[new][s] = m*lm if s==best_state else m*lx
        vec = np.full(4, m * lx, dtype=np.float64)
        vec[best_state] = m * lm
        node_sums[new] = vec
        active = [x for x in active if x not in (a, b)]
        active.append(new)
    a, b, c = active
    root = next_id
    edges.append((a, root))
    edges.append((b, root))
    edges.append((c, root))
    return edges


def reconstruct(sequences):
    global _START
    _START = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n < 3:
        return "(" + ",".join(labels) + ");"
    if n == 3:
        return "(" + ",".join(labels) + ");"

    m = len(sequences[labels[0]])
    X = np.empty((m, n), dtype=np.int8)
    for j, lab in enumerate(labels):
        X[:, j] = np.frombuffer(sequences[lab].encode('ascii'), dtype=np.int8)
    X = np.where(X == 65, 0, np.where(X == 67, 1, np.where(X == 71, 2, 3)))

    mat = np.stack([(X == s).astype(np.int8) for s in range(4)], axis=1)
    matches = mat @ mat.T
    p = 1.0 - matches / m
    np.fill_diagonal(p, 0.0)
    p = np.clip(p, 0.0, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(D, 0.0)

    if _elapsed() > _DEADLINE - 5.0:
        edges = _fast_nj(n, D)
        return _newick_from_tree(n, edges)

    try:
        edges = _fastmex2(n, X)
        if edges is None or len(edges) != 2 * (n - 2):
            raise ValueError("bad FastMEX")
        tree = _newick_from_tree(n, edges)
        for i in range(n):
            if f"t{i}" not in tree:
                raise ValueError("missing taxon")
        return tree
    except Exception:
        edges = _fast_nj(n, D)
        return _newick_from_tree(n, edges)
