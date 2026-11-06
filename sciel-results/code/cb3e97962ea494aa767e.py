"""Phylogenetic topology recovery via NJ with JC-corrected distances and ML polish."""
import math
import time
import numpy as np

_START_TIME = time.time()


def _remaining():
    return max(0.0, 180.0 * 0.85 - (time.time() - _START_TIME))


def _jc_dist_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    cm = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j in range(L):
            mat[i, j] = cm[s[j]]
    diff = (mat[:, None, :] != mat[None, :, :]).astype(np.float64)
    mismatch = diff.sum(axis=2)
    p = np.clip(mismatch / L, 0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _nj(dist, labels):
    n = len(labels)
    if n <= 3:
        return None, None
    d = dist.copy()
    nodes = list(range(n))
    trees = {i: i for i in range(n)}
    blen = {i: 0.0 for i in range(n)}
    while len(nodes) > 3:
        m = len(nodes)
        r = np.array([sum(d[a, b] for b in nodes if b != a) for a in nodes])
        best = (float('inf'), 0, 1)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                c = (m - 2) * d[a, b] - r[i] - r[j]
                if c < best[0]:
                    best = (c, i, j)
        _, i, j = best
        a, b = nodes[i], nodes[j]
        delta = 0.5 * (d[a, b] + (r[i] - r[j]) / (m - 2))
        la = max(0.0, delta)
        lb = max(0.0, d[a, b] - delta)
        new = max(nodes) + 1
        for x in nodes:
            if x not in (a, b):
                d[new, x] = d[x, new] = 0.5 * (d[a, x] + d[b, x] - d[a, b])
        nodes = [x for x in nodes if x not in (a, b)] + [new]
        trees[new] = (a, b)
        blen[a] = la
        blen[b] = lb
        blen[new] = 0.0
    return trees, blen


def _build_newick(trees, labels):
    def build(node):
        if node < len(labels):
            return labels[node]
        l, r = trees[node]
        return f"({build(l)},{build(r)})"
    root = max([k for k in trees if k >= len(labels)])
    return build(root) + ";"


def _jc_loglik(tree, labels, sequences, blen):
    n = len(labels)
    L = len(sequences[labels[0]])
    cm = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    leaf_data = {}
    for i, lab in enumerate(labels):
        s = sequences[lab]
        leaf_data[i] = np.array([cm[c] for c in s], dtype=np.int8)
    
    internal = [k for k in tree if k >= n]
    if not internal:
        return 0.0
    
    parent = {}
    def setp(node, par):
        parent[node] = par
        if node >= n:
            l, r = tree[node]
            setp(l, node)
            setp(r, node)
    root = max(internal)
    setp(root, -1)
    
    order = []
    def po(node):
        if node >= n:
            l, r = tree[node]
            po(l)
            po(r)
        order.append(node)
    po(root)
    
    nl = {}
    for i in range(n):
        data = leaf_data[i]
        vec = np.zeros((L, 4))
        vec[np.arange(L), data] = 1.0
        nl[i] = vec
    
    for node in order:
        if node < n:
            continue
        l, r = tree[node]
        t = blen.get(node, 0.0)
        ll = nl[l]
        lr = nl[r]
        prod = ll * lr
        if t < 1e-10:
            nl[node] = prod
        else:
            et = math.exp(-4.0 * t / 3.0)
            ps = 0.25 + 0.75 * et
            pd = 0.25 - 0.25 * et
            sa = prod.sum(axis=1, keepdims=True)
            nl[node] = ps * prod + pd * (sa - prod)
    
    rv = nl[root]
    return float(np.sum(np.log(rv.sum(axis=1) + 1e-300)))


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    dist = _jc_dist_matrix(sequences, labels)
    trees, blen = _nj(dist, labels)
    
    # Optimize branch lengths via coordinate descent
    internal = [k for k in trees if k >= n]
    for _ in range(8):
        if _remaining() < 0.5:
            break
        for node in internal:
            best_l = blen.get(node, 0.0)
            best_ll = _jc_loglik(trees, labels, sequences, blen)
            for d in [-0.05, -0.01, 0.01, 0.05, 0.1]:
                nl = max(0.0, best_l + d)
                blen[node] = nl
                ll = _jc_loglik(trees, labels, sequences, blen)
                if ll > best_ll:
                    best_ll = ll
                    best_l = nl
            blen[node] = best_l
    
    return _build_newick(trees, labels)
