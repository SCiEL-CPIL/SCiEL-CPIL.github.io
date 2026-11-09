"""Phylogenetic topology recovery: fast NJ + RF topological refinement."""
import time
import numpy as np

_DEADLINE = 265.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _jc_matrix(seqs):
    n = seqs.shape[0]
    L = seqs.shape[1]
    neq = (seqs[:, None, :] != seqs[None, :, :]).astype(np.float64)
    p = neq.sum(axis=2) / L
    np.fill_diagonal(p, 0.0)
    np.clip(p, 0.0, 0.7499, out=p)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    d = np.where(np.isfinite(d), d, 0.0)
    return d


def _nj_join(d):
    n = d.shape[0]
    nodes = list(range(n))
    tree = {i: str(i) for i in range(n)}
    D = d.copy()
    while len(nodes) > 3:
        m = len(nodes)
        idx = np.array(nodes, dtype=int)
        sub = D[np.ix_(idx, idx)]
        r = sub.sum(axis=1)
        Q = (m - 2) * sub - r[:, None] - r[None, :]
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        a = nodes[i]
        b = nodes[j]
        new = len(nodes)
        rest = [x for x in nodes if x != a and x != b]
        if rest:
            ri = np.array(rest, dtype=int)
            D[new, ri] = (D[a, ri] + D[b, ri] - D[a, b]) * 0.5
            D[ri, new] = D[new, ri]
        joined = "(" + tree[a] + "," + tree[b] + ")"
        tree[new] = joined
        nodes = rest + [new]
    a, b, c = nodes
    return "(" + tree[a] + "," + tree[b] + "," + tree[c] + ")"


def _parse_newick(newick):
    newick = newick.strip().rstrip(';')
    nodes = {}
    adj = {}

    def register(name):
        if name not in nodes:
            nodes[name] = len(nodes)
            adj[name] = []
        return nodes[name]

    def add_edge(u, v):
        adj[u].append(v)
        adj[v].append(u)

    pos = 0
    L = len(newick)
    stack = []
    while pos < L:
        ch = newick[pos]
        if ch == '(':
            parent = stack[-1][0] if stack else None
            child_name = "_n%d" % len(nodes)
            register(child_name)
            if parent is not None:
                add_edge(parent, child_name)
            stack.append(child_name)
            pos += 1
        elif ch == ',':
            pos += 1
        elif ch == ')':
            child = stack.pop()
            if stack:
                parent = stack[-1]
                add_edge(parent, child)
            pos += 1
        else:
            start = pos
            while pos < L and newick[pos] not in '(),;':
                pos += 1
            label = newick[start:pos]
            register(label)
            if stack:
                parent = stack[-1]
                add_edge(parent, label)
    return nodes, adj


def _refine(newick, leaf_set, iters=3000):
    nodes, adj = _parse_newick(newick)
    name_to_id = nodes
    adjl = [adj[name] for name in nodes]
    n = len(nodes)

    def splits_of(adjl):
        sp = set()
        for u in range(n):
            for v in adjl[u]:
                if u < v:
                    seen = set()
                    stack = [v]
                    while stack:
                        x = stack.pop()
                        if x in seen:
                            continue
                        seen.add(x)
                        for y in adjl[x]:
                            if y != u and y not in seen:
                                stack.append(y)
                    leaves = frozenset(name_to_id[x] for x in seen if name_to_id[x] in leaf_set)
                    if 0 < len(leaves) < len(leaf_set):
                        sp.add(leaves)
        return sp

    ref = splits_of(adjl)
    refsize = len(ref)

    def rf(sp):
        inter = len(sp & ref)
        tot = len(sp) + refsize
        return (tot - 2 * inter) / tot if tot else 0.0

    best_rf = rf(splits_of(adjl))
    best_adj = [list(x) for x in adjl]

    for it in range(iters):
        if _elapsed() > _DEADLINE:
            break
        improved = False
        for u in range(n):
            if len(adjl[u]) != 3:
                continue
            a, b, c = adjl[u]
            # three NNI moves: swap each pair of the three neighbors
            pairs = [(a, b, c), (a, c, b), (b, c, a)]
            for x, y, z in pairs:
                if y in adjl[x] or z in adjl[x]:
                    continue
                # remove u-x, u-y; add x-y, u-z
                adjl[u].remove(x)
                adjl[u].remove(y)
                adjl[x].remove(u)
                adjl[y].remove(u)
                adjl[u].append(z)
                adjl[z].append(u)
                adjl[x].append(y)
                adjl[y].append(x)
                new_rf = rf(splits_of(adjl))
                if new_rf < best_rf - 1e-12:
                    best_rf = new_rf
                    best_adj = [list(t) for t in adjl]
                    improved = True
                else:
                    adjl[u].remove(z)
                    adjl[u].append(x)
                    adjl[u].append(y)
                    adjl[z].remove(u)
                    adjl[x].remove(y)
                    adjl[y].remove(x)
                    adjl[x].append(u)
                    adjl[y].append(u)
        if not improved:
            break

    # build newick from best_adj rooted at a leaf
    root = next(iter(leaf_set))
    root_id = name_to_id[root]
    visited = set()

    def rec(u, p):
        if name_to_id[u] in leaf_set:
            return name_to_id[u]
        kids = []
        for v in best_adj[u]:
            if v == p:
                continue
            kids.append(rec(v, u))
        return "(" + ",".join(kids) + ")"

    return rec(root_id, -1) + ";"


def reconstruct(sequences):
    global _START
    _START = time.monotonic()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    code = {c: i for i, c in enumerate("ACGT")}
    mat = np.array([[code[ch] for ch in seq] for seq in (sequences[l] for l in labels)],
                   dtype=np.int8)
    d = _jc_matrix(mat)
    newick = _nj_join(d)
    if newick is None:
        return "(" + ",".join(labels) + ");"
    leaf_set = set(labels)
    if n >= 6 and _elapsed() < _DEADLINE - 5.0:
        try:
            refined = _refine(newick, leaf_set, iters=3000)
            if refined:
                newick = refined
        except Exception:
            pass
    return newick
