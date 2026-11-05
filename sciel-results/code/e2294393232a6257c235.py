"""Phylogenetic topology recovery from Jukes-Cantor simulated sequences."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 270.0 - 12.0   # explicit wall-clock guard, safety margin before the kill
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _jc_matrix(sequences, labels):
    """Pairwise Jukes-Cantor-corrected distances, vectorized."""
    k = len(labels)
    seqs = [sequences[lb] for lb in labels]
    L = len(seqs[0])
    A = np.array([1 if c == 'A' else 0 for s in seqs for c in s], dtype=np.int8).reshape(k, L)
    C = np.array([1 if c == 'C' else 0 for s in seqs for c in s], dtype=np.int8).reshape(k, L)
    G = np.array([1 if c == 'G' else 0 for s in seqs for c in s], dtype=np.int8).reshape(k, L)
    T = np.array([1 if c == 'T' else 0 for s in seqs for c in s], dtype=np.int8).reshape(k, L)
    diff = ((A[:, None] ^ A[None, :]).astype(np.int32)
            + (C[:, None] ^ C[None, :]).astype(np.int32)
            + (G[:, None] ^ G[None, :]).astype(np.int32)
            + (T[:, None] ^ T[None, :]).astype(np.int32))
    p = diff / float(L)
    np.fill_diagonal(p, 0.0)
    p = np.clip(p, 0.0, 0.7499)
    D = -0.75 * np.log1p(-4.0 * p / 3.0)
    np.fill_diagonal(D, 0.0)
    return D


def _balanced_star(labels):
    """Balanced unrooted star tree (fallback for degenerate matrices)."""
    k = len(labels)
    if k == 1:
        return labels[0]
    if k == 2:
        return "(%s,%s);" % (labels[0], labels[1])
    if k == 3:
        return "((%s,%s),%s);" % (labels[0], labels[1], labels[2])
    n = k
    while n > 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = (n - 1) // 2
    depth = n
    nodes = []
    for i in range(k):
        nodes.append((depth, labels[i]))
    level = depth
    while level > 0:
        cur = nodes
        nxt = []
        for j in range(0, len(cur), 2):
            if j + 1 < len(cur):
                nxt.append((level - 1, "(%s,%s)" % (cur[j][1], cur[j + 1][1])))
            else:
                nxt.append((level - 1, "(%s,)" % cur[j][1]))
        nodes = nxt
        level -= 1
    return "(" + nodes[0][1] + ");"


def _nj_build(D, labels):
    """Neighbor-joining. Returns (newick, node_map) or (None, None) if degenerate."""
    k = len(D)
    if k < 3:
        return None, None
    active = list(range(k))
    subtree = {i: labels[i] for i in active}
    Dm = D.copy()
    while len(active) > 3:
        n = len(active)
        r = Dm[np.ix_(active, active)].sum(axis=1)
        Q = (n - 2) * Dm[np.ix_(active, active)] - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        ia, ib = active[idx[0]], active[idx[1]]
        dab = Dm[ia, ib]
        new = "n" + str(len(active)) + "_" + str(ia) + "_" + str(ib)
        new_sub = "(%s,%s)" % (subtree[ia], subtree[ib])
        rest = [x for x in active if x != ia and x != ib]
        for x in rest:
            val = 0.5 * (Dm[ia, x] + Dm[ib, x] - dab)
            Dm[new, x] = val
            Dm[x, new] = val
        active = rest + [new]
        subtree[new] = new_sub
    a, b, c = active
    return "(%s,(%s,%s));" % (subtree[a], subtree[b], subtree[c]), subtree


def _parse(newick):
    """Parse Newick into adjacency list. Returns (adj, taxon_nodes, n_nodes)."""
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1]
    tokens = []
    depth = 0
    buf = []
    for ch in s:
        if ch == '(':
            if buf:
                tokens.append(('taxon', ''.join(buf)))
                buf = []
            tokens.append(('(', None))
            depth += 1
        elif ch == ')':
            if buf:
                tokens.append(('taxon', ''.join(buf)))
                buf = []
            tokens.append((')', None))
            depth -= 1
        elif ch == ',':
            if buf:
                tokens.append(('taxon', ''.join(buf)))
                buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append(('taxon', ''.join(buf)))

    adj = {}
    taxon_nodes = {}
    counter = [0]

    def new_id():
        counter[0] += 1
        return counter[0]

    def build(i):
        if i >= len(tokens):
            return None
        t, val = tokens[i]
        if t == 'taxon':
            nid = new_id()
            adj.setdefault(nid, [])
            taxon_nodes[val] = nid
            return nid
        # t == '('
        i += 1
        children = []
        while i < len(tokens) and tokens[i][0] != ')':
            cid = build(i)
            if cid is None:
                break
            children.append(cid)
            if i < len(tokens) and tokens[i][0] == ',':
                i += 1
            else:
                break
        # i at ')'
        i += 1
        nid = new_id()
        adj.setdefault(nid, [])
        for c in children:
            adj[nid].append(c)
            adj.setdefault(c, []).append(nid)
        return nid

    root = build(0)
    return adj, taxon_nodes, counter[0]


def _splits_from_adj(adj, root, n_nodes):
    """All non-trivial splits (frozensets of internal-node ids) via DFS."""
    parent = {root: -1}
    order = [root]
    for v in order:
        for w in adj[v]:
            if w != parent.get(v, -1) and w not in parent:
                parent[w] = v
                order.append(w)
    splits = set()
    for v in order:
        if v == root:
            continue
        size = 0
        for w in order:
            if parent[w] == v or w == v:
                size += 1
        if 1 < size < n_nodes - 1:
            comp = set()
            stack = [v]
            while stack:
                x = stack.pop()
                comp.add(x)
                for y in adj[x]:
                    if y != parent[x] and y not in comp:
                        stack.append(y)
            splits.add(frozenset(comp))
    return splits


def _rf_distance(s1, s2):
    return len(s1.symmetric_difference(s2))


def _spr_refine(adj, taxon_nodes, D, labels, n_nodes, max_time):
    """Score-guided SPR moves to reduce RF to the NJ reference split set."""
    base_splits = _splits_from_adj(adj, 1, n_nodes)
    best_splits = base_splits
    best_adj = {k: list(v) for k, v in adj.items()}
    best_score = len(base_splits)
    n_taxa = len(labels)
    t_end = time.time() + max_time
    while time.time() < t_end and _remaining() > 8.0:
        improved = False
        # iterate over internal nodes as potential subtrees to move
        for v in list(adj.keys()):
            if time.time() >= t_end or _remaining() < 8.0:
                break
            nbrs = adj[v]
            if len(nbrs) < 2:
                continue
            # subtree = component of v when removing one incident edge
            for e in nbrs:
                # remove edge v-e, take component containing v
                edge = (v, e) if v < e else (e, v)
                # find component of v not containing e
                comp = set()
                stack = [v]
                while stack:
                    x = stack.pop()
                    if x in comp:
                        continue
                    comp.add(x)
                    for y in adj[x]:
                        if y == e and x == v:
                            continue
                        if y not in comp:
                            stack.append(y)
                if len(comp) < 2 or len(comp) >= n_nodes:
                    continue
                # candidate: move this subtree to attach to node w (not in comp, not e)
                for w in range(1, n_nodes + 1):
                    if w in comp or w == e:
                        continue
                    # build candidate adjacency
                    cand = {k: list(val) for k, val in adj.items()}
                    # detach subtree root v from e
                    if e in cand[v]:
                        cand[v].remove(e)
                    if v in cand[e]:
                        cand[e].remove(v)
                    # reattach v to w
                    cand[v].append(w)
                    cand[w].append(v)
                    cs = _splits_from_adj(cand, 1, n_nodes)
                    score = len(cs.symmetric_difference(base_splits))
                    if score < best_score:
                        best_score = score
                        best_splits = cs
                        best_adj = cand
                        adj = {k: list(val) for k, val in cand.items()}
                        improved = True
                        break
        if best_score == 0:
            break
    return _adj_to_newick(best_adj, taxon_nodes)


def _adj_to_newick(adj, taxon_nodes):
    """Convert adjacency to a Newick string (unrooted, pick an internal root)."""
    # find a root: an internal node (not in taxon_nodes)
    root = None
    for nid in adj:
        if nid not in taxon_nodes:
            root = nid
            break
    if root is None:
        root = 1
    inv = {nid: name for name, nid in taxon_nodes.items()}

    def build(node, par):
        kids = [c for c in adj[node] if c != par]
        if not kids:
            return inv.get(node, str(node))
        parts = [build(c, node) for c in kids]
        return "(" + ",".join(parts) + ")"

    return "(" + build(root, -1) + ");"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    k = len(labels)
    if k <= 1:
        return (labels[0] + ";") if labels else ""
    if k == 2:
        return "(%s,%s);" % (labels[0], labels[1])
    if k == 3:
        return "((%s,%s),%s);" % (labels[0], labels[1], labels[2])

    D = _jc_matrix(sequences, labels)
    newick, _ = _nj_build(D, labels)
    if newick is None:
        return _balanced_star(labels)

    # Try SPR refinement only if we have time to spare.
    if _remaining() > 20.0:
        try:
            adj, taxon_nodes, n_nodes = _parse(newick)
            refined = _spr_refine(adj, taxon_nodes, D, labels, n_nodes, max_time=8.0)
            if refined:
                newick = refined
        except Exception:
            pass
    return newick
# EVOLVE-BLOCK-END
