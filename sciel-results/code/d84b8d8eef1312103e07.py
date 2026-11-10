"""Phylogenetic topology recovery: BIONJ + parsimony NNI re-optimization."""
import time
import math

import numpy as np

_DEADLINE = 258.0
_start_time = None


def _now():
    return time.time()


def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    enc = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        m = np.frombuffer(s.encode(), dtype=np.uint8).astype(np.int8)
        m = np.where(m == ord('A'), 0, m)
        m = np.where(m == ord('C'), 1, m)
        m = np.where(m == ord('G'), 2, m)
        m = np.where(m == ord('T'), 3, m)
        enc[i] = m
    diff = (enc[:, None, :] != enc[None, :, :]).sum(axis=2)
    P = diff / float(L)
    np.fill_diagonal(P, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        D = -0.75 * np.log(np.clip(1.0 - (4.0 / 3.0) * P, 1e-9, None))
    D = np.where(P >= 0.7499, 3.0, D)
    np.fill_diagonal(D, 0.0)
    return D


def _bionj(D, labels):
    n = len(labels)
    if n < 3:
        return "(" + ",".join(labels) + ");"
    D = D.astype(float).copy()
    active = list(range(n))
    leaf_set = {i: {i} for i in range(n)}
    tree_str = {i: labels[i] for i in range(n)}
    counter = [n]
    while len(active) > 3:
        m = len(active)
        r = np.zeros(m)
        for idx_i, i in enumerate(active):
            s = 0.0
            for j in active:
                if i != j:
                    s += D[i, j]
            r[idx_i] = s
        best = None
        for a_idx, i in enumerate(active):
            for b_idx, j in enumerate(active):
                if i == j:
                    continue
                si = leaf_set[i]
                sj = leaf_set[j]
                q = 0.0
                for k in active:
                    if k == i or k == j:
                        continue
                    sk = leaf_set[k]
                    q += D[i, k] * len(si - sk) - D[j, k] * len(sj - sk)
                q /= (2.0 * (n - 2))
                crit = D[i, j] + r[a_idx] / (m - 2) + r[b_idx] / (m - 2) + q
                if best is None or crit < best[0]:
                    best = (crit, i, j)
        _, i, j = best
        new_idx = counter[0]
        counter[0] += 1
        leaf_set[new_idx] = leaf_set[i] | leaf_set[j]
        tree_str[new_idx] = "(" + tree_str[i] + "," + tree_str[j] + ")"
        dij = D[i, j]
        for k in active:
            if k == i or k == j:
                continue
            D[new_idx, k] = 0.5 * (D[i, k] + D[j, k] - dij)
            D[k, new_idx] = D[new_idx, k]
        active = [x for x in active if x not in (i, j)] + [new_idx]
    a, b, c = active
    return "(" + tree_str[a] + "," + tree_str[b] + "," + tree_str[c] + ");"


def _parse_newick(newick):
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    pos = [0]
    nodes = []

    def parse():
        if s[pos[0]] == '(':
            pos[0] += 1
            children = []
            while True:
                child = parse()
                children.append(child)
                if s[pos[0]] == ',':
                    pos[0] += 1
                    continue
                elif s[pos[0]] == ')':
                    pos[0] += 1
                    break
                else:
                    raise ValueError("bad newick")
            lbl = None
            if pos[0] < len(s) and s[pos[0]] != '(':
                start = pos[0]
                while pos[0] < len(s) and s[pos[0]] not in ',();':
                    pos[0] += 1
                lbl = s[start:pos[0]]
            node = len(nodes)
            nodes.append(('internal', children, lbl))
            return node
        else:
            start = pos[0]
            while pos[0] < len(s) and s[pos[0]] not in ',();':
                pos[0] += 1
            lbl = s[start:pos[0]]
            node = len(nodes)
            nodes.append(('leaf', None, lbl))
            return node

    root = parse()
    children_map = {}
    parent_map = {}
    is_leaf = {}
    node_label = {}
    leaves = {}

    def build(node, parent):
        kind, ch, lbl = nodes[node]
        is_leaf[node] = (kind == 'leaf')
        node_label[node] = lbl
        if kind == 'leaf':
            children_map[node] = []
            if lbl is not None:
                leaves[lbl] = node
        else:
            children_map[node] = ch
            for c in ch:
                build(c, node)
        parent_map[node] = parent

    build(root, None)
    return {
        'root': root,
        'children': children_map,
        'parent': parent_map,
        'is_leaf': is_leaf,
        'node_label': node_label,
        'leaves': leaves,
        'n_nodes': len(nodes),
    }


def _newick_from_tree(tree, root=None):
    children = tree['children']
    node_label = tree['node_label']
    is_leaf = tree['is_leaf']
    if root is None:
        root = tree['root']

    def rec(node):
        if is_leaf[node]:
            return node_label[node]
        ch = children[node]
        if len(ch) == 0:
            return node_label[node] or "n"
        parts = [rec(c) for c in ch]
        return "(" + ",".join(parts) + ")"

    return rec(root) + ";"


def _compute_parsimony(sequences, labels, tree):
    leaves = tree['leaves']
    children = tree['children']
    is_leaf = tree['is_leaf']
    root = tree['root']
    node_label = tree['node_label']

    postorder = []

    def rec(node):
        if is_leaf[node]:
            postorder.append(node)
        else:
            for c in children[node]:
                rec(c)
            postorder.append(node)

    rec(root)

    leaf_seq = {lab: sequences[lab] for lab in labels}
    L = len(leaf_seq[labels[0]])

    state = {}
    for node in postorder:
        if is_leaf[node]:
            seq = leaf_seq[node_label[node]]
            arr = np.zeros(L, dtype=np.int32)
            for i, ch in enumerate(seq):
                if ch == 'A':
                    arr[i] = 1
                elif ch == 'C':
                    arr[i] = 2
                elif ch == 'G':
                    arr[i] = 4
                else:
                    arr[i] = 8
            state[node] = arr
        else:
            ch = children[node]
            if len(ch) == 0:
                state[node] = np.zeros(L, dtype=np.int32)
            elif len(ch) == 1:
                state[node] = state[ch[0]].copy()
            else:
                inter = None
                for c in ch:
                    sc = state[c]
                    if inter is None:
                        inter = sc.copy()
                    else:
                        inter = inter & sc
                state[node] = inter

    score = 0
    for node in postorder:
        if is_leaf[node]:
            continue
        ch = children[node]
        if len(ch) < 2:
            continue
        inter = state[node]
        for c in ch:
            sc = state[c]
            diff = sc & ~inter
            score += int(np.count_nonzero(diff))
    return score


def _build_adj(tree):
    adj = {i: set() for i in range(tree['n_nodes'])}
    for u in range(tree['n_nodes']):
        for c in tree['children'][u]:
            adj[u].add(c)
            adj[c].add(u)
    return adj


def _nni_hill_climb(tree, sequences, labels, max_iters=100):
    def build_tree_from_adj(adj, root_node):
        par = {root_node: None}
        order = [root_node]
        for node in order:
            for nb in adj[node]:
                if nb not in par:
                    par[nb] = node
                    order.append(nb)
        new_children = {i: [] for i in range(tree['n_nodes'])}
        for node in range(tree['n_nodes']):
            p = par[node]
            if p is not None:
                new_children[p].append(node)
        return {
            'root': root_node,
            'children': new_children,
            'parent': par,
            'is_leaf': tree['is_leaf'],
            'node_label': tree['node_label'],
            'leaves': tree['leaves'],
            'n_nodes': tree['n_nodes'],
        }

    best_score = _compute_parsimony(sequences, labels, tree)
    adj = _build_adj(tree)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if _now() - _start_time > _DEADLINE - 25:
            break
        for u in range(tree['n_nodes']):
            if tree['is_leaf'][u]:
                continue
            for v in sorted(adj[u]):
                if v <= u:
                    continue
                if tree['is_leaf'][v]:
                    continue
                u_others = [x for x in adj[u] if x != v]
                v_others = [x for x in adj[v] if x != u]
                if len(u_others) != 2 or len(v_others) != 2:
                    continue
                a1, a2 = u_others
                b1, b2 = v_others
                for (u_c1, u_c2, v_c1, v_c2) in [(a1, b1, a2, b2), (a1, b2, a2, b1)]:
                    new_adj = {k: set(v2) for k, v2 in adj.items()}
                    new_adj[u] = {u_c1, u_c2}
                    new_adj[v] = {v_c1, v_c2}
                    for x in (a1, a2, b1, b2):
                        new_adj[x].discard(u)
                        new_adj[x].discard(v)
                    for x in (u_c1, u_c2):
                        new_adj[x].add(u)
                    for x in (v_c1, v_c2):
                        new_adj[x].add(v)
                    new_tree = build_tree_from_adj(new_adj, u)
                    try:
                        sc = _compute_parsimony(sequences, labels, new_tree)
                    except Exception:
                        continue
                    if sc < best_score:
                        best_score = sc
                        tree = new_tree
                        adj = _build_adj(tree)
                        improved = True
                        break
            if improved:
                break
    return tree


def reconstruct(sequences):
    global _start_time
    if _start_time is None:
        _start_time = _now()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n < 3:
        return "(" + ",".join(labels) + ");"

    D = _jc_distance_matrix(sequences, labels)
    newick = _bionj(D, labels)
    tree = _parse_newick(newick)

    if _now() - _start_time < _DEADLINE - 35:
        try:
            tree = _nni_hill_climb(tree, sequences, labels, max_iters=100)
        except Exception:
            pass

    newick = _newick_from_tree(tree)
    return newick
