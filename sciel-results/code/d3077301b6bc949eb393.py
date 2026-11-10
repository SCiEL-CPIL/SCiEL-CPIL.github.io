"""Phylogenetic topology recovery: NJ + SPR local search (minimize RF)."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np

_DEADLINE = time.time() + 255.0  # safety margin before 270s kill


def _jc_distance_matrix(sequences, labels):
    """Return n x n Jukes-Cantor corrected distance matrix (vectorized)."""
    n = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    # build a char->code matrix
    code = np.zeros((n, len(sequences[labels[0]])), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        code[i] = [ord(c) for c in s]
    # pairwise differences
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diff = (code[i] != code[i + 1:]).sum(axis=1)  # n-1-i values
        p = diff / code.shape[1]
        p = np.clip(p, 0.0, 0.7499)
        d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
        D[i, i + 1:] = d
        D[i + 1:, i] = d
    return D


def _splits_from_newick(newick, labels):
    """Parse Newick into a set of canonical splits (frozensets of taxa)."""
    label_set = set(labels)
    stack = []
    # tokenise
    i = 0
    s = newick
    n = len(s)
    # We'll do a recursive parse using a pointer.
    pos = 0

    def parse_node():
        nonlocal pos
        if pos < n and s[pos] == '(':
            pos += 1
            children = []
            while True:
                children.append(parse_node())
                if pos < n and s[pos] == ',':
                    pos += 1
                    continue
                if pos < n and s[pos] == ')':
                    pos += 1
                    break
            # optional branch length
            while pos < n and s[pos] not in (',', ')', ';'):
                pos += 1
            return ('node', children)
        else:
            start = pos
            while pos < n and s[pos] not in (',', ')', ';', '('):
                pos += 1
            name = s[start:pos]
            # skip branch length
            while pos < n and s[pos] not in (',', ')', ';'):
                pos += 1
            return ('leaf', name)

    tree = parse_node()

    def taxa_of(node):
        if node[0] == 'leaf':
            return frozenset([node[1]])
        out = set()
        for c in node[1]:
            out |= set(taxa_of(c))
        return frozenset(out)

    def collect_splits(node, out):
        if node[0] == 'leaf':
            return
        for c in node[1]:
            t = taxa_of(c)
            other = label_set - set(t)
            a, b = (t, other) if len(t) <= len(other) else (other, t)
            if 1 <= len(a) <= len(label_set) - 1:
                out.add(frozenset(a))
        for c in node[1]:
            collect_splits(c, out)

    splits = set()
    collect_splits(tree, splits)
    return splits


def _rf_distance(s1, s2):
    """Normalized RF distance between two split sets (0..1)."""
    if not s1 and not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    if union == 0:
        return 0.0
    return 1.0 - inter / union


def _nj(D, labels):
    """Neighbor-Joining. Returns (newick, splits, D_final)."""
    n = len(labels)
    # work with indices
    D = D.copy()
    active = list(range(n))
    # tree representation: each active node has a 'name' (label or internal)
    name_of = {i: labels[i] for i in range(n)}
    # adjacency for building newick: keep a list of edges (a,b)
    edges = []
    while len(active) > 3:
        m = len(active)
        r = np.zeros(m)
        for idx, a in enumerate(active):
            r[idx] = D[a, active].sum()
        # find min criterion
        best = None
        best_val = float('inf')
        for i in range(m):
            for j in range(i + 1, m):
                a, b = active[i], active[j]
                crit = (m - 2) * D[a, b] - r[i] - r[j]
                if crit < best_val:
                    best_val = crit
                    best = (a, b)
        a, b = best
        # create new node
        new = -1 - len(edges)
        name_of[new] = f"n{new}"
        # update distances to new node
        for x in active:
            if x in (a, b):
                continue
            D[new, x] = D[x, new] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        D[new, new] = 0.0
        # record edge (a,b) in the tree
        edges.append((a, b))
        active = [x for x in active if x not in (a, b)]
        active.append(new)
    # final three nodes: connect them
    a, b, c = active
    edges.append((a, b))
    edges.append((b, c))

    # Build newick from edges. Reconstruct tree structure.
    # Build children map: each internal node's children.
    # We have a set of edges; build a tree rooted arbitrarily.
    nodes = set(name_of.keys())
    adj = {v: [] for v in nodes}
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)

    # root at first internal node
    root = None
    for v in nodes:
        if v not in range(n):
            root = v
            break
    if root is None:
        root = nodes.pop()

    def newick(v, parent):
        kids = [k for k in adj[v] if k != parent]
        if not kids:
            return name_of[v]
        parts = []
        for k in kids:
            parts.append(newick(k, v))
        return "(" + ",".join(parts) + ")" + name_of[v]

    nw = newick(root, None) + ";"
    splits = _splits_from_newick(nw, labels)
    return nw, splits


def _tree_from_splits(splits, labels):
    """Reconstruct a tree (as nested structure) from a split system, for SPR."""
    # Build a graph of splits to do SPR we need a tree representation.
    # We'll represent the tree as a dict: node -> list of children, with leaves = labels.
    # Rebuild via a simple method: use the splits to form a tree by
    # 'additive' construction is complex; instead we keep the Newick and
    # operate SPR on a parsed tree structure.
    raise NotImplementedError


class _Tree:
    """Unrooted tree as adjacency of nodes; leaves are labels."""
    def __init__(self, labels):
        self.labels = labels
        self.n = len(labels)
        # nodes: 0..n-1 are leaves, n+ are internal
        self.adj = {i: [] for i in range(self.n)}
        self.next_id = self.n

    def _new_node(self):
        nid = self.next_id
        self.next_id += 1
        self.adj[nid] = []
        return nid

    def add_edge(self, a, b):
        self.adj[a].append(b)
        self.adj[b].append(a)

    def neighbors(self, v):
        return self.adj[v]

    def remove_edge(self, a, b):
        self.adj[a].remove(b)
        self.adj[b].remove(a)

    def side(self, v):
        """Return set of leaves on the side of v not containing the root we pass."""
        pass

    def splits(self):
        """Return canonical split set."""
        label_set = set(self.labels)
        out = set()
        n_nodes = len(self.adj)
        # do DFS from each edge to get one side
        def leaves_from(start, parent):
            stack = [(start, parent)]
            seen = set()
            leaves = set()
            while stack:
                v, p = stack.pop()
                if v in seen:
                    continue
                seen.add(v)
                if v < self.n:
                    leaves.add(self.labels[v])
                for nb in self.adj[v]:
                    if nb != p:
                        stack.append((nb, v))
            return leaves
        # iterate over edges
        seen_edges = set()
        for v in self.adj:
            for nb in self.adj[v]:
                key = (min(v, nb), max(v, nb))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                side = leaves_from(v, nb)
                other = label_set - side
                a, b = (side, other) if len(side) <= len(other) else (other, side)
                if 1 <= len(a) <= self.n - 1:
                    out.add(frozenset(a))
        return out

    def copy(self):
        t = _Tree(self.labels)
        t.adj = {k: list(v) for k, v in self.adj.items()}
        t.next_id = self.next_id
        return t


def _nj_to_tree(D, labels):
    """Run NJ and return a _Tree object plus splits."""
    n = len(labels)
    D = D.copy()
    active = list(range(n))
    tree = _Tree(labels)
    # map active index -> tree node id
    node_of = {i: i for i in range(n)}
    while len(active) > 3:
        m = len(active)
        r = np.zeros(m)
        for idx, a in enumerate(active):
            r[idx] = D[a, active].sum()
        best = None
        best_val = float('inf')
        for i in range(m):
            for j in range(i + 1, m):
                a, b = active[i], active[j]
                crit = (m - 2) * D[a, b] - r[i] - r[j]
                if crit < best_val:
                    best_val = crit
                    best = (a, b)
        a, b = best
        na, nb = node_of[a], node_of[b]
        new_node = tree._new_node()
        # connect new node to na and nb
        tree.add_edge(new_node, na)
        tree.add_edge(new_node, nb)
        new_idx = -1 - new_node
        D[new_idx, new_idx] = 0.0
        for x in active:
            if x in (a, b):
                continue
            D[new_idx, x] = D[x, new_idx] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
        node_of[new_idx] = new_node
        active = [x for x in active if x not in (a, b)]
        active.append(new_idx)
    a, b, c = active
    na, nb, nc = node_of[a], node_of[b], node_of[c]
    tree.add_edge(na, nb)
    tree.add_edge(nb, nc)
    splits = tree.splits()
    return tree, splits


def _spr_improve(tree, D, labels, max_rounds=60, deadline=None):
    """Greedy SPR (subtree pruning and regrafting) to reduce RF-like cost.
    We use the sum of edge lengths under NJ-style as a proxy, but to directly
    target RF we instead minimize the number of 'extra' splits vs a reference.
    Here we minimize total branch length (ME) as a good proxy that correlates
    with RF for JC-simulated trees.
    """
    n = len(labels)
    label_set = set(labels)

    def side_set(v, parent):
        stack = [(v, parent)]
        seen = set()
        leaves = set()
        while stack:
            x, p = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            if x < n:
                leaves.add(labels[x])
            for nb in tree.adj[x]:
                if nb != p:
                    stack.append((nb, x))
        return leaves

    def total_length():
        # length of each edge = 0.5*(dist to subtree) heuristic: use NJ length
        # Simpler: use the 'fast' length: for edge (u,v), length = 0.5*(d(u,S)+d(v,S)-d(u,v))
        # We'll approximate using pairwise distances to the side.
        total = 0.0
        seen = set()
        for u in tree.adj:
            for v in tree.adj[u]:
                if (u, v) in seen or (v, u) in seen:
                    continue
                seen.add((u, v))
                seen.add((v, u))
                su = side_set(u, v)
                sv = label_set - su
                if not su or not sv:
                    continue
                # average distance between su and sv
                a = list(su)
                b = list(sv)
                # use precomputed D over labels
                s = 0.0
                cnt = 0
                for x in a:
                    for y in b:
                        s += D[labels.index(x), labels.index(y)]
                        cnt += 1
                if cnt:
                    avg = s / cnt
                    # edge length approx
                    total += avg
        return total

    # Precompute label index
    lidx = {lab: i for i, lab in enumerate(labels)}

    def edge_length(u, v):
        su = side_set(u, v)
        if not su or len(su) == n:
            return 0.0
        sv = label_set - su
        s = 0.0
        cnt = 0
        for x in su:
            ix = lidx[x]
            for y in sv:
                s += D[ix, lidx[y]]
                cnt += 1
        return (s / cnt) if cnt else 0.0

    def total_length2():
        total = 0.0
        seen = set()
        for u in tree.adj:
            for v in tree.adj[u]:
                if (v, u) in seen:
                    continue
                seen.add((u, v))
                total += edge_length(u, v)
        return total

    best_tree = tree.copy()
    best_len = total_length2()
    for rnd in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # iterate over edges (prune side = smaller)
        edges = []
        seen = set()
        for u in tree.adj:
            for v in tree.adj[u]:
                if (v, u) in seen:
                    continue
                seen.add((u, v))
                edges.append((u, v))
        for (u, v) in edges:
            if deadline is not None and time.time() > deadline:
                break
            # prune u side from v
            su = side_set(u, v)
            if len(su) <= 1 or len(su) >= n - 1:
                continue
            # remove edge
            tree.remove_edge(u, v)
            # find best regraft: for each edge (a,b) in remaining, try inserting u
            best_pos = None
            best_l = total_length2()  # after removal, recompute baseline
            for (a, b) in list(tree.adj.items()):
                pass
            # iterate candidate edges
            cand = []
            seen2 = set()
            for a in tree.adj:
                for b in tree.adj[a]:
                    if (b, a) in seen2:
                        continue
                    seen2.add((a, b))
                    cand.append((a, b))
            for (a, b) in cand:
                # skip if a or b is in the pruned side (would reconnect internally)
                # a,b are nodes; check if a is in su side
                # we need to know if node a is within pruned subtree
                # determine side of pruned subtree: nodes reachable from u without going to v
                pruned_nodes = set()
                stack = [(u, v)]
                while stack:
                    x, p = stack.pop()
                    if x in pruned_nodes:
                        continue
                    pruned_nodes.add(x)
                    for nb in tree.adj[x]:
                        if nb != p:
                            stack.append((nb, x))
                if a in pruned_nodes or b in pruned_nodes:
                    continue
                # try insert u between a and b
                tree.remove_edge(a, b)
                tree.add_edge(a, u)
                tree.add_edge(u, b)
                l = total_length2()
                # revert
                tree.remove_edge(a, u)
                tree.remove_edge(u, b)
                tree.add_edge(a, b)
                if l < best_l - 1e-9:
                    best_l = l
                    best_pos = (a, b)
            # revert removal
            tree.add_edge(u, v)
            if best_pos is not None:
                a, b = best_pos
                tree.remove_edge(u, v)
                tree.remove_edge(a, b)
                tree.add_edge(a, u)
                tree.add_edge(u, b)
                improved = True
                if best_l < best_len - 1e-9:
                    best_len = best_l
                    best_tree = tree.copy()
        if not improved:
            break
    return best_tree, best_len


def _tree_to_newick(tree, labels):
    n = len(labels)
    nodes = list(tree.adj.keys())
    root = None
    for v in nodes:
        if v >= n:
            root = v
            break
    if root is None:
        root = nodes[0]

    def nw(v, parent):
        kids = [k for k in tree.adj[v] if k != parent]
        if not kids:
            return labels[v]
        parts = [nw(k, v) for k in kids]
        return "(" + ",".join(parts) + ")"

    return nw(root, None) + ";"


def _random_tree(labels, rng):
    t = _Tree(labels)
    n = len(labels)
    if n <= 3:
        a, b, c = list(range(n))
        t.add_edge(a, b)
        t.add_edge(b, c)
        return t
    # build by progressive addition of random bifurcating tree
    order = list(range(n))
    rng.shuffle(order)
    # start with first three
    a, b, c = order[0], order[1], order[2]
    t.add_edge(a, b)
    t.add_edge(b, c)
    for leaf in order[3:]:
        # pick a random edge
        edges = []
        seen = set()
        for u in t.adj:
            for v in t.adj[u]:
                if (v, u) in seen:
                    continue
                seen.add((u, v))
                edges.append((u, v))
        a, b = rng.choice(edges)
        new = t._new_node()
        t.remove_edge(a, b)
        t.add_edge(a, new)
        t.add_edge(new, b)
        t.add_edge(new, leaf)
    return t


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    D = _jc_distance_matrix(sequences, labels)
    deadline = _DEADLINE - 5.0  # safety margin

    best_splits = None
    best_newick = None
    best_score = 1.0  # we don't know true tree; use ME length as proxy to rank

    # 1) NJ
    t_nj, splits_nj = _nj_to_tree(D, labels)
    nw_nj = _tree_to_newick(t_nj, labels)
    # compute a proxy cost: total NJ length
    lidx = {lab: i for i, lab in enumerate(labels)}
    def total_len(tree):
        total = 0.0
        seen = set()
        for u in tree.adj:
            for v in tree.adj[u]:
                if (v, u) in seen:
                    continue
                seen.add((u, v))
                su = set()
                stack = [(u, v)]
                while stack:
                    x, p = stack.pop()
                    if x in su:
                        continue
                    su.add(x)
                    if x < n:
                        pass
                    for nb in tree.adj[x]:
                        if nb != p:
                            stack.append((nb, x))
                leaves = [labels[x] for x in su if x < n]
                if not leaves or len(leaves) == n:
                    continue
                other = [l for l in labels if l not in leaves]
                s = 0.0
                cnt = 0
                for x in leaves:
                    ix = lidx[x]
                    for y in other:
                        s += D[ix, lidx[y]]
                        cnt += 1
                if cnt:
                    total += s / cnt
        return total

    cost_nj = total_len(t_nj)
    best_newick = nw_nj
    best_cost = cost_nj
    best_tree = t_nj

    # 2) SPR refine NJ
    if time.time() < deadline - 5:
        t_spr, spr_len = _spr_improve(t_nj.copy(), D, labels, max_rounds=40, deadline=deadline - 5)
        nw_spr = _tree_to_newick(t_spr, labels)
        c_spr = total_len(t_spr)
        if c_spr < best_cost:
            best_cost = c_spr
            best_newick = nw_spr
            best_tree = t_spr

    # 3) A few random starts + SPR (only if time allows and n is small enough)
    if n <= 24 and time.time() < deadline - 15:
        rng = random.Random(12345)
        for k in range(4):
            if time.time() > deadline - 5:
                break
            tr = _random_tree(labels, rng)
            t2, _ = _spr_improve(tr, D, labels, max_rounds=20, deadline=deadline - 5)
            nw2 = _tree_to_newick(t2, labels)
            c2 = total_len(t2)
            if c2 < best_cost:
                best_cost = c2
                best_newick = nw2
                best_tree = t2

    # Validate: ensure all taxa appear exactly once
    if best_newick is None:
        # fallback star
        best_newick = "(" + ",".join(labels) + ");"
    # sanity: count each label
    for lab in labels:
        if labels.count(lab) != 1:
            pass
    # ensure no label missing
    import re
    found = set(re.findall(r't\d+', best_newick))
    if found != set(labels):
        # rebuild a safe tree from best_tree
        best_newick = _tree_to_newick(best_tree, labels) if best_tree is not None else "(" + ",".join(labels) + ");"

    return best_newick
# EVOLVE-BLOCK-END
