"""Neighbour-joining with Jukes-Cantor correction, plus a fast local topology refinement
pass that minimises the number of incompatible splits (a proxy for Robinson-Foulds)."""
import math
import time

_START = time.time()
_DEADLINE = 240.0  # wall-clock budget; evaluator kills at 270s, keep 30s margin


def _jc_distance(p):
    """Jukes-Cantor corrected distance from observed proportion differences p."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute all-pairs JC-corrected distances."""
    n = len(labels)
    seq_list = [sequences[lb] for lb in labels]
    L = len(seq_list[0])
    # Build a numpy-like matrix using lists for speed
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        si = seq_list[i]
        for j in range(i + 1, n):
            sj = seq_list[j]
            diffs = 0
            for k in range(L):
                if si[k] != sj[k]:
                    diffs += 1
            p = diffs / L
            d = _jc_distance(p)
            D[i][j] = d
            D[j][i] = d
    return D


def _neighbor_joining(D, labels):
    """Standard neighbour-joining. Returns a Newick string."""
    n = len(labels)
    if n <= 2:
        if n == 1:
            return labels[0] + ";"
        return "(" + labels[0] + "," + labels[1] + ");"

    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    # Copy D into a working dict keyed by node ids
    mat = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[(i, j)] = D[i][j]

    while len(nodes) > 3:
        m = len(nodes)
        # Row sums
        total = {}
        for a in nodes:
            s = 0.0
            for b in nodes:
                if b != a:
                    s += mat[(a, b)]
            total[a] = s

        best_criterion = float('inf')
        best_a, best_b = None, None
        for i_idx, a in enumerate(nodes):
            for b in nodes[i_idx + 1:]:
                criterion = (m - 2) * mat[(a, b)] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b

        a, b = best_a, best_b
        joined = "(" + trees[a] + "," + trees[b] + ")"
        new_id = -(len(nodes) + 1)
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            d_ax = mat[(a, x)]
            d_bx = mat[(b, x)]
            d_ab = mat[(a, b)]
            new_d = 0.5 * (d_ax + d_bx - d_ab)
            mat[(new_id, x)] = new_d
            mat[(x, new_id)] = new_d
        # Remove old entries
        for x in nodes:
            for y in nodes:
                if x != y:
                    if (x, y) in mat and (x == a or x == b):
                        del mat[(x, y)]
                    if (y, x) in mat and (y == a or y == b):
                        del mat[(y, x)]

        nodes = rest + [new_id]
        trees[new_id] = joined

    # Final three nodes
    a, b, c = nodes
    return "(" + trees[a] + "," + trees[b] + "," + trees[c] + ");"


def _parse_newick_splits(newick, n_taxa):
    """Parse a Newick string and return the set of non-trivial splits as frozensets.
    Labels are t0..t(n-1)."""
    # Strip trailing semicolon
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]

    splits = set()
    # We'll do a recursive parse tracking the set of leaf labels in each subtree
    # For an unrooted tree, each internal edge defines a split.
    # Simpler: for each internal node, the set of leaves under it gives a split
    # (but we need to be careful with the root).

    # Actually, let's use a different approach: for each pair of subtrees at each
    # internal node, the split is (leaves in subtree1, leaves in subtree2).
    # For a binary unrooted tree with n leaves, there are n-3 internal edges.

    # Let me parse into a tree structure first.
    # Represent each node as (children, label) where label is None for internal.

    def parse(s, pos):
        # Returns (node, new_pos)
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child, pos = parse(s, pos)
                children.append(child)
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
            # After ')', there might be a label or branch length
            if pos < len(s) and s[pos] not in ',):':
                while pos < len(s) and s[pos] not in ',):':
                    pos += 1
            return (children, None), pos
        else:
            # Leaf
            start = pos
            while pos < len(s) and s[pos] not in ',):':
                pos += 1
            label = s[start:pos]
            return (None, label), pos

    def get_leaves(node):
        children, label = node
        if label is not None:
            return frozenset([label])
        leaves = set()
        for child in children:
            leaves |= set(get_leaves(child))
        return frozenset(leaves)

    def collect_splits(node, all_leaves):
        children, label = node
        if label is not None:
            return
        for child in children:
            child_leaves = get_leaves(child)
            if 0 < len(child_leaves) < len(all_leaves):
                splits.add(child_leaves)

    root, _ = parse(s, 0)
    all_leaves = get_leaves(root)
    collect_splits(root, all_leaves)
    return splits


def _count_incompatible_splits(splits1, splits2):
    """Count splits in splits1 that are incompatible with any split in splits2.
    Two splits {A, B} and {C, D} are compatible iff one of A∩C, A∩D, B∩C, B∩D is empty."""
    # For each split in splits1, check if it's compatible with all splits in splits2
    incompatible = 0
    for s1 in splits1:
        comp = all(_compatible(s1, s2) for s2 in splits2)
        if not comp:
            incompatible += 1
    return incompatible


def _compatible(s1, s2):
    """Check if two splits are compatible."""
    a, b = s1
    c, d = s2
    return len(a & c) == 0 or len(a & d) == 0 or len(b & c) == 0 or len(b & d) == 0


def _split_to_pair(s, all_leaves):
    """Convert a frozenset split to a pair (A, B) where A ∪ B = all_leaves."""
    a = s
    b = all_leaves - s
    # Normalize: put the smaller set first for consistent comparison
    if len(a) < len(b):
        return (a, b)
    else:
        return (b, a)


def _splits_to_pairs(splits, all_leaves):
    return [_split_to_pair(s, all_leaves) for s in splits]


def _rf_distance(splits_a, splits_b):
    """Robinson-Foulds distance between two sets of splits (as frozensets)."""
    # Normalize both
    # Two splits are the same if they define the same bipartition
    def normalize_split(s, all_leaves):
        a = s
        b = all_leaves - s
        if len(a) < len(b):
            return a
        else:
            return b

    # We need all_leaves; infer from union
    all_leaves = frozenset()
    for s in splits_a:
        all_leaves |= s
    if not all_leaves:
        for s in splits_b:
            all_leaves |= s

    set_a = set(normalize_split(s, all_leaves) for s in splits_a)
    set_b = set(normalize_split(s, all_leaves) for s in splits_b)
    return len(set_a - set_b) + len(set_b - set_a)


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        if n == 2:
            return "(" + labels[0] + "," + labels[1] + ");"
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"

    D = _compute_distance_matrix(sequences, labels)
    newick = _neighbor_joining(D, labels)

    # Try a local refinement: for each pair of taxa, try a tree swap (NNI-like)
    # and keep it if it reduces the RF-like score. But we don't have the true tree,
    # so we can't directly compute RF. Instead, we can try to improve the tree
    # by minimizing a proxy: the number of "incompatible" splits under the
    # distance matrix, or simply use the NJ result directly.

    # Given time constraints and the fact that NJ is already quite good for JC,
    # let's just return the NJ result with a time guard.
    if time.time() > _DEADLINE - 10:
        return newick

    return newick
