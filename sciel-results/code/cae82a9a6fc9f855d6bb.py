"""Phylogenetic topology recovery: FUAD (Fast Up-And-Down) with NJ core,
plus exhaustive quartet-score search for very small n (<=7), and NNI polish."""

import time
import math
import itertools
import numpy as np

_DEADLINE_MARGIN = 18.0
_START_TIME = None


def _get_elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _check_deadline():
    return _get_elapsed() > (270.0 - _DEADLINE_MARGIN)


def _encode_sequences(sequences, labels):
    """Convert dict of sequences to (n, L) uint8 array (A=0,C=1,G=2,T=3)."""
    n = len(labels)
    if n == 0:
        return np.empty((0, 0), dtype=np.uint8)
    L = len(sequences[labels[0]])
    if L == 0:
        return np.empty((n, 0), dtype=np.uint8)
    mat = np.empty((n, L), dtype=np.uint8)
    # Pre-build lookup table for byte->nucleotide
    lookup = np.zeros(256, dtype=np.uint8)
    lookup[65] = 0  # A
    lookup[67] = 1  # C
    lookup[71] = 2  # G
    lookup[84] = 3  # T
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        arr = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
        mat[i] = lookup[arr]
    return mat


def _jc_distance_matrix(seqs_mat, L):
    """Vectorized JC-corrected pairwise distance matrix."""
    n = seqs_mat.shape[0]
    if L == 0 or n == 0:
        return np.zeros((n, n), dtype=np.float64)
    diff = (seqs_mat[:, None, :] != seqs_mat[None, :, :])
    mismatches = diff.sum(axis=2)
    p = mismatches.astype(np.float64) / float(L)
    p = np.clip(p, 0.0, 0.75 - 1e-12)
    D = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def _fast_nj(D, labels):
    """Fast Neighbor-Joining. Returns (newick_string, newick_map, names)."""
    n = D.shape[0]
    if n == 1:
        return f"{labels[0]};", {labels[0]: labels[0]}, [labels[0]]
    if n == 2:
        return f"({labels[0]},{labels[1]});", {labels[0]: labels[0], labels[1]: labels[1]}, [labels[0], labels[1]]
    if n == 3:
        nw = f"(({labels[0]},{labels[1]}),{labels[2]});"
        m = {labels[0]: labels[0], labels[1]: labels[1], labels[2]: labels[2]}
        return nw, m, list(labels)

    names = list(labels)
    dist = D.copy()
    newick_map = {name: name for name in labels}

    while len(names) > 3:
        if _check_deadline():
            break
        m = len(names)
        r = dist.sum(axis=1)
        Q = (m - 2) * dist - r[:, None] - r[None, :]
        Q_masked = Q.copy()
        np.fill_diagonal(Q_masked, np.inf)
        idx = np.unravel_index(np.argmin(Q_masked), (m, m))
        i, j = idx[0], idx[1]
        if i == j:
            i, j = 0, 1

        a_name = names[i]
        b_name = names[j]
        d_ab = dist[i, j]

        new_name = f"n{len(names)}_{a_name}_{b_name}"
        new_newick = f"({newick_map[a_name]},{newick_map[b_name]})"

        rest = [k for k in range(m) if k not in (i, j)]
        if rest:
            d_a_r = dist[i, rest]
            d_b_r = dist[j, rest]
            d_new_r = 0.5 * (d_a_r + d_b_r - d_ab)
            new_names = [names[k] for k in rest] + [new_name]
            new_dist = np.zeros((len(rest) + 1, len(rest) + 1), dtype=np.float64)
            for idx_r, k in enumerate(rest):
                for idx_s, l in enumerate(rest):
                    if idx_r != idx_s:
                        new_dist[idx_r, idx_s] = dist[k, l]
            new_dist[:-1, -1] = d_new_r
            new_dist[-1, :-1] = d_new_r
            new_dist[-1, -1] = 0.0
            dist = new_dist
            names = new_names
        newick_map[new_name] = new_newick

    if len(names) == 3:
        a, b, c = names
        nw = f"(({newick_map[a]},{newick_map[b]}),{newick_map[c]});"
    elif len(names) == 2:
        a, b = names
        nw = f"({newick_map[a]},{newick_map[b]});"
    else:
        nw = newick_map[names[0]]
    return nw, newick_map, names


def _star(n, labels):
    return "(" + ",".join(labels) + ");"


def _quartet_score_tree(tree, labels, dist):
    """Compute total quartet score for a given tree topology."""
    n = len(labels)

    def _contains(node, target):
        if isinstance(node, int):
            return node == target
        for child in node:
            if _contains(child, target):
                return True
        return False

    def _same_clade(x, y, node):
        if isinstance(node, int):
            return node == x and node == y
        for child in node:
            if _contains(child, x) and _contains(child, y):
                return True
        return False

    total = 0.0
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        ab_same = _same_clade(a, b, tree)
        ac_same = _same_clade(a, c, tree)
        ad_same = _same_clade(a, d, tree)

        if ab_same and not ac_same and not ad_same:
            total += dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
        elif ac_same and not ab_same and not ad_same:
            total += dist[a, b] + dist[c, d] - dist[a, c] - dist[b, d]
        elif ad_same and not ab_same and not ac_same:
            total += dist[a, b] + dist[c, d] - dist[a, d] - dist[b, c]
    return total


def _generate_all_trees(n_leaves):
    """Generate all unrooted binary tree topologies as nested tuples of ints."""
    if n_leaves == 1:
        yield (0,)
    elif n_leaves == 2:
        yield ((0,), (1,))
    elif n_leaves == 3:
        yield ((0,), (1,), (2,))
    else:
        seen = set()
        for k in range(1, n_leaves // 2 + 1):
            for left in _generate_all_trees(k):
                right_indices = list(range(k, n_leaves))
                for right in _generate_all_trees(n_leaves - k):
                    def _remap(node, offset):
                        if isinstance(node, int):
                            return node + offset
                        return tuple(_remap(child, offset) for child in node)
                    right_remapped = _remap(right, k)
                    tree = (left, right_remapped)
                    canon = _canonicalize(tree)
                    if canon not in seen:
                        seen.add(canon)
                        yield tree


def _canonicalize(tree):
    if isinstance(tree, int):
        return tree
    children = [_canonicalize(child) for child in tree]
    children.sort(key=lambda x: repr(x))
    return tuple(children)


def _tree_to_newick(tree, labels):
    if isinstance(tree, int):
        return labels[tree]
    children = [_tree_to_newick(child, labels) for child in tree]
    return "(" + ",".join(children) + ")"


def _exhaustive_small_n(labels, dist):
    """Exhaustive search over all unrooted binary trees for n <= 7."""
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"(({labels[0]},{labels[1]}),{labels[2]})"
    best_tree = None
    best_score = float('inf')
    for tree in _generate_all_trees(n):
        if _check_deadline():
            break
        score = _quartet_score_tree(tree, labels, dist)
        if score < best_score:
            best_score = score
            best_tree = tree
    if best_tree is None:
        return None
    return _tree_to_newick(best_tree, labels)


def _parse_newick_to_splits(newick, labels):
    """Parse a Newick string into a set of unrooted splits (frozensets)."""
    # Strip trailing semicolon
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    if not s.startswith('('):
        return set()

    # Build a tree structure
    # We'll do a simple recursive parse
    def _parse(s):
        # Returns (node, pos) where node is a tuple of children or a leaf
        pos = 0
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child, pos = _parse(s)
                children.append(child)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                elif pos < len(s) and s[pos] == ')':
                    pos += 1
                    break
                else:
                    break
            return tuple(children), pos
        else:
            # Leaf
            end = pos
            while end < len(s) and s[end] not in ',):':
                end += 1
            leaf = s[pos:end]
            return leaf, end

    root, _ = _parse(s)

    # Collect all internal nodes and compute splits
    splits = set()
    leaves_set = set(labels)

    def _collect(node):
        if isinstance(node, str):
            return {node}
        # Internal node
        child_leaves = []
        for child in node:
            cl = _collect(child)
            child_leaves.append(cl)
        all_leaves = set()
        for cl in child_leaves:
            all_leaves.update(cl)
        # For each child, the split is (child_leaves, all_leaves - child_leaves)
        for cl in child_leaves:
            if 0 < len(cl) < len(all_leaves):
                comp = all_leaves - cl
                # Canonicalize: sort the two parts
                if len(cl) < len(comp):
                    splits.add(frozenset(cl))
                elif len(cl) > len(comp):
                    splits.add(frozenset(comp))
                else:
                    # Equal size: canonicalize by min repr
                    a = frozenset(cl)
                    b = frozenset(comp)
                    if repr(a) <= repr(b):
                        splits.add(a)
                    else:
                        splits.add(b)
        return all_leaves

    _collect(root)
    return splits


def _robins_foulds(splits1, splits2):
    """Normalized RF distance between two sets of splits."""
    if not splits1 and not splits2:
        return 0.0
    common = splits1 & splits2
    # Number of non-trivial splits in a binary tree with n leaves is n-3
    # But we just use the sizes
    d = len(splits1) + len(splits2) - 2 * len(common)
    max_splits = max(len(splits1), len(splits2), 1)
    return d / max_splits if max_splits > 0 else 0.0


def _fuad_reconstruct(D, labels):
    """
    Fast Up-And-Down (FUAD) algorithm:
    1. Iteratively prune the taxon with highest average distance to others.
    2. Reconstruct core tree via NJ on remaining taxa.
    3. Reinsert pruned taxa one by one at optimal positions.
    """
    n = len(labels)
    if n <= 3:
        return _fast_nj(D, labels)[0]

    # Determine how many to prune. Keep at least max(4, n//2) taxa in core.
    # Prune the "longest" branches first.
    prune_count = max(0, n - max(4, n // 2))
    if prune_count == 0:
        return _fast_nj(D, labels)[0]

    # Iteratively find and remove the taxon with highest avg distance
    remaining = list(range(n))
    pruned_order = []  # list of (taxon_index, distances_to_remaining_at_prune_time)
    current_D = D.copy()

    for _ in range(prune_count):
        if _check_deadline():
            break
        m = len(remaining)
        if m <= 4:
            break
        # Average distance of each remaining taxon to the others
        avg_dist = np.array([np.mean([current_D[remaining[i], remaining[j]] for j in range(m) if j != i]) for i in range(m)])
        # Pick the one with highest avg distance
        idx_to_prune = np.argmax(avg_dist)
        taxon_idx = remaining[idx_to_prune]
        # Record distances from this taxon to all remaining
        dists_to_remaining = {remaining[j]: current_D[taxon_idx, remaining[j]] for j in range(m) if j != idx_to_prune}
        pruned_order.append((taxon_idx, dists_to_remaining))
        # Remove from remaining
        remaining.pop(idx_to_prune)
        # Update current_D to just the remaining taxa
        new_m = len(remaining)
        new_D = np.zeros((new_m, new_m), dtype=np.float64)
        for i in range(new_m):
            for j in range(new_m):
                new_D[i, j] = current_D[remaining[i], remaining[j]]
        current_D = new_D

    # Reconstruct core tree
    core_labels = [labels[i] for i in remaining]
    if len(core_labels) <= 3:
        # Edge case
        if len(core_labels) == 1:
            core_newick = core_labels[0]
        elif len(core_labels) == 2:
            core_newick = f"({core_labels[0]},{core_labels[1]})"
        else:
            core_newick = f"(({core_labels[0]},{core_labels[1]}),{core_labels[2]})"
    else:
        core_newick, _, _ = _fast_nj(current_D, core_labels)

    # Now reinsert pruned taxa one by one
    # We need to work with the full distance matrix D
    # For each pruned taxon, find the best edge to insert into
    # We'll represent the tree as a list of edges (pairs of node names)
    # Start with the core tree structure

    # Parse the core newick into a set of edges (as pairs of labels)
    # This is complex; instead, we'll use a simpler approach:
    # Build the tree as a dictionary of adjacencies

    def _newick_to_adj(newick, labels_in_tree):
        """Convert newick to adjacency dict: node -> set of neighbor nodes."""
        s = newick.strip()
        if s.endswith(';'):
            s = s[:-1]

        def _parse(s):
            pos = 0
            if s[pos] == '(':
                pos += 1
                children = []
                while True:
                    child, pos = _parse(s)
                    children.append(child)
                    if pos < len(s) and s[pos] == ',':
                        pos += 1
                    elif pos < len(s) and s[pos] == ')':
                        pos += 1
                        break
                    else:
                        break
                return tuple(children), pos
            else:
                end = pos
                while end < len(s) and s[end] not in ',):':
                    end += 1
                return s[pos:end], end

        root, _ = _parse(s)

        adj = {}
        def _build(node):
            if isinstance(node, str):
                if node not in adj:
                    adj[node] = set()
                return node
            # Internal node: generate a name
            name = f"int_{id(node)}"
            if name not in adj:
                adj[name] = set()
            for child in node:
                cname = _build(child)
                adj[name].add(cname)
                adj[cname].add(name)
            return name

        _build(root)
        return adj

    adj = _newick_to_adj(core_newick, core_labels)

    # Reinsert pruned taxa in reverse order (most recently pruned first, i.e., last in list)
    for taxon_idx, dists in reversed(pruned_order):
        if _check_deadline():
            break
        taxon_label = labels[taxon_idx]
        # Find the best edge (u, v) in the current tree to insert the taxon
        # Score: d(taxon, u) + d(taxon, v) - d(u, v)  (lower is better for insertion)
        # We need d(u,v) from the original D
        best_edge = None
        best_score = float('inf')
        for u in adj:
            for v in adj[u]:
                if u > v:
                    continue  # avoid double counting
                # Get indices of u and v in labels
                try:
                    ui = labels.index(u)
                    vi = labels.index(v)
                except ValueError:
                    continue
                # u and v must be actual taxa (not internal nodes)
                # Actually, in our adjacency, internal nodes are named "int_..."
                # We only consider edges between actual taxa for simplicity
                if u.startswith('int_') or v.startswith('int_'):
                    continue
                score = D[taxon_idx, ui] + D[taxon_idx, vi] - D[ui, vi]
                if score < best_score:
                    best_score = score
                    best_edge = (u, v)

        if best_edge is None:
            # Fallback: attach to the taxon with minimum distance
            min_dist = float('inf')
            best_attach = None
            for other in adj:
                if other.startswith('int_'):
                    continue
                try:
                    oi = labels.index(other)
                except ValueError:
                    continue
                d = D[taxon_idx, oi]
                if d < min_dist:
                    min_dist = d
                    best_attach = other
            if best_attach:
                adj[taxon_label] = {best_attach}
                adj[best_attach].add(taxon_label)
            continue

        u, v = best_edge
        # Remove edge u-v, add u-taxon and taxon-v
        adj[u].discard(v)
        adj[v].discard(u)
        if taxon_label not in adj:
            adj[taxon_label] = set()
        adj[u].add(taxon_label)
        adj[v].add(taxon_label)
        adj[taxon_label].add(u)
        adj[taxon_label].add(v)

    # Convert adjacency back to Newick
    def _adj_to_newick(adj, labels):
        if not adj:
            return ""
        # Find a leaf to start from
        all_nodes = set(adj.keys())
        leaves = [n for n in all_nodes if len(adj[n]) == 1]
        if not leaves:
            leaves = list(all_nodes)
        start = leaves[0]

        def _to_newick(node, parent):
            neighbors = [n for n in adj[node] if n != parent]
            if not neighbors:
                return node
            children = [_to_newick(n, node) for n in neighbors]
            if len(children) == 1:
                return children[0]
            return "(" + ",".join(children) + ")"

        return _to_newick(start, None)

    result = _adj_to_newick(adj, labels)
    return result


def reconstruct(sequences):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"(({labels[0]},{labels[1]}),{labels[2]})"

    # Encode sequences
    seqs_mat = _encode_sequences(sequences, labels)
    L = seqs_mat.shape[1] if seqs_mat.size > 0 else 0

    if L == 0:
        return _star(n, labels)

    # Compute JC distance matrix
    D = _jc_distance_matrix(seqs_mat, L)

    # Check if all distances are zero
    if not np.any(D):
        return _star(n, labels)

    # For small n (<=7), try exhaustive search
    if n <= 7 and not _check_deadline():
        result = _exhaustive_small_n(labels, D)
        if result is not None:
            return result

    # For larger n, use FUAD
    if n >= 8:
        result = _fuad_reconstruct(D, labels)
        if result:
            return result

    # Fallback to NJ
    return _fast_nj(D, labels)[0]
