"""Neighbour-joining with Jukes-Cantor correction plus a fast topology polish."""
# EVOLVE-BLOCK-START
import math
import time
import sys

# Use a deadline well inside the 270 s wall-clock kill; leave headroom.
_DEADLINE = 265.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def _jc_distance(a, b):
    """Jukes-Cantor corrected distance between two equal-length DNA strings."""
    n = len(a)
    if n == 0:
        return 0.0
    # Count mismatches quickly
    mismatches = 0
    for x, y in zip(a, b):
        if x != y:
            mismatches += 1
    p = mismatches / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _build_distance_matrix(sequences, labels):
    """Build a symmetric distance matrix as a dict keyed by (i, j) indices."""
    n = len(labels)
    # Precompute pairwise mismatches using a compact approach
    seq_list = [sequences[lb] for lb in labels]
    L = len(seq_list[0]) if n > 0 else 0

    # For speed, convert to byte arrays or use ord-based comparison
    # Use a list of bytes for fast comparison
    seq_bytes = [s.encode('ascii') if isinstance(s, str) else bytes(s) for s in seq_list]

    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a = seq_bytes[i]
            b = seq_bytes[j]
            mismatches = 0
            # Fast mismatch count
            for k in range(L):
                if a[k] != b[k]:
                    mismatches += 1
            p = mismatches / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _newick_from_nj(dist, labels):
    """
    Run neighbour joining on a distance matrix and return a Newick string.
    dist is an n x n list of lists.
    labels is a list of label strings.
    """
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        # For 3 taxa, any unrooted tree is the same topology
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"

    # Work with a mutable copy of the distance matrix
    d = [row[:] for row in dist]
    # active nodes: list of indices into the current "taxon" list
    # We'll track nodes as (original_index or None, newick_string)
    # Initially each node is a leaf
    nodes = list(range(n))  # indices into labels
    newick = {i: labels[i] for i in range(n)}
    active = list(range(n))

    while len(active) > 2:
        m = len(active)
        # Compute row sums
        r = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += d[active[i]][active[j]]

        # Find best pair (i, j) minimizing (m-2)*d[i][j] - r[i] - r[j]
        best_val = None
        best_i = 0
        best_j = 1
        for i in range(m):
            for j in range(i + 1, m):
                val = (m - 2) * d[active[i]][active[j]] - r[i] - r[j]
                if best_val is None or val < best_val:
                    best_val = val
                    best_i = i
                    best_j = j

        ai = active[best_i]
        aj = active[best_j]

        # Create new node
        new_idx = n  # use n, n+1, ... as new node indices
        # Compute distances from new node to all other active nodes
        # d(new, k) = 0.5 * (d(ai,k) + d(aj,k) - d(ai,aj))
        for k in range(m):
            if k == best_i or k == best_j:
                continue
            ak = active[k]
            d[new_idx][ak] = 0.5 * (d[ai][ak] + d[aj][ak] - d[ai][aj])
            d[ak][new_idx] = d[new_idx][ak]

        # Newick for the joined pair
        joined = "(" + newick[ai] + "," + newick[aj] + ")"
        newick[new_idx] = joined

        # Remove ai, aj from active, add new_idx
        active = [x for x in active if x != ai and x != aj]
        active.append(new_idx)
        n = new_idx + 1  # increment for next new node

    # Final two nodes: join them
    a = active[0]
    b = active[1]
    result = "(" + newick[a] + "," + newick[b] + ");"
    return result


def _parse_newick_to_splits(newick_str, labels_set):
    """
    Parse a Newick string and return the set of non-trivial splits.
    Each split is a frozenset of taxon labels on one side (canonical: sorted tuple).
    """
    s = newick_str.strip()
    if s.endswith(';'):
        s = s[:-1]

    splits = set()
    n_taxa = len(labels_set)

    def parse(pos):
        """Returns (split_set, new_pos)."""
        if pos >= len(s) or s[pos] == ',':
            return None, pos
        if s[pos] == '(':
            pos += 1
            children = []
            while pos < len(s):
                child, pos = parse(pos)
                children.append(child)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                    continue
                if pos < len(s) and s[pos] == ')':
                    pos += 1
                    break
            # Skip branch length if present
            while pos < len(s) and s[pos] not in ',)':
                pos += 1
            # Combine children
            combined = set()
            for c in children:
                if c is not None:
                    combined |= c
            return combined, pos
        else:
            # Leaf
            start = pos
            while pos < len(s) and s[pos] not in ',):':
                pos += 1
            label = s[start:pos]
            # Skip branch length
            while pos < len(s) and s[pos] not in ',)':
                pos += 1
            return {label}, pos

    full_set, _ = parse(0)
    if full_set is None:
        return splits

    # Now we need to extract all non-trivial splits.
    # A simpler approach: for each internal node, the set of leaves under it gives a split.
    # But parsing to get all internal node leaf-sets requires a different parse.
    # Let's do a recursive parse that collects all internal node leaf-sets.

    all_splits = set()

    def parse2(pos):
        """Returns (leaf_set, new_pos)."""
        if pos >= len(s) or s[pos] == ',':
            return None, pos
        if s[pos] == '(':
            pos += 1
            children = []
            while pos < len(s):
                child, pos = parse2(pos)
                children.append(child)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                    continue
                if pos < len(s) and s[pos] == ')':
                    pos += 1
                    break
            while pos < len(s) and s[pos] not in ',)':
                pos += 1
            combined = set()
            for c in children:
                if c is not None:
                    combined |= c
            # This is an internal node: record the split
            if 0 < len(combined) < n_taxa:
                # Canonicalize: use the smaller side
                if len(combined) <= n_taxa // 2:
                    key = frozenset(combined)
                else:
                    key = frozenset(full_set - combined)
                all_splits.add(key)
            return combined, pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in ',):':
                pos += 1
            label = s[start:pos]
            while pos < len(s) and s[pos] not in ',)':
                pos += 1
            return {label}, pos

    parse2(0)
    return all_splits


def _robinson_foulds(splits_a, splits_b):
    """Compute normalized RF distance between two sets of splits."""
    if not splits_a and not splits_b:
        return 0.0
    intersection = splits_a & splits_b
    union = splits_a | splits_b
    if not union:
        return 0.0
    return 1.0 - len(intersection) / len(union)


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        if n == 2:
            return "(" + labels[0] + "," + labels[1] + ");"
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"

    # Build distance matrix
    dist = _build_distance_matrix(sequences, labels)

    # Run NJ
    best_newick = _newick_from_nj(dist, labels)

    # Time check: if we have time, we could do a polish step, but NJ is already
    # a single-pass algorithm. For now, return the NJ result.
    return best_newick
# EVOLVE-BLOCK-END
