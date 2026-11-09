"""Rewritten program for phylogenetic topology recovery."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from collections import defaultdict

_START_TIME = time.time()
_DEADLINE = _START_TIME + 260.0  # safety margin before 270s kill


def _time_remaining():
    return _DEADLINE - time.time()


def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using numpy."""
    n = len(labels)
    # Convert sequences to integer arrays
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequences[labels[0]])
    A = np.zeros((n, L), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j in range(L):
            A[i, j] = base_to_int[seq[j]]
    
    # Hamming distances
    diff = A[:, None, :] != A[None, :, :]
    hamming = np.sum(diff, axis=2)
    p = hamming.astype(np.float64) / L
    
    # JC correction: d = -3/4 * ln(1 - 4p/3)
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    # Handle edge cases
    d = np.where(np.isfinite(d), d, 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _run_nj(distance_matrix):
    """Run Neighbor Joining on a distance matrix. Returns splits as frozensets."""
    n = len(distance_matrix)
    if n <= 2:
        return []
    
    # Convert to list of lists for easier manipulation
    D = [list(row) for row in distance_matrix]
    active = list(range(n))
    splits = []
    
    while len(active) > 3:
        m = len(active)
        
        # Compute r_i = sum of distances from i to all other active nodes
        r = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += D[active[i]][active[j]]
        
        # Find best pair (i, j) minimizing Q(i,j) = (m-2)*D[i][j] - r[i] - r[j]
        best_val = float('inf')
        best_i, best_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                val = (m - 2) * D[active[i]][active[j]] - r[i] - r[j]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j
        
        # Join active[best_i] and active[best_j]
        ai, aj = active[best_i], active[best_j]
        
        # Create new node
        new_node = n + len(splits)
        splits.append((ai, aj))
        
        # Compute distances from new node to all other active nodes
        new_distances = {}
        for k in range(m):
            if k == best_i or k == best_j:
                continue
            ak = active[k]
            d = 0.5 * (D[ai][ak] + D[aj][ak] - D[ai][aj])
            new_distances[ak] = d
        
        # Build new distance matrix
        new_active = []
        for k in range(m):
            if k == best_i or k == best_j:
                continue
            new_active.append(active[k])
        new_active.append(new_node)
        
        new_D = [[0.0] * len(new_active) for _ in range(len(new_active))]
        for i, ni in enumerate(new_active):
            for j, nj in enumerate(new_active):
                if i == j:
                    new_D[i][j] = 0.0
                elif ni == new_node:
                    new_D[i][j] = new_distances[nj]
                elif nj == new_node:
                    new_D[i][j] = new_distances[ni]
                else:
                    new_D[i][j] = D[ni][nj]
        
        D = new_D
        active = new_active
    
    # For 3 remaining nodes, add the split
    if len(active) == 3:
        splits.append((active[0], active[1]))
    
    return splits


def _extract_splits_from_nj(distance_matrix, labels):
    """Run NJ and return the set of splits (as frozensets of leaf indices)."""
    n = len(labels)
    splits = _run_nj(distance_matrix)
    
    # Convert splits to frozensets of leaf labels
    split_set = set()
    # We need to track which original leaves are on each side
    # This requires tracking the tree structure. Let's use a different approach.
    
    # Actually, let's reconstruct the tree and extract splits properly
    return _nj_splits_with_leaves(distance_matrix, labels)


def _nj_splits_with_leaves(distance_matrix, labels):
    """Run NJ and return splits as frozensets of leaf indices."""
    n = len(labels)
    if n <= 3:
        return set()
    
    D = [list(row) for row in distance_matrix]
    active = list(range(n))
    # Track the set of leaves for each node
    leaf_sets = {i: {i} for i in range(n)}
    splits = set()
    
    while len(active) > 3:
        m = len(active)
        
        # Compute r_i
        r = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += D[active[i]][active[j]]
        
        # Find best pair
        best_val = float('inf')
        best_i, best_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                val = (m - 2) * D[active[i]][active[j]] - r[i] - r[j]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j
        
        ai, aj = active[best_i], active[best_j]
        
        # The split created by joining ai and aj
        # One side is leaf_sets[ai] ∪ leaf_sets[aj], the other is the rest
        side1 = leaf_sets[ai] | leaf_sets[aj]
        side2 = set(range(n)) - side1
        if 0 < len(side1) < n:
            splits.add(frozenset(side1))
        
        # Create new node
        new_node = n + len(splits)
        leaf_sets[new_node] = side1
        
        # Compute distances from new node
        new_distances = {}
        for k in range(m):
            if k == best_i or k == best_j:
                continue
            ak = active[k]
            d = 0.5 * (D[ai][ak] + D[aj][ak] - D[ai][aj])
            new_distances[ak] = d
        
        # Build new structures
        new_active = []
        for k in range(m):
            if k == best_i or k == best_j:
                continue
            new_active.append(active[k])
        new_active.append(new_node)
        
        new_D = [[0.0] * len(new_active) for _ in range(len(new_active))]
        for i, ni in enumerate(new_active):
            for j, nj in enumerate(new_active):
                if i == j:
                    new_D[i][j] = 0.0
                elif ni == new_node:
                    new_D[i][j] = new_distances[nj]
                elif nj == new_node:
                    new_D[i][j] = new_distances[ni]
                else:
                    new_D[i][j] = D[ni][nj]
        
        D = new_D
        active = new_active
    
    # For 3 remaining nodes, add one split
    if len(active) == 3:
        side1 = leaf_sets[active[0]] | leaf_sets[active[1]]
        side2 = set(range(n)) - side1
        if 0 < len(side1) < n:
            splits.add(frozenset(side1))
    
    return splits


def _nj_newick(distance_matrix, labels):
    """Run NJ and return a Newick string."""
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    D = [list(row) for row in distance_matrix]
    active = list(range(n))
    tree_str = {i: labels[i] for i in range(n)}
    
    while len(active) > 3:
        m = len(active)
        
        r = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += D[active[i]][active[j]]
        
        best_val = float('inf')
        best_i, best_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                val = (m - 2) * D[active[i]][active[j]] - r[i] - r[j]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j
        
        ai, aj = active[best_i], active[best_j]
        
        new_node = n + len([k for k in tree_str if k >= n])
        tree_str[new_node] = "(" + tree_str[ai] + "," + tree_str[aj] + ")"
        
        new_distances = {}
        for k in range(m):
            if k == best_i or k == best_j:
                continue
            ak = active[k]
            d = 0.5 * (D[ai][ak] + D[aj][ak] - D[ai][aj])
            new_distances[ak] = d
        
        new_active = []
        for k in range(m):
            if k == best_i or k == best_j:
                continue
            new_active.append(active[k])
        new_active.append(new_node)
        
        new_D = [[0.0] * len(new_active) for _ in range(len(new_active))]
        for i, ni in enumerate(new_active):
            for j, nj in enumerate(new_active):
                if i == j:
                    new_D[i][j] = 0.0
                elif ni == new_node:
                    new_D[i][j] = new_distances[nj]
                elif nj == new_node:
                    new_D[i][j] = new_distances[ni]
                else:
                    new_D[i][j] = D[ni][nj]
        
        D = new_D
        active = new_active
    
    # For 3 remaining nodes
    if len(active) == 3:
        result = "(" + tree_str[active[0]] + "," + tree_str[active[1]] + "," + tree_str[active[2]] + ");"
    else:
        result = "(" + ",".join(tree_str[x] for x in active) + ");"
    
    return result


def _fast_distance_matrix(sequences, labels):
    """Fast distance matrix computation using numpy vectorization."""
    n = len(labels)
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequences[labels[0]])
    
    A = np.zeros((n, L), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j in range(L):
            A[i, j] = base_to_int[seq[j]]
    
    # Hamming distances via broadcasting
    diff = A[:, None, :] != A[None, :, :]
    hamming = np.sum(diff, axis=2)
    p = hamming.astype(np.float64) / L
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    d = np.where(np.isfinite(d), d, 3.0)
    np.fill_diagonal(d, 0.0)
    return d


def _nj_fast(distance_matrix, labels):
    """Optimized NJ returning Newick string."""
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    D = np.array(distance_matrix, dtype=np.float64)
    active = list(range(n))
    tree_str = {i: labels[i] for i in range(n)}
    next_node_id = n
    
    while len(active) > 3:
        m = len(active)
        
        # Compute r vector
        r = np.sum(D[np.ix_(active, active)], axis=1)
        
        # Q matrix: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        sub_D = D[np.ix_(active, active)]
        Q = (m - 2) * sub_D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find minimum
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = idx
        if i > j:
            i, j = j, i
        
        ai, aj = active[i], active[j]
        
        new_node = next_node_id
        next_node_id += 1
        tree_str[new_node] = "(" + tree_str[ai] + "," + tree_str[aj] + ")"
        
        # Compute distances from new node to others
        rest = [k for k in active if k != ai and k != aj]
        if rest:
            d_ai_aj = D[ai, aj]
            new_dists = 0.5 * (D[ai, rest] + D[aj, rest] - d_ai_aj)
        else:
            new_dists = np.array([])
        
        # Build new active list
        new_active = rest + [new_node]
        
        # Build new distance matrix
        new_m = len(new_active)
        new_D = np.zeros((new_m, new_m), dtype=np.float64)
        for idx1, ni in enumerate(new_active):
            for idx2, nj in enumerate(new_active):
                if idx1 == idx2:
                    new_D[idx1, idx2] = 0.0
                elif ni == new_node:
                    if nj == new_node:
                        new_D[idx1, idx2] = 0.0
                    else:
                        # Find position of nj in rest
                        pos = rest.index(nj) if nj in rest else -1
                        if pos >= 0:
                            new_D[idx1, idx2] = new_dists[pos]
                elif nj == new_node:
                    pos = rest.index(ni) if ni in rest else -1
                    if pos >= 0:
                        new_D[idx1, idx2] = new_dists[pos]
                else:
                    new_D[idx1, idx2] = D[ni, nj]
        
        # Extend D to include new node
        old_n = D.shape[0]
        D_ext = np.zeros((old_n + 1, old_n + 1), dtype=np.float64)
        D_ext[:old_n, :old_n] = D
        D_ext[old_n, :old_n] = new_dists
        D_ext[:old_n, old_n] = new_dists
        D_ext[old_n, old_n] = 0.0
        D = D_ext
        
        active = new_active
    
    if len(active) == 3:
        result = "(" + tree_str[active[0]] + "," + tree_str[active[1]] + "," + tree_str[active[2]] + ");"
    else:
        result = "(" + ",".join(tree_str[x] for x in active) + ");"
    
    return result


def _nj_splits_fast(distance_matrix, labels):
    """Optimized NJ returning splits as frozensets."""
    n = len(labels)
    if n <= 3:
        return set()
    
    D = np.array(distance_matrix, dtype=np.float64)
    active = list(range(n))
    leaf_sets = {i: frozenset({i}) for i in range(n)}
    splits = set()
    next_node_id = n
    
    while len(active) > 3:
        m = len(active)
        
        r = np.sum(D[np.ix_(active, active)], axis=1)
        sub_D = D[np.ix_(active, active)]
        Q = (m - 2) * sub_D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = idx
        if i > j:
            i, j = j, i
        
        ai, aj = active[i], active[j]
        
        side1 = leaf_sets[ai] | leaf_sets[aj]
        side2 = frozenset(set(range(n)) - set(side1))
        if 0 < len(side1) < n:
            splits.add(side1)
        
        new_node = next_node_id
        next_node_id += 1
        leaf_sets[new_node] = side1
        
        rest = [k for k in active if k != ai and k != aj]
        if rest:
            d_ai_aj = D[ai, aj]
            new_dists = 0.5 * (D[ai, rest] + D[aj, rest] - d_ai_aj)
        else:
            new_dists = np.array([])
        
        new_active = rest + [new_node]
        
        old_n = D.shape[0]
        D_ext = np.zeros((old_n + 1, old_n + 1), dtype=np.float64)
        D_ext[:old_n, :old_n] = D
        D_ext[old_n, :old_n] = new_dists
        D_ext[:old_n, old_n] = new_dists
        D_ext[old_n, old_n] = 0.0
        D = D_ext
        
        active = new_active
    
    if len(active) == 3:
        side1 = leaf_sets[active[0]] | leaf_sets[active[1]]
        side2 = frozenset(set(range(n)) - set(side1))
        if 0 < len(side1) < n:
            splits.add(side1)
    
    return splits


def _score_splits(candidate_splits, reference_splits):
    """Compute normalized Robinson-Foulds distance between two split sets."""
    if not reference_splits:
        return 0.0
    
    # Normalize: only keep non-trivial splits
    def normalize(splits):
        n = max(max(s) for s in splits) + 1 if splits else 0
        norm = set()
        for s in splits:
            if 0 < len(s) < n:
                norm.add(frozenset(s))
        return norm
    
    cand = normalize(candidate_splits)
    ref = normalize(reference_splits)
    
    if not ref:
        return 0.0
    
    # RF distance = |cand - ref| + |ref - cand|
    # But we need to be careful: a split and its complement are the same
    # Our splits are already canonicalized as frozensets, but we need to ensure
    # we don't count both a split and its complement
    
    # Actually, for unrooted trees, a split {A, B} where A ∪ B = all leaves
    # is the same as {B, A}. We represent each split by the smaller side
    # or just by the frozenset. Since we're comparing sets, we need consistency.
    
    # Let's use a canonical form: for each split, represent it by the frozenset
    # where we always pick the side with the smaller index first, or just use
    # the frozenset as-is. Since both candidate and reference use the same
    # leaf indices, direct set comparison works.
    
    symmetric_diff = cand.symmetric_difference(ref)
    total = len(ref)
    
    if total == 0:
        return 0.0
    
    return len(symmetric_diff) / (2.0 * total)


def _random_tree_splits(n, rng):
    """Generate random splits for a random tree (for testing/sanity)."""
    # Simple random split generation
    splits = set()
    for _ in range(2 * (n - 3)):
        size = rng.randint(1, n - 1)
        leaves = rng.sample(range(n), size)
        splits.add(frozenset(leaves))
    return splits


def _parsimony_tree(sequences, labels):
    """Simple parsimony-based tree construction (Fitch algorithm on a starting tree)."""
    # This is a placeholder - for now, just use NJ
    return None


def reconstruct(sequences):
    """Reconstruct phylogenetic tree from sequences. Returns Newick string."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    
    # Compute distance matrix
    dist_matrix = _fast_distance_matrix(sequences, labels)
    
    # Run NJ to get initial tree
    newick = _nj_fast(dist_matrix, labels)
    
    # Check time
    if _time_remaining() < 10:
        return newick
    
    # Try to improve with some refinement
    # For now, return NJ result
    return newick
# EVOLVE-BLOCK-END
