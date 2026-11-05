"""
Phylogenetic topology recovery using vectorized Neighbor Joining.
Combines fast NumPy-based distance matrix computation with an efficient NJ
implementation that uses NumPy operations where possible and a strict
deadline guard.
"""
import time
import math
import numpy as np

# Global deadline: 270s wall-clock limit, use 260s as safety margin
_DEADLINE = time.time() + 260.0


def _time_exceeded():
    return time.time() > _DEADLINE


# Lookup table for fast ASCII -> int mapping
_LOOKUP = np.zeros(256, dtype=np.int8)
_LOOKUP[ord('A')] = 0
_LOOKUP[ord('C')] = 1
_LOOKUP[ord('G')] = 2
_LOOKUP[ord('T')] = 3
_LOOKUP[ord('a')] = 0
_LOOKUP[ord('c')] = 1
_LOOKUP[ord('g')] = 2
_LOOKUP[ord('t')] = 3


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distance matrix using vectorized NumPy."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    seqs = [sequences[label] for label in labels]
    L = len(seqs[0]) if seqs else 0

    if L == 0:
        return np.zeros((n, n), dtype=np.float64)

    # Convert sequences to int8 array (n, L)
    seq_array = np.empty((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        b = s.encode('ascii')
        seq_array[i] = _LOOKUP[np.frombuffer(b, dtype=np.uint8)]

    # Compute pairwise Hamming distances via broadcasting
    # diff_matrix[i, j] = number of positions where seq_array[i] != seq_array[j]
    diff_matrix = np.sum(seq_array[:, None, :] != seq_array[None, :, :], axis=2)

    # Convert to proportions
    P = diff_matrix / L

    # Apply JC correction: d = -0.75 * ln(1 - 4/3 * p)
    # For p >= 0.75, the log term is undefined; cap distance at 3.0
    mask = P < 0.75
    D = np.zeros((n, n), dtype=np.float64)

    # Calculate for valid entries
    term = 1.0 - (4.0 / 3.0) * P[mask]
    # Avoid log(0) or log(negative) due to floating point issues
    term = np.maximum(term, 1e-15)
    D[mask] = -0.75 * np.log(term)

    # Set invalid entries (p >= 0.75) to 3.0
    D[~mask] = 3.0

    # Diagonal should be 0
    np.fill_diagonal(D, 0.0)

    return D


def _neighbor_joining(D, labels):
    """
    Perform Neighbor Joining on distance matrix D.
    Returns a Newick string.
    D is an n x n numpy array, labels is a list of n strings.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"

    dist_matrix = D.copy()
    tree_substrings = list(labels)
    num_nodes = n

    while num_nodes > 3:
        if _time_exceeded():
            break

        m = num_nodes
        # Compute total distances (row sums)
        r = np.sum(dist_matrix, axis=1)

        # Find the pair that minimizes the NJ criterion
        # Q(i, j) = (m - 2) * d(i, j) - r(i) - r(j)
        # We want to find min Q over i < j

        # Vectorized approach: compute Q matrix
        # Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        # We can compute the full matrix and then find the minimum off-diagonal
        # But we only need i < j, so we can use the upper triangle

        # Compute Q for all pairs
        Q = (m - 2) * dist_matrix - r[:, None] - r[None, :]

        # Set diagonal to inf so it's not selected
        np.fill_diagonal(Q, np.inf)

        # We need min over i < j, but Q is symmetric so min over all off-diagonal works
        # Find the minimum value and its indices
        min_val = np.min(Q)
        # Get indices of the minimum
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = idx[0], idx[1]

        # Ensure i < j for consistency
        if i > j:
            i, j = j, i

        # Compute branch lengths
        delta = (r[i] - r[j]) / (2.0 * (m - 2))
        d_ij = dist_matrix[i, j]
        len_i = 0.5 * d_ij + 0.5 * delta
        len_j = 0.5 * d_ij - 0.5 * delta

        # Ensure non-negative
        if len_i < 0:
            len_i = 0.0
        if len_j < 0:
            len_j = 0.0

        # Create new node
        new_name = f"({tree_substrings[i]}:{len_i:.6f},{tree_substrings[j]}:{len_j:.6f})"

        # Compute distances from new node to all other nodes
        # Newick's formula: d(new, k) = 0.5 * (d(i,k) + d(j,k) - d(i,j))
        # Keep track of which nodes remain
        new_active = []
        new_distances = np.zeros(m - 2)
        for k in range(m):
            if k == i or k == j:
                continue
            new_distances[len(new_active)] = 0.5 * (dist_matrix[i, k] + dist_matrix[j, k] - d_ij)
            new_active.append(k)

        # Build new distance matrix
        # The new matrix has (m-1) x (m-1) size
        # The last row/column corresponds to the new node
        new_dist = np.zeros((m - 1, m - 1), dtype=np.float64)

        # Copy distances between remaining nodes
        for idx, k in enumerate(new_active):
            for idx2, k2 in enumerate(new_active):
                new_dist[idx, idx2] = dist_matrix[k, k2]
            new_dist[idx, m - 2] = new_distances[idx]
            new_dist[m - 2, idx] = new_distances[idx]

        # Update structures
        tree_substrings = [tree_substrings[k] for k in new_active] + [new_name]
        dist_matrix = new_dist
        num_nodes = m - 1

    # Combine remaining nodes
    if num_nodes == 3:
        # For 3 nodes, compute branch lengths for the star
        len_a = 0.5 * (dist_matrix[0, 1] + dist_matrix[0, 2] - dist_matrix[1, 2])
        len_b = 0.5 * (dist_matrix[0, 1] + dist_matrix[1, 2] - dist_matrix[0, 2])
        len_c = 0.5 * (dist_matrix[0, 2] + dist_matrix[1, 2] - dist_matrix[0, 1])
        if len_a < 0:
            len_a = 0.0
        if len_b < 0:
            len_b = 0.0
        if len_c < 0:
            len_c = 0.0
        result = f"({tree_substrings[0]}:{len_a:.6f},{tree_substrings[1]}:{len_b:.6f},{tree_substrings[2]}:{len_c:.6f})"
    elif num_nodes == 2:
        half_d = dist_matrix[0, 1] / 2.0
        result = f"({tree_substrings[0]}:{half_d:.6f},{tree_substrings[1]}:{half_d:.6f})"
    else:
        result = "(" + ",".join(tree_substrings) + ")"

    return result + ";"


def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Returns a Newick string.
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"

    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)

    # Run Neighbor Joining
    result = _neighbor_joining(D, labels)

    return result
