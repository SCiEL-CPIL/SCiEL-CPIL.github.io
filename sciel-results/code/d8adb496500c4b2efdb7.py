"""Reconstruct phylogenetic topology from Jukes-Cantor simulated sequences."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 255.0  # seconds wall-clock; evaluator kills at 270s


def _jc_distance(p):
    """Jukes-Cantor correction for proportion of differences p."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix using numpy."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])

    # Encode sequences as numpy arrays of ASCII codes for fast comparison
    enc = np.zeros((n, seq_len), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        enc[i] = np.frombuffer(s.encode('ascii'), dtype=np.int8)

    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.count_nonzero(enc[i] != enc[j])
            p = diffs / seq_len
            D[i, j] = D[j, i] = _jc_distance(p)
    return D


def _neighbor_joining(D, labels):
    """
    Perform neighbor joining on a distance matrix D (numpy array, symmetric, zero diag).
    labels: list of taxon names corresponding to rows/cols of D.
    Returns a Newick string (no trailing semicolon).
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    D = D.astype(np.float64).copy()
    node_names = list(labels)

    while len(node_names) > 3:
        m = len(node_names)
        # Compute row sums (only among active nodes)
        r = D.sum(axis=1)  # shape (m,)

        # Q matrix: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        Q = (m - 2) * D - r[:, None] - r[None, :]
        # Set diagonal to inf
        np.fill_diagonal(Q, np.inf)
        # Find min
        idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = idx
        if i > j:
            i, j = j, i

        # Create new node
        new_name = f"({node_names[i]},{node_names[j]})"

        # Compute new distances
        new_dist = 0.5 * (D[i, :] + D[j, :] - D[i, j])

        # Remove rows/cols i and j, add new row/col
        keep = [k for k in range(m) if k != i and k != j]
        new_m = m - 1
        new_D = np.zeros((new_m, new_m), dtype=np.float64)
        for a, ia in enumerate(keep):
            for b, ib in enumerate(keep):
                new_D[a, b] = D[ia, ib]
        # Add new node at position new_m (last)
        new_node_idx = m - 2
        for a, ia in enumerate(keep):
            new_D[a, new_node_idx] = new_dist[ia]
            new_D[new_node_idx, a] = new_dist[ia]
        new_D[new_node_idx, new_node_idx] = 0.0

        # Update node_names
        new_names = [node_names[k] for k in keep] + [new_name]

        D = new_D
        node_names = new_names

    # Now we have 3 nodes
    return f"({node_names[0]},{node_names[1]},{node_names[2]})"


def _nj_with_deadline(sequences, labels, deadline):
    """Run NJ with deadline guard. Returns Newick string or None if timeout."""
    start = time.time()
    D = _compute_distance_matrix(sequences, labels)
    elapsed = time.time() - start
    if elapsed > deadline - 5:
        return None
    return _neighbor_joining(D, labels)


def reconstruct(sequences):
    """
    Reconstruct unrooted phylogenetic tree topology.
    Returns Newick string over labels t0..t(n-1).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"

    # Try neighbor joining
    result = _nj_with_deadline(sequences, labels, deadline - start_time)
    if result is not None:
        return result + ";"

    # Fallback: star tree (should not happen given time budget)
    return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
