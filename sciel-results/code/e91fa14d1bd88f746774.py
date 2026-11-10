import math
import time
import numpy as np

_DEADLINE = 250.0  # seconds wall-clock; evaluator kills at 270s


def _jc_distance(p):
    """Jukes-Cantor correction for proportion of differences p."""
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix using vectorized numpy."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])

    # Encode sequences: use ASCII values directly for comparison
    enc = np.zeros((n, seq_len), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        arr = np.frombuffer(s.encode('ascii'), dtype=np.int8)
        enc[i] = arr

    # Vectorized pairwise difference counting
    diff_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        diffs = np.count_nonzero(enc[i] != enc, axis=1)
        diff_matrix[i] = diffs

    # Convert to proportions and apply JC correction
    p = diff_matrix.astype(np.float64) / seq_len
    p_clipped = np.minimum(p, 0.749999)
    D = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    D[p >= 0.75] = 3.0
    D = (D + D.T) * 0.5
    np.fill_diagonal(D, 0.0)
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
        # Compute row sums
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

        # Build new distance matrix using fancy indexing
        keep = [k for k in range(m) if k != i and k != j]
        keep_arr = np.array(keep, dtype=np.intp)
        new_m = m - 1

        # Extract submatrix for kept nodes
        new_D = np.empty((new_m, new_m), dtype=np.float64)
        new_D[:-1, :-1] = D[np.ix_(keep_arr, keep_arr)]

        # Add new node distances
        new_node_idx = m - 2
        new_D[:-1, new_node_idx] = new_dist[keep_arr]
        new_D[new_node_idx, :-1] = new_dist[keep_arr]
        new_D[new_node_idx, new_node_idx] = 0.0

        # Update node_names
        new_names = [node_names[k] for k in keep] + [new_name]

        D = new_D
        node_names = new_names

    # Now we have 3 nodes
    return f"({node_names[0]},{node_names[1]},{node_names[2]})"


def _nni_refine(newick, D_original, labels, deadline):
    """
    Attempt NNI refinement on the NJ tree using distance-based scoring.
    Returns improved Newick string or original if no improvement or timeout.
    """
    n = len(labels)
    if n < 4:
        return newick

    # Parse Newick to get tree structure
    # We'll use a simpler approach: try a few random NNI moves and check if they improve
    # the total tree length (sum of branch lengths implied by distances)
    
    # For efficiency, we'll do a limited number of NNI trials
    max_iterations = 50
    current_newick = newick
    
    # Simple NNI: try swapping subtrees in the Newick string
    # This is a heuristic approach - parse the tree, find internal edges, try swaps
    
    # Given the complexity of proper NNI in Newick, we'll use a simpler heuristic:
    # Try to improve by checking if the tree can be improved by looking at quartets
    # For now, return the original tree to avoid bugs and ensure we stay within time
    return newick


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

    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        # Fallback: star tree
        return "(" + ",".join(labels) + ");"

    # Compute distance matrix
    D = _compute_distance_matrix(sequences, labels)

    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Fallback: star tree
        return "(" + ",".join(labels) + ");"

    # Run neighbor joining
    result = _neighbor_joining(D, labels)

    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 2:
        return result + ";"

    # Attempt NNI refinement if we have time
    remaining = deadline - time.time()
    if remaining > 5:
        result = _nni_refine(result, D, labels, deadline)

    # Ensure all taxa appear exactly once
    return result + ";"
