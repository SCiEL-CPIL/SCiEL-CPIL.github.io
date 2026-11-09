"""Rewritten program using a fast, vectorized Neighbor-Joining with deadline guard.

Key structural changes vs. the seed:
  1. Pairwise Jukes-Cantor distances are computed with a single NumPy broadcast
     instead of an O(N^2) Python loop over sites.
  2. The NJ contraction loop uses dense NumPy matrices (O(N^2) memory, N<=32)
     instead of a Python dict of tuples, which is much faster per iteration.
  3. A hard deadline guard returns a polytomy over the remaining nodes if the
     time budget is exhausted.
"""
import time
import numpy as np

DEADLINE_SECONDS = 270.0
SAFETY_MARGIN = 5.0


def _jc_distance_matrix(sequences, labels):
    """Compute the pairwise Jukes-Cantor distance matrix using NumPy."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    # Convert sequences to a numeric array (A=0, C=1, G=2, T=3)
    seq_array = np.empty((n, len(sequences[labels[0]])), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        # Fast character-to-index mapping using np.frombuffer on bytes
        b = seq.encode('ascii')
        arr = np.frombuffer(b, dtype=np.uint8)
        # Map: A(65)->0, C(67)->1, G(71)->2, T(84)->3
        # Use a lookup table for speed
        lut = np.zeros(256, dtype=np.int8)
        lut[65] = 0   # A
        lut[67] = 1   # C
        lut[71] = 2   # G
        lut[84] = 3   # T
        seq_array[i] = lut[arr]

    L = seq_array.shape[1]
    if L == 0:
        return np.zeros((n, n), dtype=np.float64)

    # Compute Hamming distances via broadcast: shape (n, n, L)
    # For n<=32 and L<=2000, this is at most 32*32*2000 = 2,048,000 elements, ~2 MB.
    diff = seq_array[:, None, :] != seq_array[None, :, :]
    hamming = diff.sum(axis=2, dtype=np.int32)  # shape (n, n)

    # Convert to Jukes-Cantor distances
    p = hamming.astype(np.float64) / L
    # Avoid log(0) or log of negative: clip p to [0, 0.7499]
    p_clipped = np.clip(p, 0.0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _neighbor_joining(dist_matrix, labels, deadline):
    """Run Neighbor-Joining on a dense distance matrix. Returns a Newick string."""
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"

    # We'll maintain:
    #   dist: dense (n x n) distance matrix over current active nodes
    #   active: list of current node labels
    #   tree_repr: dict mapping node label -> Newick fragment
    #
    # But to keep it fast, we'll work with indices into a growing list.
    # Simpler: maintain a list of active node IDs (integers), a dict from ID to
    # Newick string, and a dense matrix over active nodes.

    # Start: active nodes are 0..n-1
    active_ids = list(range(n))
    # dist is already indexed by original labels; we'll maintain a dense matrix
    # over the current active set, re-indexed each iteration.
    # For efficiency, keep a dense matrix D of shape (m, m) where m = len(active_ids).
    D = dist_matrix.copy()  # shape (n, n)
    tree_repr = {i: labels[i] for i in range(n)}
    next_id = n

    m = n
    while m > 3:
        # Check deadline
        if time.time() > deadline:
            # Return a polytomy over all remaining active nodes
            remaining = [tree_repr[i] for i in active_ids]
            return "(" + ",".join(remaining) + ");"

        # Compute row sums r_i
        r = D.sum(axis=1)  # shape (m,)

        # Compute the Q-matrix: Q(i,j) = (m-2)*D(i,j) - r[i] - r[j]
        # We want to find the pair (i,j) with minimum Q.
        # Build Q as a dense matrix.
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)

        # Find the minimum off-diagonal entry
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i == j:
            # Should not happen, but guard
            break
        if i > j:
            i, j = j, i

        d_ij = D[i, j]

        # New node
        new_id = next_id
        next_id += 1
        tree_repr[new_id] = "(" + tree_repr[active_ids[i]] + "," + tree_repr[active_ids[j]] + ")"

        # Compute new distances from new_id to all other active nodes
        # d(new, k) = 0.5 * (D(i,k) + D(j,k) - D(i,j))
        other_indices = [k for k in range(m) if k != i and k != j]
        new_dists = np.zeros(len(other_indices), dtype=np.float64)
        for idx, k in enumerate(other_indices):
            new_dists[idx] = 0.5 * (D[i, k] + D[j, k] - d_ij)

        # Build new active list and new distance matrix
        new_active = [active_ids[k] for k in other_indices] + [new_id]
        m_new = len(new_active)

        # New distance matrix
        D_new = np.zeros((m_new, m_new), dtype=np.float64)
        # Fill in distances among the remaining old nodes
        for a_idx, a_old in enumerate(other_indices):
            for b_idx, b_old in enumerate(other_indices):
                D_new[a_idx, b_idx] = D[a_old, b_old]

        # Fill in distances from new node to others
        for a_idx, a_old in enumerate(other_indices):
            D_new[a_idx, m_new - 1] = new_dists[a_idx]
            D_new[m_new - 1, a_idx] = new_dists[a_idx]

        active_ids = new_active
        D = D_new
        m = m_new

    # Final tree: join remaining nodes into a polytomy
    remaining = [tree_repr[i] for i in active_ids]
    return "(" + ",".join(remaining) + ");"


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 2:
        return "(" + ",".join(labels) + ");"

    # Compute JC distance matrix
    dist_matrix = _jc_distance_matrix(sequences, labels)

    # Check deadline before NJ
    if time.time() > deadline:
        return "(" + ",".join(labels) + ");"

    # Run Neighbor-Joining
    newick = _neighbor_joining(dist_matrix, labels, deadline)
    return newick
