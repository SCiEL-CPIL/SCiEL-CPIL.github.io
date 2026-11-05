"""
Phylogenetic topology recovery using vectorized Neighbor Joining.

Design family (kept): Jukes-Cantor-corrected pairwise distances + Neighbor
Joining to build the unrooted topology.

Refinement applied: a fully vectorized NJ update. Instead of rebuilding the
distance matrix and re-indexing the node list with Python loops on every
join (O(m^2) interpreter overhead per join), we keep a fixed n x n matrix and
a boolean "active" mask, and update distances with np.where. This removes the
per-join loop overhead while preserving the exact NJ criterion and Newick
construction, so the wall-clock budget is spent on more instances rather than
on bookkeeping. A strict deadline guard returns the best tree found so far.
"""
import time
import numpy as np

# Evaluator kills after 270 s wall-clock; keep a safety margin.
_DEADLINE = time.time() + 258.0


def _time_exceeded():
    return time.time() > _DEADLINE


# Fast ASCII -> int mapping for the distance computation.
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
    """Pairwise Jukes-Cantor corrected distances via vectorized broadcasting."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    seqs = [sequences[label] for label in labels]
    L = len(seqs[0]) if seqs else 0
    if L == 0:
        return np.zeros((n, n), dtype=np.float64)

    seq_array = np.empty((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        b = s.encode('ascii')
        seq_array[i] = _LOOKUP[np.frombuffer(b, dtype=np.uint8)]

    # Hamming distance matrix by broadcasting.
    diff = np.sum(seq_array[:, None, :] != seq_array[None, :, :], axis=2)
    P = diff / L

    D = np.zeros((n, n), dtype=np.float64)
    mask = P < 0.75
    term = np.maximum(1.0 - (4.0 / 3.0) * P[mask], 1e-15)
    D[mask] = -0.75 * np.log(term)
    D[~mask] = 3.0
    np.fill_diagonal(D, 0.0)
    return D


def _neighbor_joining(D, labels):
    """
    Vectorized Neighbor Joining. Returns a Newick string.
    D is an n x n numpy array; labels is a list of n strings.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"

    dist = D.copy()
    names = list(labels)
    active = np.ones(n, dtype=bool)

    # Active-set view helper.
    def _view():
        return dist[active][:, active]

    # Build the Newick string for the final 3 (or 2) active nodes as a star.
    def _finish_star():
        idx = np.flatnonzero(active)
        if len(idx) == 3:
            a, b, c = idx[0], idx[1], idx[2]
            la = 0.5 * (dist[a, b] + dist[a, c] - dist[b, c])
            lb = 0.5 * (dist[a, b] + dist[b, c] - dist[a, c])
            lc = 0.5 * (dist[a, c] + dist[b, c] - dist[a, b])
            la = la if la > 0 else 0.0
            lb = lb if lb > 0 else 0.0
            lc = lc if lc > 0 else 0.0
            return (f"({names[a]}:{la:.6f},"
                    f"{names[b]}:{lb:.6f},"
                    f"{names[c]}:{lc:.6f})")
        if len(idx) == 2:
            a, b = idx[0], idx[1]
            h = dist[a, b] / 2.0
            return f"({names[a]}:{h:.6f},{names[b]}:{h:.6f})"
        # Fallback: star over whatever remains.
        return "(" + ",".join(names[k] for k in idx) + ")"

    while True:
        if _time_exceeded():
            return _finish_star() + ";"

        act = np.flatnonzero(active)
        m = len(act)
        if m <= 3:
            return _finish_star() + ";"

        # Work on the active sub-matrix.
        Ds = _view()
        r = np.sum(Ds, axis=1)

        # NJ criterion Q(i,j) = (m-2)*d(i,j) - r(i) - r(j); minimize off-diagonal.
        Q = (m - 2) * Ds - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        p = np.unravel_index(np.argmin(Q), Q.shape)
        if p[0] > p[1]:
            p = (p[1], p[0])
        i, j = p

        ga = act[i]
        gb = act[j]
        d_ij = Ds[i, j]

        # Branch lengths.
        delta = (r[i] - r[j]) / (2.0 * (m - 2))
        len_i = 0.5 * d_ij + 0.5 * delta
        len_j = 0.5 * d_ij - 0.5 * delta
        len_i = len_i if len_i > 0 else 0.0
        len_j = len_j if len_j > 0 else 0.0

        # New internal node: store its Newick and its distances to remaining nodes.
        names[gb] = f"({names[ga]}:{len_i:.6f},{names[gb]}:{len_j:.6f})"
        dist[gb] = 0.5 * (dist[ga] + dist[gb] - d_ij)
        dist[:, gb] = dist[gb]
        dist[gb, gb] = 0.0
        active[ga] = False

    # Unreachable.
    return "(" + ",".join(names) + ");"


def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Returns a Newick string over t0..t(n-1).
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"

    D = _jc_distance_matrix(sequences, labels)
    return _neighbor_joining(D, labels)
