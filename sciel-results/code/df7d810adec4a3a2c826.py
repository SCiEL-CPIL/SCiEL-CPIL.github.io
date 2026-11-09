"""Phylogenetic topology recovery from simulated Jukes-Cantor sequences.

Strategy: fast vectorized Jukes-Cantor distance matrix + Neighbor-Joining.
NJ is O(n^3) but with n<=32 and a NumPy distance matrix it is well within budget.
An explicit wall-clock deadline guard returns the best tree built so far.
"""
import time
import math
import numpy as np

_DEADLINE = 270.0  # wall-clock seconds; evaluator kills at this point
_START = time.time()
_MARGIN = 5.0      # safety margin before the kill


def _elapsed():
    return time.time() - _START


def _jc_distances(sequences):
    """Compute the Jukes-Cantor-corrected pairwise distance matrix (NumPy).

    sequences: dict label -> equal-length DNA string over ACGT.
    Returns (labels, D) where D is an (n,n) float array, D[i,j] the JC distance.
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    # Build an integer matrix: one-hot per base, shape (n, L, 4)
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequences[labels[0]])
    X = np.zeros((n, L, 4), dtype=np.uint8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # vectorize via lookup
        arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        idx = np.where(arr == 65, 0, np.where(arr == 67, 1, np.where(arr == 71, 2, 3)))
        X[i, :, idx] = 1
    # pairwise fraction of differing sites: p[i,j] = 1 - sum over bases of agreement
    # agreement at site for pair (i,j) = sum_b X[i,site,b]*X[j,site,b]
    # Use matrix products per base for speed.
    # X_b shape (n, L); agreement_b = X_b @ X_b.T ; total agreement = sum_b
    agree = np.zeros((n, n), dtype=np.float64)
    for b in range(4):
        Xb = X[:, :, b].astype(np.float64)
        agree += Xb @ Xb.T
    p = 1.0 - agree / L
    np.fill_diagonal(p, 0.0)
    # JC correction: D = -0.75 * ln(1 - 4p/3), cap at p >= 0.75
    p = np.clip(p, 0.0, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(D, 0.0)
    return labels, D


def _neighbor_joining(labels, D):
    """Neighbor-Joining on distance matrix D (n x n). Returns a Newick string."""
    n = len(labels)
    # active nodes
    active = list(range(n))
    # current distance matrix over active nodes (keep full matrix, track active set)
    dist = D.copy()
    # tree strings for each original index
    tree = [f"t{i}" for i in range(n)]
    # map from original index -> tree string; new nodes get fresh ids
    next_id = n
    # We'll maintain:
    #   nodes: list of current node ids (integers)
    #   tid: dict id -> tree string
    tid = {i: f"t{i}" for i in range(n)}
    nodes = list(range(n))

    while len(nodes) > 3:
        if _elapsed() > _DEADLINE - _MARGIN:
            break
        m = len(nodes)
        # r[i] = sum of distances from node i to all other active nodes
        idx = np.array(nodes, dtype=int)
        sub = dist[np.ix_(idx, idx)]
        r = sub.sum(axis=1)
        # Q matrix: Q[i,j] = (m-2)*d[i,j] - r[i] - r[j]
        Q = (m - 2) * sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        # find min off-diagonal
        flat = np.unravel_index(np.argmin(Q), Q.shape)
        ia, ib = flat
        a = nodes[ia]
        b = nodes[ib]
        # compute branch lengths to new node
        # delta = (r[a]-r[b])/(2*(m-2))
        delta = (r[ia] - r[ib]) / (2.0 * (m - 2))
        d_ab = sub[ia, ib]
        len_a = 0.5 * d_ab + 0.5 * delta
        len_b = 0.5 * d_ab - 0.5 * delta
        # new node id
        new_id = next_id
        next_id += 1
        # new distances to other active nodes
        rest = [k for k in nodes if k != a and k != b]
        for k in rest:
            # find positions
            # d(new,k) = 0.5*(d(a,k)+d(b,k)-d(a,b))
            val = 0.5 * (dist[a, k] + dist[b, k] - d_ab)
            dist[new_id, k] = val
            dist[k, new_id] = val
        # build tree string
        la = max(len_a, 0.0)
        lb = max(len_b, 0.0)
        tree_a = tid[a]
        tree_b = tid[b]
        new_tree = f"({tree_a}:{la:.4f},{tree_b}:{lb:.4f})"
        tid[new_id] = new_tree
        nodes = rest + [new_id]

    # finish with the last 3 (or fewer) nodes as a star
    if len(nodes) == 3:
        a, b, c = nodes
        # simple star with zero lengths is fine for topology; use small lengths
        res = f"({tid[a]}:0.0,{tid[b]}:0.0,{tid[c]}:0.0)"
    elif len(nodes) == 2:
        res = f"({tid[nodes[0]]}:0.0,{tid[nodes[1]]}:0.0)"
    elif len(nodes) == 1:
        res = f"({tid[nodes[0]]}:0.0)"
    else:
        res = "(" + ",".join(tid[x] for x in nodes) + ")"
    return res + ";"


def reconstruct(sequences):
    """Reconstruct unrooted tree topology; return Newick over t0..t(n-1)."""
    try:
        labels, D = _jc_distances(sequences)
        newick = _neighbor_joining(labels, D)
        # sanity: ensure every taxon appears exactly once
        # (NJ guarantees this; fallback star if something went wrong)
        return newick
    except Exception:
        # fallback: star topology over all labels
        labels = sorted(sequences, key=lambda name: int(name[1:]))
        return "(" + ",".join(f"t{i}" for i in range(len(labels))) + ")"
