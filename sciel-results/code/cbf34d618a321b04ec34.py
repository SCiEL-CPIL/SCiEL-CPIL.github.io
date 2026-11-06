"""Phylogenetic topology recovery via spectral embedding and balanced clustering."""
import time
import math
import numpy as np
from scipy.linalg import eigh

_DEADLINE = 270.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_exceeded(margin=5.0):
    return _elapsed() > _DEADLINE - margin

def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected pairwise distance matrix using NumPy."""
    n = len(labels)
    # Build alignment matrix: n x L, each entry is 0-3 for A,C,G,T
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequences[labels[0]])
    # Use integer encoding
    mat = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j, ch in enumerate(seq):
            mat[i, j] = base_map[ch]
    
    # Pairwise differences: for each pair, count positions where they differ
    # d[i,j] = number of positions where mat[i] != mat[j]
    # Use broadcasting: (n, 1, L) vs (1, n, L) -> (n, n, L)
    # This is memory-heavy for large n, so do it in chunks
    D = np.zeros((n, n), dtype=np.float64)
    chunk = 32
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        # (chunk, 1, L) vs (1, n, L)
        diff = (mat[i:end, None, :] != mat[None, :, :])
        D[i:end, :] = diff.sum(axis=2).astype(np.float64)
    
    # Convert to proportions
    D /= L
    np.fill_diagonal(D, 0.0)
    
    # JC correction: d = -3/4 * ln(1 - 4p/3)
    # Clip to avoid log of non-positive
    p = np.clip(D, 0.0, 0.7499)
    D = -0.75 * np.log(np.maximum(1.0 - 4.0 * p / 3.0, 1e-10))
    np.fill_diagonal(D, 0.0)
    
    return D

def _spectral_embedding(D, n):
    """
    Embed taxa in Euclidean space using Laplacian eigenvectors.
    For a tree metric, the Laplacian's eigenvectors encode the topology.
    We use the smallest non-trivial eigenvectors.
    """
    # Build weighted Laplacian from the distance matrix
    # Use a Gaussian kernel to create a similarity graph, then Laplacian
    # Actually, for tree metrics, we can use the distance matrix directly.
    # The standard approach: use the double-centered distance matrix.
    
    # Double centering: B = -1/2 * H * D * H, where H = I - 1/n * J
    H = np.eye(n) - 1.0 / n
    B = -0.5 * H @ D @ H
    
    # Eigendecomposition of B
    # B is symmetric, so use eigh
    eigenvalues, eigenvectors = eigh(B)
    
    # Sort by eigenvalue descending (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top k eigenvectors (k = n-1 for full embedding, but we use fewer)
    # For topology recovery, the top (n-1) eigenvectors span the space
    # Use the first (n-1) or a subset
    k = min(n - 1, 10)  # Use at most 10 dimensions for efficiency
    coords = eigenvectors[:, :k] * np.sqrt(np.maximum(eigenvalues[:k], 0))
    
    return coords

def _balanced_binary_cluster(coords, n, labels, deadline_check=None):
    """
    Recover unrooted tree topology by recursively splitting the point set
    into two balanced halves using a hyperplane through the centroid.
    
    This mimics the structure of a binary tree where each internal node
    corresponds to a split of the leaf set.
    """
    if n <= 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        # For 3 taxa, any unrooted tree is the same topology
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    if deadline_check and deadline_check():
        # Fallback: return a star tree
        return "(" + ",".join(labels) + ")"
    
    # Find the best hyperplane to split the points into two roughly equal halves
    # Use the direction of the first principal component (or a random direction)
    # to project points, then split at the median
    
    # Center the coordinates
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    
    # Use PCA direction (first eigenvector of covariance)
    cov = centered.T @ centered / n
    eigvals, eigvecs = eigh(cov)
    # Largest eigenvalue direction
    idx = np.argmax(eigvals)
    direction = eigvecs[:, idx]
    
    # Project points onto this direction
    projections = centered @ direction
    
    # Split at median to get roughly balanced halves
    mid = np.median(projections)
    left_mask = projections <= mid
    right_mask = ~left_mask
    
    # Ensure both sides are non-empty and reasonably balanced
    n_left = left_mask.sum()
    n_right = n - n_left
    
    if n_left == 0 or n_right == 0:
        # Try a different direction
        # Use a random direction
        rng = np.random.RandomState(42)
        direction = rng.randn(coords.shape[1])
        direction /= np.linalg.norm(direction)
        projections = centered @ direction
        mid = np.median(projections)
        left_mask = projections <= mid
        right_mask = ~left_mask
        n_left = left_mask.sum()
        n_right = n - n_left
    
    if n_left == 0 or n_right == 0:
        # Fallback: split by index
        mid_idx = n // 2
        left_mask = np.zeros(n, dtype=bool)
        left_mask[:mid_idx] = True
        right_mask = ~left_mask
        n_left = mid_idx
        n_right = n - n_left
    
    # Get the labels for each half
    left_labels = [labels[i] for i in range(n) if left_mask[i]]
    right_labels = [labels[i] for i in range(n) if right_mask[i]]
    
    # Get the coordinates for each half
    left_coords = coords[left_mask]
    right_coords = coords[right_mask]
    
    # Recursively cluster each half
    left_tree = _balanced_binary_cluster(left_coords, n_left, left_labels, deadline_check)
    right_tree = _balanced_binary_cluster(right_coords, n_right, right_labels, deadline_check)
    
    return f"({left_tree},{right_tree})"

def _nj_reconstruct(D, labels):
    """Neighbor joining as fallback or polish."""
    n = len(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    dist = D.copy()
    
    while len(nodes) > 3:
        n_nodes = len(nodes)
        total = {a: sum(dist[a, b] for b in nodes if b != a) for a in nodes}
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n_nodes - 2) * dist[a, b] - total[a] - total[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        _, a, b = best
        joined = f"({trees[a]},{trees[b]})"
        new_idx = max(nodes) + 1
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            dist[new_idx, x] = 0.5 * (dist[a, x] + dist[b, x] - dist[a, b])
            dist[x, new_idx] = dist[new_idx, x]
        nodes = rest + [new_idx]
        trees[new_idx] = joined
    
    return "(" + ",".join(trees[x] for x in nodes) + ")"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Compute JC distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Try spectral embedding + balanced clustering
    try:
        coords = _spectral_embedding(D, n)
        tree = _balanced_binary_cluster(coords, n, labels, deadline_check=_deadline_exceeded)
    except Exception:
        # Fallback to NJ
        tree = _nj_reconstruct(D, labels)
    
    # Verify the tree has all labels exactly once
    # Simple check: count occurrences
    tree_clean = tree.rstrip(';')
    for lab in labels:
        if tree_clean.count(lab) != 1:
            # Fallback to NJ
            tree = _nj_reconstruct(D, labels)
            break
    
    return tree
