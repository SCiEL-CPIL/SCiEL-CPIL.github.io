"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    degree = adj.sum(axis=1)
    total_edges = degree.sum() / 2.0

    deadline = time.time() + 140  # 10s safety margin before 150s limit

    def _cut_size(labels):
        # Vectorized cut size computation
        cross = (labels[:, None] != labels[None, :])
        return int((adj * cross).sum() / 2.0)

    def _local_search(labels):
        """Vectorized local search: repeatedly find all beneficial single flips and apply them."""
        labels = labels.copy()
        best_cut = _cut_size(labels)
        
        for _ in range(200):
            if time.time() > deadline:
                break
            # gain[v] = degree[v] - 2 * same_side_neighbors[v]
            same_mask = (labels[:, None] == labels[None, :])
            same_count = (adj * same_mask).sum(axis=1)
            gain = degree - 2.0 * same_count
            
            to_flip = gain > 0
            if not to_flip.any():
                break
            labels[to_flip] = 1 - labels[to_flip]
            new_cut = _cut_size(labels)
            if new_cut > best_cut:
                best_cut = new_cut
        
        return labels, best_cut

    def _spectral_init():
        """Fiedler vector initialization."""
        L = np.diag(degree) - adj
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
            fiedler = eigvecs[:, 1]
            labels = (fiedler >= 0).astype(np.int8)
            if labels.sum() == 0 or labels.sum() == n:
                labels = np.arange(n) % 2
            return labels
        except Exception:
            return np.arange(n) % 2

    def _random_init():
        """Random 50/50 initialization."""
        return (np.random.rand(n) < 0.5).astype(np.int8)

    def _greedy_init():
        """Greedy construction: assign vertices one by one to maximize local cut contribution."""
        labels = np.zeros(n, dtype=np.int8)
        # Random order for diversity
        order = np.random.permutation(n)
        assigned = np.zeros(n, dtype=bool)
        for v in order:
            # Count neighbors already assigned to 0 and 1
            neigh_0 = adj[v, (labels == 0) & assigned].sum()
            neigh_1 = adj[v, (labels == 1) & assigned].sum()
            labels[v] = 0 if neigh_0 >= neigh_1 else 1
            assigned[v] = True
        return labels

    # Generate diverse initializations
    inits = []
    inits.append(_spectral_init())
    inits.append(_random_init())
    inits.append(_greedy_init())
    # Add more random inits for diversity
    for _ in range(10):
        inits.append(_random_init())

    best_labels = None
    best_cut = -1

    for init_labels in inits:
        if time.time() > deadline:
            break
        labels, cut = _local_search(init_labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # If we still have time, do additional random restarts
    while time.time() < deadline:
        if best_labels is None:
            break
        # Perturb the best solution: flip a random subset of vertices
        perturb_mask = np.random.rand(n) < 0.2
        perturbed = best_labels.copy()
        perturbed[perturb_mask] = 1 - perturbed[perturb_mask]
        labels, cut = _local_search(perturbed)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    if best_labels is None:
        best_labels = np.arange(n) % 2

    return best_labels
# EVOLVE-BLOCK-END
