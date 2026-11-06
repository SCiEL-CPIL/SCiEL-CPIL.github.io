"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    deadline = time.time() + 145.0  # safety margin before 150s kill

    def cut_size(labels):
        # Count edges crossing the partition
        # For each edge (i,j), it crosses if labels[i] != labels[j]
        # Sum over all i<j where labels differ
        # Efficient: (sum of adj * (labels[:,None] != labels[None,:])) / 2
        # But let's use a faster method
        # Cut size = sum over v of (number of neighbors in opposite set) / 2
        # Or: for each v, count neighbors with different label, sum, divide by 2
        diff = labels[:, None] != labels[None, :]
        return int(adj @ diff.sum(axis=1)) // 2

    def flip_gain(labels, v):
        # Gain in cut size if we flip vertex v
        # Current contribution of v to cut: neighbors in opposite set
        # After flip: neighbors in same set become opposite
        # Gain = (neighbors in same set) - (neighbors in opposite set)
        # = 2 * (neighbors in same set) - degree[v]
        same = labels == labels[v]
        n_same = adj[v] @ same
        n_opp = adj[v].sum() - n_same
        return int(2 * n_same - adj[v].sum())

    def local_search(labels, max_iters=5000):
        labels = labels.copy()
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 2.0:
            improved = False
            iters += 1
            # Random order to avoid cycles
            order = np.random.permutation(n)
            for v in order:
                gain = flip_gain(labels, v)
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels

    def simulated_annealing(labels, max_iters=20000):
        labels = labels.copy()
        T = 1.0
        cooling = 0.9995
        for it in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            v = np.random.randint(n)
            gain = flip_gain(labels, v)
            if gain > 0 or (np.random.rand() < np.exp(gain / T)):
                labels[v] = 1 - labels[v]
            T *= cooling
        return local_search(labels, max_iters=2000)

    # Generate multiple initial solutions and pick the best
    best_labels = None
    best_cut = -1

    # Initial solution 1: alternating
    labels1 = np.arange(n) % 2
    labels1 = local_search(labels1)
    c1 = cut_size(labels1)
    if c1 > best_cut:
        best_cut = c1
        best_labels = labels1.copy()

    # Initial solution 2: random
    for restart in range(5):
        if time.time() > deadline - 5.0:
            break
        labels_r = np.random.randint(0, 2, n)
        labels_r = local_search(labels_r)
        c_r = cut_size(labels_r)
        if c_r > best_cut:
            best_cut = c_r
            best_labels = labels_r.copy()

    # Spectral initialization using Fiedler vector
    try:
        deg = adj.sum(axis=1).astype(float)
        L = np.diag(deg) - adj.astype(float)
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the second smallest eigenvalue
        fiedler = eigvecs[:, 1]
        # Partition based on sign of Fiedler vector
        labels_spec = (fiedler > 0).astype(int)
        labels_spec = local_search(labels_spec)
        c_spec = cut_size(labels_spec)
        if c_spec > best_cut:
            best_cut = c_spec
            best_labels = labels_spec.copy()
    except Exception:
        pass

    # More random restarts with simulated annealing if time permits
    restart_count = 0
    while time.time() < deadline - 10.0 and restart_count < 20:
        restart_count += 1
        labels_sa = np.random.randint(0, 2, n)
        labels_sa = simulated_annealing(labels_sa, max_iters=5000)
        c_sa = cut_size(labels_sa)
        if c_sa > best_cut:
            best_cut = c_sa
            best_labels = labels_sa.copy()

    # Final local search polish on best solution
    if best_labels is not None:
        best_labels = local_search(best_labels, max_iters=10000)

    return best_labels
# EVOLVE-BLOCK-END
