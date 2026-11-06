"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    deadline = time.time() + 145.0

    # Spectral initialization via Fiedler vector
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
        median_val = np.median(fiedler)
        labels = (fiedler > median_val).astype(np.int8)
    except Exception:
        labels = np.arange(n) % 2

    def _cut_size(labels):
        cross = (labels[None, :] != labels[:, None])
        return (adj * cross).sum() // 2

    def _compute_gains(labels):
        """Compute gain of flipping each vertex."""
        gains = np.zeros(n)
        for v in range(n):
            same = adj[v][labels == labels[v]].sum()
            other = adj[v][labels != labels[v]].sum()
            gains[v] = other - same
        return gains

    def _local_search(labels):
        """1-flip local search until no improvement."""
        for _ in range(20):
            gains = _compute_gains(labels)
            if gains.max() <= 0:
                break
            # Flip the vertex with maximum gain
            v = np.argmax(gains)
            if gains[v] <= 0:
                break
            labels[v] = 1 - labels[v]
        return labels

    def _two_flip_search(labels):
        """2-flip neighborhood exploration."""
        current_cut = _cut_size(labels)
        improved = False
        # Precompute same/other counts
        same_counts = np.zeros(n)
        other_counts = np.zeros(n)
        for v in range(n):
            same_counts[v] = adj[v][labels == labels[v]].sum()
            other_counts[v] = adj[v][labels != labels[v]].sum()

        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    # Both in same group
                    adj_ij = adj[i, j]
                    gain = (same_counts[i] - adj_ij) - other_counts[i] + (same_counts[j] - adj_ij) - other_counts[j]
                else:
                    # Different groups
                    # Flipping both: i goes to j's group, j goes to i's group
                    # Edge (i,j): before cut, after non-cut -> loss of 1
                    # For i: neighbors in i's old group (same_i) become cut (gain same_i), neighbors in other group (other_i) become non-cut (loss other_i)
                    # But j was in other group for i, so edge (i,j) is in other_i. After flip, j is in same group as i (new), so edge (i,j) is non-cut.
                    # Net for i: gain same_i - other_i, but we need to account for j specifically
                    # Let me just compute: new_labels = labels with i,j flipped
                    # gain = cut(new) - cut(old)
                    # Change for edge (i,u), u!=j: if u in same group as i originally: +1, else -1
                    # Change for edge (j,u), u!=i: if u in same group as j originally: +1, else -1
                    # Change for edge (i,j): was 1 (cut), now 0 (non-cut): -1
                    gain = (same_counts[i] - adj[i, j]) - other_counts[i] + (same_counts[j] - adj[j, i]) - other_counts[j] - 1
                if gain > 0:
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    improved = True
                    # Recompute counts
                    same_counts = np.zeros(n)
                    other_counts = np.zeros(n)
                    for v in range(n):
                        same_counts[v] = adj[v][labels == labels[v]].sum()
                        other_counts[v] = adj[v][labels != labels[v]].sum()
                    break
            if improved:
                improved = False
        return labels

    # First pass: local search
    labels = _local_search(labels)
    best_labels = labels.copy()
    best_cut = _cut_size(labels)

    # Multiple restarts
    np.random.seed(42)
    for restart in range(30):
        if time.time() > deadline - 10:
            break

        if restart == 0:
            current = best_labels.copy()
        else:
            current = best_labels.copy()
            k = max(1, n // 10)
            flip_indices = np.random.choice(n, k, replace=False)
            current[flip_indices] = 1 - current[flip_indices]

        current = _local_search(current)
        cut_val = _cut_size(current)

        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = current.copy()

        if time.time() > deadline - 5:
            break

        current2 = _two_flip_search(current)
        cut_val2 = _cut_size(current2)
        if cut_val2 > best_cut:
            best_cut = cut_val2
            best_labels = current2.copy()
            # Do another local search after 2-flip
            best_labels = _local_search(best_labels)
            best_cut = _cut_size(best_labels)

    return best_labels
# EVOLVE-BLOCK-END
