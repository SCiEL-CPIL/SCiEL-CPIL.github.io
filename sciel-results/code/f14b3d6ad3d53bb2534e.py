"""Maximum Cut via Iterated Local Search with Spectral Init and Exhaustive 2-Swaps."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    if n == 0:
        return np.array([], dtype=np.int8)

    adj = np.array(adjacency, dtype=np.int32)
    # Ensure symmetric
    adj = (adj + adj.T) // 2
    degs = np.sum(adj, axis=1).astype(np.int32)
    
    # Precompute neighbor masks for faster updates? 
    # For N=100, boolean indexing is fine.
    
    def compute_cut(labels):
        diff = (labels[:, None] != labels[None, :])
        return int(np.sum(adj * diff)) // 2

    def compute_same_counts(labels):
        same_mask = (labels[:, None] == labels[None, :]).astype(np.int32)
        return (adj * same_mask).sum(axis=1).astype(np.int32)

    def local_search_1_flip(labels):
        """Greedy 1-flip until local optimum. Vectorized updates."""
        labels = labels.copy()
        same_counts = compute_same_counts(labels)
        
        while True:
            if time.time() > deadline:
                break
            # Gain for flipping v: 2 * same_counts[v] - degs[v]
            gains = 2 * same_counts - degs
            best_gain = np.max(gains)
            if best_gain <= 0:
                break
            
            v = np.argmax(gains)
            # Flip v
            old_label_v = labels[v]
            labels[v] = 1 - old_label_v
            
            # Update same_counts
            # v's new same count is degs[v] - old_same_count
            same_counts[v] = degs[v] - same_counts[v]
            
            # Update neighbors
            neighbors = np.where(adj[v] > 0)[0]
            if len(neighbors) > 0:
                # For each neighbor u:
                # If u had same label as old_label_v, it is now different. same_counts[u] -= 1
                # If u had diff label from old_label_v, it is now same. same_counts[u] += 1
                # labels[neighbors] contains current labels.
                # old_label_v is the label BEFORE flip.
                # So if labels[u] == old_label_v, they were same. Now diff.
                was_same = (labels[neighbors] == old_label_v)
                updates = np.where(was_same, -1, 1)
                same_counts[neighbors] += updates
        
        return labels

    def local_search_2_swap(labels):
        """Greedy 2-swap until local optimum. Exhaustive search over all pairs."""
        labels = labels.copy()
        same_counts = compute_same_counts(labels)
        
        while True:
            if time.time() > deadline:
                break
            
            # Compute gain for all pairs (i, j)
            # Gain(i, j) = 2 * same_counts[i] + 2 * same_counts[j] - degs[i] - degs[j] - 2 * adj[i, j]
            S = same_counts.astype(np.float64)
            d = degs.astype(np.float64)
            
            gain_mat = 2.0 * S[:, None] + 2.0 * S[None, :] - d[:, None] - d[None, :] - 2.0 * adj
            
            # Only consider i < j
            tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            gain_mat[~tri_mask] = -1e9
            
            max_gain = np.max(gain_mat)
            if max_gain <= 0:
                break
            
            idx = np.unravel_index(np.argmax(gain_mat), gain_mat.shape)
            i, j = idx
            
            # Perform swap
            labels[i], labels[j] = labels[j], labels[i]
            
            # Recompute same_counts (Safe and fast for N=100)
            same_counts = compute_same_counts(labels)
            
        return labels

    # Spectral Initialization
    def spectral_init():
        D = np.diag(degs)
        L = D - adj
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            idx = np.argsort(eigenvalues)
            fiedler = eigenvectors[:, idx[1]]
        except:
            fiedler = np.random.randn(n)
        
        labels = (fiedler > 0).astype(np.int8)
        return labels

    def random_init():
        return np.random.randint(0, 2, size=n, dtype=np.int8)

    best_labels = None
    best_cut = -1

    def update_best(labels):
        nonlocal best_labels, best_cut
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            return True
        return False

    # Initial candidates
    inits = [
        spectral_init(),
        random_init(),
        np.arange(n) % 2
    ]
    
    # Try each init with deep local search
    for init_labels in inits:
        if time.time() > deadline:
            break
        
        lab = local_search_1_flip(init_labels)
        update_best(lab)
        
        lab = local_search_2_swap(lab)
        update_best(lab)

    # ILS Loop
    current_labels = best_labels.copy()
    kick_size = 5
    
    # We can also try different kick sizes
    kick_sizes = [3, 5, 8, 12]
    
    while time.time() < deadline - 5:
        kick = random.choice(kick_sizes)
        perturbed = current_labels.copy()
        for _ in range(kick):
            v = np.random.randint(n)
            perturbed[v] = 1 - perturbed[v]
            
        lab = local_search_1_flip(perturbed)
        lab = local_search_2_swap(lab)
        
        new_cut = compute_cut(lab)
        if new_cut > best_cut:
            current_labels = lab.copy()
            best_cut = new_cut
            best_labels = lab.copy()

    # Final Polish
    if best_labels is not None:
        polished = local_search_2_swap(best_labels)
        if compute_cut(polished) > best_cut:
            best_labels = polished

    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
