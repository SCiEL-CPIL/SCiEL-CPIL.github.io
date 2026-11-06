"""
Strategy: Multi-Start Spectral Initialization with Vectorized Simulated Annealing and 2-Flip Polish.
Combines the multi-start SA approach with a vectorized 2-flip neighborhood exploration to escape
local optima that are stable against single-vertex moves. The 2-flip moves allow simultaneous
relabeling of two vertices, which can improve the cut even when no single flip is beneficial.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses multi-start spectral initialization followed by vectorized simulated annealing,
    with a final 2-flip polish pass to escape 1-flip local optima.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s safety margin
    
    # Check if A is all zeros
    if not np.any(adj):
        return np.zeros(n, dtype=np.int8)
    
    # Precompute degrees
    degrees = np.sum(adj, axis=1)
    
    # Build Laplacian for spectral methods
    D = np.diag(degrees)
    L = D - adj
    
    def compute_cut_size(labels):
        """Compute the number of cut edges for a given labeling."""
        diff_mask = (labels[:, None] != labels[None, :])
        return 0.5 * np.sum(adj * diff_mask)
    
    # Generate multiple diverse initializations
    initializations = []
    
    # 1. Fiedler vector initialization
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        fiedler_vec = eigenvectors[:, 1]
        labels_fiedler = (fiedler_vec > 0).astype(np.int8)
        initializations.append(labels_fiedler.copy())
        
        # Also try median split
        median_val = np.median(fiedler_vec)
        labels_median = (fiedler_vec >= median_val).astype(np.int8)
        initializations.append(labels_median.copy())
        
        # Balanced split by rank
        sorted_idx = np.argsort(fiedler_vec)
        labels_rank = np.zeros(n, dtype=np.int8)
        labels_rank[sorted_idx[n//2:]] = 1
        initializations.append(labels_rank.copy())
    except Exception:
        pass
    
    # 2. Top eigenvector of adjacency matrix
    try:
        eigenvalues_a, eigenvectors_a = np.linalg.eigh(adj)
        idx_a = np.argsort(eigenvalues_a)[::-1]
        top_eigvec = eigenvectors_a[:, idx_a[0]]
        labels_top = (top_eigvec > 0).astype(np.int8)
        initializations.append(labels_top.copy())
    except Exception:
        pass
    
    # 3. Random initialization (a few variants)
    rng = np.random.RandomState(42)
    for _ in range(2):
        labels_rand = rng.randint(0, 2, n).astype(np.int8)
        initializations.append(labels_rand.copy())
    
    # 4. Balanced random initialization
    labels_balanced = np.zeros(n, dtype=np.int8)
    half = n // 2
    labels_balanced[:half] = 1
    perm = rng.permutation(n)
    labels_balanced = labels_balanced[perm]
    initializations.append(labels_balanced.copy())
    
    def vectorized_sa(labels, adj, degrees, start_time, deadline, max_iter=30000):
        """
        Vectorized Simulated Annealing.
        At each step, compute gains for all vertices, then select a move.
        """
        labels = labels.copy()
        current_cut = compute_cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        # SA parameters
        T0 = 10.0
        T_min = 0.01
        cooling = 0.9998
        
        T = T0
        rng_sa = np.random.RandomState(int(start_time * 1000) % 10000)
        
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            # Compute gains for flipping each vertex
            same_mask = (labels[:, None] == labels[None, :])
            same_counts = np.sum(adj * same_mask, axis=1)
            gains = 2 * same_counts - degrees
            
            # Select a move
            if rng_sa.random() < 0.3:
                max_gain = np.max(gains)
                if max_gain <= 1e-9:
                    flip_idx = rng_sa.randint(n)
                else:
                    flip_idx = np.argmax(gains)
            else:
                flip_idx = rng_sa.randint(n)
            
            gain = gains[flip_idx]
            
            if gain > 0:
                accept = True
            else:
                if T > T_min:
                    prob = np.exp(gain / T)
                    accept = (rng_sa.random() < prob)
                else:
                    accept = False
            
            if accept:
                labels[flip_idx] = 1 - labels[flip_idx]
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_labels, best_cut
    
    def two_flip_polish(labels, max_rounds=50):
        """
        Vectorized 2-flip neighborhood exploration.
        For each pair (i, j), compute the gain of flipping both simultaneously.
        Accept improving moves greedily.
        """
        labels = labels.copy()
        current_cut = compute_cut_size(labels)
        
        for _ in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gains for all pairs of flips
            # Gain of flipping i and j simultaneously:
            # = gain_i + gain_j + 2*adj[i,j]*(1 if labels[i]==labels[j] else -1)
            # where gain_i = 2*same_count_i - degree_i
            
            same_mask = (labels[:, None] == labels[None, :])
            same_counts = np.sum(adj * same_mask, axis=1)
            gains = 2 * same_counts - degrees
            
            # For pair (i,j): combined_gain = gains[i] + gains[j] + 2*adj[i,j]*(1 if same else -1)
            # = gains[i] + gains[j] + 2*adj[i,j]*(2*same_mask[i,j] - 1)
            
            # Build the combined gain matrix
            # same_factor = 2*same_mask - 1  (1 if same, -1 if different)
            same_factor = 2 * same_mask.astype(np.float64) - 1
            pair_gain = gains[:, None] + gains[None, :] + 2 * adj * same_factor
            
            # We only consider i < j to avoid double counting and self-pairs
            # Set diagonal to -inf
            np.fill_diagonal(pair_gain, -np.inf)
            
            # Find the best pair
            max_pair_gain = np.max(pair_gain)
            
            if max_pair_gain <= 1e-9:
                break
            
            # Get the pair indices
            i, j = np.unravel_index(np.argmax(pair_gain), pair_gain.shape)
            if i > j:
                i, j = j, i
            
            # Flip both vertices
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            current_cut += max_pair_gain
        
        return labels, current_cut
    
    # Run SA from each initialization, keeping the best result
    best_labels = None
    best_cut = -1
    
    for init_labels in initializations:
        if time.time() > deadline - 20.0:
            break
        
        sa_labels, sa_cut = vectorized_sa(
            init_labels, adj, degrees, start_time, deadline,
            max_iter=25000
        )
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Final polish: 2-flip neighborhood exploration on the best result
    if best_labels is not None and time.time() < deadline - 10.0:
        polished_labels, polished_cut = two_flip_polish(best_labels, max_rounds=100)
        if polished_cut > best_cut:
            best_cut = polished_cut
            best_labels = polished_labels
    
    # Final 1-flip polish
    if best_labels is not None and time.time() < deadline - 5.0:
        labels = best_labels.copy()
        improved = True
        while improved:
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            same_mask = (labels[:, None] == labels[None, :])
            same_counts = np.sum(adj * same_mask, axis=1)
            gains = 2 * same_counts - degrees
            
            max_gain = np.max(gains)
            if max_gain > 1e-9:
                flip_idx = np.argmax(gains)
                labels[flip_idx] = 1 - labels[flip_idx]
                improved = True
        best_labels = labels
    
    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)
    
    return best_labels
# EVOLVE-BLOCK-END
