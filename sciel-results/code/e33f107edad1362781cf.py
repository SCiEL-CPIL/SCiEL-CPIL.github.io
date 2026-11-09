"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(A, labels):
    """Number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((A & diff).sum() // 2)


def _local_search(A, labels, deadline):
    """Vectorized greedy local improvement: flip vertices that increase cut size."""
    n = len(labels)
    labels = labels.copy()
    # Precompute: for each vertex, number of neighbors in each partition
    # gain[v] = (neighbors in opposite) - (neighbors in same) if we flip v
    # Actually: current cut contribution of v = neighbors in opposite set
    # After flip: neighbors in same set become crossing
    # Gain = (same_count) - (opp_count)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        # Compute for each vertex: count of neighbors in same and opposite set
        # same[v] = sum of A[v] where labels == labels[v]
        # opp[v] = sum of A[v] where labels != labels[v]
        # Vectorized: 
        # For each v, same_count = A[v] @ (labels == labels[v])
        # This is O(n^2) per round but vectorized
        same_mask = (labels[None, :] == labels[:, None])  # n x n
        # same_count[v] = sum over j where labels[j]==labels[v] of A[v,j]
        same_count = (A * same_mask).sum(axis=1)
        row_sums = A.sum(axis=1)
        opp_count = row_sums - same_count
        gain = same_count - opp_count  # positive means flip improves
        # Find best vertex to flip (largest positive gain)
        max_gain = gain.max()
        if max_gain > 0:
            # Flip all vertices with positive gain simultaneously? No, greedy one at a time
            # But for speed, flip the best one
            v = int(np.argmax(gain))
            labels[v] = 1 - labels[v]
            improved = True
    return labels


def _local_search_vectorized(A, labels, deadline):
    """Faster vectorized local search: flip all improving vertices each round."""
    n = len(labels)
    labels = labels.copy()
    improved = True
    rounds = 0
    while improved:
        improved = False
        if time.time() > deadline:
            break
        rounds += 1
        same_mask = (labels[None, :] == labels[:, None])
        same_count = (A * same_mask).sum(axis=1)
        row_sums = A.sum(axis=1)
        opp_count = row_sums - same_count
        gain = same_count - opp_count
        # Flip all vertices with positive gain
        to_flip = gain > 0
        if to_flip.any():
            labels[to_flip] = 1 - labels[to_flip]
            improved = True
    return labels


def _spectral_partition(A, n, rng):
    """Spectral partitioning using Fiedler vector."""
    # Compute graph Laplacian
    degrees = A.sum(axis=1)
    L = np.diag(degrees) - A.astype(np.float64)
    # Eigendecomposition
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the second eigenvector
        fiedler = eigvecs[:, 1]
        # Partition based on sign
        labels = (fiedler >= 0).astype(np.int8)
        # Ensure both partitions are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            # Use median split
            median_val = np.median(fiedler)
            labels = (fiedler >= median_val).astype(np.int8)
        return labels
    except:
        return np.arange(n) % 2


def _simulated_annealing(A, labels, deadline, rng):
    """Simulated annealing with adaptive temperature."""
    n = len(labels)
    labels = labels.copy()
    best_labels = labels.copy()
    best_cut = _cut_size(A, labels)
    
    # Initial temperature
    T = 10.0
    T_min = 0.01
    T_max = 100.0
    
    current_cut = best_cut
    row_sums = A.sum(axis=1)
    
    while time.time() < deadline:
        # Choose a random vertex to flip
        v = rng.integers(0, n)
        old_label = labels[v]
        # Compute gain
        same_mask_v = (labels == old_label)
        same_count = (A[v] & same_mask_v).sum()
        opp_count = row_sums[v] - same_count
        gain = same_count - opp_count
        
        if gain >= 0 or rng.random() < np.exp(gain / T):
            labels[v] = 1 - old_label
            current_cut += gain
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        # Adaptive cooling
        # Estimate acceptance rate over recent iterations (simplified: just cool)
        T *= 0.999
        if T < T_min:
            T = T_max  # Reheat
    return best_labels


def partition(adjacency):
    n = len(adjacency)
    A = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 120s limit, use 110s
    
    rng = np.random.RandomState(42)
    
    best_labels = None
    best_cut = -1
    
    # Generate diverse initial partitions
    initializations = []
    
    # 1. Spectral partition
    spec_labels = _spectral_partition(A, n, rng)
    initializations.append(spec_labels)
    
    # 2. Alternating
    alt_labels = (np.arange(n) % 2).astype(np.int8)
    initializations.append(alt_labels)
    
    # 3. Random
    rand_labels = rng.randint(0, 2, size=n).astype(np.int8)
    initializations.append(rand_labels)
    
    # 4. More random
    for _ in range(3):
        initializations.append(rng.randint(0, 2, size=n).astype(np.int8))
    
    # 5. Spectral with perturbation
    for _ in range(2):
        spec = _spectral_partition(A, n, rng)
        # Perturb a few vertices
        perturbed = spec.copy()
        flip_indices = rng.choice(n, size=n//10, replace=False)
        perturbed[flip_indices] = 1 - perturbed[flip_indices]
        initializations.append(perturbed)
    
    # Local search from each initialization
    for init_labels in initializations:
        if time.time() > deadline:
            break
        ls_labels = _local_search_vectorized(A, init_labels, deadline)
        cut = _cut_size(A, ls_labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = ls_labels.copy()
    
    # Simulated annealing from best so far
    if time.time() < deadline - 5.0 and best_labels is not None:
        sa_labels = _simulated_annealing(A, best_labels, deadline - 1.0, rng)
        cut = _cut_size(A, sa_labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = sa_labels.copy()
    
    # Final local search polish from best
    if best_labels is not None and time.time() < deadline - 1.0:
        final_labels = _local_search_vectorized(A, best_labels, deadline)
        cut = _cut_size(A, final_labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = final_labels.copy()
    
    # Ensure we have a valid result
    if best_labels is None:
        best_labels = (np.arange(n) % 2).astype(np.int8)
    
    return best_labels
# EVOLVE-BLOCK-END
