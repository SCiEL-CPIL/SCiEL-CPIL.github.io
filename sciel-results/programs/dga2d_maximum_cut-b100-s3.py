# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _flip_gain(adjacency, labels, v):
    """Compute the change in cut size if we flip vertex v."""
    same_count = adjacency[v][labels == labels[v]].sum()
    opp_count = adjacency[v][labels != labels[v]].sum()
    return int(same_count - opp_count)


def _local_search(adjacency, labels, deadline):
    """Iterative local search: flip vertices if it improves cut size."""
    n = len(labels)
    labels = labels.copy()
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for v in range(n):
            if time.time() > deadline:
                break
            gain = _flip_gain(adjacency, labels, v)
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _random_labels(n, seed):
    """Generate random 0/1 labels."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n)


def _greedy_partition(adjacency, n):
    """Greedy construction: add vertices one by one, assigning the label that maximizes cut."""
    labels = np.zeros(n, dtype=np.int8)
    unassigned = set(range(n))
    
    # Start with a random vertex
    first = random.randint(0, n - 1)
    labels[first] = 0
    unassigned.remove(first)
    
    while unassigned:
        if time.time() > 100:  # Safety
            break
        # For each unassigned vertex, compute gain if assigned 0 or 1
        best_v = None
        best_gain = -1
        best_label = 0
        
        for v in list(unassigned):
            # If v is assigned 0: cut edges to already-assigned 1s
            cut_if_0 = adjacency[v][labels == 1].sum()
            # If v is assigned 1: cut edges to already-assigned 0s
            cut_if_1 = adjacency[v][labels == 0].sum()
            
            gain_0 = int(cut_if_0)
            gain_1 = int(cut_if_1)
            
            if gain_0 > best_gain:
                best_gain = gain_0
                best_v = v
                best_label = 0
            elif gain_1 > best_gain:
                best_gain = gain_1
                best_v = v
                best_label = 1
        
        if best_v is None:
            break
        
        labels[best_v] = best_label
        unassigned.remove(best_v)
    
    return labels


def _spectral_partition(adjacency, n):
    """Spectral partition using the Fiedler vector."""
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        # Build Laplacian L = D - A
        A = adjacency.astype(np.float64)
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Get smallest eigenvalue (should be 0) and second smallest (Fiedler)
        # Use eigsh for the smallest eigenvalues
        # k=2 to get the two smallest
        try:
            vals, vecs = eigsh(L, k=2, which='SA')
            # Sort by eigenvalue
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]
        except Exception:
            # Fallback to dense
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]
        
        # Partition: vertices with positive Fiedler component -> 1, negative -> 0
        labels = (fiedler_vec > 0).astype(np.int8)
        
        # Ensure both partitions are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            # If all same, use median split
            median = np.median(fiedler_vec)
            labels = (fiedler_vec > median).astype(np.int8)
            if labels.sum() == 0 or labels.sum() == n:
                labels = np.arange(n) % 2
        
        return labels
    except Exception:
        return np.arange(n) % 2


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin: 115s out of 150s limit
    
    best_labels = None
    best_cut = -1
    
    def _update_best(labels):
        nonlocal best_labels, best_cut
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Strategy 1: Spectral initialization
    if n >= 2:
        spec_labels = _spectral_partition(adjacency, n)
        spec_labels = _local_search(adjacency, spec_labels, deadline)
        _update_best(spec_labels)
    
    # Strategy 2: Greedy construction
    greedy_labels = _greedy_partition(adjacency, n)
    greedy_labels = _local_search(adjacency, greedy_labels, deadline)
    _update_best(greedy_labels)
    
    # Strategy 3: Alternating
    alt_labels = np.arange(n) % 2
    alt_labels = _local_search(adjacency, alt_labels, deadline)
    _update_best(alt_labels)
    
    # Strategy 4: Multiple random restarts
    base_seed = 42
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        seed = base_seed + restart
        labels = _random_labels(n, seed)
        labels = _local_search(adjacency, labels, deadline)
        _update_best(labels)
    
    # Strategy 5: If time remains, do perturbation-based restarts from best
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline:
                break
            # Perturb the best solution
            perturbed = best_labels.copy()
            # Flip 5% of random vertices
            num_flips = max(1, n // 20)
            flip_indices = np.random.choice(n, size=num_flips, replace=False)
            perturbed[flip_indices] = 1 - perturbed[flip_indices]
            perturbed = _local_search(adjacency, perturbed, deadline)
            _update_best(perturbed)
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
