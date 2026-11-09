# EVOLVE-BLOCK-START
import numpy as np
import time


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


def _spectral_partition(adjacency, n):
    """Use Fiedler vector for initial partition."""
    # Build Laplacian L = D - A
    A = adjacency.astype(float)
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Find eigenvector corresponding to second smallest eigenvalue
    # For n=100, we can use full eigendecomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Fiedler vector is eigenvector for second smallest eigenvalue
        fiedler = eigenvectors[:, 1]
        # Partition: positive -> 1, negative -> 0
        labels = (fiedler >= 0).astype(np.int8)
        return labels
    except:
        return np.random.randint(0, 2, size=n).astype(np.int8)


def _greedy_partition(adjacency, n, deadline):
    """Greedy construction: add vertices one by one, choosing label that maximizes cut."""
    labels = np.zeros(n, dtype=np.int8)
    # Order vertices by degree (highest first)
    degrees = adjacency.sum(axis=1)
    order = np.argsort(-degrees)
    
    for v in order:
        if time.time() > deadline:
            break
        # Try both labels, pick the one with higher cut contribution
        # If label is 0: cut edges to neighbors with label 1
        # If label is 1: cut edges to neighbors with label 0
        cut_if_0 = adjacency[v][labels == 1].sum()
        cut_if_1 = adjacency[v][labels == 0].sum()
        labels[v] = 0 if cut_if_0 >= cut_if_1 else 1
    
    return labels


def _random_labels(n, seed):
    """Generate random 0/1 labels."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n).astype(np.int8)


def _perturb(labels, n, num_flips, rng):
    """Perturb a solution by flipping a few random vertices."""
    perturbed = labels.copy()
    indices = rng.choice(n, size=min(num_flips, n), replace=False)
    perturbed[indices] = 1 - perturbed[indices]
    return perturbed


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin: 140s out of 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Multiple diverse initializations
    initializations = []
    
    # 1. Spectral partition
    initializations.append(_spectral_partition(adjacency, n))
    
    # 2. Greedy partition
    initializations.append(_greedy_partition(adjacency, n, deadline))
    
    # 3. Alternating
    initializations.append(np.arange(n) % 2)
    
    # 4-12. Random initializations with different seeds
    for seed in range(9):
        initializations.append(_random_labels(n, 100 + seed))
    
    # Phase 2: Local search from each initialization
    for init_labels in initializations:
        if time.time() > deadline:
            break
        labels = _local_search(adjacency, init_labels, deadline)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
        best_cut = _cut_size(adjacency, best_labels)
    
    # Phase 3: Perturbation-based restarts from best solution
    rng = np.random.RandomState(42)
    num_perturbations = 50
    for i in range(num_perturbations):
        if time.time() > deadline:
            break
        # Increase perturbation size over time
        num_flips = 1 + (i % 5)
        perturbed = _perturb(best_labels, n, num_flips, rng)
        labels = _local_search(adjacency, perturbed, deadline)
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
