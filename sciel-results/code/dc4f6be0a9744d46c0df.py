"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    cross_mask = labels[:, None] != labels[None, :]
    return int((adj * cross_mask).sum() // 2)


def _spectral_init(adj, n, k=1):
    """
    Initialize using the (k+1)-th eigenvector of the graph Laplacian.
    Returns a 0/1 label array.
    """
    degrees = np.sum(adj, axis=1).astype(float)
    D = np.diag(degrees)
    A = adj.astype(float)
    L = D - A

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        if k >= len(eigenvectors):
            k = len(eigenvectors) - 1
        fiedler = eigenvectors[:, k]
        labels = (fiedler > 0).astype(np.int8)
    except Exception:
        labels = np.random.randint(0, 2, size=n).astype(np.int8)

    return labels


def _multi_spectral_init(adj, n, num_vectors=3):
    """
    Use multiple Fiedler vectors to create diverse high-quality initializations.
    Combines eigenvectors with different weightings.
    """
    degrees = np.sum(adj, axis=1).astype(float)
    D = np.diag(degrees)
    A = adj.astype(float)
    L = D - A

    inits = []
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Use the first few non-trivial eigenvectors
        for k in range(1, min(num_vectors + 1, len(eigenvectors))):
            fiedler = eigenvectors[:, k]
            labels = (fiedler > 0).astype(np.int8)
            inits.append(labels)
        
        # Also try a weighted combination of the first two Fiedler vectors
        if len(eigenvectors) >= 3:
            combined = 0.7 * eigenvectors[:, 1] + 0.3 * eigenvectors[:, 2]
            labels = (combined > 0).astype(np.int8)
            inits.append(labels)
    except Exception:
        pass

    if not inits:
        inits.append(np.random.randint(0, 2, size=n).astype(np.int8))

    return inits


def _random_init(n):
    """Random 0/1 initialization."""
    return np.random.randint(0, 2, size=n).astype(np.int8)


def _vectorized_greedy_improve(adj, labels, max_passes=50):
    """
    Vectorized greedy local search: repeatedly flip vertices that improve the cut.
    Returns improved labels.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()

    for _ in range(max_passes):
        # Vectorized: compute for all vertices at once
        # same_side_mask[v] = 1 if neighbor has same label as v
        same_side_mask = (labels[None, :] == labels[:, None]).astype(np.int8)
        # For each vertex v, count edges to same side and other side
        same_counts = (adj_int * same_side_mask).sum(axis=1)
        other_counts = adj_int.sum(axis=1) - same_counts
        
        # Find vertices where flipping improves the cut
        # Flipping v: gain = same_counts[v] - other_counts[v]
        # If gain > 0, flipping v improves the cut
        gains = same_counts - other_counts
        to_flip = np.where(gains > 0)[0]
        
        if len(to_flip) == 0:
            break
        
        labels[to_flip] = 1 - labels[to_flip]

    return labels


def _goemans_maxwell(adj, labels, max_passes=100):
    """
    Goemans-Williamson style local improvement:
    For each vertex, compute the net gain of flipping it.
    Use a priority queue approach to find the best flip.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)

    for _ in range(max_passes):
        # Compute gains for all vertices
        # gain[v] = (edges to opposite side) - (edges to same side)
        same_counts = (adj_int * (labels[None, :] == labels[:, None]).astype(np.int8)).sum(axis=1)
        other_counts = adj_int.sum(axis=1) - same_counts
        gains = other_counts - same_counts

        # Find vertex with maximum positive gain
        max_gain = gains.max()
        if max_gain <= 0:
            break

        max_gain_idx = np.argmax(gains)
        labels[max_gain_idx] = 1 - labels[max_gain_idx]
        current_cut += max_gain

    return labels


def _simulated_annealing(adj, labels, start_temp=10.0, cool_rate=0.995, max_steps=5000):
    """
    Simple simulated annealing to escape local optima.
    Returns improved labels.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    temp = start_temp

    for step in range(max_steps):
        v = np.random.randint(0, n)
        old_label = labels[v]
        new_label = 1 - old_label

        # Compute change in cut size
        same_count = np.sum(adj_int[v, labels == old_label])
        other_count = np.sum(adj_int[v, labels != old_label])
        
        delta = same_count - other_count

        new_cut = current_cut + delta

        if new_cut > current_cut:
            accept = True
        else:
            if temp < 1e-10:
                accept = False
            else:
                if delta / temp > -50:
                    accept = np.random.random() < np.exp(delta / temp)
                else:
                    accept = False

        if accept:
            labels[v] = new_label
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

        temp *= cool_rate

    return best_labels


def _smooth_local_search(adj, labels, max_passes=30, temperature=1.0):
    """
    Local search using a smooth surrogate for the gain function.
    Uses a soft-max approximation to handle ties more gracefully.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)

    for _ in range(max_passes):
        # Compute gains for all vertices
        same_side_mask = (labels[None, :] == labels[:, None]).astype(np.int8)
        same_counts = (adj_int * same_side_mask).sum(axis=1)
        other_counts = adj_int.sum(axis=1) - same_counts
        gains = same_counts - other_counts

        # Use smooth max: instead of just taking argmax, use temperature-based selection
        # This helps escape shallow local optima by occasionally trying suboptimal flips
        max_gain = gains.max()
        if max_gain <= 0:
            break

        # Create a soft-max distribution over gains
        # Normalize gains to create probabilities
        shifted_gains = gains - max_gain  # shift so max is 0
        exp_gains = np.exp(shifted_gains / temperature)
        probs = exp_gains / exp_gains.sum()

        # Sample a vertex to flip based on the distribution
        v = np.random.choice(n, p=probs)

        # Only flip if it improves or is neutral (with some probability)
        if gains[v] > 0 or (gains[v] == 0 and np.random.random() < 0.5):
            labels[v] = 1 - labels[v]
            current_cut += gains[v]

    return labels


def partition(adjacency):
    """
    Partition the graph into two sets to maximize the cut.
    Uses multi-spectral initialization + vectorized greedy local search + smooth local search + simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)

    # Generate multiple high-quality initializations using multi-spectral approach
    inits = _multi_spectral_init(adj, n, num_vectors=3)

    # Add alternating initialization
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)

    # Add a couple of random initializations for diversity
    for _ in range(2):
        inits.append(_random_init(n))

    best_labels = None
    best_cut = -1

    for init_labels in inits:
        if time.time() > deadline:
            break

        # Vectorized greedy local search
        greedy_labels = _vectorized_greedy_improve(adj, init_labels)
        greedy_cut = _cut_size(adj, greedy_labels)

        if greedy_cut > best_cut:
            best_cut = greedy_cut
            best_labels = greedy_labels.copy()

        # Smooth local search refinement
        if time.time() < deadline - 15:
            smooth_labels = _smooth_local_search(adj, greedy_labels, max_passes=20, temperature=1.0)
            smooth_cut = _cut_size(adj, smooth_labels)

            if smooth_cut > best_cut:
                best_cut = smooth_cut
                best_labels = smooth_labels.copy()

        # Goemans-Williamson refinement (single flip improvements)
        if time.time() < deadline - 10:
            gm_labels = _goemans_maxwell(adj, best_labels)
            gm_cut = _cut_size(adj, gm_labels)

            if gm_cut > best_cut:
                best_cut = gm_cut
                best_labels = gm_labels.copy()

        # Simulated annealing from the best result found so far
        if time.time() < deadline - 20:
            sa_labels = _simulated_annealing(adj, best_labels, start_temp=5.0, cool_rate=0.995, max_steps=2000)
            sa_cut = _cut_size(adj, sa_labels)

            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()

    # Final vectorized greedy pass on the best found
    if best_labels is not None and time.time() < deadline:
        final_labels = _vectorized_greedy_improve(adj, best_labels)
        final_cut = _cut_size(adj, final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels.copy()

    # Final polish with smooth local search
    if best_labels is not None and time.time() < deadline - 5:
        polish_labels = _smooth_local_search(adj, best_labels, max_passes=10, temperature=0.5)
        polish_cut = _cut_size(adj, polish_labels)
        if polish_cut > best_cut:
            best_cut = polish_cut
            best_labels = polish_labels.copy()

    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2

    return best_labels
# EVOLVE-BLOCK-END
