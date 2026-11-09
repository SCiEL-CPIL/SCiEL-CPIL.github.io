"""Rewritten program for Maximum Cut with vectorized multi-start local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = labels[None, :] != labels[:, None]
    return int((adjacency & diff).sum() // 2)

def _all_flip_gains(adjacency, labels):
    """
    Compute the gain (change in cut size) for flipping each vertex.
    gain[v] = (# neighbors with same label as v) - (# neighbors with diff label as v)
    If we flip v, edges that were cut become uncut (loss) and edges that were uncut become cut (gain).
    So delta = same_count - diff_count.
    """
    n = len(labels)
    # For each vertex v, we want:
    # same_count[v] = sum_u A[v,u] * (labels[u] == labels[v])
    # diff_count[v] = sum_u A[v,u] * (labels[u] != labels[v])
    
    # Vectorized approach:
    # Let L be the label vector (0/1).
    # same_matrix = (labels[:, None] == labels[None, :])  # N x N boolean
    # same_counts = (adjacency & same_matrix).sum(axis=1)
    # diff_counts = (adjacency & ~same_matrix).sum(axis=1)
    # gains = same_counts - diff_counts
    
    # More efficient:
    # Note that diff_counts = degree[v] - same_counts[v] (since A[v,v]=0)
    # So gains = same_counts - (degree - same_counts) = 2 * same_counts - degree
    
    same_matrix = (labels[:, None] == labels[None, :])
    same_counts = (adjacency & same_matrix).sum(axis=1)
    degrees = adjacency.sum(axis=1)
    gains = 2 * same_counts - degrees
    return gains

def _local_search(adjacency, labels, deadline):
    """
    Greedy local search: repeatedly flip the vertex with the maximum positive gain
    until no positive gain exists (local optimum).
    Uses vectorized gain computation.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    start_time = time.time()
    
    while True:
        if time.time() > deadline:
            break
        
        gains = _all_flip_gains(adjacency, labels)
        max_gain = gains.max()
        
        if max_gain <= 0:
            break  # Local optimum
        
        # Flip the vertex with max gain
        v = np.argmax(gains)
        labels[v] = 1 - labels[v]
        current_cut += int(max_gain)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut

def _spectral_init(adjacency):
    """Initialize using the sign of the Fiedler vector."""
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)
    L = np.diag(degrees) - adjacency.astype(float)
    
    try:
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]
        labels = (fiedler >= 0).astype(int)
        return labels
    except Exception:
        return np.random.randint(0, 2, size=n)

def _random_init(n):
    """Random initialization."""
    return np.random.randint(0, 2, size=n)

def _simulated_annealing(adjacency, labels, start_temp, end_temp, max_iters, deadline):
    """
    Simulated Annealing with vectorized gain computation.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    start_time = time.time()
    
    if max_iters <= 0:
        return best_labels, best_cut
    
    alpha = (end_temp / start_temp) ** (1.0 / max_iters)
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        T = start_temp * (alpha ** i)
        
        # Pick a random vertex
        v = np.random.randint(0, n)
        
        # Calculate gain of flipping v
        lv = labels[v]
        same = np.sum(adjacency[v] & (labels == lv))
        diff = np.sum(adjacency[v] & (labels != lv))
        delta = same - diff
        
        # Acceptance
        if delta > 0:
            accept = True
        else:
            if T > 1e-10:
                prob = np.exp(-delta / T)
                accept = np.random.rand() < prob
            else:
                accept = False
        
        if accept:
            labels[v] = 1 - lv
            current_cut += int(delta)
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
    
    return best_labels, best_cut

def partition(adjacency):
    """
    Main partition function.
    Strategy: Multi-start local search with spectral and random initializations.
    1. Spectral init + local search
    2. Multiple random restarts with local search
    3. Use remaining time for SA polishing if available
    """
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral Initialization + Local Search
    labels = _spectral_init(adjacency)
    labels, cut = _local_search(adjacency, labels, deadline - 10.0)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Phase 2: Random Restarts with Local Search
    # Estimate time per local search iteration
    # For N=100, each gain computation is O(N^2) = 10,000 ops
    # A local search might take ~10-100 flips
    # Let's do as many restarts as possible
    
    remaining = deadline - time.time() - 5.0
    if remaining > 2.0:
        # Try multiple random restarts
        num_restarts = 20
        time_per_restart = remaining / num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            
            labels = _random_init(n)
            # Local search with a tighter deadline for each restart
            restart_deadline = min(time.time() + time_per_restart, deadline - 5.0)
            labels, cut = _local_search(adjacency, labels, restart_deadline)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
    
    # Phase 3: SA polishing on the best solution found
    remaining = deadline - time.time() - 3.0
    if remaining > 2.0 and best_labels is not None:
        max_deg = adjacency.sum(axis=1).max()
        start_temp = max_deg * 1.0
        end_temp = 0.01
        
        # Estimate iterations
        est_time_per_iter = 0.00005  # Rough guess
        max_iters = int(remaining / est_time_per_iter)
        max_iters = max(1000, min(max_iters, 200000))
        
        labels, cut = _simulated_annealing(adjacency, best_labels, start_temp, end_temp, max_iters, deadline - 3.0)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Final local search on SA result
        if time.time() < deadline - 2.0:
            labels, cut = _local_search(adjacency, best_labels, deadline - 2.0)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = _spectral_init(adjacency)
    
    return best_labels
# EVOLVE-BLOCK-END
