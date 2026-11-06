"""Combined program for Maximum Cut: spectral init + fast greedy + tabu search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the cut size (number of edges crossing the partition)."""
    labels_int = labels.astype(np.float64)
    degrees = adjacency.sum(axis=1)
    cut = np.dot(labels_int, degrees) - labels_int @ adjacency @ labels_int
    return cut


def _spectral_init(adjacency):
    """Generate initial partition using the Fiedler vector (2nd eigenvector of Laplacian)."""
    n = len(adjacency)
    adj = adjacency.astype(np.float64)
    
    # Compute Laplacian: L = D - A
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    
    # Find the second smallest eigenvector (Fiedler vector)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
    except:
        fiedler = np.random.randn(n)
    
    # Use the sign of the Fiedler vector to create initial partition
    threshold = np.mean(fiedler)
    labels = (fiedler > threshold).astype(int)
    
    # Ensure both sides are non-empty
    if labels.sum() == 0 or labels.sum() == n:
        threshold = np.median(fiedler)
        labels = (fiedler > threshold).astype(int)
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.random.randint(0, 2, size=n)
    
    return labels


def _greedy_improve(adjacency, labels, deadline):
    """
    Perform greedy local search: flip vertices that improve the cut.
    Uses efficient delta computation to avoid recomputing full cut.
    """
    n = len(labels)
    labels = labels.copy()
    
    # Precompute degrees
    degrees = adjacency.sum(axis=1)
    
    # Track for each vertex: number of neighbors on same side vs different side
    # same_side[v] = sum(adj[v] & (labels == labels[v]))
    # diff_side[v] = sum(adj[v] & (labels != labels[v]))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle order to break symmetry
        order = list(range(n))
        random.shuffle(order)
        
        for v in order:
            if time.time() >= deadline:
                return labels
            
            # Count neighbors with same and different labels
            same = np.sum(adjacency[v] & (labels == labels[v]))
            diff = np.sum(adjacency[v] & (labels != labels[v]))
            
            if diff < same:
                # Flipping v would improve the cut
                labels[v] = 1 - labels[v]
                improved = True
    
    return labels


def _tabu_search(adjacency, labels, deadline, tabu_tenure=7, max_iters=2000):
    """
    Tabu search to escape local optima.
    Forbids recent moves to avoid cycling.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Tabu list: dictionary mapping vertex -> iteration when it becomes non-tabu
    tabu_until = {}
    current_iter = 0
    
    # Precompute degrees
    degrees = adjacency.sum(axis=1)
    
    for _ in range(max_iters):
        if time.time() >= deadline:
            break
        current_iter += 1
        
        best_gain = -np.inf
        best_v = -1
        best_is_tabu = True
        
        # Evaluate all possible single-flip moves
        for v in range(n):
            same = np.sum(adjacency[v] & (labels == labels[v]))
            diff = np.sum(adjacency[v] & (labels != labels[v]))
            gain = diff - same  # change in cut if we flip v
            
            is_tabu = v in tabu_until and tabu_until[v] > current_iter
            
            # Aspiration: allow tabu move if it improves best
            if is_tabu and gain <= best_cut:
                continue
            
            if gain > best_gain:
                best_gain = gain
                best_v = v
                best_is_tabu = is_tabu
        
        if best_v == -1:
            break
        
        # Perform the best move
        labels[best_v] = 1 - labels[best_v]
        current_cut += best_gain
        
        # Add to tabu list
        tabu_until[best_v] = current_iter + tabu_tenure
        
        # Update best
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
        
        # Clear expired tabu entries
        if current_iter % 10 == 0:
            tabu_until = {k: v for k, v in tabu_until.items() if v > current_iter}
    
    return best_labels, best_cut


def _simulated_annealing(adjacency, labels, deadline, start_temp=5.0, cooling=0.9995):
    """Perform simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    temp = start_temp
    best_labels = labels.copy()
    best_cut = _cut_size(adjacency, labels)
    
    while time.time() < deadline and temp > 0.01:
        # Pick a random vertex to flip
        v = random.randint(0, n - 1)
        
        # Compute delta in cut size if we flip v
        same = np.sum(adjacency[v] & (labels == labels[v]))
        diff = np.sum(adjacency[v] & (labels != labels[v]))
        delta = same - diff  # change in cut size if we flip v
        
        if delta > 0:
            # Accept (improvement)
            labels[v] = 1 - labels[v]
            new_cut = _cut_size(adjacency, labels)
            if new_cut > best_cut:
                best_cut = new_cut
                best_labels = labels.copy()
        else:
            # Accept with probability exp(delta / temp)
            if random.random() < np.exp(delta / temp):
                labels[v] = 1 - labels[v]
        
        temp *= cooling
    
    return best_labels, best_cut


def partition(adjacency):
    """
    Solve Maximum Cut using spectral initialization, greedy local search,
    tabu search, and simulated annealing with multiple restarts.
    
    Returns a 0/1 label array for each vertex.
    """
    start_time = time.time()
    DEADLINE = start_time + 115.0  # 115s safety margin out of 120s limit
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adjacency = np.array(adjacency, dtype=bool)
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral initialization + greedy refinement
    spectral_labels = _spectral_init(adjacency)
    refined = _greedy_improve(adjacency, spectral_labels.copy(), DEADLINE)
    cut = _cut_size(adjacency, refined)
    best_labels = refined.copy()
    best_cut = cut
    
    # Phase 2: Tabu search from the greedy-refined spectral solution
    tabu_labels, tabu_cut = _tabu_search(adjacency, refined.copy(), DEADLINE, 
                                          tabu_tenure=7, max_iters=3000)
    if tabu_cut > best_cut:
        best_cut = tabu_cut
        best_labels = tabu_labels.copy()
    
    # Phase 3: Simulated annealing from the best solution so far
    if time.time() < DEADLINE - 5.0:
        sa_labels, sa_cut = _simulated_annealing(adjacency, best_labels.copy(), 
                                                 DEADLINE - 2.0, start_temp=5.0, 
                                                 cooling=0.9995)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Phase 4: Multiple random restarts with greedy + tabu
    num_restarts = 0
    while time.time() < DEADLINE - 10.0:
        num_restarts += 1
        
        # Random initial assignment
        labels = np.random.randint(0, 2, size=n)
        
        # Greedy refinement
        refined = _greedy_improve(adjacency, labels.copy(), DEADLINE - 5.0)
        cut = _cut_size(adjacency, refined)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = refined.copy()
        
        # Tabu search if time permits
        remaining = DEADLINE - time.time()
        if remaining > 8.0:
            tabu_labels, tabu_cut = _tabu_search(adjacency, refined.copy(), 
                                                  DEADLINE - 3.0, 
                                                  tabu_tenure=5, max_iters=1000)
            if tabu_cut > best_cut:
                best_cut = tabu_cut
                best_labels = tabu_labels.copy()
        
        # Limit restarts to avoid excessive computation
        if num_restarts >= 20:
            break
    
    # Final polish: one more greedy pass on the best solution
    if best_labels is not None and time.time() < DEADLINE:
        polished = _greedy_improve(adjacency, best_labels.copy(), DEADLINE)
        polished_cut = _cut_size(adjacency, polished)
        if polished_cut > best_cut:
            best_cut = polished_cut
            best_labels = polished.copy()
    
    if best_labels is None:
        # Fallback: return alternating pattern
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
