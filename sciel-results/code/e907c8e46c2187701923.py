"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _all_flip_gains(adjacency, labels):
    """Compute flip gains for all vertices at once.
    
    gain(v) = same_degree(v) - opp_degree(v)
    where same_degree(v) = number of neighbors of v with same label as v,
    and opp_degree(v) = number of neighbors of v with opposite label.
    
    We compute this without building the full n×n mask:
    - For label 0 vertices: same = col-sum of A[:, labels==0], opp = col-sum of A[:, labels==1]
    - For label 1 vertices: same = col-sum of A[:, labels==1], opp = col-sum of A[:, labels==0]
    """
    n = len(labels)
    mask0 = labels == 0
    mask1 = labels == 1
    
    # For each vertex v, count neighbors in group 0 and group 1
    # A[v, :] is row v; we want sum over neighbors
    # same_count[v] = A[v, mask0].sum() if labels[v]==0 else A[v, mask1].sum()
    # opp_count[v]  = A[v, mask1].sum() if labels[v]==0 else A[v, mask0].sum()
    
    # Compute group sums for all rows at once using matrix multiplication
    # A @ indicator gives for each row the sum of A[row, :] over the indicated columns
    group0_sums = adjacency @ mask0.astype(np.int32)  # for each v: # neighbors in group 0
    group1_sums = adjacency @ mask1.astype(np.int32)  # for each v: # neighbors in group 1
    
    # For vertices in group 0: same = group0_sums[v], opp = group1_sums[v]
    # For vertices in group 1: same = group1_sums[v], opp = group0_sums[v]
    same_counts = np.where(mask0, group0_sums, group1_sums)
    opp_counts = np.where(mask0, group1_sums, group0_sums)
    
    gains = same_counts - opp_counts
    return gains.astype(np.int32)


def _local_search(adjacency, labels, deadline, max_rounds=200):
    """Iterative local search: flip vertices if it improves cut size."""
    n = len(labels)
    labels = labels.copy()
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        gains = _all_flip_gains(adjacency, labels)
        positive = np.where(gains > 0)[0]
        if len(positive) == 0:
            break
        labels[positive] = 1 - labels[positive]
    
    return labels


def _spectral_partition(adjacency, n):
    """Use Fiedler vector for initial partition."""
    A = adjacency.astype(float)
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]
        labels = (fiedler >= 0).astype(np.int8)
        return labels
    except:
        return np.random.randint(0, 2, size=n).astype(np.int8)


def _greedy_partition(adjacency, n, deadline):
    """Greedy construction: add vertices one by one, choosing label that maximizes cut."""
    labels = np.zeros(n, dtype=np.int8)
    degrees = adjacency.sum(axis=1)
    order = np.argsort(-degrees)
    
    for v in order:
        if time.time() > deadline:
            break
        cut_if_0 = adjacency[v][labels == 1].sum()
        cut_if_1 = adjacency[v][labels == 0].sum()
        labels[v] = 0 if cut_if_0 >= cut_if_1 else 1
    
    return labels


def _random_labels(n, seed):
    """Generate random 0/1 labels."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n).astype(np.int8)


def _simulated_annealing(adjacency, labels, deadline, start_temp=20.0, end_temp=0.01, cooling=0.997):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    temp = start_temp
    rng = np.random.RandomState(123)
    
    # Pre-compute adjacency as int for faster ops
    adj_int = adjacency.astype(np.int8)
    
    # Batch SA: process multiple flips per time check for efficiency
    batch_size = 50
    
    while time.time() < deadline and temp > end_temp:
        # Process a batch of flips
        for _ in range(batch_size):
            if time.time() > deadline:
                break
            
            # Pick a random vertex to flip
            v = rng.randint(n)
            
            # Compute gain for flipping v
            same_count = adj_int[v][labels == labels[v]].sum()
            opp_count = adj_int[v][labels != labels[v]].sum()
            gain = same_count - opp_count
            
            if gain >= 0:
                # Accept improving or neutral moves
                labels[v] = 1 - labels[v]
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                # Accept worsening moves with probability exp(gain/temp)
                # gain is negative here
                if rng.random() < np.exp(gain / temp):
                    labels[v] = 1 - labels[v]
                    current_cut += gain
        
        temp *= cooling
    
    return best_labels, best_cut


def _tabu_search(adjacency, labels, deadline, tabu_length=15, max_iters=5000):
    """Tabu search to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Tabu list: dict mapping vertex -> iteration when it was last flipped
    tabu = {}
    rng = np.random.RandomState(456)
    adj_int = adjacency.astype(np.int8)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        gains = _all_flip_gains(adjacency, labels)
        
        # Find best non-tabu move
        best_v = -1
        best_gain = -np.inf
        for v in range(n):
            if v in tabu and it - tabu[v] < tabu_length:
                continue
            if gains[v] > best_gain:
                best_gain = gains[v]
                best_v = v
        
        if best_v == -1:
            # All moves are tabu, pick best tabu move
            best_v = int(np.argmax(gains))
            best_gain = gains[best_v]
        
        if best_gain <= 0:
            # No improving moves; occasionally do a random move to diversify
            if rng.random() < 0.3:
                best_v = rng.randint(n)
                best_gain = gains[best_v]
        
        # Make the move
        labels[best_v] = 1 - labels[best_v]
        tabu[best_v] = it
        current_cut += best_gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Multiple diverse initializations
    initializations = []
    
    # 1. Spectral partition
    initializations.append(_spectral_partition(adjacency, n))
    
    # 2. Greedy partition
    initializations.append(_greedy_partition(adjacency, n, deadline))
    
    # 3. Alternating
    initializations.append((np.arange(n) % 2).astype(np.int8))
    
    # 4-13. Random initializations
    for seed in range(10):
        initializations.append(_random_labels(n, 200 + seed))
    
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
        best_labels = (np.arange(n) % 2).astype(np.int8)
        best_cut = _cut_size(adjacency, best_labels)
    
    # Phase 3: Simulated annealing from the best local search result
    if time.time() < deadline - 10.0:
        sa_labels, sa_cut = _simulated_annealing(adjacency, best_labels, deadline)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Phase 4: Tabu search from the best result
    if time.time() < deadline - 5.0:
        tabu_labels, tabu_cut = _tabu_search(adjacency, best_labels, deadline)
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels
    
    # Phase 5: Final high-precision polish pass
    # Run multiple rounds of local search to squeeze out any remaining improvements
    if time.time() < deadline - 2.0:
        polish_deadline = deadline - 1.0
        for _ in range(3):
            if time.time() > polish_deadline:
                break
            final_labels = _local_search(adjacency, best_labels, polish_deadline)
            final_cut = _cut_size(adjacency, final_labels)
            if final_cut > best_cut:
                best_cut = final_cut
                best_labels = final_labels
            else:
                break
    
    return best_labels
# EVOLVE-BLOCK-END
