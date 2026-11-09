"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute cut size: number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adj & diff).sum() // 2)


def _all_flip_gains(adj, labels):
    """Compute flip gains for all vertices at once (vectorized)."""
    same_mask = (labels[:, None] == labels[None, :])
    opp_mask = ~same_mask
    same_counts = (adj & same_mask).sum(axis=1)
    opp_counts = (adj & opp_mask).sum(axis=1)
    gains = same_counts - opp_counts
    return gains.astype(np.int32)


def _local_search(adj, labels, deadline, max_rounds=200):
    """Iterative local search: flip all vertices with positive gain per round."""
    n = len(labels)
    labels = labels.copy().astype(np.int8)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        gains = _all_flip_gains(adj, labels)
        positive = np.where(gains > 0)[0]
        if len(positive) == 0:
            break
        labels[positive] = 1 - labels[positive]
    
    return labels


def _spectral_partition(adj, n):
    """Use Fiedler vector of Laplacian for initial partition."""
    A = adj.astype(np.float64)
    D = np.diag(A.sum(axis=1))
    L = D - A
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]
        labels = (fiedler >= 0).astype(np.int8)
        return labels
    except Exception:
        return np.random.randint(0, 2, size=n).astype(np.int8)


def _greedy_partition(adj, n, deadline):
    """Greedy construction: add vertices by descending degree, assign to maximize cut."""
    labels = np.zeros(n, dtype=np.int8)
    degrees = adj.sum(axis=1)
    order = np.argsort(-degrees)
    
    for v in order:
        if time.time() > deadline:
            break
        cut_if_0 = adj[v][labels == 1].sum()
        cut_if_1 = adj[v][labels == 0].sum()
        labels[v] = 0 if cut_if_0 >= cut_if_1 else 1
    
    return labels


def _simulated_annealing(adj, labels, deadline, rng, start_temp=30.0, end_temp=0.01, cooling=0.9985):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy().astype(np.int8)
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    temp = start_temp
    adj_int = adj.astype(np.int8)
    
    while time.time() < deadline and temp > end_temp:
        v = rng.integers(0, n)
        
        same_count = adj_int[v][labels == labels[v]].sum()
        opp_count = adj_int[v][labels != labels[v]].sum()
        gain = int(same_count - opp_count)
        
        if gain >= 0:
            labels[v] = 1 - labels[v]
            current_cut += gain
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            if rng.random() < np.exp(gain / temp):
                labels[v] = 1 - labels[v]
                current_cut += gain
        
        temp *= cooling
    
    return best_labels, best_cut


def _tabu_search(adj, labels, deadline, rng, tabu_length=12, max_iters=8000):
    """Tabu search to escape local optima."""
    n = len(labels)
    labels = labels.copy().astype(np.int8)
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    tabu_until = np.full(n, -1, dtype=np.int32)
    adj_int = adj.astype(np.int8)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        gains = _all_flip_gains(adj, labels)
        
        # Find best non-tabu move
        non_tabu = it >= tabu_until
        masked_gains = gains.copy()
        masked_gains[~non_tabu] = -999999
        
        best_v = int(np.argmax(masked_gains))
        best_gain = gains[best_v]
        
        if best_gain <= 0 and rng.random() < 0.4:
            # Diversify: random move
            best_v = int(rng.integers(0, n))
            best_gain = gains[best_v]
        
        labels[best_v] = 1 - labels[best_v]
        tabu_until[best_v] = it + tabu_length
        current_cut += best_gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adj = np.asarray(adjacency, dtype=bool)
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Generate diverse initializations
    initializations = []
    
    # Spectral
    initializations.append(_spectral_partition(adj, n))
    
    # Greedy
    initializations.append(_greedy_partition(adj, n, deadline))
    
    # Alternating
    initializations.append((np.arange(n) % 2).astype(np.int8))
    
    # Degree-based
    degrees = adj.sum(axis=1)
    initializations.append((degrees > degrees.mean()).astype(np.int8))
    
    # Random initializations
    for seed in range(8):
        initializations.append(rng.integers(0, 2, n).astype(np.int8))
    
    # Phase 2: Local search from each initialization
    for init_labels in initializations:
        if time.time() > deadline - 20.0:
            break
        labels = _local_search(adj, init_labels, deadline - 20.0)
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = (np.arange(n) % 2).astype(np.int8)
        best_cut = _cut_size(adj, best_labels)
    
    # Phase 3: SA from best local search result
    if time.time() < deadline - 15.0:
        sa_deadline = deadline - 15.0
        sa_labels, sa_cut = _simulated_annealing(adj, best_labels, sa_deadline, rng)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Polish after SA
        polished = _local_search(adj, best_labels, deadline - 15.0)
        pol_cut = _cut_size(adj, polished)
        if pol_cut > best_cut:
            best_cut = pol_cut
            best_labels = polished
    
    # Phase 4: Tabu search from best result
    if time.time() < deadline - 10.0:
        tabu_deadline = deadline - 10.0
        tabu_labels, tabu_cut = _tabu_search(adj, best_labels, tabu_deadline, rng)
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels.copy()
        
        # Polish after tabu
        polished = _local_search(adj, best_labels, deadline - 10.0)
        pol_cut = _cut_size(adj, polished)
        if pol_cut > best_cut:
            best_cut = pol_cut
            best_labels = polished
    
    # Phase 5: Additional SA restarts from perturbed best if time remains
    if time.time() < deadline - 5.0:
        remaining = deadline - time.time()
        if remaining > 8.0:
            for extra in range(3):
                if time.time() > deadline - 5.0:
                    break
                perturbed = best_labels.copy().astype(np.int8)
                num_perturb = max(1, n // 12)
                perturbation_indices = rng.choice(n, size=num_perturb, replace=False)
                perturbed[perturbation_indices] = 1 - perturbed[perturbation_indices]
                
                sa_deadline = deadline - 5.0
                sa_labels, sa_cut = _simulated_annealing(adj, perturbed, sa_deadline, rng)
                if sa_cut > best_cut:
                    best_cut = sa_cut
                    best_labels = sa_labels.copy()
                
                polished = _local_search(adj, best_labels, deadline - 5.0)
                pol_cut = _cut_size(adj, polished)
                if pol_cut > best_cut:
                    best_cut = pol_cut
                    best_labels = polished
    
    # Final polish
    if time.time() < deadline - 2.0:
        final_labels = _local_search(adj, best_labels, deadline - 2.0)
        final_cut = _cut_size(adj, final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
