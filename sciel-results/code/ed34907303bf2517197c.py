"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _random_labels(n):
    """Generate random 0/1 labels."""
    return np.random.randint(0, 2, size=n)


def _greedy_partition(adjacency):
    """Greedy construction: add vertices one by one, placing each to maximize immediate cut gain."""
    n = len(adjacency)
    labels = np.zeros(n, dtype=int)
    unassigned = set(range(n))
    
    # Start with a random vertex
    start_v = np.random.randint(n)
    labels[start_v] = np.random.randint(0, 2)
    unassigned.remove(start_v)
    
    while unassigned:
        best_v = None
        best_gain = -np.inf
        best_label = 0
        
        # Sample a subset if too many unassigned
        if len(unassigned) > 50:
            candidates = random.sample(list(unassigned), 50)
        else:
            candidates = list(unassigned)
        
        for v in candidates:
            # Try both labels
            for l in [0, 1]:
                # Count edges to already assigned vertices with different label
                assigned_mask = np.zeros(n, dtype=bool)
                assigned_mask[np.array(list(unassigned))] = False
                # Actually, let's compute differently
                pass
        
        # Simpler approach: just pick random vertex and best label
        v = random.choice(list(unassigned))
        # Count neighbors with label 0 and label 1 among assigned
        assigned = ~np.isin(np.arange(n), list(unassigned))
        n0 = (adjacency[v] & assigned & (labels == 0)).sum()
        n1 = (adjacency[v] & assigned & (labels == 1)).sum()
        
        if n0 > n1:
            labels[v] = 1
        elif n1 > n0:
            labels[v] = 0
        else:
            labels[v] = np.random.randint(0, 2)
        
        unassigned.remove(v)
    
    return labels


def _flip_gain(adjacency, labels, v):
    """Compute the change in cut size if we flip vertex v.
    Positive gain means flipping improves the cut.
    """
    same_mask = (labels == labels[v])
    other_mask = (labels != labels[v])
    same_count = adjacency[v, same_mask].sum()
    other_count = adjacency[v, other_mask].sum()
    return int(same_count - other_count)


def _all_flip_gains(adjacency, labels):
    """Vectorized computation of flip gains for all vertices."""
    n = len(adjacency)
    # For each vertex v, gain = sum(adj[v] & (labels == labels[v])) - sum(adj[v] & (labels != labels[v]))
    # = sum(adj[v]) - 2 * sum(adj[v] & (labels != labels[v]))
    # Let's compute: for vertex v, let s = labels[v], then
    # same_count = sum(adj[v, labels == s]), other_count = sum(adj[v, labels != s])
    # gain = same_count - other_count = 2*same_count - degree[v]
    
    # Compute same_count for each vertex
    # same_count[v] = sum(adj[v] & (labels == labels[v]))
    # We can do this with matrix operations
    # labels is 0/1 array
    # For vertex v with label 0: same_count = sum(adj[v, labels==0])
    # For vertex v with label 1: same_count = sum(adj[v, labels==1])
    
    adj_float = adjacency.astype(float)
    degree = adj_float.sum(axis=1)
    
    # same_count[v] = sum over u where labels[u] == labels[v] of adj[v,u]
    # This is (adj_float @ (labels == labels[:, None]).astype(float)) ... no that's not right
    
    # Better: same_count[v] = sum_u adj[v,u] * (labels[u] == labels[v])
    # = sum_u adj[v,u] * (1 - |labels[u] - labels[v]|) ... for 0/1 labels
    # Actually for 0/1: (labels[u] == labels[v]) = 1 if same, 0 if diff
    # = 1 - (labels[u] != labels[v])
    # So same_count = degree[v] - other_count
    # And gain = same_count - other_count = degree[v] - 2*other_count
    
    # other_count[v] = sum_u adj[v,u] * (labels[u] != labels[v])
    # For 0/1 labels: labels[u] != labels[v] = labels[u] XOR labels[v]
    # = labels[u] + labels[v] - 2*labels[u]*labels[v] ... for 0/1
    # Actually: if labels[v]=0: diff = labels[u], if labels[v]=1: diff = 1-labels[u]
    # So other_count[v] = sum_u adj[v,u] * (labels[u] if labels[v]==0 else 1-labels[u])
    # = sum_u adj[v,u] * labels[u] if labels[v]==0, else sum_u adj[v,u] * (1-labels[u])
    
    # Let's just compute it directly
    # other_count = adj_float @ labels for label 0 vertices, and adj_float @ (1-labels) for label 1 vertices
    adj_labels = adj_float @ labels.astype(float)  # sum_u adj[v,u]*labels[u]
    adj_one_minus_labels = adj_float @ (1 - labels).astype(float)  # sum_u adj[v,u]*(1-labels[u])
    
    # For vertices with label 0: other_count = adj_labels[v]
    # For vertices with label 1: other_count = adj_one_minus_labels[v]
    other_count = np.where(labels == 0, adj_labels, adj_one_minus_labels)
    
    gains = degree - 2 * other_count
    return gains.astype(int)


def _local_search(adjacency, labels, max_time=None):
    """Iterated local search via single-flip improvement with vectorized gain computation."""
    n = len(adjacency)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    if max_time is None:
        max_time = 145.0
    
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        order = np.random.permutation(n)
        for v in order:
            if max_time is not None and time.time() - start_time > max_time:
                return labels, current_cut
            
            gain = _flip_gain(adjacency, labels, v)
            
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                improved = True
    
    return labels, current_cut


def _two_flip_local_search(adjacency, labels, max_time=None):
    """Local search allowing pairs of flips to escape local optima."""
    n = len(adjacency)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    if max_time is None:
        max_time = 145.0
    
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        # Try single flips first
        order = np.random.permutation(n)
        for v in order:
            if max_time is not None and time.time() - start_time > max_time:
                return labels, current_cut
            
            gain = _flip_gain(adjacency, labels, v)
            
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                improved = True
                break
        
        # If no single flip helps, try a few random double flips
        if not improved:
            for _ in range(10):
                if max_time is not None and time.time() - start_time > max_time:
                    return labels, current_cut
                
                i, j = np.random.choice(n, size=2, replace=False)
                if labels[i] == labels[j]:
                    # Flipping two same-label vertices
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    new_cut = _cut_size(adjacency, labels)
                    if new_cut > current_cut:
                        current_cut = new_cut
                        improved = True
                        break
                    else:
                        labels[i] = 1 - labels[i]
                        labels[j] = 1 - labels[j]
    
    return labels, current_cut


def _spectral_partition(adjacency):
    """Use the Fiedler vector (second eigenvector of Laplacian) for initialization."""
    n = len(adjacency)
    degree = adjacency.sum(axis=1)
    L = np.diag(degree) - adjacency.astype(float)
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
        labels = (fiedler >= 0).astype(int)
        return labels
    except Exception:
        return np.random.randint(0, 2, size=n)


def _simulated_annealing(adjacency, labels, max_time=None, T0=10.0, cooling=0.999, min_T=0.1):
    """Simulated annealing for maximum cut with quantum-inspired schedule.
    
    Uses a modified cooling schedule that slows down near convergence to allow
    better exploration, and accepts non-improving moves with probability exp(-delta_E / T).
    """
    n = len(adjacency)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    if max_time is None:
        max_time = 145.0
    
    start_time = time.time()
    T = T0
    no_improve_count = 0
    max_no_improve = 500
    
    while T > min_T:
        if time.time() - start_time > max_time:
            break
        
        # Pick a random vertex to flip
        v = np.random.randint(n)
        
        gain = _flip_gain(adjacency, labels, v)
        
        if gain >= 0:
            # Always accept improving or neutral moves
            labels[v] = 1 - labels[v]
            current_cut += gain
            no_improve_count = 0
        else:
            # Accept with probability exp(gain / T) since gain is negative
            prob = np.exp(gain / T)
            if np.random.random() < prob:
                labels[v] = 1 - labels[v]
                current_cut += gain
            no_improve_count += 1
        
        # Track best
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            no_improve_count = 0
        
        # Quantum-inspired: if stuck, temporarily increase temperature
        if no_improve_count > max_no_improve:
            T = max(T, 0.5 * T0)
            no_improve_count = 0
        
        # Cool down with adaptive rate
        # Slow down cooling when close to convergence
        if T < 0.5 * T0:
            T *= cooling * 0.95
        else:
            T *= cooling
    
    return best_labels, best_cut


def _iterated_simulated_annealing(adjacency, n, max_time=None, num_restarts=10):
    """Run simulated annealing from multiple random starts."""
    if max_time is None:
        max_time = 145.0
    
    start_time = time.time()
    best_labels = None
    best_cut = -1
    
    for i in range(num_restarts):
        if time.time() - start_time > max_time - 5:
            break
        
        # Random initialization
        init_labels = np.random.randint(0, 2, size=n)
        
        labels, cut = _simulated_annealing(adjacency, init_labels, max_time=max_time - (time.time() - start_time))
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def _tabu_search(adjacency, labels, max_time=None, tabu_size=10):
    """Tabu search for maximum cut."""
    n = len(adjacency)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    if max_time is None:
        max_time = 145.0
    
    start_time = time.time()
    
    # Tabu list: (vertex, time_added)
    tabu = {}
    
    while time.time() - start_time < max_time:
        # Find best move (including tabu moves)
        best_gain = -np.inf
        best_v = None
        best_is_tabu = False
        
        order = np.random.permutation(n)
        for v in order:
            gain = _flip_gain(adjacency, labels, v)
            is_tabu = v in tabu and (time.time() - tabu[v]) < tabu_size
            
            if not is_tabu:
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
                    best_is_tabu = False
            else:
                # Aspiration: allow tabu move if it improves best
                if gain > best_cut:
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                        best_is_tabu = True
        
        if best_v is None:
            break
        
        # Make the move
        labels[best_v] = 1 - labels[best_v]
        current_cut += best_gain
        tabu[best_v] = time.time()
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    def _update_best(labels, cut):
        nonlocal best_labels, best_cut
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Phase 1: Local search from diverse initializations
    initializations = []
    
    # 1. Spectral partition
    initializations.append(_spectral_partition(adjacency))
    
    # 2. Random
    initializations.append(_random_labels(n))
    
    # 3. Greedy
    initializations.append(_greedy_partition(adjacency))
    
    # 4. Alternating
    initializations.append(np.arange(n) % 2)
    
    # 5. Another random
    initializations.append(_random_labels(n))
    
    for init_labels in initializations:
        if time.time() > deadline - 25:
            break
        
        labels, cut = _local_search(adjacency, init_labels, max_time=deadline - 5)
        _update_best(labels, cut)
    
    # Phase 2: Two-flip local search from best
    if best_labels is not None and time.time() < deadline - 20:
        labels, cut = _two_flip_local_search(adjacency, best_labels, max_time=deadline - 10)
        _update_best(labels, cut)
    
    # Phase 3: Simulated annealing from best and random starts
    if best_labels is not None:
        if time.time() < deadline - 15:
            labels, cut = _simulated_annealing(adjacency, best_labels, max_time=deadline - 5, T0=5.0, cooling=0.9995, min_T=0.05)
            _update_best(labels, cut)
    
    # Run SA from a couple random starts
    for _ in range(2):
        if time.time() > deadline - 15:
            break
        
        init_labels = _random_labels(n)
        labels, cut = _simulated_annealing(adjacency, init_labels, max_time=deadline - 5, T0=5.0, cooling=0.9995, min_T=0.05)
        _update_best(labels, cut)
    
    # Phase 4: Tabu search from best
    if best_labels is not None and time.time() < deadline - 10:
        labels, cut = _tabu_search(adjacency, best_labels, max_time=deadline - 5, tabu_size=8)
        _update_best(labels, cut)
    
    # Phase 5: Final local search polish on the best
    if best_labels is not None and time.time() < deadline - 3:
        labels, cut = _local_search(adjacency, best_labels, max_time=deadline - 2)
        _update_best(labels, cut)
    
    if best_labels is None:
        best_labels = _random_labels(n)
    
    return best_labels
# EVOLVE-BLOCK-END
