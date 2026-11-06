"""
Strategy: Multi-Start with Diverse Initializations + Vectorized Simulated Annealing.
Combines multiple initialization strategies (Fiedler vector, top eigenvector, random, balanced random)
with a vectorized simulated annealing local search that accepts temporary quality drops to escape
local optima. The diverse initializations increase the chance of finding a globally better basin.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses multi-start initialization (Fiedler, spectral, random) + vectorized simulated annealing.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin before 150s limit
    
    # Check if A is all zeros
    if not np.any(adj):
        return np.zeros(n, dtype=np.int8)
    
    degrees = np.sum(adj, axis=1)
    
    # Helper: compute cut size for a given labeling (0/1)
    def cut_size(labels):
        # cut = sum over edges (i,j) where labels[i] != labels[j]
        # = 0.5 * sum_{i,j} adj[i,j] * (labels[i] != labels[j])
        # Using: (labels[i] != labels[j]) = 1 - (labels[i] == labels[j])
        # labels in {0,1}: labels[i]*labels[j] + (1-labels[i])*(1-labels[j]) = same
        # same = labels[i]*labels[j] + 1 - labels[i] - labels[j] + labels[j] = ... 
        # Simpler: diff = labels[i] XOR labels[j] = labels[i] + labels[j] - 2*labels[i]*labels[j]
        # For binary: if same, diff=0; if different, diff=1
        # cut = sum adj[i,j] * (labels[i] != labels[j])
        # = sum adj[i,j] * (labels[i] - labels[j])^2 / ... no
        # Let's use: for binary labels, labels[i] != labels[j] iff labels[i] + labels[j] = 1
        # cut = sum_{i<j} adj[i,j] * (labels[i] != labels[j])
        # Vectorized: 
        # same = (labels[:, None] == labels[None, :])
        # cut = np.sum(adj * (1 - same)) / 2  # since adj is symmetric, each edge counted twice
        same = (labels[:, None] == labels[None, :])
        return np.sum(adj * (1 - same)) / 2.0
    
    # Helper: greedy 1-flip polish
    def greedy_polish(labels, time_limit):
        improved = True
        while improved:
            if time.time() > time_limit:
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
        return labels
    
    # Helper: vectorized simulated annealing
    def simulated_annealing(labels, time_limit, seed=0):
        rng = np.random.RandomState(seed)
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        # Temperature schedule
        t0 = 5.0  # initial temperature
        t_end = 0.01
        # We'll run for a fixed number of iterations or until time limit
        # Estimate iterations per second: each iteration is O(n^2) for gain computation
        # For n=100, that's 10000 ops, very fast. Let's do many iterations.
        
        # Precompute degree for gain calculation
        # Gain of flipping vertex i:
        # If label[i] = 0, flipping to 1:
        #   Edges from i to same-side (0) become cut: +same_counts[i]
        #   Edges from i to diff-side (1) become non-cut: -diff_counts[i]
        #   Net gain = same_counts[i] - diff_counts[i] = same_counts[i] - (degree[i] - same_counts[i]) = 2*same_counts[i] - degree[i]
        # If label[i] = 1, flipping to 0:
        #   Same logic, gain = 2*same_counts[i] - degree[i]
        # So gain is the same formula regardless of current label.
        
        # We'll iterate, trying to flip each vertex with some probability
        # Vectorized approach: compute all gains at once, pick a random vertex, decide to accept
        
        max_iters = 500000
        iters_per_sec_estimate = 20000  # rough estimate
        
        # Adaptive: run until time limit
        while time.time() < time_limit:
            # Compute current gains for all vertices
            same_mask = (labels[:, None] == labels[None, :])
            same_counts = np.sum(adj * same_mask, axis=1)
            gains = 2 * same_counts - degrees
            
            # Pick a random vertex to flip
            idx = rng.randint(n)
            delta = gains[idx]
            
            # Annealing temperature: linearly cool
            # We need to know how many iterations we've done to set temperature
            # Instead, use time-based cooling
            elapsed = time.time() - start_time
            remaining = max(time_limit - time.time(), 0)
            # Simple: use a fixed number of iterations and cool linearly
            # Let's use iteration-based cooling
            break  # placeholder, we'll restructure
        
        # Better approach: fixed iterations with linear cooling
        return best_labels, best_cut
    
    # Let me restructure the simulated annealing to be cleaner
    def sa_search(labels, time_limit, seed=0):
        rng = np.random.RandomState(seed)
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        t0 = 10.0
        t_end = 0.001
        # Estimate how many iterations we can do
        # Each iteration: compute gains (O(n^2)), pick random vertex, decide accept
        # For n=100, gains computation is ~100*100 = 10000 multiply-adds, very fast
        # Let's aim for ~200k iterations total
        
        # First, let's do a quick test to estimate speed
        test_start = time.time()
        same_mask = (labels[:, None] == labels[None, :])
        same_counts = np.sum(adj * same_mask, axis=1)
        gains = 2 * same_counts - degrees
        test_elapsed = time.time() - test_start
        
        # Estimate iterations per second
        if test_elapsed > 0:
            iters_per_sec = 1.0 / test_elapsed
        else:
            iters_per_sec = 50000
        
        # Available time for SA
        avail_time = max(time_limit - time.time(), 0.1)
        max_iters = int(avail_time * iters_per_sec * 0.8)  # 80% of available time
        max_iters = max(max_iters, 1000)
        
        # Linear cooling schedule
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Compute gains
            same_mask = (labels[:, None] == labels[None, :])
            same_counts = np.sum(adj * same_mask, axis=1)
            gains = 2 * same_counts - degrees
            
            # Pick random vertex
            idx = rng.randint(n)
            delta = gains[idx]
            
            # Temperature
            progress = it / max(max_iters - 1, 1)
            T = t0 * (t_end / t0) ** progress
            
            # Decide to accept
            if delta > 0:
                accept = True
            else:
                # Accept with probability exp(delta / T)
                if T < 1e-10:
                    accept = False
                else:
                    accept = rng.rand() < np.exp(delta / T)
            
            if accept:
                labels[idx] = 1 - labels[idx]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        return best_labels, best_cut
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fiedler vector (second smallest eigenvalue of Laplacian)
    try:
        L = np.diag(degrees) - adj
        # For n=100, we can compute all eigenvalues
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is eigenvector corresponding to second smallest eigenvalue
        fiedler = eigvecs[:, 1]
        # Round to 0/1
        init_fiedler = (fiedler > 0).astype(np.int8)
        initializations.append(init_fiedler)
    except Exception:
        pass
    
    # 2. Top eigenvector of adjacency matrix
    try:
        np.random.seed(42)
        x = np.random.randn(n)
        x = x / np.linalg.norm(x)
        for _ in range(200):
            Ax = adj @ x
            norm = np.linalg.norm(Ax)
            if norm < 1e-12:
                break
            x = Ax / norm
        init_top = (x > 0).astype(np.int8)
        initializations.append(init_top)
    except Exception:
        pass
    
    # 3. Random balanced initialization
    for seed in [1, 2, 3]:
        rng = np.random.RandomState(seed)
        init_rand = rng.randint(0, 2, size=n).astype(np.int8)
        initializations.append(init_rand)
    
    # 4. Greedy initialization (start all 0s, greedily flip)
    init_greedy = np.zeros(n, dtype=np.int8)
    initializations.append(init_greedy)
    
    # Remove duplicates (by comparing as tuples)
    unique_inits = []
    seen = set()
    for init in initializations:
        key = tuple(init.tolist())
        if key not in seen:
            seen.add(key)
            unique_inits.append(init)
    
    # For each initialization, run greedy polish then SA
    best_overall = None
    best_cut_overall = -1
    
    # Budget: split time among initializations
    num_inits = len(unique_inits)
    # Reserve some time for the best one to do more SA
    per_init_time = (deadline - 5.0 - start_time) / max(num_inits, 1)
    
    for i, init in enumerate(unique_inits):
        if time.time() > deadline - 5.0:
            break
        
        # Greedy polish first
        greedy_limit = min(time.time() + per_init_time * 0.3, deadline - 10.0)
        polished = greedy_polish(init.copy(), greedy_limit)
        cut_val = cut_size(polished)
        
        # Simulated annealing
        sa_limit = min(time.time() + per_init_time * 0.7, deadline - 5.0)
        if sa_limit > time.time() + 0.1:
            sa_result, sa_cut = sa_search(polished, sa_limit, seed=i + 10)
            if sa_cut > cut_val:
                polished = sa_result
                cut_val = sa_cut
        
        if cut_val > best_cut_overall:
            best_cut_overall = cut_val
            best_overall = polished.copy()
    
    # Final greedy polish on the best result
    if best_overall is not None and time.time() < deadline - 2.0:
        final = greedy_polish(best_overall, deadline - 1.0)
        if cut_size(final) >= best_cut_overall - 1e-9:
            best_overall = final
    
    if best_overall is None:
        # Fallback: random
        best_overall = np.random.randint(0, 2, size=n).astype(np.int8)
    
    return best_overall
# EVOLVE-BLOCK-END
