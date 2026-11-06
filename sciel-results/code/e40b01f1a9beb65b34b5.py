"""Simulated Annealing + Local Search for Maximum Cut."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _compute_cut(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Cut = sum of adj[v,u] where labels[v] != labels[u]
    # Using matrix: cut = (labels[:, None] != labels[None, :]).sum() * adj / 2
    # More efficient: use the fact that cut = sum_v sum_u adj[v,u] * (labels[v] != labels[u])
    # Since adj is symmetric, we can compute:
    # For each vertex v, the cut contribution is sum of adj[v, u] for u where labels[u] != labels[v]
    # Total cut = (1/2) * sum over all directed edges crossing
    # Alternatively: cut = total_edges - (sum of edges within same partition)
    # Let's use: cut = (labels @ adj @ labels_complement) ... no that's not right.
    
    # Simple vectorized approach:
    # cut = np.sum(adj * (labels[:, None] != labels[None, :])) // 2
    # But this creates an n x n boolean matrix. For n=100, that's fine.
    
    diff = (labels[:, None] != labels[None, :])
    return np.sum(adj * diff) // 2


def _local_search(adj, labels, max_iters=1000, time_limit=None, start_time=None):
    """Greedy local search: flip vertices if it improves the cut."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _compute_cut(adj, labels)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # For each vertex, compute the gain from flipping
        # If vertex v is in group 0, flipping to 1:
        #   gain = (edges from v to group 1) - (edges from v to group 0)
        #         = deg_to_1[v] - deg_to_0[v]
        # If vertex v is in group 1, flipping to 0:
        #   gain = deg_to_0[v] - deg_to_1[v]
        
        # deg_to_0[v] = sum of adj[v, u] for u in group 0
        # deg_to_1[v] = sum of adj[v, u] for u in group 1
        
        mask0 = (labels == 0)
        mask1 = (labels == 1)
        
        deg_to_0 = adj @ mask0.astype(np.int8)
        deg_to_1 = adj @ mask1.astype(np.int8)
        
        # For vertices in group 0, gain from flipping to 1 = deg_to_1 - deg_to_0
        # For vertices in group 1, gain from flipping to 0 = deg_to_0 - deg_to_1
        gain = np.where(labels == 0, deg_to_1 - deg_to_0, deg_to_0 - deg_to_1)
        
        # Find the vertex with maximum positive gain
        max_gain = np.max(gain)
        if max_gain <= 0:
            break
        
        # Flip all vertices with positive gain? No, let's do one at a time for safety
        # Actually, let's find all vertices with positive gain and flip them simultaneously
        # This is a common heuristic but may not be optimal. Let's do sequential.
        
        best_v = np.argmax(gain)
        labels[best_v] = 1 - labels[best_v]
        current_cut += gain[best_v]
        improved = True
    
    return labels, current_cut


def _simulated_annealing(adj, n, start_time, time_limit):
    """Run simulated annealing for the remaining time."""
    labels = np.random.randint(0, 2, n)
    current_cut = _compute_cut(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Cooling schedule
    T_start = 10.0
    T_end = 0.1
    # We'll run for the remaining time
    
    while time.time() - start_time < time_limit - 5:  # 5 second safety margin
        # Pick a random vertex to flip
        v = np.random.randint(n)
        old_label = labels[v]
        
        # Compute gain from flipping v
        mask0 = (labels == 0)
        mask1 = (labels == 1)
        deg_to_0 = adj[v] @ mask0.astype(np.int8)
        deg_to_1 = adj[v] @ mask1.astype(np.int8)
        
        if old_label == 0:
            delta = deg_to_1 - deg_to_0
        else:
            delta = deg_to_0 - deg_to_1
        
        labels[v] = 1 - old_label
        current_cut += delta
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
        
        # Accept or reject
        if delta < 0:
            # Compute temperature based on progress
            elapsed = time.time() - start_time
            total_time = time_limit - start_time
            if total_time > 0:
                progress = min(1.0, elapsed / total_time)
            else:
                progress = 1.0
            T = T_start * (T_end / T_start) ** progress
            
            if np.random.random() < np.exp(delta / max(T, 1e-10)):
                pass  # Accept
            else:
                # Reject
                labels[v] = old_label
                current_cut -= delta
    
    return best_labels, best_cut


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses multiple restarts with local search and simulated annealing.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    start_time = time.time()
    time_limit = 110.0  # Safety margin from 120s limit
    
    best_labels = np.zeros(n, dtype=np.int8)
    best_cut = 0
    
    # Strategy: Use remaining time for multiple approaches
    # 1. Quick local search from several random starts
    # 2. Simulated annealing for the rest of the time
    
    # Phase 1: Multiple quick local searches from random starts
    phase1_time = min(30.0, time_limit * 0.3)
    phase1_deadline = start_time + phase1_time
    
    np.random.seed(42)  # Reproducible but varied
    
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > phase1_deadline:
            break
        
        # Random start
        labels = np.random.randint(0, 2, n)
        
        # Local search
        labels, cut = _local_search(adj, labels, max_iters=500, 
                                     time_limit=phase1_deadline - time.time(),
                                     start_time=start_time)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Phase 2: Simulated annealing from the best found so far
    if time.time() < time_limit - 5:
        sa_start = time.time()
        sa_labels, sa_cut = _simulated_annealing(adj, n, sa_start, time_limit - 5)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Final local search polish
        if time.time() < time_limit - 2:
            best_labels, best_cut = _local_search(adj, best_labels, max_iters=1000,
                                                   time_limit=time_limit - time.time(),
                                                   start_time=start_time)
    
    # Ensure we return within the time limit
    return best_labels
# EVOLVE-BLOCK-END
