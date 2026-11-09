"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # labels is a 0/1 array of length n
    # cut = sum_{v,u: adj[v,u]=1, labels[v]!=labels[u]} 1
    # Using matrix operations:
    # For each v, count neighbors u where labels[u] != labels[v]
    # If labels[v]=0: count neighbors with label 1
    # If labels[v]=1: count neighbors with label 0
    # Total cut = sum_v adj[v] @ (1 - labels) if labels[v]=0, or adj[v] @ labels if labels[v]=1
    # More cleanly: cut = 0.5 * sum_{v,u} adj[v,u] * (labels[v] != labels[u])
    # = 0.5 * (sum_v labels[v] * sum_u adj[v,u]*(1-labels[u]) + sum_v (1-labels[v]) * sum_u adj[v,u]*labels[u])
    # Simpler vectorized: 
    # Let x = labels (0/1). The cut is sum_{v<u} adj[v,u] * (x[v] != x[u])
    # = sum_{v<u} adj[v,u] * (x[v] + x[u] - 2*x[v]*x[u])  [since x in {0,1}, x[v]!=x[u] iff x[v]+x[u]=1]
    # Actually simpler: cut = x^T (D - A) x / ... no.
    # Let's just do: for each v, the number of neighbors not in same set.
    # neighbors_same = adj @ labels  (if labels are 0/1, this gives for each v the count of neighbors with label 1)
    # If labels[v] == 1: same_count = (adj @ labels)[v], diff_count = degree[v] - same_count
    # If labels[v] == 0: same_count = degree[v] - (adj @ labels)[v], diff_count = (adj @ labels)[v]
    # Total cut = sum of diff_count over all v, but this counts each edge twice? No.
    # Wait: each edge (v,u) with different labels is counted once from v's perspective and once from u's? 
    # No: from v's perspective, we count edge (v,u) if labels differ. From u's perspective, we also count it.
    # So sum_v diff_count[v] = 2 * cut_size.
    # Thus cut_size = 0.5 * sum_v diff_count[v].
    
    degrees = adj.sum(axis=1)
    adj_labels = adj @ labels  # for each v, number of neighbors with label 1
    # For v with label 1: diff = degrees[v] - adj_labels[v]
    # For v with label 0: diff = adj_labels[v]
    diff = np.where(labels == 1, degrees - adj_labels, adj_labels)
    return 0.5 * diff.sum()


def _spectral_init(adj, n):
    """Initialize labels using the Fiedler vector (second eigenvector of the Laplacian)."""
    degrees = adj.sum(axis=1)
    D = np.diag(degrees)
    L = D - adj
    
    # Get eigenvalues and eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Sort by eigenvalue (eigh returns in ascending order)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # For a connected graph, the first eigenvector is constant, second is Fiedler
        fiedler = eigvecs[:, 1]
        # Partition: vertices with positive fiedler value go to 0, negative to 1
        labels = (fiedler < 0).astype(np.int64)
        return labels
    except Exception:
        # Fallback to random
        return np.random.randint(0, 2, n)


def _greedy_flip_improve(adj, labels, max_iter=100, time_budget=None, start_time=None):
    """Greedy local search: flip a vertex if it improves the cut."""
    n = len(labels)
    labels = labels.copy()
    degrees = adj.sum(axis=1)
    
    # Precompute current cut
    current_cut = _cut_size(adj, labels)
    
    # For efficiency, track for each vertex the gain of flipping
    # Gain of flipping v = (neighbors in opposite set) - (neighbors in same set)
    # = 2 * (neighbors in opposite set) - degree[v]
    
    for iteration in range(max_iter):
        if time_budget is not None and start_time is not None:
            if time.time() - start_time > time_budget:
                break
        
        improved = False
        for v in range(n):
            # Count neighbors in same and opposite sets
            same_count = (adj[v] & (labels == labels[v])).sum()
            opp_count = degrees[v] - same_count
            
            # If we flip v, the new contribution from v's edges changes
            # Current contribution: opp_count (edges from v crossing)
            # After flip: same_count (edges from v crossing, since same becomes opposite)
            # Gain = same_count - opp_count
            if same_count > opp_count:
                labels[v] = 1 - labels[v]
                current_cut += same_count - opp_count
                improved = True
        
        if not improved:
            break
    
    return labels, current_cut


def _simulated_annealing(adj, labels, time_budget, start_time):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    degrees = adj.sum(axis=1)
    
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Temperature schedule
    T_start = max(degrees) * 0.5 if max(degrees) > 0 else 1.0
    T_end = 1e-6
    cooling = 0.995
    
    T = T_start
    
    rng = np.random.default_rng(42)
    
    while T > T_end:
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random vertex to flip
        v = rng.integers(0, n)
        
        # Compute gain of flipping v
        same_count = (adj[v] & (labels == labels[v])).sum()
        opp_count = degrees[v] - same_count
        delta = same_count - opp_count
        
        # Accept if improvement, or with probability exp(delta/T) if worsening
        if delta >= 0:
            labels[v] = 1 - labels[v]
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            prob = np.exp(delta / T)
            if rng.random() < prob:
                labels[v] = 1 - labels[v]
                current_cut += delta
        
        T *= cooling
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Set a deadline: leave 5 seconds margin for safety
    deadline = time.time() + 115  # 115 seconds out of 120
    
    best_labels = None
    best_cut = -1
    
    # Strategy 1: Spectral initialization + greedy improvement
    labels_spec = _spectral_init(adj, n)
    labels_greedy, cut_greedy = _greedy_flip_improve(adj, labels_spec, max_iter=50)
    if cut_greedy > best_cut:
        best_cut = cut_greedy
        best_labels = labels_greedy.copy()
    
    # Strategy 2: Random restarts with greedy improvement
    rng = np.random.default_rng(123)
    num_random_restarts = 5
    
    for _ in range(num_random_restarts):
        if time.time() > deadline:
            break
        labels_rand = rng.integers(0, 2, n).astype(np.int64)
        labels_greedy, cut_greedy = _greedy_flip_improve(adj, labels_rand, max_iter=30)
        if cut_greedy > best_cut:
            best_cut = cut_greedy
            best_labels = labels_greedy.copy()
    
    # Strategy 3: Simulated annealing from the best solution found so far
    if time.time() < deadline - 10:
        remaining_time = min(30, deadline - time.time() - 5)
        if remaining_time > 2:
            labels_sa, cut_sa = _simulated_annealing(adj, best_labels, remaining_time, time.time())
            if cut_sa > best_cut:
                best_cut = cut_sa
                best_labels = labels_sa.copy()
    
    # Final greedy polish
    if time.time() < deadline:
        labels_final, cut_final = _greedy_flip_improve(adj, best_labels, max_iter=20, 
                                                        time_budget=deadline - time.time() - 2,
                                                        start_time=time.time())
        if cut_final > best_cut:
            best_cut = cut_final
            best_labels = labels_final.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
