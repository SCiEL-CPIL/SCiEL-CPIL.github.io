"""Combined approach: Spectral init + Fast local search + SA with optimized delta updates."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.linalg import eigh


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _spectral_partition(adjacency, degrees, adj_float):
    """Use spectral method (Fiedler vector of Laplacian) to get initial partition."""
    n = len(adjacency)
    try:
        L = np.diag(degrees.astype(float)) - adj_float
        eigenvalues, eigenvectors = eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # If the graph is disconnected, there are multiple zero eigenvalues.
        # We pick index 1 which is generally the Fiedler vector for connected graphs.
        fiedler = eigenvectors[:, 1]
        labels = (fiedler > 0).astype(np.int32)
        # Ensure both partitions are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.arange(n) % 2
        return labels
    except Exception:
        return np.arange(n) % 2


def _initialize_state(labels, adj, degrees, neighbors):
    """Initialize same_count array and current cut."""
    n = len(labels)
    same_count = np.zeros(n, dtype=np.int32)
    for i in range(n):
        # same_count[i] = number of neighbors of i with same label as i
        # adj[i] is boolean/int array of neighbors
        same_count[i] = np.sum(adj[i] * (labels == labels[i]))
    
    # Cut size = 0.5 * sum(degrees - same_count)
    current_cut = int(np.sum(degrees - same_count) // 2)
    return same_count, current_cut


def _local_search(adjacency, labels, deadline, max_rounds=100):
    """Iterative local search: flip vertices if it improves cut size.
    Uses optimized delta updates."""
    n = len(labels)
    adj = np.asarray(adjacency, dtype=np.int8)
    degrees = adj.sum(axis=1)
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]
    
    labels = labels.copy()
    same_count, current_cut = _initialize_state(labels, adj, degrees, neighbors)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        # Compute flip gains for all vertices
        # delta = same_count[v] - (degrees[v] - same_count[v])
        # If delta > 0, flipping v increases the cut size
        same_v = same_count
        opp_v = degrees - same_v
        gains = same_v - opp_v
        
        # Find vertices where flipping improves the cut
        flip_candidates = np.where(gains > 0)[0]
        
        if len(flip_candidates) == 0:
            break
        
        # Flip one vertex at a time in the order of best gain to avoid interference
        # Or flip all at once? Flipping all at once can be unstable.
        # Let's flip the best one first, then re-evaluate? 
        # The previous program flipped all at once. Let's stick to sequential for stability in local search.
        # Actually, for speed, flipping all "improving" vertices simultaneously is a common heuristic (parallel update).
        # Let's try sequential but efficient.
        
        # Sort candidates by gain descending
        sorted_candidates = flip_candidates[np.argsort(-gains[flip_candidates])]
        
        for v in sorted_candidates:
            if time.time() > deadline:
                break
            
            same_v = same_count[v]
            opp_v = degrees[v] - same_v
            delta = same_v - opp_v
            
            if delta > 0:
                # Flip v
                old_label = labels[v]
                labels[v] = 1 - old_label
                
                # Update same_count for v and its neighbors
                same_count[v] = opp_v
                
                for u in neighbors[v]:
                    if labels[u] == old_label:
                        # u was same as v, now u is opposite to v
                        # u's same_count decreases by 1
                        same_count[u] -= 1
                    else:
                        # u was opposite to v, now u is same as v
                        # u's same_count increases by 1
                        same_count[u] += 1
                
                current_cut += delta
                improved = True
        
        if not improved:
            break
    
    return labels, current_cut


def _simulated_annealing(adjacency, labels, deadline, max_iters=5000):
    """Simulated annealing to escape local optima with optimized delta updates."""
    n = len(labels)
    adj = np.asarray(adjacency, dtype=np.int8)
    degrees = adj.sum(axis=1)
    neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]
    
    labels = labels.copy()
    same_count, current_cut = _initialize_state(labels, adj, degrees, neighbors)
    
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.1
    cooling = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    rng = np.random.default_rng()
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random vertex to flip
        v = rng.integers(0, n)
        
        # Delta calculation
        same_v = same_count[v]
        opp_v = degrees[v] - same_v
        delta = same_v - opp_v
        
        if delta >= 0 or rng.random() < np.exp(delta / T):
            # Flip v
            old_label = labels[v]
            labels[v] = 1 - old_label
            
            # Update same_count for v and its neighbors
            same_count[v] = opp_v
            
            for u in neighbors[v]:
                if labels[u] == old_label:
                    same_count[u] -= 1
                else:
                    same_count[u] += 1
            
            current_cut += delta
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        T *= cooling
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    # Precompute common structures
    adj = np.asarray(adjacency, dtype=np.int8)
    adj_float = adj.astype(float)
    degrees = adj.sum(axis=1)
    
    # 1. Spectral initialization
    labels = _spectral_partition(adjacency, degrees, adj_float)
    
    # 2. Local search to polish
    labels, best_cut = _local_search(adjacency, labels, deadline)
    best_labels = labels.copy()
    
    # 3. Simulated annealing with restarts to escape local optima
    num_restarts = 5
    rng_master = np.random.default_rng(12345)
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:  # Leave 10s for final polish
            break
        
        # Start from best found or random perturbation
        if r == 0:
            sa_labels = best_labels.copy()
        else:
            sa_labels = best_labels.copy()
            # Perturb some vertices
            num_perturb = max(1, n // 10)
            perturb_indices = rng_master.choice(n, num_perturb, replace=False)
            sa_labels[perturb_indices] = 1 - sa_labels[perturb_indices]
        
        sa_result, sa_cut = _simulated_annealing(adjacency, sa_labels, deadline)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_result.copy()
    
    # 4. Final local search polish
    if time.time() < deadline - 5.0:
        best_labels, best_cut = _local_search(adjacency, best_labels, deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
