"""Improved program for Maximum Independent Set greedy scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csc_matrix, csr_matrix


def _greedy_size(adjacency, order, neighbor_lists=None):
    """Compute independent set size given a vertex order (indices sorted by priority)."""
    n = len(order)
    available = np.ones(n, dtype=bool)
    count = 0
    for v in order:
        if available[v]:
            count += 1
            available[v] = False
            if neighbor_lists is not None:
                for u in neighbor_lists[v]:
                    available[u] = False
            else:
                neighbors = adjacency[v].astype(bool)
                available[neighbors] = False
    return count


def _evaluate_order(adjacency, priority_vec, neighbor_lists=None):
    """Evaluate a priority vector by running the greedy decoder."""
    n = len(priority_vec)
    order = np.argsort(-priority_vec, kind='stable')
    return _greedy_size(adjacency, order, neighbor_lists)


def _power_iteration_eigenvector(M, k=200, tol=1e-6, max_iter=50):
    """Power iteration to find the dominant eigenvector of M."""
    n = M.shape[0]
    v = np.random.randn(n)
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(max_iter):
        v_new = M.dot(v)
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            break
        v_new /= norm
        if np.linalg.norm(v_new - v) < tol:
            v = v_new
            break
        v = v_new
    return v


def _spectral_priority(adjacency, n, deadline):
    """Compute spectral-based priority using Fiedler vector of Laplacian."""
    A = csc_matrix(adjacency)
    degrees = np.asarray(A.sum(axis=1)).flatten()
    
    # Build Laplacian
    D = csc_matrix((degrees, (np.arange(n), np.arange(n))))
    L = D - A
    L = L.tocsr()
    
    # Get Fiedler vector via eigsh
    try:
        from scipy.sparse.linalg import eigsh
        if n > 10:
            vals, vecs = eigsh(L, k=min(3, n-1), which='SM', return_eigenvectors=True)
            # Use the eigenvector with smallest positive eigenvalue
            positive_mask = vals > 1e-10
            if np.any(positive_mask):
                pos_indices = np.where(positive_mask)[0]
                idx = pos_indices[np.argmin(vals[pos_indices])]
                fiedler_vec = vecs[:, idx]
            else:
                fiedler_vec = vecs[:, -1]
        else:
            L_dense = L.toarray()
            vals, vecs = np.linalg.eigh(L_dense)
            fiedler_vec = vecs[:, 1] if n > 1 else vecs[:, 0]
        
        fiedler_abs = np.abs(fiedler_vec)
        fiedler_abs = fiedler_abs / (np.max(fiedler_abs) + 1e-12)
        return fiedler_abs
    except Exception:
        return np.ones(n) / n


def _marginal_loss_scores(adjacency, n, neighbor_lists):
    """
    Compute marginal loss scores: for each vertex, how many vertices would be
    blocked if we picked it. Lower marginal loss = better to pick earlier.
    """
    # For each vertex v, marginal loss = number of available neighbors
    # that would be blocked. Initially all are available, so it's just degree.
    # But we can be smarter: vertices that neighbor many low-degree vertices
    # have lower effective marginal loss.
    
    degrees = np.array([len(neighbor_lists[i]) for i in range(n)], dtype=float)
    
    # Effective marginal loss: weighted by neighbor degrees
    # A vertex that neighbors high-degree vertices has higher marginal loss
    # because those neighbors would block many other vertices anyway.
    marginal_loss = np.zeros(n)
    for v in range(n):
        if degrees[v] == 0:
            marginal_loss[v] = 1.0  # Picking an isolated vertex always helps
        else:
            # Sum of degrees of neighbors / degree = average neighbor degree
            # Higher average neighbor degree means picking v blocks "more valuable" vertices
            avg_neighbor_deg = np.mean([degrees[u] for u in neighbor_lists[v]])
            marginal_loss[v] = 1.0 + avg_neighbor_deg
    
    return marginal_loss


def priority(adjacency):
    """
    Assign a score to each vertex such that a greedy decoder picking vertices
    in descending score order (skipping neighbors of already-picked vertices)
    yields a large independent set.
    
    Strategy:
    1. Compute initial priority from degree, spectral info, and marginal loss.
    2. Use local search (swap-based) to improve the ordering.
    """
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s limit
    
    n = adjacency.shape[0]
    A = csc_matrix(adjacency)
    
    # Precompute neighbor lists
    neighbor_lists = []
    for i in range(n):
        nbrs = np.nonzero(adjacency[i])[0]
        neighbor_lists.append(list(nbrs))
    
    # Compute base signals
    degrees = np.array([len(neighbor_lists[i]) for i in range(n)], dtype=float)
    
    # Spectral signal
    fiedler_abs = _spectral_priority(adjacency, n, DEADLINE)
    
    # Marginal loss
    marginal_loss = _marginal_loss_scores(adjacency, n, neighbor_lists)
    
    # Normalize signals to [0, 1]
    deg_norm = degrees / (np.max(degrees) + 1e-12)
    ml_norm = marginal_loss / (np.max(marginal_loss) + 1e-12)
    
    # Initial score: we want high score for vertices we pick early
    # Low degree, low marginal loss, low fiedler centrality are good
    score = -(0.35 * deg_norm + 0.4 * ml_norm + 0.25 * fiedler_abs)
    
    # Evaluate initial score
    current_size = _evaluate_order(adjacency, score, neighbor_lists)
    best_score = score.copy()
    best_size = current_size
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return best_score.astype(float)
    
    # Local search: try swaps to improve the greedy outcome
    # We work with the order (permutation) and try to improve it
    order = np.argsort(-score, kind='stable')
    best_order = order.copy()
    
    # Simulated annealing style local search on the order
    T = 1.0
    cooling = 0.995
    min_T = 0.01
    
    # Precompute adjacency as boolean matrix for fast lookups
    adj_bool = adjacency.astype(bool)
    
    while time.time() - start_time < DEADLINE:
        # Pick a random position in the order
        i = np.random.randint(0, n - 1)
        j = i + np.random.randint(1, n - i)
        if j >= n:
            j = n - 1
        
        # Swap positions i and j
        old_order = best_order.copy()
        best_order[i], best_order[j] = best_order[j], best_order[i]
        
        new_size = _greedy_size(adjacency, best_order, neighbor_lists)
        delta = new_size - best_size
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            best_size = new_size
        else:
            best_order = old_order
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Convert best_order back to scores: earlier in order = higher score
    # Assign scores so that argsort(-scores) gives best_order
    final_score = np.zeros(n)
    for rank, vertex in enumerate(best_order):
        final_score[vertex] = n - rank  # Higher score = earlier in greedy
    
    return final_score.astype(float)
# EVOLVE-BLOCK-END
