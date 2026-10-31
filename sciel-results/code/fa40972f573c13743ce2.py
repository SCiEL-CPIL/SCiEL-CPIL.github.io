"""Minimax-inspired TSP constructive heuristic with per-instance parameter fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 130.0  # safety margin within 150s wall-clock kill

# Per-instance fitted parameters
_FIT_PARAMS = {}  # key: (n, seed_hash) -> (w_greedy, w_dest, w_minimax)
_FIT_DONE = False
_FIT_AT = 0.0

# Default parameters (will be tuned per-instance)
_W_GREEDY_DEFAULT = 1.0
_W_DEST_DEFAULT = 0.25
_W_MINIMAX_DEFAULT = 0.15

# Parameters to try during fitting
_PARAM_GRID = [
    (1.0, 0.2, 0.1),
    (1.0, 0.3, 0.15),
    (1.0, 0.4, 0.2),
    (0.9, 0.3, 0.2),
    (0.9, 0.4, 0.25),
    (0.8, 0.4, 0.3),
    (0.8, 0.5, 0.35),
    (0.7, 0.5, 0.4),
    (1.0, 0.15, 0.1),
    (1.0, 0.35, 0.2),
]


def _compute_scores(current_node, destination_node, unvisited_nodes, distance_matrix,
                    w_greedy, w_dest, w_minimax):
    """Compute composite scores for all unvisited nodes."""
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return None
    if n_unvisited == 1:
        return np.array([0.0])
    
    d = distance_matrix
    
    # Distance from current to each unvisited
    d_curr = d[current_node, unvisited_nodes]
    
    # Distance from each unvisited to destination
    d_dest = d[unvisited_nodes, destination_node]
    
    # Minimax term: distance from each unvisited to the farthest unvisited
    # This is a structural change: instead of minimizing sum, we penalize
    # nodes that would leave a very far node for later
    d_unvisited = d[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(d_unvisited, 0.0)
    max_dist_to_unvisited = np.max(d_unvisited, axis=1)
    
    # Also compute min distance to nearest other unvisited (connectivity)
    d_unvisited_min = d_unvisited.copy()
    np.fill_diagonal(d_unvisited_min, np.inf)
    min_dist_to_unvisited = np.min(d_unvisited_min, axis=1)
    
    # Composite score
    scores = (w_greedy * d_curr + 
              w_dest * d_dest + 
              w_minimax * max_dist_to_unvisited -
              0.1 * min_dist_to_unvisited)  # small bonus for being well-connected
    
    return scores


def _tour_length_heuristic(n, d, params, seed):
    """Simulate the full tour using the given parameters and return tour length."""
    rng = np.random.RandomState(seed)
    # Regenerate points to match the instance (approximation for fitting)
    # Actually, we should use the actual distance matrix, not regenerate
    # So this function is for testing with a known matrix
    
    cur = 0
    start = 0
    unv = set(range(1, n))
    L = 0.0
    w_greedy, w_dest, w_minimax = params
    
    while unv:
        arr = np.fromiter(unv, dtype=int)
        if len(arr) == 1:
            cur = int(arr[0])
            L += d[cur, cur]  # zero, just for structure
            unv.discard(cur)
            continue
        
        scores = _compute_scores(cur, start, arr, d, w_greedy, w_dest, w_minimax)
        if scores is None:
            break
        best_idx = int(np.argmin(scores))
        next_node = int(arr[best_idx])
        L += d[cur, next_node]
        cur = next_node
        unv.discard(cur)
    
    L += d[cur, start]
    return L


def _fit_params_for_instance(n, d, seed, budget):
    """Fit parameters for the current instance within a time budget."""
    best_params = (_W_GREEDY_DEFAULT, _W_DEST_DEFAULT, _W_MINIMAX_DEFAULT)
    best_tour = np.inf
    
    t_end = time.time() + budget
    for params in _PARAM_GRID:
        if time.time() > t_end:
            break
        tour = _tour_length_heuristic(n, d, params, seed)
        if tour < best_tour:
            best_tour = tour
            best_params = params
    
    return best_params


def _get_params(n, d, seed):
    """Get fitted parameters for the current instance."""
    global _FIT_DONE, _FIT_AT
    
    key = (n, seed)
    if key in _FIT_PARAMS:
        return _FIT_PARAMS[key]
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        return (_W_GREEDY_DEFAULT, _W_DEST_DEFAULT, _W_MINIMAX_DEFAULT)
    
    # Fit parameters if not done yet
    if not _FIT_DONE:
        # Use a small budget for fitting
        budget = min(8.0, max(0.5, _DEADLINE - time.time() - 10.0))
        if budget > 0.5:
            params = _fit_params_for_instance(n, d, seed, budget)
            _FIT_PARAMS[key] = params
            _FIT_DONE = True
            _FIT_AT = time.time()
            return params
    
    return (_W_GREEDY_DEFAULT, _W_DEST_DEFAULT, _W_MINIMAX_DEFAULT)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city in a TSP tour using a minimax-inspired composite heuristic.
    
    Strategy: For each candidate node j in unvisited_nodes, compute a composite score:
      1. Distance from current_node to j (greedy)
      2. Distance from j to destination_node (goal-seeking)
      3. Max distance from j to any other unvisited node (minimax/bottleneck)
      4. Small bonus for being well-connected to other unvisited nodes
    
    The minimax term fundamentally changes the objective from cumulative cost
    to bottleneck cost, aiming to prevent any single long edge from dominating.
    
    Parameters are fitted per-instance to adapt to the specific geometry.
    """
    unvisited = np.asarray(unvisited_nodes)
    n_unvisited = len(unvisited)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited[0])
    
    # Check deadline
    if time.time() > _DEADLINE - 2.0:
        # Fallback: pure greedy nearest neighbor
        d_curr = distance_matrix[current_node, unvisited]
        return int(unvisited[int(np.argmin(d_curr))])
    
    n = int(distance_matrix.shape[0])
    d = distance_matrix
    
    # Derive a seed from the instance (use hash of a few elements)
    seed = int(hash((n, tuple(d[:3, :3].flatten()))) % (2**31))
    
    # Get fitted parameters
    w_greedy, w_dest, w_minimax = _get_params(n, d, seed)
    
    # Compute scores
    scores = _compute_scores(current_node, destination_node, unvisited, d,
                             w_greedy, w_dest, w_minimax)
    
    if scores is None:
        return int(current_node)
    
    best_idx = int(np.argmin(scores))
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
