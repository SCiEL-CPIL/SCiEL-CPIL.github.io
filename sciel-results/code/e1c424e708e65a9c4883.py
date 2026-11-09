# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Fixed problem data
N = 10
DEMANDS = [
    (0, 5, 3),
    (1, 7, 2),
    (2, 9, 4),
    (3, 6, 1),
    (0, 4, 5),
    (2, 8, 2),
    (5, 9, 3),
    (1, 4, 2),
    (6, 0, 4),
    (3, 8, 1),
    (7, 2, 3),
    (4, 9, 2),
    (8, 1, 5),
    (0, 6, 1),
    (5, 2, 2),
]
D = len(DEMANDS)

# Precompute route incidence vectors for clockwise routing
CW_ROUTES = np.zeros((D, N), dtype=np.int8)
for i, (s, t, w) in enumerate(DEMANDS):
    dist = (t - s) % N
    for k in range(dist):
        CW_ROUTES[i, (s + k) % N] = 1

# Precompute weights vector
WEIGHTS = np.array([w for _, _, w in DEMANDS], dtype=np.float64)

# Precompute CCW routes
CCW_ROUTES = 1 - CW_ROUTES


def _compute_max_load(routing: np.ndarray) -> float:
    """Compute the maximum edge load given a routing decision."""
    cw_mask = (routing < 0.5)
    ccw_mask = ~cw_mask
    
    cw_weights = np.where(cw_mask, WEIGHTS, 0.0)
    ccw_weights = np.where(ccw_mask, WEIGHTS, 0.0)
    
    cw_load = CW_ROUTES.T @ cw_weights
    ccw_load = CCW_ROUTES.T @ ccw_weights
    total_load = cw_load + ccw_load
    
    return float(np.max(total_load))


def _exhaustive_search() -> np.ndarray:
    """Exhaustive search over all 2^15 = 32768 possibilities using vectorized numpy."""
    masks = np.arange(1 << D, dtype=np.int32)
    routing_matrix = np.zeros((1 << D, D), dtype=np.float64)
    for j in range(D):
        routing_matrix[:, j] = (masks >> j) & 1
    
    cw_mask = routing_matrix < 0.5
    ccw_mask = ~cw_mask
    
    cw_weights = cw_mask * WEIGHTS[np.newaxis, :]
    ccw_weights = ccw_mask * WEIGHTS[np.newaxis, :]
    
    cw_loads = cw_weights @ CW_ROUTES
    ccw_loads = ccw_weights @ CCW_ROUTES
    
    total_loads = cw_loads + ccw_loads
    max_loads = total_loads.max(axis=1)
    
    best_idx = np.argmin(max_loads)
    best_routing = routing_matrix[best_idx].copy()
    
    return best_routing


def _lp_relaxation_solve():
    """
    Solve the LP relaxation of the ring routing problem.
    Variables: x_0..x_14 (fractional routing), t (max load)
    Minimize t subject to:
      For each edge e: sum_i (CW_ROUTES[i,e]*x_i + CCW_ROUTES[i,e]*(1-x_i)) * w_i <= t
      0 <= x_i <= 1
    """
    # Total variables: D + 1 (x_0..x_{D-1}, t)
    n_vars = D + 1
    
    # Objective: minimize t (coefficient 1 for t, 0 for x's)
    c = np.zeros(n_vars)
    c[D] = 1.0  # t is the last variable
    
    # Constraints: A_ub @ vars <= b_ub
    # For each edge e:
    #   sum_i [CW_ROUTES[i,e]*x_i + CCW_ROUTES[i,e]*(1-x_i)] * w_i <= t
    #   sum_i [CW_ROUTES[i,e]*w_i*x_i + CCW_ROUTES[i,e]*w_i - CCW_ROUTES[i,e]*w_i*x_i] <= t
    #   sum_i [(CW_ROUTES[i,e] - CCW_ROUTES[i,e])*w_i*x_i] + sum_i CCW_ROUTES[i,e]*w_i <= t
    #   sum_i [(CW_ROUTES[i,e] - CCW_ROUTES[i,e])*w_i*x_i] - t <= -sum_i CCW_ROUTES[i,e]*w_i
    
    A_ub = np.zeros((N, n_vars))
    b_ub = np.zeros(N)
    
    for e in range(N):
        for i in range(D):
            coeff = (CW_ROUTES[i, e] - CCW_ROUTES[i, e]) * WEIGHTS[i]
            A_ub[e, i] = coeff
        A_ub[e, D] = -1.0  # -t
        b_ub[e] = -sum(CCW_ROUTES[i, e] * WEIGHTS[i] for i in range(D))
    
    # Bounds: 0 <= x_i <= 1, t >= 0
    bounds = [(0, 1)] * D + [(0, None)]
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.status == 0:
        lp_solution = result.x[:D].copy()
        lp_t = result.x[D]
        return lp_solution, lp_t
    return None, None


def _round_from_lp(lp_solution: np.ndarray) -> np.ndarray:
    """
    Round the LP solution using a greedy approach.
    Sort demands by how close they are to 0 or 1, fix them first,
    then determine the rest greedily.
    """
    routing = np.zeros(D)
    fixed = np.zeros(D, dtype=bool)
    
    # Sort by distance to nearest integer (0 or 1)
    dist_to_int = np.minimum(lp_solution, 1 - lp_solution)
    order = np.argsort(dist_to_int)
    
    for idx in order:
        if fixed[idx]:
            continue
        # Fix this variable to nearest integer
        routing[idx] = 1.0 if lp_solution[idx] > 0.5 else 0.0
        fixed[idx] = True
    
    return routing


def _local_search_from_lp(lp_solution: np.ndarray, time_limit: float) -> np.ndarray:
    """
    Local search starting from the LP-rounded solution.
    Uses simulated annealing with 1-flip and 2-flip moves.
    """
    start_time = time.time()
    
    # Start from LP-rounded solution
    current_routing = _round_from_lp(lp_solution)
    current_max = _compute_max_load(current_routing)
    
    best_routing = current_routing.copy()
    best_max = current_max
    
    rng = np.random.default_rng(123)
    temp = 1.5
    cooling = 0.999
    
    step = 0
    while time.time() - start_time < time_limit:
        step += 1
        
        # Choose move type: 1-flip (80%) or 2-flip (20%)
        if rng.random() < 0.8:
            i = rng.integers(0, D)
            current_routing[i] = 1.0 - current_routing[i]
            new_max = _compute_max_load(current_routing)
            
            delta = new_max - current_max
            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
                current_max = new_max
                if current_max < best_max - 1e-12:
                    best_max = current_max
                    best_routing = current_routing.copy()
            else:
                current_routing[i] = 1.0 - current_routing[i]
        else:
            i = rng.integers(0, D)
            j = rng.integers(0, D)
            if i != j:
                current_routing[i] = 1.0 - current_routing[i]
                current_routing[j] = 1.0 - current_routing[j]
                new_max = _compute_max_load(current_routing)
                
                delta = new_max - current_max
                if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
                    current_max = new_max
                    if current_max < best_max - 1e-12:
                        best_max = current_max
                        best_routing = current_routing.copy()
                else:
                    current_routing[i] = 1.0 - current_routing[i]
                    current_routing[j] = 1.0 - current_routing[j]
        
        temp *= cooling
    
    return best_routing


def _greedy_initial_solution() -> np.ndarray:
    """Greedy construction: start with all clockwise, then flip demands to reduce max load."""
    routing = np.zeros(D)
    current_max = _compute_max_load(routing)
    
    for _ in range(5):
        improved = False
        for i in range(D):
            routing[i] = 1.0 - routing[i]
            new_max = _compute_max_load(routing)
            if new_max < current_max - 1e-9:
                current_max = new_max
                improved = True
            else:
                routing[i] = 1.0 - routing[i]
        if not improved:
            break
    
    return routing


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    
    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin
    
    # Phase 1: Exhaustive search gives the optimal integer solution immediately
    # 2^15 = 32768 possibilities, very fast with vectorized numpy
    exhaustive_routing = _exhaustive_search()
    exhaustive_max = _compute_max_load(exhaustive_routing)
    
    # Phase 2: LP relaxation to get a potentially better starting point
    lp_solution, lp_t = _lp_relaxation_solve()
    
    best_routing = exhaustive_routing.copy()
    best_max = exhaustive_max
    
    if lp_solution is not None:
        # Try local search from LP-rounded solution
        remaining = deadline - time.time() - 10.0
        if remaining > 2.0:
            lp_routing = _local_search_from_lp(lp_solution, remaining)
            lp_max = _compute_max_load(lp_routing)
            if lp_max < best_max - 1e-12:
                best_max = lp_max
                best_routing = lp_routing
    
    # Phase 3: Greedy + local search as backup
    remaining = deadline - time.time() - 5.0
    if remaining > 1.0:
        greedy_routing = _greedy_initial_solution()
        greedy_max = _compute_max_load(greedy_routing)
        
        if greedy_max < best_max - 1e-12:
            best_max = greedy_max
            best_routing = greedy_routing
    
    # Final safety: ensure valid output
    best_routing = np.where(best_routing >= 0.5, 1.0, 0.0).astype(np.float64)
    
    return best_routing


# EVOLVE-BLOCK-END
