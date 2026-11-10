# EVOLVE-BLOCK-START
import numpy as np
import time
import random

# Precompute the ring structure
N_NODES = 10
N_EDGES = 10
N_DEMANDS = 15

# Demands: (source, target, weight)
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


def _build_edge_masks():
    """
    For each demand, precompute which edges are used if routed clockwise
    and which if routed counter-clockwise.
    
    Clockwise from s to t: edges s, s+1, ..., t-1 (mod N_NODES)
    Counter-clockwise: all other edges.
    """
    cw_masks = []
    ccw_masks = []
    for s, t, w in DEMANDS:
        cw_mask = np.zeros(N_EDGES)
        # Clockwise: edges s, s+1, ..., t-1 mod N
        if s <= t:
            for e in range(s, t):
                cw_mask[e] = 1
        else:
            for e in range(s, N_NODES):
                cw_mask[e] = 1
            for e in range(0, t):
                cw_mask[e] = 1
        
        ccw_mask = 1.0 - cw_mask
        cw_masks.append(cw_mask)
        ccw_masks.append(ccw_mask)
    
    return np.array(cw_masks), np.array(ccw_masks)


# Precompute masks
CW_MASKS, CCW_MASKS = _build_edge_masks()

# Weights array
WEIGHTS = np.array([w for s, t, w in DEMANDS], dtype=np.float64)

# Precompute delta vectors for fast load updates
# delta[i] = w_i * (ccw_mask_i - cw_mask_i)
# If we switch from CW to CCW: loads += delta[i]
# If we switch from CCW to CW: loads -= delta[i]
DELTAS = WEIGHTS[:, np.newaxis] * (CCW_MASKS - CW_MASKS)


def _compute_loads(routing):
    """
    Given a routing vector (0=cw, 1=ccw) of shape (N_DEMANDS,),
    compute the load on each edge.
    """
    cw_weights = (1.0 - routing) * WEIGHTS
    ccw_weights = routing * WEIGHTS
    loads = CW_MASKS.T @ cw_weights + CCW_MASKS.T @ ccw_weights
    return loads


def _max_load(routing):
    """Compute the maximum edge load for a given routing."""
    loads = _compute_loads(routing)
    return np.max(loads)


def _local_search_1bit(routing, max_iters=100000):
    """
    1-bit flip local search: try flipping each bit, keep if it improves.
    Repeat until no improvement.
    """
    routing = routing.copy()
    current_max = _max_load(routing)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(N_DEMANDS):
            new_routing = routing.copy()
            new_routing[i] = 1 - new_routing[i]
            new_max = _max_load(new_routing)
            if new_max < current_max - 1e-12:
                routing = new_routing
                current_max = new_max
                improved = True
    
    return routing, current_max


def _local_search_2bit(routing, max_iters=50000):
    """
    2-bit flip local search: try flipping each pair of bits, keep if it improves.
    This helps escape 1-bit local optima.
    """
    routing = routing.copy()
    current_max = _max_load(routing)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(N_DEMANDS):
            for j in range(i + 1, N_DEMANDS):
                new_routing = routing.copy()
                new_routing[i] = 1 - new_routing[i]
                new_routing[j] = 1 - new_routing[j]
                new_max = _max_load(new_routing)
                if new_max < current_max - 1e-12:
                    routing = new_routing
                    current_max = new_max
                    improved = True
    
    return routing, current_max


def _simulated_annealing(routing, start_temp=10.0, end_temp=0.01, max_iters=50000):
    """
    Simulated annealing to escape local optima.
    """
    routing = routing.copy()
    current_max = _max_load(routing)
    best_routing = routing.copy()
    best_max = current_max
    
    n_iters = max_iters
    for it in range(n_iters):
        # Temperature schedule
        T = start_temp * (end_temp / start_temp) ** (it / n_iters)
        
        # Random move: flip a random bit
        i = random.randint(0, N_DEMANDS - 1)
        new_routing = routing.copy()
        new_routing[i] = 1 - new_routing[i]
        new_max = _max_load(new_routing)
        
        delta = new_max - current_max
        if delta < 0 or random.random() < np.exp(-delta / (T + 1e-12)):
            routing = new_routing
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_routing = routing.copy()
    
    return best_routing, best_max


def _lp_relaxation_with_rounding():
    """
    Use LP relaxation approach: formulate as minimizing max load with binary routing variables.
    Since we have only 15 demands, we can use a simple bisection on max load with
    feasibility check via integer programming (brute force over 2^15 = 32768).
    """
    # Actually, let's just do exhaustive search since 2^15 is manageable
    # But use a smarter approach: bisection on max load
    # For each candidate max_load M, check if there exists a routing where all edge loads <= M
    
    # Precompute all possible routings' max loads using Gray code for efficiency
    # This is the same as the current program's approach but we'll combine with local search
    
    # Instead, let's use a hybrid: start from LP-like heuristic
    # LP relaxation: minimize max_e sum_i x_i * w_i * mask_i[e]
    # where x_i = 0 or 1
    
    # Simple heuristic: route each demand to minimize the current max load
    # Start with all CW
    routing = np.zeros(N_DEMANDS)
    loads = _compute_loads(routing)
    
    # Greedy: for each demand, choose direction that minimizes max load
    # Process demands in order of decreasing weight
    order = np.argsort(-WEIGHTS)
    
    for idx in order:
        # Try CW
        routing_cw = routing.copy()
        routing_cw[idx] = 0
        loads_cw = _compute_loads(routing_cw)
        max_cw = np.max(loads_cw)
        
        # Try CCW
        routing_ccw = routing.copy()
        routing_ccw[idx] = 1
        loads_ccw = _compute_loads(routing_ccw)
        max_ccw = np.max(loads_ccw)
        
        if max_cw <= max_ccw:
            routing[idx] = 0
        else:
            routing[idx] = 1
    
    return routing, _max_load(routing)


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    
    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
        Entry >= 0.5 routes counter-clockwise, < 0.5 routes clockwise.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_routing = None
    best_max = float('inf')
    
    # Method 1: LP-like greedy heuristic
    if time.time() < deadline - 300:
        routing, max_load = _lp_relaxation_with_rounding()
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 2: Hill climbing from all-zero
    if time.time() < deadline - 200:
        routing, max_load = _local_search_1bit(np.zeros(N_DEMANDS))
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 3: Hill climbing from all-one
    if time.time() < deadline - 200:
        routing, max_load = _local_search_1bit(np.ones(N_DEMANDS))
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 4: 2-bit local search from best found
    if time.time() < deadline - 150 and best_routing is not None:
        routing, max_load = _local_search_2bit(best_routing)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 5: Random restarts with 1-bit + 2-bit local search
    rng = np.random.default_rng(42)
    for trial in range(30):
        if time.time() > deadline - 100:
            break
        random_routing = rng.integers(0, 2, N_DEMANDS).astype(np.float64)
        # 1-bit local search
        hc_routing, hc_max = _local_search_1bit(random_routing, max_iters=20000)
        # 2-bit local search
        ls_routing, ls_max = _local_search_2bit(hc_routing, max_iters=10000)
        if ls_max < best_max:
            best_max = ls_max
            best_routing = ls_routing.copy()
    
    # Method 6: Simulated annealing from best found
    if time.time() < deadline - 60 and best_routing is not None:
        routing, max_load = _simulated_annealing(best_routing, start_temp=5.0, end_temp=0.01, max_iters=30000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Method 7: More random restarts with SA
    for trial in range(15):
        if time.time() > deadline - 30:
            break
        random_routing = rng.integers(0, 2, N_DEMANDS).astype(np.float64)
        # First 1-bit local search
        hc_routing, hc_max = _local_search_1bit(random_routing, max_iters=10000)
        # Then SA from there
        sa_routing, sa_max = _simulated_annealing(hc_routing, start_temp=3.0, end_temp=0.01, max_iters=15000)
        if sa_max < best_max:
            best_max = sa_max
            best_routing = sa_routing.copy()
    
    # Final polish: 1-bit + 2-bit local search
    if time.time() < deadline - 10 and best_routing is not None:
        routing, max_load = _local_search_1bit(best_routing, max_iters=20000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
        routing, max_load = _local_search_2bit(best_routing, max_iters=5000)
        if max_load < best_max:
            best_max = max_load
            best_routing = routing.copy()
    
    # Ensure the output is valid: 0 for CW, 1 for CCW
    # The contract says: >= 0.5 routes CCW, < 0.5 routes CW
    # Our routing uses 0 for CW, 1 for CCW, which satisfies this
    if best_routing is None:
        # Fallback: all clockwise
        best_routing = np.zeros(N_DEMANDS)
    
    return best_routing


# EVOLVE-BLOCK-END
