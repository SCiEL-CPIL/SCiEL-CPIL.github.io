# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute edge sets for each demand for clockwise and counter-clockwise
    # For demand (s, t, w):
    # Clockwise: edges s, s+1, ..., t-1 (mod n)
    # Counter-clockwise: all other edges

    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        # Clockwise edges
        edges_cw = []
        if s <= t:
            edges_cw = list(range(s, t))
        else:
            edges_cw = list(range(s, n)) + list(range(0, t))
        cw_edges.append(edges_cw)
        
        # Counter-clockwise edges (complement)
        all_edges = set(range(n))
        edges_ccw = list(all_edges - set(edges_cw))
        ccw_edges.append(edges_ccw)
    
    # Convert to numpy arrays for fast computation
    # Create incidence matrix: M[i, e] = 1 if demand i uses edge e when routed clockwise
    M_cw = np.zeros((D, n))
    for i in range(D):
        for e in cw_edges[i]:
            M_cw[i, e] = 1.0
    
    # For counter-clockwise, it's the complement
    M_ccw = np.zeros((D, n))
    for i in range(D):
        for e in ccw_edges[i]:
            M_ccw[i, e] = 1.0
    
    weights = np.array([w for _, _, w in demands])
    
    def compute_loads(routing):
        """Compute edge loads given routing decisions (0=cw, 1=ccw)"""
        # routing is array of shape (D,), values 0 or 1
        # For each demand i:
        #   if routing[i] == 0: use M_cw[i]
        #   if routing[i] == 1: use M_ccw[i]
        # Load on edge e = sum over all demands i of w_i * indicator(i uses e)
        
        # Using matrix operations:
        # load = weights @ (routing @ M_ccw + (1-routing) @ M_cw) ... not quite
        # Actually: for each i, contribution is w_i * (routing[i] * M_ccw[i] + (1-routing[i]) * M_cw[i])
        # So total load on edge e = sum_i w_i * (routing[i] * M_ccw[i,e] + (1-routing[i]) * M_cw[i,e])
        
        # Let's compute:
        # term1 = (routing * weights) @ M_ccw  -> shape (n,)
        # term2 = ((1-routing) * weights) @ M_cw  -> shape (n,)
        # loads = term1 + term2
        
        r = routing.astype(float)
        term1 = (r * weights) @ M_ccw
        term2 = ((1 - r) * weights) @ M_cw
        return term1 + term2
    
    def max_load(routing):
        loads = compute_loads(routing)
        return np.max(loads)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 620  # 10 second safety margin
    
    # Initialize with greedy
    def greedy_init():
        routing = np.zeros(D, dtype=int)
        # Start all clockwise
        for i in range(D):
            # Try both directions, pick the one that gives lower max load
            routing[i] = 0
            load0 = max_load(routing)
            routing[i] = 1
            load1 = max_load(routing)
            if load1 <= load0:
                routing[i] = 1
            else:
                routing[i] = 0
        return routing
    
    best_routing = greedy_init()
    best_score = max_load(best_routing)
    
    # Simulated annealing
    random.seed(42)
    np.random.seed(42)
    
    T = 10.0  # initial temperature
    T_min = 0.01
    cooling = 0.995
    current_routing = best_routing.copy()
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
        
        # Random perturbation: flip one or more bits
        if random.random() < 0.7:
            # Flip one bit
            idx = random.randint(0, D - 1)
            new_routing = current_routing.copy()
            new_routing[idx] = 1 - new_routing[idx]
        else:
            # Flip two bits
            idx1 = random.randint(0, D - 1)
            idx2 = random.randint(0, D - 1)
            if idx1 == idx2:
                continue
            new_routing = current_routing.copy()
            new_routing[idx1] = 1 - new_routing[idx1]
            new_routing[idx2] = 1 - new_routing[idx2]
        
        new_score = max_load(new_routing)
        delta = new_score - current_score
        
        if delta < 0 or random.random() < np.exp(-delta / max(T, T_min)):
            current_routing = new_routing
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_routing = current_routing.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        T = max(T * cooling, T_min)
    
    # Final polish: local search with 1-flip and 2-flip moves
    def local_search(routing, score, time_limit=5):
        start = time.time()
        improved = True
        while improved and time.time() < start + time_limit:
            improved = False
            # 1-flip moves
            for i in range(D):
                new_routing = routing.copy()
                new_routing[i] = 1 - new_routing[i]
                new_score = max_load(new_routing)
                if new_score < score:
                    routing = new_routing
                    score = new_score
                    improved = True
            # 2-flip moves
            for i in range(D):
                for j in range(i + 1, D):
                    new_routing = routing.copy()
                    new_routing[i] = 1 - new_routing[i]
                    new_routing[j] = 1 - new_routing[j]
                    new_score = max_load(new_routing)
                    if new_score < score:
                        routing = new_routing
                        score = new_score
                        improved = True
        return routing, score
    
    best_routing, best_score = local_search(best_routing, best_score, time_limit=30)
    
    # Ensure output is in the correct format: >= 0.5 means counter-clockwise
    # Our routing uses 0 for clockwise, 1 for counter-clockwise
    # So we return the routing array directly (0 or 1)
    result = best_routing.astype(float)
    return result


# EVOLVE-BLOCK-END
