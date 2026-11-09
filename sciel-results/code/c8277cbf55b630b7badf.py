# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Uses an exact ILP formulation via scipy's linprog (HiGHS backend).

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    demands = [(0,5,3), (1,7,2), (2,9,4), (3,6,1), (0,4,5), (2,8,2),
               (5,9,3), (1,4,2), (6,0,4), (3,8,1), (7,2,3), (4,9,2),
               (8,1,5), (0,6,1), (5,2,2)]
    D = len(demands)
    
    # Precompute for each demand and each edge whether the edge is used
    # if routed clockwise (cw) or counter-clockwise (ccw)
    # x[d] = 1 means demand d is routed counter-clockwise
    # x[d] = 0 means demand d is routed clockwise
    
    # For demand d with source s, target t:
    # CW route uses edges: s, s+1, ..., t-1 (mod n)
    # CCW route uses edges: all other edges
    
    edge_cw = np.zeros((D, n), dtype=int)  # edge_cw[d][e] = 1 if edge e is on CW route for demand d
    edge_ccw = np.zeros((D, n), dtype=int)  # edge_ccw[d][e] = 1 if edge e is on CCW route for demand d
    
    for d, (s, t, w) in enumerate(demands):
        # CW edges from s to t
        e = s
        while True:
            edge_cw[d, e] = 1
            if e == t:
                break
            e = (e + 1) % n
        # CCW edges: all edges NOT on CW route
        edge_ccw[d] = 1 - edge_cw[d]
    
    # ILP formulation:
    # Variables: x[0..D-1] (binary), y (continuous, = max load)
    # Minimize: y
    # Subject to: for each edge e: sum_d (edge_cw[d,e]*(1-x[d]) + edge_ccw[d,e]*x[d]) * w[d] <= y
    #            x[d] in {0,1}
    
    # Rewrite constraint for edge e:
    # sum_d edge_cw[d,e]*w[d] - sum_d edge_cw[d,e]*w[d]*x[d] + sum_d edge_ccw[d,e]*w[d]*x[d] <= y
    # sum_d (edge_ccw[d,e] - edge_cw[d,e]) * w[d] * x[d] - y <= -sum_d edge_cw[d,e]*w[d]
    
    # Variables: [x_0, ..., x_{D-1}, y]
    # Objective: minimize y -> c = [0, ..., 0, 1]
    c = np.zeros(D + 1)
    c[D] = 1.0  # minimize y
    
    # Constraints: A_ub @ vars <= b_ub
    # For each edge e: sum_d (edge_ccw[d,e] - edge_cw[d,e]) * w[d] * x[d] - y <= -base_cw_load[e]
    # where base_cw_load[e] = sum_d edge_cw[d,e] * w[d]
    
    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    
    for e in range(n):
        base_load = 0.0
        for d in range(D):
            s, t, w = demands[d]
            if edge_cw[d, e]:
                base_load += w
            A_ub[e, d] = (edge_ccw[d, e] - edge_cw[d, e]) * demands[d][2]
        A_ub[e, D] = -1.0
        b_ub[e] = -base_load
    
    # Bounds: x[d] in [0,1] for d in 0..D-1, y in [0, inf)
    bounds = [(0, 1)] * D + [(0, None)]
    
    # Solve with HiGHS (exact for LP relaxation, and since the constraint matrix 
    # has special structure, we can try to get integer solution)
    # Actually, let's use the method that handles integer constraints if available,
    # or we can enumerate since D=15 is small enough for branch and bound via linprog
    
    # First try: solve LP relaxation and check if integer
    deadline = time.time() + 600
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        x = result.x[:D]
        # Round to nearest integer (0 or 1)
        x_int = np.round(x).astype(int)
        
        # Verify it's feasible and compute the actual max load
        max_load = 0.0
        for e in range(n):
            load = 0.0
            for d in range(D):
                s, t, w = demands[d]
                if x_int[d] == 0:  # CW
                    if edge_cw[d, e]:
                        load += w
                else:  # CCW
                    if edge_ccw[d, e]:
                        load += w
            max_load = max(max_load, load)
        
        # If the LP solution is already integer (or rounds cleanly), use it
        # Check if rounded solution is valid
        if np.allclose(x, x_int, atol=1e-6):
            routing = x_int.astype(float)
            return routing
    
    # If LP relaxation didn't give integer solution, do exhaustive search over 2^15 = 32768 possibilities
    # This is fast enough
    best_routing = np.zeros(D)
    best_max_load = float('inf')
    
    for mask in range(1 << D):
        if time.time() > deadline - 30:
            break
        x = np.array([(mask >> d) & 1 for d in range(D)])
        
        max_load = 0.0
        for e in range(n):
            load = 0.0
            for d in range(D):
                s, t, w = demands[d]
                if x[d] == 0:  # CW
                    if edge_cw[d, e]:
                        load += w
                else:  # CCW
                    if edge_ccw[d, e]:
                        load += w
            max_load = max(max_load, load)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = x.copy()
    
    return best_routing.astype(float)


# EVOLVE-BLOCK-END
