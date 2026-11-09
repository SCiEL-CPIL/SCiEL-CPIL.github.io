# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds


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
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute the set of edges used for clockwise and counter-clockwise routing
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
    # Counter-clockwise: all other edges
    
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        cw_set = set()
        if s == t:
            # This case shouldn't happen given the demands, but handle it
            pass
        else:
            e = s
            while e != t:
                cw_set.add(e)
                e = (e + 1) % n
        
        all_edges = set(range(n))
        ccw_set = all_edges - cw_set
        
        cw_edges.append(cw_set)
        ccw_edges.append(ccw_set)
    
    # ILP Formulation:
    # Variables:
    #   x_d for d in 0..D-1: binary, 1 if demand d is routed counter-clockwise, 0 if clockwise
    #   L: continuous, the maximum load
    #
    # Objective: minimize L
    #
    # Constraints:
    #   For each edge e in 0..n-1:
    #     Load_e <= L
    #     Load_e = sum over demands d:
    #               w_d * (1 if d uses e clockwise and x_d=0 else 0)
    #               + w_d * (1 if d uses e counter-clockwise and x_d=1 else 0)
    #
    # Let's rewrite the load constraint for edge e:
    #   sum_{d: e in cw_edges[d]} w_d * (1 - x_d) + sum_{d: e in ccw_edges[d]} w_d * x_d <= L
    #
    #   sum_{d: e in cw_edges[d]} w_d - sum_{d: e in cw_edges[d]} w_d * x_d 
    #     + sum_{d: e in ccw_edges[d]} w_d * x_d <= L
    #
    #   sum_{d: e in cw_edges[d]} w_d + sum_{d} w_d * (1_{e in ccw_edges[d]} - 1_{e in cw_edges[d]}) * x_d <= L
    #
    # Let c_e = sum_{d: e in cw_edges[d]} w_d
    # Let a_{e,d} = w_d * (1_{e in ccw_edges[d]} - 1_{e in cw_edges[d]})
    #
    # Constraint: c_e + sum_d a_{e,d} * x_d <= L
    # Rearranging for standard form (<= 0 or >= 0):
    # sum_d a_{e,d} * x_d - L <= -c_e
    
    # Variables vector: [x_0, ..., x_{D-1}, L]
    # Total variables: D + 1
    
    # Objective: minimize L
    # c_obj = [0, 0, ..., 0, 1]
    c_obj = np.zeros(D + 1)
    c_obj[D] = 1.0  # index D corresponds to L
    
    # Constraints:
    # For each edge e:
    # sum_{d=0}^{D-1} a_{e,d} * x_d - L <= -c_e
    
    A = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    b_lb = np.full(n, -np.inf)
    
    for e in range(n):
        c_e = 0.0
        for d in range(D):
            s, t, w = demands[d]
            if e in cw_edges[d]:
                c_e += w
                # a_{e,d} = w * (0 - 1) = -w
                A[e, d] = -w
            else:
                # e is in ccw_edges[d] (since cw and ccw partition the ring for s!=t)
                # a_{e,d} = w * (1 - 0) = w
                A[e, d] = w
        A[e, D] = -1.0  # coefficient for L
        b_ub[e] = -c_e
    
    # We want: A * x <= b_ub and A * x >= b_lb
    # But wait, the constraint is:
    # c_e + sum a_{e,d} x_d <= L
    # sum a_{e,d} x_d - L <= -c_e
    # So A * vars <= -c_e
    # And A * vars >= -inf
    
    constraint = LinearConstraint(A, b_lb, b_ub)
    
    # Bounds:
    # x_d in [0, 1] for d in 0..D-1
    # L in [0, inf)
    lb = np.zeros(D + 1)
    ub = np.ones(D + 1)
    ub[D] = np.inf
    
    bounds = Bounds(lb, ub)
    
    # Integrality: x_d are integer (binary), L is continuous
    integrality = np.zeros(D + 1)
    integrality[:D] = 1  # binary variables
    integrality[D] = 0   # continuous variable
    
    start_time = time.time()
    deadline = start_time + 600.0  # 600s deadline
    
    try:
        res = milp(
            c=c_obj,
            constraints=constraint,
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0:  # Optimal
            x_sol = res.x
            routing = np.zeros(D)
            for d in range(D):
                # x_sol[d] is 1 if counter-clockwise, 0 if clockwise
                routing[d] = 1.0 if x_sol[d] > 0.5 else 0.0
            return routing
        else:
            # Fallback: all clockwise
            return np.zeros(D)
            
    except Exception:
        # Fallback: all clockwise
        return np.zeros(D)


# EVOLVE-BLOCK-END
