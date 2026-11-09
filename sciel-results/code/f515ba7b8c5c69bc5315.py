# EVOLVE-BLOCK-START
import numpy as np
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

    # Precompute clockwise and counter-clockwise edge sets for each demand
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = []
        e = s
        while e != t:
            cw.append(e)
            e = (e + 1) % n
        cw_edges.append(cw)
        # Counter-clockwise: all edges not in clockwise
        all_edges = set(range(n))
        ccw = list(all_edges - set(cw))
        ccw_edges.append(ccw)

    # Formulate MIP
    # Variables: x_0, ..., x_{D-1} (binary, 1 if CCW, 0 if CW)
    #            M (continuous, >= 0)
    # Total variables: D + 1
    # Indices: 0..D-1 for x, D for M
    
    # Objective: Minimize M
    # c vector: [0, 0, ..., 0, 1]
    c = np.zeros(D + 1)
    c[D] = 1.0

    # Constraints:
    # For each edge e (0..n-1):
    # Load_e = sum_{i: e in CW(i)} w_i * (1 - x_i) + sum_{i: e in CCW(i)} w_i * x_i <= M
    # sum_{i: e in CW(i)} w_i - sum_{i: e in CW(i)} w_i * x_i + sum_{i: e in CCW(i)} w_i * x_i <= M
    # sum_{i: e in CW(i)} w_i + sum_{i} (A_{e,i}) x_i - M <= 0
    # where A_{e,i} = -w_i if e in CW(i), w_i if e in CCW(i)
    
    A = np.zeros((n, D + 1))
    b_lower = np.zeros(n)
    b_upper = np.zeros(n) # We want LHS <= 0, so upper bound is 0
    
    for e in range(n):
        b_lower[e] = 0
        b_upper[e] = 0
        # Coefficient for M is -1
        A[e, D] = -1.0
        
        for i in range(D):
            s, t, w = demands[i]
            if e in cw_edges[i]:
                # Term is -w_i * x_i
                A[e, i] = -w
                # Constant term sum w_i is moved to RHS? 
                # Equation: Const + Sum(A*x) - M <= 0
                # So Const = sum_{i: e in CW(i)} w_i
                # We can add this to b_lower? No, standard form is A x >= b_lower and A x <= b_upper.
                # Let's use: Load_e - M <= 0
                # Load_e = Const_e + Sum_{i} (Coeff_{e,i} * x_i)
                # So: Sum_{i} (Coeff_{e,i} * x_i) - M <= -Const_e
                # This means b_upper[e] = -Const_e
                pass
        
        # Let's re-calculate properly.
        # Load_e <= M
        # Const_e + Sum_{i: e in CW(i)} (-w_i) x_i + Sum_{i: e in CCW(i)} (w_i) x_i - M <= 0
        # Sum_{i} (Coeff_{e,i} x_i) - 1*M <= -Const_e
        # So b_upper[e] = -Const_e
        
        Const_e = 0
        for i in range(D):
            if e in cw_edges[i]:
                A[e, i] = -demands[i][2]
                Const_e += demands[i][2]
            elif e in ccw_edges[i]:
                A[e, i] = demands[i][2]
            else:
                A[e, i] = 0
        b_upper[e] = -Const_e

    # Bounds
    # x_i in {0, 1}
    # M in [0, infinity)
    integrality = np.zeros(D + 1)
    integrality[:D] = 1.0 # x_i are integer (binary)
    
    bounds = Bounds(lb=np.zeros(D + 1), ub=np.concatenate([np.ones(D), [np.inf]]))
    
    # Solve MIP
    res = milp(c=c, constraints=LinearConstraint(A, b_lower, b_upper), 
               integrality=integrality, bounds=bounds)
    
    if res.status != 0:
        # Fallback to simple heuristic if MIP fails
        routing = np.zeros(D)
        return routing
        
    # Extract x variables
    x = res.x[:D]
    # The solution is binary, but might be 0.0 or 1.0
    routing = np.where(x > 0.5, 1.0, 0.0)
    
    return routing


# EVOLVE-BLOCK-END
