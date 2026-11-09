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
    D = 15
    N = 10
    
    # Demands: (source, target, weight)
    demands = [
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
        (5, 2, 2)
    ]
    
    # Precompute edge usage matrices
    # For each demand d, compute which edges are used if routed CW vs CCW
    # CW from s to t uses edges s, s+1, ..., t-1 (mod N)
    # CCW uses all other edges
    
    # A[d, e] = 1 if demand d uses edge e when routed CW
    # B[d, e] = 1 if demand d uses edge e when routed CCW
    
    A = np.zeros((D, N))
    B = np.zeros((D, N))
    
    for d, (s, t, w) in enumerate(demands):
        # Determine edges for CW
        if s < t:
            cw_edges = range(s, t)
        else:
            cw_edges = list(range(s, N)) + list(range(0, t))
        
        for e in cw_edges:
            A[d, e] = 1
        
        # CCW uses all other edges
        for e in range(N):
            if A[d, e] == 0:
                B[d, e] = 1
    
    # Weights
    W = np.array([d[2] for d in demands])
    
    # Formulate ILP:
    # Variables: x_d (binary, 0=CW, 1=CCW) for d=0..D-1, and t (continuous, peak load)
    # Total variables: D + 1
    
    # Minimize t
    # Constraints: For each edge e, sum_d W_d * (x_d * B[d,e] + (1-x_d) * A[d,e]) <= t
    # Which is: sum_d W_d * (A[d,e] + x_d * (B[d,e] - A[d,e])) <= t
    # => sum_d W_d * (A[d,e] + x_d * (B[d,e] - A[d,e])) - t <= 0
    
    # Objective: minimize t (index D in the variable vector)
    c = np.zeros(D + 1)
    c[D] = 1.0  # minimize t
    
    # Constraints:
    # For each edge e: sum_d W_d * (A[d,e] + x_d * (B[d,e] - A[d,e])) - t <= 0
    # This is a linear inequality: M @ vars <= 0 where M has shape (N, D+1)
    
    M = np.zeros((N, D + 1))
    for e in range(N):
        for d in range(D):
            # Coefficient of x_d: W_d * (B[d,e] - A[d,e])
            M[e, d] = W[d] * (B[d, e] - A[d, e])
            # Constant term: W_d * A[d,e] (but this goes to rhs, not matrix)
        # Actually, let's reformulate:
        # sum_d W_d * A[d,e] + sum_d W_d * x_d * (B[d,e] - A[d,e]) - t <= 0
        # => sum_d W_d * x_d * (B[d,e] - A[d,e]) - t <= -sum_d W_d * A[d,e]
        # So M @ vars <= rhs where rhs[e] = -sum_d W_d * A[d,e]
    
    rhs = np.zeros(N)
    for e in range(N):
        rhs[e] = -np.sum(W * A[:, e])
    
    # Also need: t >= 0 (but this is implied by loads being non-negative, still add for safety)
    # And 0 <= x_d <= 1 (binary)
    
    # Bounds: x_d in [0, 1] integer, t in [0, inf)
    lb = np.zeros(D + 1)
    ub = np.ones(D + 1)
    ub[D] = np.inf  # t can be large
    
    # Integrality: x_d are binary (integer), t is continuous
    integrality = np.ones(D + 1)
    integrality[D] = 0  # t is continuous
    
    # Linear constraint: M @ vars <= rhs
    # scipy.optimize.milp uses LinearConstraint(A, lb, ub) where A @ x >= lb and A @ x <= ub
    # Our constraint: M @ vars <= rhs
    # So we need: M @ vars >= -inf AND M @ vars <= rhs
    # LinearConstraint(M, -np.inf, rhs)
    
    lin_cons = LinearConstraint(M, -np.inf, rhs)
    bounds = Bounds(lb, ub)
    
    # Solve
    try:
        result = milp(
            c=c,
            constraints=lin_cons,
            integrality=integrality,
            bounds=bounds
        )
        
        if result.success:
            # Extract routing decisions
            routing = np.round(result.x[:D]).astype(float)
            # Ensure values are 0 or 1 (or 0.5 for CCW, 0 for CW per task spec)
            # Task says: >= 0.5 routes CCW, < 0.5 routes CW
            # So 0 -> CW, 1 -> CCW is correct
            return routing
        else:
            # Fallback: local search
            pass
    except Exception:
        pass
    
    # Fallback: simple local search
    # Start with all CW
    routing = np.zeros(D)
    
    def compute_max_load(r):
        loads = np.zeros(N)
        for d in range(D):
            s, t, w = demands[d]
            if r[d] < 0.5:  # CW
                if s < t:
                    edges = range(s, t)
                else:
                    edges = list(range(s, N)) + list(range(0, t))
                for e in edges:
                    loads[e] += w
            else:  # CCW
                for e in range(N):
                    if A[d, e] == 0:
                        loads[e] += w
        return np.max(loads)
    
    current_max = compute_max_load(routing)
    
    # Local search: try flipping each bit
    improved = True
    while improved:
        improved = False
        for d in range(D):
            old_val = routing[d]
            routing[d] = 1.0 - old_val
            new_max = compute_max_load(routing)
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                routing[d] = old_val
    
    return routing


# EVOLVE-BLOCK-END
