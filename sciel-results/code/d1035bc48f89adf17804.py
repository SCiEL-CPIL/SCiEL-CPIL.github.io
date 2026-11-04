import time
import random

def _petersen_edges():
    return [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]

def _lift_petersen(Q, voltages):
    n = 10 * Q
    edges = []
    base_edges = _petersen_edges()
    for u, v in base_edges:
        c = voltages[(u, v)]
        for i in range(Q):
            u_idx = u * Q + i
            v_idx = v * Q + ((i + c) % Q)
            if u_idx < v_idx:
                edges.append((u_idx, v_idx))
            else:
                edges.append((v_idx, u_idx))
    return n, edges

def _construct_petersen_lift(k, g, deadline):
    if k != 3:
        return None
    
    # Moore bound for k=3, g=16 is 510.
    # n = 10 * Q. So Q >= 51.
    start_Q = 52
    max_Q = 200 # 2000 vertices is still better than 6561
    
    for Q in range(start_Q, max_Q + 1):
        if time.time() > deadline - 10.0:
            break
        for attempt in range(20): # 20 random tries per Q
            if time.time() > deadline - 5.0:
                break
            voltages = {}
            for u, v in _petersen_edges():
                voltages[(u, v)] = random.randint(0, Q - 1)
            
            n, edges = _lift_petersen(Q, voltages)
            # Quick check: degree is 3.
            # Check girth
            if _check_girth(n, edges, g):
                return n, edges
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    # Try the Petersen lift first (potentially much smaller)
    best = _construct_petersen_lift(k, g, deadline)
    
    if best is None:
        # Fall back to LUW
        d_min = max(3, (g - 2) // 2)
        for d in range(d_min, d_min + 6):
            if time.time() > deadline - 20.0:
                break
            try:
                n, edges = _luw(k, d)
            except Exception:
                continue
            if time.time() > deadline - 10.0:
                if best is None:
                    best = (n, edges)
                break
            if _check_girth(n, edges, g):
                best = (n, edges)
                break

    if best is None:
        best = _fallback(k, g)
        
    return best
