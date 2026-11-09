import numpy as np
import time

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0,5,3), (1,7,2), (2,9,4), (3,6,1), (0,4,5), 
        (2,8,2), (5,9,3), (1,4,2), (6,0,4), (3,8,1), 
        (7,2,3), (4,9,2), (8,1,5), (0,6,1), (5,2,2)
    ]
    
    # Precompute edge masks or direct edge lists
    # For each demand, list of edges for CW and CCW
    cw_edges = []
    ccw_edges = []
    weights = []
    
    for s, t, w in demands:
        weights.append(w)
        # CW: s -> t
        edges = []
        e = s
        while e != t:
            edges.append(e)
            e = (e + 1) % N
        cw_edges.append(edges)
        
        # CCW: t -> s
        edges = []
        e = t
        while e != s:
            edges.append(e)
            e = (e + 1) % N
        ccw_edges.append(edges)
        
    best_max_load = float('inf')
    best_assignment = None
    
    # Brute force 2^15
    for mask in range(1 << D):
        loads = np.zeros(N)
        valid = True
        # Early termination if we already exceed best? 
        # Hard to do easily without tracking partial sums, but 32k is small.
        
        for i in range(D):
            w = weights[i]
            if (mask >> i) & 1:
                # CCW
                for e in ccw_edges[i]:
                    loads[e] += w
            else:
                # CW
                for e in cw_edges[i]:
                    loads[e] += w
        
        current_max = np.max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_assignment = mask
            
    # Convert best_assignment to output array
    # Entry >= 0.5 -> CCW (1), < 0.5 -> CW (0)
    routing = np.zeros(D)
    for i in range(D):
        if (best_assignment >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
