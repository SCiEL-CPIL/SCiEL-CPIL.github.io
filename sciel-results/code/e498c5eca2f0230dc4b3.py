import numpy as np
import time

def ring_routing() -> np.ndarray:
    n_nodes = 10
    n_edges = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute edge usage for each demand in both directions
    # A[i, d, e] is 1 if demand i in direction d uses edge e
    A = np.zeros((D, 2, n_edges), dtype=np.int8)
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise (d=0): edges s, s+1, ..., t-1 (mod n)
        # If s == t, it's a loop? No, s != t in ring usually, but here s!=t.
        # Path: s -> s+1 -> ... -> t. Edges are (u, u+1).
        # Edge e is between e and (e+1)%10.
        # So edge e is used if e is in the set {s, s+1, ..., t-1} mod 10.
        
        # Clockwise
        if s <= t:
            edges_cw = range(s, t)
        else:
            edges_cw = list(range(s, n_nodes)) + list(range(0, t))
            
        for e in edges_cw:
            A[i, 0, e] = 1
            
        # Counter-clockwise (d=1): all other edges
        for e in range(n_edges):
            if A[i, 0, e] == 0:
                A[i, 1, e] = 1
            # Note: if s==t, both would be empty/full? 
            # Here s != t for all demands.
            # If s == t, clockwise path is empty? No, usually 0 or full ring.
            # Standard definition: s to t clockwise.
            # If s=0, t=5. Edges 0,1,2,3,4.
            # If s=5, t=0. Edges 5,6,7,8,9.
            
    # Weights
    W = np.array([d[2] for d in demands])
    
    # Precompute delta vectors for flipping
    # delta[i, e] = change in load on edge e if we flip demand i
    # If currently 0 (CW), flip to 1 (CCW): change is A[i,1,e] - A[i,0,e]
    # If currently 1 (CCW), flip to 0 (CW): change is A[i,0,e] - A[i,1,e]
    
    start_time = time.time()
    deadline = start_time + 590  # Leave 10s margin
    
    best_routing = np.zeros(D, dtype=np.float64)
    best_max_load = np.inf
    
    # Helper to compute max load given routing
    def get_max_load(routing):
        loads = np.zeros(n_edges)
        for i in range(D):
            if routing[i] < 0.5:
                loads += W[i] * A[i, 0, :]
            else:
                loads += W[i] * A[i, 1, :]
        return np.max(loads)
    
    # Initial best: all clockwise
    init_routing = np.zeros(D)
    best_max_load = get_max_load(init_routing)
    best_routing = init_routing.copy()
    
    # Local search function
    def local_search(current_routing):
        current_max = get_max_load(current_routing)
        improved = True
        while improved:
            improved = False
            # Try flipping each demand
            for i in range(D):
                new_routing = current_routing.copy()
                if new_routing[i] < 0.5:
                    new_routing[i] = 1.0
                else:
                    new_routing[i] = 0.0
                
                new_max = get_max_load(new_routing)
                
                if new_max < current_max:
                    current_routing = new_routing
                    current_max = new_max
                    improved = True
                    # Optionally break here to restart scan, or continue
                    # Continuing might allow multiple improvements in one pass
                    # But restarting is safer for hill climbing
                    break 
            # If no improvement, try random perturbations? 
            # For now, simple hill climbing.
        
        return current_routing, current_max
    
    # Multi-start hill climbing
    rng = np.random.default_rng(42)
    
    # Run a few random restarts
    num_restarts = 10000  # We have lots of time, let's do many
    # But each restart is fast. Let's check time.
    
    while time.time() < deadline:
        # Random start
        current_routing = rng.integers(0, 2, size=D).astype(np.float64)
        
        routing, max_load = local_search(current_routing)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = routing.copy()
            
        # Also try starting from best_routing and perturbing
        if rng.random() < 0.5:
            perturbed = best_routing.copy()
            # Flip 1-3 random bits
            num_flips = rng.integers(1, 4)
            indices = rng.choice(D, size=num_flips, replace=False)
            for idx in indices:
                if perturbed[idx] < 0.5:
                    perturbed[idx] = 1.0
                else:
                    perturbed[idx] = 0.0
            routing, max_load = local_search(perturbed)
            if max_load < best_max_load:
                best_max_load = max_load
                best_routing = routing.copy()
                
    return best_routing
