# EVOLVE-BLOCK-START
import numpy as np


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using Local Search with Random Restarts.
    """
    n_nodes = 10
    n_demands = 15
    
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    weights = np.array([d[2] for d in demands], dtype=np.float64)
    
    # Precompute incidence matrices
    # M_cw[i, e] = 1 if demand i uses edge e when routed CW, else 0
    # M_ccw[i, e] = 1 if demand i uses edge e when routed CCW, else 0
    M_cw = np.zeros((n_demands, n_nodes))
    M_ccw = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        # CW path
        edge = s
        while edge != t:
            M_cw[i, edge] = 1
            edge = (edge + 1) % n_nodes
        
        # CCW path is the complement
        M_ccw[i] = 1 - M_cw[i]
    
    def compute_max_load(routing):
        """
        routing: array of 0s (CW) and 1s (CCW)
        """
        # If routing[i] is 0 (CW), load contribution is weights[i] * M_cw[i]
        # If routing[i] is 1 (CCW), load contribution is weights[i] * M_ccw[i]
        
        # Vectorized calculation:
        # load_e = sum_i w_i * ( (1-r_i)*M_cw[i,e] + r_i*M_ccw[i,e] )
        # We can compute this as:
        # Base load if all CW: sum_i w_i * M_cw[i]
        # Delta for flipping i to CCW: w_i * (M_ccw[i] - M_cw[i])
        
        # Let's just compute directly for simplicity and speed in numpy
        # Construct the selection matrix
        # S[i, e] = M_cw[i, e] if r[i]==0 else M_ccw[i, e]
        
        # Efficient way:
        # loads = M_cw.T @ (weights * (1 - routing)) + M_ccw.T @ (weights * routing)
        
        w_cw = weights * (1 - routing)
        w_ccw = weights * routing
        
        loads = M_cw.T @ w_cw + M_ccw.T @ w_ccw
        return np.max(loads)
    
    def local_search(start_routing, max_iters=1000):
        """
        Perform steepest descent local search.
        """
        current = start_routing.copy()
        current_load = compute_max_load(current)
        
        improved = True
        while improved:
            improved = False
            best_flip = -1
            best_new_load = current_load
            
            # Try flipping each demand
            for i in range(n_demands):
                # Create candidate
                candidate = current.copy()
                candidate[i] = 1 - candidate[i]
                
                # Calculate new load
                # Instead of full recompute, we can update incrementally, 
                # but for n=10, full recompute is fast enough.
                new_load = compute_max_load(candidate)
                
                if new_load < best_new_load:
                    best_new_load = new_load
                    best_flip = i
            
            if best_flip != -1:
                current[best_flip] = 1 - current[best_flip]
                current_load = best_new_load
                improved = True
        
        return current, current_load

    np.random.seed(42)
    
    best_routing = None
    best_load = float('inf')
    
    # Multiple restarts
    n_restarts = 200
    
    for _ in range(n_restarts):
        # Random initial solution
        initial = np.random.choice([0, 1], size=n_demands)
        
        # Run local search
        local_sol, local_load = local_search(initial)
        
        if local_load < best_load:
            best_load = local_load
            best_routing = local_sol
            
            # Early termination if we hit a theoretical lower bound?
            # Total weight = sum(w) = 3+2+4+1+5+2+3+2+4+1+3+2+5+1+2 = 40
            # Average load per edge if perfectly distributed = 40 * (avg_path_len) / 10
            # Min path length is 1, max is 9.
            # A very good solution would have max load around 20-30.
            if best_load <= 20:
                break

    return best_routing.astype(np.float64)

# EVOLVE-BLOCK-END
