import numpy as np
import time

def ring_routing() -> np.ndarray:
    # Data
    demands = [
        (0,5,3), (1,7,2), (2,9,4), (3,6,1), (0,4,5), 
        (2,8,2), (5,9,3), (1,4,2), (6,0,4), (3,8,1), 
        (7,2,3), (4,9,2), (8,1,5), (0,6,1), (5,2,2)
    ]
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute edges
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = set()
        ccw = set()
        # CW: s, s+1, ..., t-1
        cur = s
        while cur != t:
            cw.add(cur)
            cur = (cur + 1) % n_nodes
        # CCW: all others
        all_edges = set(range(n_nodes))
        ccw = all_edges - cw
        cw_edges.append(cw)
        ccw_edges.append(ccw)
        
    # Convert to lists for faster access or numpy arrays
    # Let's use numpy arrays of shape (15, 10) with 0/1 indicating edge usage
    cw_mask = np.zeros((n_demands, n_nodes))
    ccw_mask = np.zeros((n_demands, n_nodes))
    weights = np.array([d[2] for d in demands])
    
    for i, (s, t, w) in enumerate(demands):
        cur = s
        while cur != t:
            cw_mask[i, cur] = 1
            cur = (cur + 1) % n_nodes
        ccw_mask[i] = 1 - cw_mask[i]
        
    # Precompute delta for flipping each demand
    # If demand i is flipped from CW to CCW, the change in load on edge e is:
    #   w_i * (ccw_mask[i,e] - cw_mask[i,e])
    # If flipped from CCW to CW, the change is the negative.
    # delta_flip[i] = w_i * (ccw_mask[i] - cw_mask[i])  (shape: 10)
    # Flipping i when it's CW: loads += delta_flip[i]
    # Flipping i when it's CCW: loads -= delta_flip[i]
    delta_flip = weights[:, None] * (ccw_mask - cw_mask)  # shape (15, 10)
    
    def compute_loads(routing):
        """Compute edge loads given routing (0=CW, 1=CCW)."""
        # load_e = sum_i w_i * (routing_i * ccw_mask[i,e] + (1-routing_i) * cw_mask[i,e])
        #        = sum_i w_i * cw_mask[i,e] + sum_i w_i * routing_i * (ccw_mask[i,e] - cw_mask[i,e])
        base = np.sum(weights[:, None] * cw_mask, axis=0)  # all CW
        extra = np.sum(weights[:, None] * routing[:, None] * (ccw_mask - cw_mask), axis=0)
        return base + extra
    
    def local_search(initial_routing, deadline=None):
        current_routing = initial_routing.copy()
        current_loads = compute_loads(current_routing)
        current_max = np.max(current_loads)
        
        improved = True
        while improved:
            if deadline is not None and time.time() >= deadline:
                break
            improved = False
            for i in range(n_demands):
                # Try flipping i using incremental update
                if current_routing[i] == 0:
                    # CW -> CCW: add delta_flip[i]
                    new_loads = current_loads + delta_flip[i]
                else:
                    # CCW -> CW: subtract delta_flip[i]
                    new_loads = current_loads - delta_flip[i]
                
                new_max = np.max(new_loads)
                
                if new_max < current_max - 1e-12:
                    current_routing[i] = 1 - current_routing[i]
                    current_loads = new_loads
                    current_max = new_max
                    improved = True
        return current_routing, current_max
    
    def pair_flip_search(initial_routing, deadline=None):
        """Local search allowing pair flips to escape single-flip local optima."""
        current_routing = initial_routing.copy()
        current_loads = compute_loads(current_routing)
        current_max = np.max(current_loads)
        
        improved = True
        while improved:
            if deadline is not None and time.time() >= deadline:
                break
            improved = False
            # Try all single flips first
            for i in range(n_demands):
                if current_routing[i] == 0:
                    new_loads = current_loads + delta_flip[i]
                else:
                    new_loads = current_loads - delta_flip[i]
                new_max = np.max(new_loads)
                if new_max < current_max - 1e-12:
                    current_routing[i] = 1 - current_routing[i]
                    current_loads = new_loads
                    current_max = new_max
                    improved = True
            # Try pair flips
            for i in range(n_demands):
                for j in range(i + 1, n_demands):
                    # Compute combined delta
                    if current_routing[i] == 0:
                        di = delta_flip[i]
                    else:
                        di = -delta_flip[i]
                    if current_routing[j] == 0:
                        dj = delta_flip[j]
                    else:
                        dj = -delta_flip[j]
                    new_loads = current_loads + di + dj
                    new_max = np.max(new_loads)
                    if new_max < current_max - 1e-12:
                        current_routing[i] = 1 - current_routing[i]
                        current_routing[j] = 1 - current_routing[j]
                        current_loads = new_loads
                        current_max = new_max
                        improved = True
        return current_routing, current_max

    # Run multiple restarts with deadline guard
    best_routing = None
    best_max = float('inf')
    
    start_time = time.time()
    deadline = start_time + 580  # 580s to be safe within 630s limit
    
    def update_best(r, m):
        nonlocal best_routing, best_max
        if m < best_max:
            best_max = m
            best_routing = r.copy()
    
    # Phase 1: Deterministic starts with local search
    for init in [np.zeros(n_demands), np.ones(n_demands)]:
        if time.time() >= deadline:
            break
        r, m = local_search(init, deadline)
        update_best(r, m)
        # Follow up with pair-flip search
        if time.time() >= deadline:
            break
        r, m = pair_flip_search(r, deadline)
        update_best(r, m)
    
    # Phase 2: Simulated annealing for better exploration
    if time.time() < deadline:
        sa_deadline = deadline
        current_routing = best_routing.copy()
        current_loads = compute_loads(current_routing)
        current_max = np.max(current_loads)
        
        # Temperature schedule
        t_start = 2.0
        t_end = 0.01
        sa_iterations = 0
        max_sa_iterations = 200000  # cap to avoid infinite loop
        
        while time.time() < sa_deadline and sa_iterations < max_sa_iterations:
            sa_iterations += 1
            # Temperature annealing
            progress = min(1.0, sa_iterations / max_sa_iterations)
            temperature = t_start * (t_end / t_start) ** progress
            
            # Pick a random demand to flip
            i = np.random.randint(0, n_demands)
            if current_routing[i] == 0:
                delta_load = delta_flip[i]
            else:
                delta_load = -delta_flip[i]
            new_loads = current_loads + delta_load
            new_max = np.max(new_loads)
            
            # Accept if better, or with probability exp(-delta/T)
            if new_max < current_max:
                current_routing[i] = 1 - current_routing[i]
                current_loads = new_loads
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_routing = current_routing.copy()
            else:
                delta = new_max - current_max
                if delta > 0:
                    prob = np.exp(-delta / max(temperature, 1e-10))
                    if np.random.random() < prob:
                        current_routing[i] = 1 - current_routing[i]
                        current_loads = new_loads
                        current_max = new_max
        
        # Final polish after SA
        if time.time() < deadline:
            r, m = local_search(current_routing, deadline)
            update_best(r, m)
            if time.time() < deadline:
                r, m = pair_flip_search(r, deadline)
                update_best(r, m)
    
    # Phase 3: Random restarts with local search
    rng = np.random.default_rng(42)
    while time.time() < deadline:
        random_routing = rng.integers(0, 2, size=n_demands)
        r, m = local_search(random_routing, deadline)
        update_best(r, m)
        if time.time() < deadline:
            r, m = pair_flip_search(r, deadline)
            update_best(r, m)
    
    return best_routing
