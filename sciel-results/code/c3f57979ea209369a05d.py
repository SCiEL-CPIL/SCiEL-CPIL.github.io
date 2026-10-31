import numpy as np
import time
import itertools

def construct():
    n_points = 16
    d = 2
    
    def objective(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        mask = ~np.eye(n_points, dtype=bool)
        dist_sq = dist_sq[mask]
        
        if len(dist_sq) == 0:
            return 1e10
            
        min_dist_sq = np.min(dist_sq)
        max_dist_sq = np.max(dist_sq)
        
        if min_dist_sq < 1e-10:
            return 1e10
            
        return max_dist_sq / min_dist_sq

    def anneal(initial_points, time_limit=10):
        current_points = initial_points.copy()
        current_cost = objective(current_points)
        best_points = current_points.copy()
        best_cost = current_cost
        
        T = 0.5
        T_min = 1e-5
        cooling = 0.9995
        start_time = time.time()
        
        rng = np.random.default_rng(42)
        
        while time.time() - start_time < time_limit:
            idx = rng.integers(0, n_points)
            step_size = 0.1 * T
            move = rng.normal(0, step_size, size=d)
            new_points = current_points.copy()
            new_points[idx] += move
            
            new_cost = objective(new_points)
            
            if new_cost < current_cost or rng.random() < np.exp(-(new_cost - current_cost) / T):
                current_points = new_points
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_points = current_points.copy()
            
            T *= cooling
            if T < T_min:
                break
                
        return best_points, best_cost

    # Generate Hex Lattice Start
    sqrt3 = np.sqrt(3)
    cart_points = []
    # Center
    cart_points.append((0.0, 0.0))
    # Ring 1
    dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    for q, r in dirs:
        x = q + r/2.0
        y = sqrt3/2.0 * r
        cart_points.append((x, y))
    # Ring 2
    ring2_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) == 2:
                ring2_points.append((q, r))
    for q, r in ring2_points:
        x = q + r/2.0
        y = sqrt3/2.0 * r
        cart_points.append((x, y))
        
    outer_indices = list(range(7, 19))
    best_min_max_dist = float('inf')
    best_subset = None
    for remove_combo in itertools.combinations(outer_indices, 3):
        remove_set = set(remove_combo)
        current_points = []
        for i in range(19):
            if i not in remove_set:
                current_points.append(cart_points[i])
        max_dist = 0.0
        n = len(current_points)
        for i in range(n):
            for j in range(i+1, n):
                dx = current_points[i][0] - current_points[j][0]
                dy = current_points[i][1] - current_points[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > max_dist:
                    max_dist = dist
        if max_dist < best_min_max_dist:
            best_min_max_dist = max_dist
            best_subset = current_points
            
    init_hex = np.array(best_subset, dtype=float)
    
    # Other Starts
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    init1 = np.column_stack([np.cos(angles), np.sin(angles)])
    
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    init2 = np.column_stack([gx.ravel(), gy.ravel()])
    
    configs = [init_hex, init1, init2]
    
    best_overall = None
    best_cost_overall = float('inf')
    
    total_time = 110
    start_total = time.time()
    
    for i, init in enumerate(configs):
        if time.time() - start_total > total_time:
            break
        time_left = total_time - (time.time() - start_total)
        if time_left <= 0:
            break
        alloc_time = min(5.0, time_left / (len(configs) - i))
        res_points, res_cost = anneal(init, time_limit=alloc_time)
        
        if res_cost < best_cost_overall:
            best_cost_overall = res_cost
            best_overall = res_points
            
    if best_overall is None:
        return init_hex
        
    return best_overall
