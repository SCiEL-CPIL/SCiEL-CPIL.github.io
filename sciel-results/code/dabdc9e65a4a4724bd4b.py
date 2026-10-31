"""Circle packing in unit square: Greedy Maximin Packing with Local Search Scaling."""
import numpy as np

N = 26


def _valid(arr, eps=1e-9):
    """Check that all circles are inside [0,1]^2 and have disjoint interiors."""
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < 0) or np.any(x < -eps) or np.any(y < -eps) or np.any(x + r > 1 + eps) or np.any(y + r > 1 + eps):
        return False
    # pairwise distance check
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - eps:
                return False
    return True


def _score(arr):
    """Sum of radii, only if valid; else -inf."""
    if not _valid(arr):
        return -np.inf
    return float(np.sum(arr[:, 2]))


def _greedy_maximin_pack(grid_size=50):
    """
    Greedy algorithm: place circles one by one at the location 
    that maximizes the possible radius (farthest from existing circles and walls).
    """
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Grid for searching
    xs = np.linspace(0, 1, grid_size)
    ys = np.linspace(0, 1, grid_size)
    
    for k in range(N):
        best_r = -1
        best_x, best_y = 0.5, 0.5
        
        for gx in xs:
            d_wall_x = min(gx, 1 - gx)
            for gy in ys:
                d_wall_y = min(gy, 1 - gy)
                d_wall = min(d_wall_x, d_wall_y)
                
                if d_wall <= best_r:
                    continue
                
                # Distance to existing circles
                d_circ = np.inf
                for i in range(k):
                    d = np.hypot(gx - x[i], gy - y[i])
                    avail = d - r[i]
                    if avail < d_circ:
                        d_circ = avail
                    if d_circ <= best_r:
                        break
                
                current_r = min(d_wall, d_circ)
                
                if current_r > best_r:
                    best_r = current_r
                    best_x, best_y = gx, gy
        
        x[k] = best_x
        y[k] = best_y
        r[k] = best_r
        
    return np.column_stack([x, y, r])


def _local_search_scale(arr, max_iters=500, step_size=0.05, scale_factor=1.005, rng=None):
    """
    Local search to resolve overlaps after scaling radii.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    r = arr[:, 2].copy()
    n = len(r)
    
    r *= scale_factor
    
    if _valid(np.column_stack([x, y, r]), eps=1e-8):
        return arr
    
    best_cost = np.inf
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    def calc_cost(x, y, r):
        cost = 0.0
        for i in range(n):
            cost += max(0, r[i] - x[i])**2
            cost += max(0, r[i] - (1 - x[i]))**2
            cost += max(0, r[i] - y[i])**2
            cost += max(0, r[i] - (1 - y[i]))**2
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                overlap = (r[i] + r[j]) - d
                if overlap > 0:
                    cost += overlap**2
        return cost
    
    cur_cost = calc_cost(x, y, r)
    best_cost = cur_cost
    
    for it in range(max_iters):
        i = rng.integers(n)
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        
        old_x, old_y = x[i], y[i]
        x[i] += dx
        y[i] += dy
        x[i] = np.clip(x[i], 0.0, 1.0)
        y[i] = np.clip(y[i], 0.0, 1.0)
        
        new_cost = calc_cost(x, y, r)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
            
        if new_cost < cur_cost:
            cur_cost = new_cost
        else:
            x[i] = old_x
            y[i] = old_y
            
        if best_cost < 1e-8:
            break
            
    if best_cost < 1e-8:
        return np.column_stack([best_x, best_y, best_r])
    return np.column_stack([best_x, best_y, best_r])


def construct():
    best_arr = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        # Slight variation in greedy grid
        grid_size = 50
        
        arr1 = _greedy_maximin_pack(grid_size=grid_size)
        current = arr1.copy()
        
        for _ in range(20):
            new_arr = _local_search_scale(current, max_iters=500, step_size=0.05, scale_factor=1.005, rng=rng)
            
            sc = _score(new_arr)
            if sc > best_score:
                best_score = sc
                best_arr = new_arr.copy()
            
            if _valid(new_arr, eps=1e-8):
                current = new_arr
            else:
                break
                
    if best_arr is None:
        best_arr = _greedy_maximin_pack(grid_size=50)
        
    return best_arr
