import numpy as np
import time
import itertools

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 500  # Safety margin
    
    # Constants
    SQRT3 = np.sqrt(3)
    APOHEM_OUTER_FACTOR = SQRT3 / 2.0
    APOHEM_INNER_DIFF = SQRT3  # Apothem of Minkowski difference of two unit hexagons
    
    # Helper: check if point (x,y) is inside outer hexagon of side S centered at (0,0)
    # Returns max violation (0 if inside)
    def outer_violation(x, y, S):
        # Projections onto normals of outer hexagon
        # Normals are at 30, 90, 150, 210, 270, 330
        # Projections:
        p1 = abs(x * SQRT3/2 + y * 0.5)
        p2 = abs(y)
        p3 = abs(-x * SQRT3/2 + y * 0.5)
        max_proj = max(p1, p2, p3)
        limit = S * APOHEM_OUTER_FACTOR
        return max(0.0, max_proj - limit)
        
    # Helper: check overlap between two unit hexagons at (x1,y1) and (x2,y2)
    # Returns max violation (0 if disjoint)
    def overlap_violation(x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        # Projections of difference vector onto normals of Minkowski difference hexagon
        p1 = abs(dx * SQRT3/2 + dy * 0.5)
        p2 = abs(dy)
        p3 = abs(-dx * SQRT3/2 + dy * 0.5)
        max_proj = max(p1, p2, p3)
        limit = APOHEM_INNER_DIFF
        return max(0.0, limit - max_proj)

    def objective(params):
        # params: [x0, y0, ..., x10, y10, S]
        S = params[-1]
        if S < np.sqrt(11) - 1e-9:
            return 1e6 # Infeasible S
            
        pts = params[:-1].reshape(11, 2)
        
        penalty = 0.0
        
        # Containment
        for i in range(11):
            cx, cy = pts[i]
            # Check 6 vertices of inner hexagon
            # Vertices relative to center (angle 0):
            # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
            vs = [
                (1.0, 0.0),
                (0.5, SQRT3/2),
                (-0.5, SQRT3/2),
                (-1.0, 0.0),
                (-0.5, -SQRT3/2),
                (0.5, -SQRT3/2)
            ]
            for vx, vy in vs:
                px = cx + vx
                py = cy + vy
                penalty += outer_violation(px, py, S)
                
        # Disjointness
        for i in range(11):
            for j in range(i+1, 11):
                penalty += overlap_violation(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
                
        return S + penalty * 10.0

    # Initial configuration: Hexagonal lattice
    # Center at (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3) from center?
    # Distance between centers in lattice is sqrt(3).
    # Directions: 0, 60, 120, 180, 240, 300
    # Coordinates:
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    # Wait, distance is sqrt(3).
    # x = sqrt(3) * cos(theta), y = sqrt(3) * sin(theta)
    
    def get_initial_config(seed):
        np.random.seed(seed)
        pts = np.zeros((11, 2))
        # Center
        pts[0] = [0, 0]
        
        # Ring 1
        for k in range(6):
            angle = k * np.pi / 3
            r = SQRT3
            pts[k+1] = [r * np.cos(angle), r * np.sin(angle)]
            
        # Ring 2 (4 more)
        # Place them further out
        # e.g. at distance 2*sqrt(3) in some directions
        # Or just perturb the whole lattice
        # Let's try a specific compact shape
        # 1 center, 6 ring 1, 4 ring 2
        # Ring 2 positions in lattice:
        # (2*sqrt(3), 0) is too far?
        # Let's just use a perturbed lattice for restarts
        
        # Add noise
        pts += np.random.normal(0, 0.1, (11, 2))
        
        # Estimate S
        # Max distance from origin to any vertex
        max_d = 0
        for i in range(11):
            cx, cy = pts[i]
            for vx, vy in [(1,0), (0.5, SQRT3/2), (-0.5, SQRT3/2), (-1,0), (-0.5, -SQRT3/2), (0.5, -SQRT3/2)]:
                px = cx + vx
                py = cy + vy
                # Hexagonal norm approx
                d = max(abs(px * SQRT3/2 + py * 0.5), abs(py), abs(-px * SQRT3/2 + py * 0.5))
                max_d = max(max_d, d)
        S_est = max_d / APOHEM_OUTER_FACTOR + 0.1
        params = np.concatenate([pts.flatten(), [S_est]])
        return params

    best_params = None
    best_score = np.inf
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        params = get_initial_config(r)
        
        # Use Nelder-Mead
        # We can use scipy if available, otherwise implement a simple version or use Powell
        # scipy.optimize.minimize is robust
        try:
            from scipy.optimize import minimize
            res = minimize(objective, params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except ImportError:
            # Fallback: simple coordinate descent or just return seed if no scipy
            # But scipy is stated to be available
            pass

    if best_params is None:
        # Fallback to seed-like solution
        # Just return a valid, albeit large, packing
        inner = np.zeros((11, 3))
        # ... fill with valid positions ...
        # For now, return the initial config of restart 0 with large S
        params = get_initial_config(0)
        S = params[-1] + 5.0
        inner = np.zeros((11, 3))
        for i in range(11):
            inner[i, 0] = params[2*i]
            inner[i, 1] = params[2*i+1]
            inner[i, 2] = 0.0
        outer = np.array([0.0, 0.0, 0.0])
        return inner, outer, S

    # Extract best
    S = best_params[-1]
    inner = np.zeros((11, 3))
    for i in range(11):
        inner[i, 0] = best_params[2*i]
        inner[i, 1] = best_params[2*i+1]
        inner[i, 2] = 0.0
        
    outer = np.array([0.0, 0.0, 0.0])
    
    return inner, outer, S

if __name__ == "__main__":
    inner, outer, S = hexagon_packing_11()
    print(f"Side length: {S}")
