import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM_IN = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def get_initial_points(A):
        # Generate points on a triangular lattice
        # Scale factor such that distance between neighbors is approx 2*A / 2?
        # Just generate a grid and scale
        pts = []
        # Generate a 5x5 grid of axial coords
        for q in range(-2, 3):
            for r in range(-2, 3):
                pts.append((q, r))
        
        # Convert to cartesian (scaled by 1 for now)
        scale = 1.0
        x = scale * math.sqrt(3) * (p[0] + p[1] / 2.0) if False else 0 # dummy
        # Standard axial to cartesian:
        # x = sqrt(3) * (q + r/2)
        # y = 3/2 * r
        
        raw_pts = []
        for q, r in pts:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            raw_pts.append([x, y])
        
        raw_pts = np.array(raw_pts)
        
        # Find the point furthest from origin to determine scale
        # We want all points to be inside Hex(A).
        # The "radius" of the hexagon in terms of distance to vertex is 2*A/sqrt(3).
        # Distance to flat is A.
        # Let's just scale so that the max distance from origin is A * 0.9 (safety margin)
        max_dist = np.max(np.linalg.norm(raw_pts, axis=1))
        if max_dist == 0:
             return np.zeros((N, 2))
        
        # We need exactly N points. Take the first N?
        # Better: take the N closest to origin? Or just generate a specific cluster.
        # For N=11, a 3x3 block + 2 is good.
        # Let's just use the first 11 points from the grid centered at 0.
        # Sort by distance to origin
        dists = np.linalg.norm(raw_pts, axis=1)
        idx = np.argsort(dists)
        selected = raw_pts[idx[:N]]
        
        # Scale to fit
        # The apothem of the selected cluster's bounding hexagon?
        # Simpler: Scale so that the max norm is A * 0.8
        max_norm = np.max(np.linalg.norm(selected, axis=1))
        if max_norm > 0:
             scale_factor = (A * 0.8) / max_norm
        else:
             scale_factor = 1.0
             
        return selected * scale_factor

    def solve_for_A(A):
        """
        Maximize t (min squared distance) subject to points in Hex(A)
        Returns max_t
        """
        if time.time() - START_TIME > TIME_LIMIT:
            return -1.0

        x0 = get_initial_points(A)
        # Variables: x1, y1, ..., x11, y11, t
        # Total 22 + 1 = 23 variables
        x0_full = np.append(x0.flatten(), 1.0) # initial t guess
        
        # Objective: Minimize -t
        def objective(v):
            t = v[-1]
            return -t

        # Constraints
        # 1. Distance constraints: (xi-xj)^2 + (yi-yj)^2 - t >= 0
        # 2. Hexagon constraints for each point i:
        #    |xi| <= A
        #    |sqrt(3)/2 * xi + 1/2 * yi| <= A
        #    |sqrt(3)/2 * xi - 1/2 * yi| <= A
        
        cons = []
        
        # Distance constraints
        for i in range(N):
            for j in range(i+1, N):
                xi, yi = v[2*i], v[2*i+1]
                xj, yj = v[2*j], v[2*j+1]
                # (xi-xj)^2 + (yi-yj)^2 - t >= 0
                # This is non-linear. SLSQP handles it.
                # We define constraint as g(v) >= 0
                def dist_con(v, i=i, j=j):
                    xi, yi = v[2*i], v[2*i+1]
                    xj, yj = v[2*j], v[2*j+1]
                    t = v[-1]
                    return (xi - xj)**2 + (yi - yj)**2 - t
                cons.append({'type': 'ineq', 'fun': dist_con})
        
        # Hexagon constraints
        for i in range(N):
            xi, yi = v[2*i], v[2*i+1]
            
            # |xi| <= A  =>  A - |xi| >= 0
            def hex_c1(v, i=i):
                xi = v[2*i]
                return A - abs(xi)
            cons.append({'type': 'ineq', 'fun': hex_c1})
            
            # |sqrt(3)/2 * xi + 1/2 * yi| <= A
            def hex_c2(v, i=i):
                xi, yi = v[2*i], v[2*i+1]
                val = math.sqrt(3)/2 * xi + 0.5 * yi
                return A - abs(val)
            cons.append({'type': 'ineq', 'fun': hex_c2})
            
            # |sqrt(3)/2 * xi - 1/2 * yi| <= A
            def hex_c3(v, i=i):
                xi, yi = v[2*i], v[2*i+1]
                val = math.sqrt(3)/2 * xi - 0.5 * yi
                return A - abs(val)
            cons.append({'type': 'ineq', 'fun': hex_c3})
        
        # Bounds for x, y
        bounds = []
        for i in range(N):
            bounds.append((-A, A))
            bounds.append((-A, A))
        bounds.append((0, 10.0)) # t bound
        
        try:
            res = minimize(objective, x0_full, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 200, 'ftol': 1e-9})
            if res.success:
                return res.x[-1] # return t
            else:
                # If it fails, return a low value to indicate failure
                return -1.0
        except Exception:
            return -1.0

    # Binary search for A
    # S = 2*A/sqrt(3) + 1
    # Min S ~ 2.0 (for 7 hexes). For 11, maybe 2.5?
    # A for S=2.5: 2.5 = 2*A/sqrt(3) + 1 => 1.5 = 2*A/sqrt(3) => A = 1.5*sqrt(3)/2 = 1.299
    # A for S=3.0: A = 2.0*sqrt(3)/2 = 1.732
    
    A_low = 1.0  # S ~ 2.15
    A_high = 2.0 # S ~ 3.31
    
    # Check if A_high is feasible
    if solve_for_A(A_high) < 3.0:
        # Even A_high is not enough? Increase A_high
        A_high = 3.0
        if solve_for_A(A_high) < 3.0:
             # Fallback
             return [[0.0, 0.0, 0.0]], [0.0, 0.0], 3.0, 0.0

    best_A = A_high
    best_t = -1.0
    best_x = None
    
    for _ in range(20): # 20 iterations of binary search
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        A_mid = (A_low + A_high) / 2.0
        t_val = solve_for_A(A_mid)
        
        if t_val >= 3.0:
            A_high = A_mid
            best_A = A_mid
            # We need to recover the positions for best_A
            # Re-run solve_for_A to get x?
            # To save time, we can just store the result of the last successful solve
            # But solve_for_A only returns t.
            # Let's modify solve_for_A to return x as well, or just re-run at the end.
            # Re-running at the end is safer.
        else:
            A_low = A_mid
            
    # Final run to get positions for best_A
    # We need a version of solve that returns the full solution
    def solve_final(A):
        x0 = get_initial_points(A)
        x0_full = np.append(x0.flatten(), 1.0)
        
        def objective(v):
            return -v[-1]

        cons = []
        for i in range(N):
            for j in range(i+1, N):
                def dist_con(v, i=i, j=j):
                    xi, yi = v[2*i], v[2*i+1]
                    xj, yj = v[2*j], v[2*j+1]
                    t = v[-1]
                    return (xi - xj)**2 + (yi - yj)**2 - t
                cons.append({'type': 'ineq', 'fun': dist_con})
        
        for i in range(N):
            def hex_c1(v, i=i):
                return A - abs(v[2*i])
            cons.append({'type': 'ineq', 'fun': hex_c1})
            
            def hex_c2(v, i=i):
                xi, yi = v[2*i], v[2*i+1]
                return A - abs(math.sqrt(3)/2 * xi + 0.5 * yi)
            cons.append({'type': 'ineq', 'fun': hex_c2})
            
            def hex_c3(v, i=i):
                xi, yi = v[2*i], v[2*i+1]
                return A - abs(math.sqrt(3)/2 * xi - 0.5 * yi)
            cons.append({'type': 'ineq', 'fun': hex_c3})
        
        bounds = [(-A, A), (-A, A)] * N + [(0, 10.0)]
        
        try:
            res = minimize(objective, x0_full, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 300, 'ftol': 1e-10})
            if res.success:
                return res.x
            else:
                return None
        except Exception:
            return None

    x_sol = solve_final(best_A)
    
    if x_sol is None:
        # Fallback to a known good configuration (Flower)
        # Side length 2.5
        S = 2.5
        A = (S - 1) * math.sqrt(3) / 2.0
        # Just return the initial points scaled
        pts = get_initial_points(A)
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [0.0, 0.0], S, 0.0

    # Extract positions
    pts = []
    for i in range(N):
        x = x_sol[2*i]
        y = x_sol[2*i+1]
        pts.append([float(x), float(y), 0.0])
    
    # Calculate Side Length
    S = 2.0 * best_A / math.sqrt(3) + 1.0
    
    # Center is 0,0
    center = [0.0, 0.0]
    rotation = 0.0
    
    return pts, center, float(S), rotation

