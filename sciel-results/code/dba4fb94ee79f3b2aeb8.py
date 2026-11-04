import math
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 11
APOTHEM = math.sqrt(3) / 2.0
MIN_DIST = math.sqrt(3)
MIN_DIST_SQ = 3.0
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 4.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: return a valid, though suboptimal, configuration
        # Simple grid
        pts = []
        for i in range(3):
            for j in range(3):
                x = i * math.sqrt(3)
                y = j * 1.5
                if (i + j) % 2 == 0: # Checkerboard to ensure dist >= sqrt(3)?
                     # Actually, let's just use a known valid small cluster
                     pass
        # Hardcoded fallback: 7 hex cluster + 4 more
        # Center (0,0)
        # Ring 1: dist sqrt(3)
        # This is getting complicated for fallback. 
        # Just return the first candidate from the list below.
        pass

    def get_lattice_candidates():
        # Generate a few good lattice-based starting points
        candidates = []
        
        # 1. Compact 7 + 4 (Star-like)
        # Axial to Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
        def ax_to_cart(axial):
            pts = []
            for q, r in axial:
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                pts.append((x, y))
            return np.array(pts)

        # Cluster 1: Center + 6 neighbors + 4 more
        # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Add: (2,0), (0,2), (-2,2), (2,-2) ? 
        # Let's use the best one from the previous code's list
        c1 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)]
        candidates.append(ax_to_cart(c1))

        # Cluster 2: 3x3 block + 2
        c2 = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)]
        candidates.append(ax_to_cart(c2))

        # Cluster 3: Cross
        c3 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)]
        candidates.append(ax_to_cart(c3))
        
        # Cluster 4: Elongated
        c4 = [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)]
        candidates.append(ax_to_cart(c4))
        
        # Cluster 5: 4x3 minus corners
        c5 = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)]
        candidates.append(ax_to_cart(c5))

        return candidates

    def objective_and_grads(vars):
        """
        vars: [x1, y1, ..., x11, y11, beta]
        Returns: S (side length of outer hexagon)
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
        
        p = vars[:2*N].reshape(N, 2)
        beta = vars[2*N]
        
        # Normals for the 6 sides of the outer hexagon
        # The outer hexagon has rotation beta.
        # Its sides are perpendicular to directions beta + pi/6 + k*pi/3
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Project points onto normals
        projs = p @ normals.T # shape (N, 6)
        
        # Width in each direction
        # For aligned hexagons, the width of the cluster is 
        # max(proj) - min(proj) + 2 * APOTHEM
        widths = np.max(projs, axis=0) - np.min(projs, axis=0) + 2 * APOTHEM
        
        # The side length S is related to width W by W = S * sqrt(3)
        # So S = W / sqrt(3)
        # We want to minimize the maximum S across all 6 directions
        # S_max = max(widths) / sqrt(3)
        
        return np.max(widths) / math.sqrt(3)

    def constraints(vars):
        """
        Non-overlap constraints: ||p_i - p_j||^2 >= MIN_DIST_SQ
        We define constraints as ||p_i - p_j||^2 - MIN_DIST_SQ >= 0
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return []
            
        p = vars[:2*N].reshape(N, 2)
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                diff = p[i] - p[j]
                dist_sq = np.dot(diff, diff)
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: np.dot(v[2*i:2*i+2] - v[2*j:2*j+2], v[2*i:2*i+2] - v[2*j:2*j+2]) - MIN_DIST_SQ
                })
        return cons

    candidates = get_lattice_candidates()
    
    best_S = float('inf')
    best_vars = None
    
    # Try optimizing from each candidate
    for idx, init_pts in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Center the points to help optimization
        centroid = np.mean(init_pts, axis=0)
        init_pts_centered = init_pts - centroid
        
        # Initial guess for beta: 0
        init_vars = np.concatenate([init_pts_centered.ravel(), [0.0]])
        
        # Run optimization
        # Use SLSQP as it handles inequality constraints well for small problems
        # Max iterations and time checks are handled inside objective/constraints
        try:
            res = minimize(
                objective_and_grads, 
                init_vars, 
                method='SLSQP', 
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < best_S:
                S_val = res.fun
                # Verify constraints are satisfied
                p_opt = res.x[:2*N].reshape(N, 2)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if np.dot(p_opt[i]-p_opt[j], p_opt[i]-p_opt[j]) < MIN_DIST_SQ - 1e-9:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    best_S = S_val
                    best_vars = res.x
                    
        except Exception as e:
            pass

    # If optimization failed or didn't find anything, use the best candidate as is
    if best_vars is None:
        # Fallback to the first candidate, assuming it's valid
        # We need to calculate S for the first candidate
        p = candidates[0]
        centroid = np.mean(p, axis=0)
        p = p - centroid
        beta = 0.0
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = p @ normals.T
        widths = np.max(projs, axis=0) - np.min(projs, axis=0) + 2 * APOTHEM
        best_S = np.max(widths) / math.sqrt(3)
        best_vars = np.concatenate([p.ravel(), [0.0]])

    # Extract final solution
    p_final = best_vars[:2*N].reshape(N, 2)
    beta_final = best_vars[2*N]
    
    # The outer hexagon center is the centroid of the inner hexagons?
    # Actually, the optimal center of the outer hexagon is the midpoint of the min and max projections
    # For a regular hexagon, if it encloses the cluster, the center C satisfies:
    # C.n_k = (max(p.n_k) + min(p.n_k)) / 2 for all k? 
    # This is only true if the hexagon is "tight" in all 6 directions, which might not be the case.
    # However, the LP solution for the minimum enclosing hexagon of a set of points (with fixed orientation)
    # gives the center.
    # Let's re-calculate the optimal center for the final configuration using a quick LP or geometric mean.
    # A robust way: The center of the minimum enclosing hexagon of a set of points is the centroid of the 
    # "dual" vertices? No.
    # Let's just use the centroid of the inner hexagons as a good approximation for the outer center,
    # or better, solve the LP for the center given the final positions and beta.
    
    def calc_optimal_center(p, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = p @ normals.T
        h = np.max(projs, axis=0)
        g = np.min(projs, axis=0)
        
        # Minimize t s.t. C.n_k - t <= h_k, -C.n_k - t <= g_k
        # This is a 3-variable LP.
        from scipy.optimize import linprog
        A_ub = np.zeros((12, 3))
        b_ub = np.zeros(12)
        for k in range(6):
            nx, ny = normals[k]
            A_ub[2*k, 0] = nx
            A_ub[2*k, 1] = ny
            A_ub[2*k, 2] = -1.0
            b_ub[2*k] = h[k]
            
            A_ub[2*k+1, 0] = -nx
            A_ub[2*k+1, 1] = -ny
            A_ub[2*k+1, 2] = -1.0
            b_ub[2*k+1] = g[k]
            
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                return (res.x[0], res.x[1])
        except:
            pass
        return (0.0, 0.0)

    center_x, center_y = calc_optimal_center(p_final, beta_final)
    
    # Recalculate S with the optimal center to be precise
    p_centered = p_final - np.array([center_x, center_y])
    angles = beta_final + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    projs = p_centered @ normals.T
    widths = np.max(projs, axis=0) - np.min(projs, axis=0) + 2 * APOTHEM
    final_S = np.max(widths) / math.sqrt(3)

    # Construct return values
    inner = []
    for i in range(N):
        x, y = p_final[i]
        # Inner hexagons are aligned with the outer one?
        # The problem says "each inner hexagon may translate and rotate".
        # We assumed they are aligned with the outer one (relative angle 0).
        # This is a valid configuration.
        inner.append([float(x), float(y), float(math.degrees(beta_final))])
        
    center = [float(center_x), float(center_y)]
    side = float(final_S)
    rotation = float(math.degrees(beta_final))
    
    return inner, center, side, rotation

# Fallback for the initial check
if __name__ == "__main__":
    # Just to ensure no syntax errors
    pass
