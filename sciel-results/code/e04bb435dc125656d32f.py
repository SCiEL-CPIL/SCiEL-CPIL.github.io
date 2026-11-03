import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Unit hexagon vertices (flat-top orientation, side=1)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_angles = np.array([k * math.pi / 3.0 for k in range(6)])
    hex_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])  # 6x2

    # For separation constraint: two unit hexagons are disjoint iff
    # the minimum over all 6*6 vertex-pair projections along 6 directions >= sqrt(3)
    # Actually, for two regular hexagons, the separation is determined by
    # the 6 directions of the normals. The minimum projection difference
    # along any of the 6 normal directions must be >= sqrt(3) (the distance between opposite sides).
    # 
    # More precisely: two convex polygons P and Q are disjoint iff there exists a separating axis.
    # For two regular hexagons, the candidate separating axes are the 6 normal directions
    # (which are the same for both if they have the same orientation).
    #
    # For two hexagons with the SAME orientation, they are disjoint iff for all 6 normal directions n_k:
    #   (max_{v in P} v.n_k) - (min_{w in Q} w.n_k) <= 0  OR
    #   (max_{w in Q} w.n_k) - (min_{v in P} v.n_k) <= 0
    # i.e., their projections on every normal direction do not overlap.
    #
    # The projection of a unit hexagon centered at c onto direction n is [c.n - APOTHEM, c.n + APOTHEM].
    # So two hexagons centered at c1, c2 are disjoint iff for all 6 directions:
    #   |c1.n - c2.n| >= 2 * APOTHEM = sqrt(3)
    #
    # This is the key simplification! For same-orientation hexagons, separation is just
    # a set of 6 linear constraints on the center difference.

    # Normal directions for the 6 sides
    norm_angles = np.array([math.pi / 6.0 + k * math.pi / 3.0 for k in range(6)])
    normals = np.column_stack([np.cos(norm_angles), np.sin(norm_angles)])  # 6x2

    def objective(params):
        """Objective: outer hexagon side length S."""
        # params layout: [x1,y1,...,xN,yN, cx, cy, beta]
        centers = params[:2 * N].reshape(N, 2)
        cx, cy = params[2 * N], params[2 * N + 1]
        beta = params[2 * N + 2]

        # Outer hexagon normals (rotated by beta)
        cos_b, sin_b = math.cos(beta), math.sin(beta)
        rot = np.array([[cos_b, -sin_b], [sin_b, cos_b]])
        outer_normals = normals @ rot.T  # 6x2

        # Project all inner centers onto each outer normal
        # proj[i, k] = (center_i - outer_center) . outer_normal_k
        rel = centers - np.array([cx, cy])  # Nx2
        projs = rel @ outer_normals.T  # Nx6

        # D = max over all projections of |proj|
        D = np.max(np.abs(projs))
        S = (D + APOTHEM) / APOTHEM
        return S

    def constraints_func(params):
        """Return list of constraint dicts: |c_i.n_k - c_j.n_k| >= sqrt(3) for all i<j, k."""
        centers = params[:2 * N].reshape(N, 2)
        sqrt3 = math.sqrt(3)
        cons = []
        for k in range(6):
            n = normals[k]
            proj = centers @ n  # N
            for i in range(N):
                for j in range(i + 1, N):
                    d = proj[i] - proj[j]
                    # Constraint: d >= sqrt3 OR d <= -sqrt3
                    # We need to handle this. Use a smooth penalty approach instead.
                    # For now, collect both.
                    cons.append({'type': 'ineq', 'fun': lambda d=d: d - sqrt3})
                    cons.append({'type': 'ineq', 'fun': lambda d=d: -d - sqrt3})
        return cons

    def smooth_constraints(params):
        """Smooth constraint: for each pair and direction, (|d| - sqrt3) <= 0 where d = proj_i - proj_j.
        Use (d^2 - 3) <= 0 as a smooth approximation (d^2 >= 3 means |d| >= sqrt3)."""
        centers = params[:2 * N].reshape(N, 2)
        # For each direction k, compute all pairwise differences
        # Use vectorized computation
        # proj = centers @ normals.T  # Nx6
        # For each k, we need all pairwise |proj_i - proj_j|^2 >= 3
        # This is equivalent to: for each k, the minimum pairwise squared difference >= 3
        
        # Better: use a penalty-based approach in the objective
        pass

    # Alternative: use penalty method
    def penalized_objective(params, penalty=1000.0):
        """Objective with penalty for constraint violations."""
        S = objective(params)
        centers = params[:2 * N].reshape(N, 2)
        sqrt3 = math.sqrt(3)
        
        # Compute constraint violations
        penalty_val = 0.0
        for k in range(6):
            n = normals[k]
            proj = centers @ n
            for i in range(N):
                for j in range(i + 1, N):
                    d = proj[i] - proj[j]
                    # Violation if |d| < sqrt3
                    if abs(d) < sqrt3:
                        violation = sqrt3 - abs(d)
                        penalty_val += violation * violation
        return S + penalty * penalty_val

    def generate_initial_configurations():
        """Generate diverse initial configurations."""
        np.random.seed(42)
        configs = []
        
        # Hexagonal spiral
        axial_spiral = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (1, -2)
        ]
        pts = np.array([[math.sqrt(3) * (q + r / 2.0), 1.5 * r] for q, r in axial_spiral])
        configs.append(pts)
        
        # Compact hexagonal arrangement
        axial_2 = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (1, 1), (-1, 2), (-2, 1), (2, -1)
        ]
        pts2 = np.array([[math.sqrt(3) * (q + r / 2.0), 1.5 * r] for q, r in axial_2])
        configs.append(pts2)
        
        # Random
        for _ in range(3):
            pts_r = np.random.randn(N, 2) * 2.0
            configs.append(pts_r)
        
        # Ring + center
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N) * 1.8
        pts_ring = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        configs.append(pts_ring)
        
        return configs

    def run_optimization(pts_init, max_time_per=30.0):
        """Run nonlinear optimization from a given initial configuration."""
        t0 = time.time()
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None, None
        
        # Initial outer hexagon parameters
        cx0 = float(np.mean(pts_init[:, 0]))
        cy0 = float(np.mean(pts_init[:, 1]))
        beta0 = 0.0
        
        x0 = np.concatenate([pts_init.flatten(), [cx0, cy0, beta0]])
        
        best_S = float('inf')
        best_x = x0.copy()
        
        # Try multiple penalty values
        for penalty in [100.0, 500.0, 2000.0]:
            if time.time() - t0 > max_time_per or time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            try:
                res = minimize(
                    penalized_objective,
                    x0,
                    args=(penalty,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_S:
                    best_S = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Also try with the raw objective (no penalty) to see if we're already feasible
        try:
            res_raw = minimize(
                objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res_raw.fun < best_S:
                best_S = res_raw.fun
                best_x = res_raw.x.copy()
        except Exception:
            pass
        
        # Extract result
        centers = best_x[:2 * N].reshape(N, 2)
        cx, cy = best_x[2 * N], best_x[2 * N + 1]
        beta = best_x[2 * N + 2]
        
        # Verify feasibility
        sqrt3 = math.sqrt(3)
        feasible = True
        for k in range(6):
            n = normals[k]
            proj = centers @ n
            for i in range(N):
                for j in range(i + 1, N):
                    if abs(proj[i] - proj[j]) < sqrt3 - 1e-6:
                        feasible = False
                        break
                if not feasible:
                    break
            if not feasible:
                break
        
        if not feasible:
            # Scale up to make feasible
            com = np.mean(centers, axis=0)
            for scale in np.linspace(1.0, 3.0, 100):
                centers_scaled = com + (centers - com) * scale
                feasible2 = True
                for k in range(6):
                    n = normals[k]
                    proj = centers_scaled @ n
                    for i in range(N):
                        for j in range(i + 1, N):
                            if abs(proj[i] - proj[j]) < sqrt3 - 1e-6:
                                feasible2 = False
                                break
                        if not feasible2:
                            break
                    if not feasible2:
                        break
                if feasible2:
                    centers = centers_scaled
                    break
        
        # Compute final S
        S = objective(best_x)
        return S, centers, (cx, cy), beta

    # Run optimization from multiple starting points
    configs = generate_initial_configurations()
    best_global_S = float('inf')
    best_global_centers = None
    best_global_center = None
    best_global_beta = 0.0
    
    for i, pts_init in enumerate(configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        S, centers, center, beta = run_optimization(pts_init)
        if S < best_global_S:
            best_global_S = S
            best_global_centers = centers
            best_global_center = center
            best_global_beta = beta
        
        # Early stop if very good
        if best_global_S < 3.0:
            break
    
    # Fallback
    if best_global_centers is None or best_global_S == float('inf'):
        best_global_centers = configs[0]
        best_global_center = (float(np.mean(best_global_centers[:, 0])), float(np.mean(best_global_centers[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_centers[:, 0] * nx + best_global_centers[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
    
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Verify and adjust
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (best_global_centers[:, 0] - cx) * nx + (best_global_centers[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in best_global_centers:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
