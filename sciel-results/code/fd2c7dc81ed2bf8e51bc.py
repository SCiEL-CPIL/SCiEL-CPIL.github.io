import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Generate diverse initial configurations
    def generate_initial_configs():
        configs = []
        # Hexagonal shell + center
        configs.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
        # Row patterns
        configs.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
        configs.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
        configs.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
        # Triangular
        configs.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
        configs.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
        # Cross/star
        configs.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
        # Dense cluster
        configs.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
        # Elongated
        configs.append([(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(1,1),(2,1),(3,1),(4,1),(2,2)])
        # Compact
        configs.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(0,2),(-1,2),(2,-1)])
        return configs

    def smooth_objective(params, pts_init, temperature=0.01):
        """
        params: [x1,y1, ..., xN,yN, cx, cy, beta]
        Returns smooth approximation of outer hexagon side length.
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10

        n = N
        inner_pts = params[:2*n].reshape(n, 2)
        cx, cy, beta = params[2*n], params[2*n+1], params[2*n+2]

        # Normalize beta to [0, pi/6)
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)

        # Outer hexagon normals (6 directions)
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])

        # Disjointness: pairwise distance >= 2
        # Smooth penalty for overlapping pairs
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = inner_pts[i,0] - inner_pts[j,0]
                dy = inner_pts[i,1] - inner_pts[j,1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 2.0:
                    # Smooth penalty: (2 - dist)^2
                    overlap_penalty += (2.0 - dist) ** 2

        # Containment: each inner hexagon's vertices must be inside outer hexagon
        # Inner hexagon i has center (xi, yi) and same orientation as outer (for simplicity)
        # Actually, inner hexagons can have any orientation. Let's assume they're aligned with outer.
        # Vertices of inner hexagon i:
        # v_k = (xi, yi) + [cos(beta + k*pi/3), sin(beta + k*pi/3)] for k=0..5
        
        # For each vertex, check it's inside outer hexagon centered at (cx,cy) with side S
        # A point p is inside outer hexagon if for all 6 normals n_k:
        #   n_k . (p - c) <= S * apothem = S * sqrt(3)/2
        # and n_k . (p - c) >= -S * sqrt(3)/2
        
        # We want to find minimum S such that all constraints hold.
        # S = max over all vertices and all normals of |n_k . (v - c)| / apothem
        
        # Collect all vertex projections
        all_projs = []
        for i in range(n):
            for k in range(6):
                vx = inner_pts[i,0] + math.cos(beta + k * math.pi / 3.0)
                vy = inner_pts[i,1] + math.sin(beta + k * math.pi / 3.0)
                for m in range(6):
                    proj = normals[m,0] * (vx - cx) + normals[m,1] * (vy - cy)
                    all_projs.append(proj)
        
        all_projs = np.array(all_projs)
        
        # Smooth max of |proj| using log-sum-exp
        abs_projs = np.abs(all_projs)
        max_val = np.max(abs_projs)
        if max_val > 500:
            return 1e10
        shifted = (abs_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D_smooth = temperature * log_sum
        
        S = (D_smooth + APOTHEM) / APOTHEM
        
        total = S + 10.0 * overlap_penalty
        return total

    def exact_objective(params, pts_init):
        """Exact (nonsmooth) objective for final evaluation."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10

        n = N
        inner_pts = params[:2*n].reshape(n, 2)
        cx, cy, beta = params[2*n], params[2*n+1], params[2*n+2]
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)

        # Check disjointness
        for i in range(n):
            for j in range(i+1, n):
                dx = inner_pts[i,0] - inner_pts[j,0]
                dy = inner_pts[i,1] - inner_pts[j,1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 2.0 - 1e-9:
                    return 1e10

        # Compute exact S
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        max_proj = 0.0
        for i in range(n):
            for k in range(6):
                vx = inner_pts[i,0] + math.cos(beta + k * math.pi / 3.0)
                vy = inner_pts[i,1] + math.sin(beta + k * math.pi / 3.0)
                for m in range(6):
                    proj = abs(normals[m,0] * (vx - cx) + normals[m,1] * (vy - cy))
                    if proj > max_proj:
                        max_proj = proj
        
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    # Generate initial configurations
    initial_configs = generate_initial_configs()
    
    best_S = float('inf')
    best_params = None
    best_center = None
    best_beta = 0.0
    best_inner = None

    # For each initial config, run optimization
    for init_idx, axial in enumerate(initial_configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        pts_init = axial_to_cartesian(axial)
        
        # Center the configuration
        cx0 = float(np.mean(pts_init[:, 0]))
        cy0 = float(np.mean(pts_init[:, 1]))
        beta0 = 0.0

        # Build initial params
        params0 = np.zeros(2*N + 3)
        for i in range(N):
            params0[2*i] = pts_init[i, 0]
            params0[2*i+1] = pts_init[i, 1]
        params0[2*N] = cx0
        params0[2*N+1] = cy0
        params0[2*N+2] = beta0

        # Try multiple temperatures for the smooth surrogate
        temperatures = [0.02, 0.01, 0.005]
        for temp in temperatures:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break

            # Try a few different starting betas
            for sb in [0.0, math.pi/12, -math.pi/12]:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break

                params_try = params0.copy()
                params_try[2*N+2] = ((sb % (math.pi/6)) + math.pi/6) % (math.pi/6)

                try:
                    res = minimize(
                        smooth_objective,
                        params_try,
                        args=(pts_init, temp),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                    if res.fun < 1e10:
                        # Evaluate exact objective
                        S_exact = exact_objective(res.x, pts_init)
                        if S_exact < best_S:
                            best_S = S_exact
                            best_params = res.x.copy()
                            best_center = (float(res.x[2*N]), float(res.x[2*N+1]))
                            best_beta = ((res.x[2*N+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            best_inner = res.x[:2*N].reshape(N, 2)
                except Exception:
                    pass

        # Polish with Nelder-Mead on the exact objective
        if best_params is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            try:
                res = minimize(
                    exact_objective,
                    best_params,
                    args=(pts_init,),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                )
                if res.fun < best_S:
                    best_S = res.fun
                    best_params = res.x.copy()
                    best_center = (float(res.x[2*N]), float(res.x[2*N+1]))
                    best_beta = ((res.x[2*N+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    best_inner = res.x[:2*N].reshape(N, 2)
            except Exception:
                pass

    # Fallback if nothing worked
    if best_inner is None:
        pts_init = axial_to_cartesian(initial_configs[0])
        cx0 = float(np.mean(pts_init[:, 0]))
        cy0 = float(np.mean(pts_init[:, 1]))
        best_center = (cx0, cy0)
        best_beta = 0.0
        best_inner = pts_init
        # Compute S
        angles = best_beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        max_proj = 0.0
        for i in range(N):
            for k in range(6):
                vx = best_inner[i,0] + math.cos(best_beta + k * math.pi / 3.0)
                vy = best_inner[i,1] + math.sin(best_beta + k * math.pi / 3.0)
                for m in range(6):
                    proj = abs(normals[m,0] * (vx - cx0) + normals[m,1] * (vy - cy0))
                    if proj > max_proj:
                        max_proj = proj
        best_S = (max_proj + APOTHEM) / APOTHEM

    # Final polish pass with very tight tolerance
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        try:
            res = minimize(
                exact_objective,
                best_params,
                args=(np.zeros((N,2)),),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True}
            )
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
                best_center = (float(res.x[2*N]), float(res.x[2*N+1]))
                best_beta = ((res.x[2*N+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                best_inner = res.x[:2*N].reshape(N, 2)
        except Exception:
            pass

    # Build output
    cx, cy = best_center
    angle_deg = math.degrees(best_beta)

    # Slight scale to ensure no numerical violations
    SCALE = 1.0 + 1e-12
    inner_pts = best_inner.copy()
    inner_pts[:, 0] = cx + (inner_pts[:, 0] - cx) * SCALE
    inner_pts[:, 1] = cy + (inner_pts[:, 1] - cy) * SCALE

    # Recompute S with scaled points
    angles = best_beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    max_proj = 0.0
    for i in range(N):
        for k in range(6):
            vx = inner_pts[i,0] + math.cos(best_beta + k * math.pi / 3.0)
            vy = inner_pts[i,1] + math.sin(best_beta + k * math.pi / 3.0)
            for m in range(6):
                proj = abs(normals[m,0] * (vx - cx) + normals[m,1] * (vy - cy))
                if proj > max_proj:
                    max_proj = proj
    S_final = (max_proj + APOTHEM) / APOTHEM

    inner = []
    for i in range(N):
        inner.append([float(inner_pts[i, 0]), float(inner_pts[i, 1]), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
