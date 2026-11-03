import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 12.0
EPS = 1e-9

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

    # Generate diverse starting configurations
    candidates = []
    # Standard hexagonal cluster
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    # Cross
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # 3x3 block + 2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    # Compact 3x3+2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    # Star
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    # Elongated
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    # Asymmetric star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    # Compact with top
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    # Wide base
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    # Dense cluster
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    # Zigzag
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    # Star with top cluster
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    # Trapezoid
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    # 4x3 variant
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    # Hexagonal ring
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # Offset grid
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)])
    # Dense diamond
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2),(1,3),(2,3),(0,2)])
    # Wide with top
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    # Symmetric star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # 3x3+row
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])

    def compute_S(pts, cx, cy, beta):
        """Compute exact S (outer side length) for given positions, center, rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T  # (n_pts, 6)
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def smooth_S(pts, cx, cy, beta, temp=0.02):
        """Smooth surrogate using log-sum-exp for the max."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T  # (n_pts, 6)
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 500:
            return 1e6
        shifted = (all_projs - max_val) / temp
        shifted = np.clip(shifted, -100, 100)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temp
        D = temp * log_sum
        return (D + APOTHEM) / APOTHEM

    def full_objective(params, temp=0.0):
        """
        Full objective: all positions (2N), center (2), beta (1).
        params layout: [x1,y1,...,xN,yN, cx, cy, beta]
        Returns S with non-overlap penalty.
        """
        n = N
        pos = params[:2*n].reshape(n, 2)
        cx = params[2*n]
        cy = params[2*n+1]
        beta = params[2*n+2]
        beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)

        if temp > 0:
            S = smooth_S(pos, cx, cy, beta, temp)
        else:
            S = compute_S(pos, cx, cy, beta)

        # Non-overlap penalty: for each pair, distance must be >= sqrt(3)
        min_dist = math.sqrt(3)
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    penalty += (min_dist - d) ** 2

        return S + 10.0 * penalty

    def smooth_objective(params, temp=0.02):
        """Smooth version for gradient-based methods."""
        n = N
        pos = params[:2*n].reshape(n, 2)
        cx = params[2*n]
        cy = params[2*n+1]
        beta = params[2*n+2]
        beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)

        S = smooth_S(pos, cx, cy, beta, temp)

        # Smooth non-overlap penalty
        min_dist = math.sqrt(3)
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    penalty += (min_dist - d) ** 2

        return S + 5.0 * penalty

    def optimize_from_start(pts0, beta0=0.0):
        """Full joint optimization from a starting configuration."""
        n = N
        pts0 = np.array(pts0, dtype=float)
        cx0 = float(np.mean(pts0[:, 0]))
        cy0 = float(np.mean(pts0[:, 1]))

        # Initial parameter vector
        x0 = np.zeros(2*n + 3)
        x0[:2*n] = pts0.flatten()
        x0[2*n] = cx0
        x0[2*n+1] = cy0
        x0[2*n+2] = beta0

        best_S = float('inf')
        best_x = x0.copy()

        # Stage 1: Coarse Nelder-Mead with smooth surrogate
        temps = [0.1, 0.05, 0.02, 0.01, 0.005]
        for temp in temps:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            try:
                res = minimize(smooth_objective, best_x, args=(temp,),
                             method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Stage 2: Exact objective polish with multiple starts
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # Create perturbed starts
            starts = [best_x.copy()]
            for perturb in [0.01, 0.005, -0.01, -0.005]:
                for dim in range(len(best_x)):
                    s = best_x.copy()
                    s[dim] += perturb
                    starts.append(s)
            # Also try different betas
            for b in [0.0, math.pi/12, math.pi/6 - 0.01, 0.04, 0.08]:
                s = best_x.copy()
                s[2*n+2] = b
                starts.append(s)

            for start in starts[:15]:  # Limit number of starts
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                try:
                    res = minimize(full_objective, start, args=(0.0,),
                                 method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_x = res.x
                except Exception:
                    pass

        # Stage 3: Very fine polish
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            try:
                res = minimize(full_objective, best_x, args=(0.0,),
                             method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_x = res.x
            except Exception:
                pass

        # Extract result
        pos = best_x[:2*n].reshape(n, 2)
        cx = best_x[2*n]
        cy = best_x[2*n+1]
        beta = ((best_x[2*n+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)

        # Verify non-overlap
        min_dist = math.sqrt(3)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist - 1e-6:
                    # Invalid, return inf
                    return float('inf'), None, None, None

        S = compute_S(pos, cx, cy, beta)
        return S, pos, (cx, cy), beta

    # Evaluate all candidates
    best_global_S = float('inf')
    best_global_pos = None
    best_global_center = None
    best_global_beta = 0.0

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)

        # Try multiple initial betas
        for b0 in [0.0, math.pi/12, math.pi/6 - 0.01]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            S, pos, center, beta = optimize_from_start(pts, b0)
            if S < best_global_S:
                best_global_S = S
                best_global_pos = pos
                best_global_center = center
                best_global_beta = beta

    # Additional: try random perturbations of best found
    if best_global_pos is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        np.random.seed(42)
        for trial in range(10):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            perturbed = best_global_pos.copy()
            perturbed += np.random.randn(N, 2) * 0.05
            S, pos, center, beta = optimize_from_start(perturbed, best_global_beta + np.random.randn()*0.02)
            if S < best_global_S:
                best_global_S = S
                best_global_pos = pos
                best_global_center = center
                best_global_beta = beta

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback
        pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_global_beta = 0.0
        best_global_pos = pts
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Final computation with safety
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (best_global_pos[:, 0] - cx) * nx + (best_global_pos[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))

    D = max_proj
    S = (D + APOTHEM) / APOTHEM

    inner = []
    for x, y in best_global_pos:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(angle_deg)

    return inner, center, side, rotation
