"""Heilbronn triangles: place 11 points in a unit equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3.0)
# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])
TRI_AREA = SQRT3 / 4.0  # area of the unit equilateral triangle

# Deadline guard
T_START = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill


def bary_to_cart(b):
    """Convert barycentric coords (a,b,c) with a+b+c=1 to Cartesian.
    We use a,b as free variables, c=1-a-b."""
    a = b[:, 0]
    b_ = b[:, 1]
    x = a * V1[0] + b_ * V2[0]
    y = a * V1[1] + b_ * V2[1]
    return np.column_stack([x, y])


def cart_to_bary(pts):
    """Convert Cartesian points to barycentric (a, b) where c=1-a-b."""
    # Solve: p = a*V1 + b*V2 (since V0 is origin)
    # [x]   [1.0  0.5 ][a]
    # [y] = [0.0  s3/2 ][b]
    s3_2 = SQRT3 / 2.0
    x = pts[:, 0]
    y = pts[:, 1]
    b_ = y / s3_2
    a = x - 0.5 * b_
    return np.column_stack([a, b_])


def min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area


def soft_min_objective(b, beta):
    """Return negative of soft-min of triangle areas (for minimization).
    b: (N, 2) array of barycentric coords (a, b) with c=1-a-b."""
    pts = bary_to_cart(b)
    n = len(pts)
    # Collect all triangle areas
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                areas.append(area)
    areas = np.array(areas)
    # Soft-min via log-sum-exp: min_i a_i ≈ -(1/beta)*log(sum(exp(-beta*a_i)))
    # But we want to maximize the min, so we minimize -soft_min
    # soft_min = -(1/beta) * log(sum(exp(-beta * areas)))
    # We minimize -soft_min = (1/beta) * log(sum(exp(-beta * areas)))
    # For numerical stability:
    max_neg = np.max(-beta * areas)
    log_sum = max_neg + np.log(np.sum(np.exp(-beta * areas - max_neg)))
    soft_min_val = -(1.0 / beta) * log_sum
    return -soft_min_val


def soft_min_grad(b, beta):
    """Gradient of -soft_min w.r.t. barycentric coords."""
    pts = bary_to_cart(b)
    n = len(pts)
    # Build area array and compute gradient
    areas = np.empty(165)  # C(11,3) = 165
    idx = 0
    tri_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = dx1 * dy2 - dy1 * dx2
                area = 0.5 * abs(cross)
                areas[idx] = area
                tri_indices.append((i, j, k, cross))
                idx += 1
    
    # Soft-min weights
    neg_beta_areas = -beta * areas
    max_neg = np.max(neg_beta_areas)
    exp_vals = np.exp(neg_beta_areas - max_neg)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # weights for soft-min
    
    # Gradient of soft_min w.r.t. each point
    # soft_min = -(1/beta) * log(sum(exp(-beta * area_m)))
    # d(soft_min)/d(area_m) = weights[m]
    # d(area_m)/d(p) for each triangle
    
    grad_pts = np.zeros_like(pts)
    for m, (i, j, k, cross) in enumerate(tri_indices):
        if weights[m] == 0:
            continue
        # area = 0.5 * |cross| where cross = (p_j - p_i) x (p_k - p_i)
        # d(area)/d(p_i), d(area)/d(p_j), d(area)/d(p_k)
        sign = np.sign(cross) if cross != 0 else 1.0
        # cross = (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)
        # d(cross)/d(p_i) = (-(y_k-y_i), (x_k-x_i))
        # d(cross)/d(p_j) = ((y_k-y_i), -(x_k-x_i))
        # d(cross)/d(p_k) = (-(y_j-y_i), (x_j-x_i))
        dy_k_i = pts[k, 1] - pts[i, 1]
        dx_k_i = pts[k, 0] - pts[i, 0]
        dy_j_i = pts[j, 1] - pts[i, 1]
        dx_j_i = pts[j, 0] - pts[i, 0]
        
        dcross_di = np.array([-dy_k_i, dx_k_i])
        dcross_dj = np.array([dy_k_i, -dx_k_i])
        dcross_dk = np.array([-dy_j_i, dx_j_i])
        
        darea_di = 0.5 * sign * dcross_di
        darea_dj = 0.5 * sign * dcross_dj
        darea_dk = 0.5 * sign * dcross_dk
        
        grad_pts[i] += weights[m] * darea_di
        grad_pts[j] += weights[m] * darea_dj
        grad_pts[k] += weights[m] * darea_dk
    
    # Map Cartesian gradient to barycentric gradient
    # x = a*1.0 + b*0.5, y = a*0 + b*(s3/2)
    # dx/da = 1, dx/db = 0.5, dy/da = 0, dy/db = s3/2
    # grad_a = grad_x * 1 + grad_y * 0 = grad_x
    # grad_b = grad_x * 0.5 + grad_y * (s3/2)
    grad_b = np.zeros_like(b)
    grad_b[:, 0] = grad_pts[:, 0]
    grad_b[:, 1] = grad_pts[:, 0] * 0.5 + grad_pts[:, 1] * (SQRT3 / 2.0)
    
    return -grad_b  # negative because we're minimizing -soft_min


def hexagonal_init(n_points):
    """Generate structured hexagonal lattice points in barycentric coordinates."""
    # Start with a grid of points in barycentric space
    # We want roughly sqrt(n) x sqrt(n) grid
    # For 11 points, use a structured pattern
    points = []
    
    # Try a few structured patterns and pick the best
    candidates = []
    
    # Pattern 1: Roughly uniform grid
    # 4x3 grid = 12 points, take 11
    rows = 4
    cols = 3
    for r in range(rows):
        for c in range(cols):
            a = (c + 0.5) / cols
            b = (r + 0.5) / rows
            if a + b < 0.95:  # stay inside
                candidates.append([a, b])
    
    # Pattern 2: Include some edge/midpoints
    candidates2 = [
        [0.5, 0.5],      # center
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.5, 0.05],
        [0.05, 0.5],
        [0.5, 0.9],
        [0.9, 0.05],
        [0.05, 0.9],
        [0.3, 0.1],
        [0.7, 0.1],
        [0.1, 0.3],
        [0.9, 0.3],
        [0.3, 0.5],
        [0.7, 0.5],
        [0.5, 0.3],
    ]
    # Filter to valid barycentric (a>=0, b>=0, a+b<=1)
    valid = []
    for p in candidates2:
        if p[0] >= 0 and p[1] >= 0 and p[0] + p[1] <= 1.0:
            valid.append(p)
    
    # Combine and take first 11 valid
    all_cands = candidates + valid
    seen = set()
    unique = []
    for p in all_cands:
        key = (round(p[0], 3), round(p[1], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    # Take 11
    init = np.array(unique[:n_points])
    if len(init) < n_points:
        # Fill with random valid points
        rng = np.random.RandomState(42)
        while len(init) < n_points:
            a = rng.random() * 0.9
            b = rng.random() * (0.9 - a) if a < 0.9 else 0.05
            if a >= 0 and b >= 0 and a + b <= 1.0:
                init = np.vstack([init, [[a, b]]])
    
    return init[:n_points]


def construct():
    global T_START
    T_START = time.time()
    
    # Initial structured points
    b0 = hexagonal_init(N)
    
    # Ensure valid
    b0 = np.clip(b0, 0.01, 0.99)
    # Project to valid barycentric
    for i in range(N):
        if b0[i, 0] + b0[i, 1] > 0.98:
            scale = 0.98 / (b0[i, 0] + b0[i, 1])
            b0[i] *= scale
    
    # Evaluate initial
    pts0 = bary_to_cart(b0)
    best_area = min_triangle_area_cart(pts0)
    best_b = b0.copy()
    
    # Multi-stage optimization with increasing beta
    betas = [50, 100, 200, 500, 1000, 2000, 5000]
    
    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    # But we need a+b <= 1, which is a constraint. L-BFGS-B doesn't handle that directly.
    # We'll use a penalty or just rely on the bounds and check validity after.
    # Actually, let's use a change of variables: let a = u, b = v*(1-u) where u,v in [0,1]
    # This guarantees a+b <= 1 automatically.
    
    def uvar_to_bary(uv):
        """uv: (N,2) with u,v in [0,1] -> barycentric (a,b)"""
        a = uv[:, 0]
        b = uv[:, 1] * (1.0 - uv[:, 0])
        return np.column_stack([a, b])
    
    def bary_to_uvar(b):
        a = b[:, 0]
        b_ = b[:, 1]
        u = a
        # Use np.where to avoid division by zero or near-zero issues and array boolean errors
        denom = 1.0 - a
        safe_denom = np.where(np.abs(denom) > 1e-10, denom, 1.0)
        v = b_ / safe_denom
        # If denom was effectively zero, v should be 0 (or handled by bounds later)
        v = np.where(np.abs(denom) > 1e-10, v, 0.0)
        return np.column_stack([u, v])
    
    # Convert initial to uvar
    uv0 = bary_to_uvar(b0)
    uv0 = np.clip(uv0, 0.001, 0.999)
    
    bounds_uv = [(0.001, 0.999), (0.001, 0.999)] * N
    
    current_uv = uv0.copy()
    
    for beta in betas:
        if time.time() - T_START > TIME_LIMIT - 5:
            break
        
        def obj(uv):
            b = uvar_to_bary(uv)
            return soft_min_objective(b, beta)
        
        def grad(uv):
            b = uvar_to_bary(uv)
            gb = soft_min_grad(b, beta)
            # Chain rule: gb = [da/dx, db/dx] -> guv
            # a = u, b = v*(1-u)
            # da/du = 1, da/dv = 0
            # db/du = -v, db/dv = (1-u)
            gu = gb[:, 0] * 1.0 + gb[:, 1] * (-uv[:, 1])
            gv = gb[:, 0] * 0.0 + gb[:, 1] * (1.0 - uv[:, 0])
            return np.column_stack([gu, gv]).flatten()
        
        # Flatten for scipy
        def obj_flat(x):
            uv = x.reshape(N, 2)
            return obj(uv)
        
        def grad_flat(x):
            uv = x.reshape(N, 2)
            return grad(uv)
        
        result = minimize(
            obj_flat,
            current_uv.flatten(),
            jac=grad_flat,
            method='L-BFGS-B',
            bounds=bounds_uv,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.success or result.fun < obj_flat(current_uv.flatten()):
            new_uv = result.x.reshape(N, 2)
            # Check validity
            new_b = uvar_to_bary(new_uv)
            new_pts = bary_to_cart(new_b)
            # Verify all points are in triangle
            valid = True
            for i in range(N):
                if new_pts[i, 0] < -1e-9 or new_pts[i, 0] > 1.0 + 1e-9:
                    valid = False
                    break
                if new_pts[i, 1] < -1e-9 or new_pts[i, 1] > SQRT3/2 + 1e-9:
                    valid = False
                    break
                # Check above bottom edge and below the other two edges
                # For equilateral triangle with vertices (0,0), (1,0), (0.5, s3/2)
                # y >= 0, y <= s3/2 * x for x in [0, 0.5], y <= s3/2 * (1-x) for x in [0.5, 1]
                x, y = new_pts[i]
                if y < -1e-9:
                    valid = False
                    break
                if x <= 0.5:
                    if y > SQRT3/2 * x + 1e-9:
                        valid = False
                        break
                else:
                    if y > SQRT3/2 * (1 - x) + 1e-9:
                        valid = False
                        break
            
            if valid:
                area = min_triangle_area_cart(new_pts)
                if area > best_area:
                    best_area = area
                    best_b = new_b
                    current_uv = new_uv
        
        if time.time() - T_START > TIME_LIMIT - 10:
            break
    
    # Final polish: try a few more iterations with very high beta
    if time.time() - T_START < TIME_LIMIT - 5:
        for beta in [10000, 20000]:
            if time.time() - T_START > TIME_LIMIT - 3:
                break
            
            def obj_flat(x):
                uv = x.reshape(N, 2)
                b = uvar_to_bary(uv)
                return soft_min_objective(b, beta)
            
            def grad_flat(x):
                uv = x.reshape(N, 2)
                b = uvar_to_bary(uv)
                gb = soft_min_grad(b, beta)
                gu = gb[:, 0] + gb[:, 1] * (-uv[:, 1])
                gv = gb[:, 1] * (1.0 - uv[:, 0])
                return np.column_stack([gu, gv]).flatten()
            
            result = minimize(
                obj_flat,
                current_uv.flatten(),
                jac=grad_flat,
                method='L-BFGS-B',
                bounds=bounds_uv,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.success or result.fun < obj_flat(current_uv.flatten()):
                new_uv = result.x.reshape(N, 2)
                new_b = uvar_to_bary(new_uv)
                new_pts = bary_to_cart(new_b)
                valid = True
                for i in range(N):
                    x, y = new_pts[i]
                    if x < -1e-9 or x > 1.0 + 1e-9 or y < -1e-9 or y > SQRT3/2 + 1e-9:
                        valid = False
                        break
                    if x <= 0.5:
                        if y > SQRT3/2 * x + 1e-9:
                            valid = False
                            break
                    else:
                        if y > SQRT3/2 * (1 - x) + 1e-9:
                            valid = False
                            break
                
                if valid:
                    area = min_triangle_area_cart(new_pts)
                    if area > best_area:
                        best_area = area
                        best_b = new_b
                        current_uv = new_uv
    
    # Return the best points found
    best_pts = bary_to_cart(best_b)
    
    # Final validation and clamping
    for i in range(N):
        x, y = best_pts[i]
        if x < 0: x = 0.001
        if x > 1: x = 0.999
        if y < 0: y = 0.001
        if y > SQRT3/2: y = SQRT3/2 - 0.001
        # Ensure inside triangle
        # y <= sqrt(3)/2 * x (for x < 0.5) and y <= sqrt(3)/2 * (1-x) (for x >= 0.5)
        max_y = SQRT3/2 * min(x, 1.0-x)
        if y > max_y:
            y = max_y * 0.99
        best_pts[i] = [x, y]
        
    # Nelder-Mead polish step (structural addition for robustness)
    if time.time() - T_START < TIME_LIMIT - 10:
        try:
            # Define objective for Nelder-Mead: negative min area
            def nm_objective(pts_flat):
                pts = pts_flat.reshape(N, 2)
                # Check validity
                for i in range(N):
                    x, y = pts[i]
                    if x < 0 or x > 1 or y < 0 or y > SQRT3/2:
                        return 1.0 # Large penalty
                    max_y = SQRT3/2 * min(x, 1.0-x)
                    if y > max_y:
                        return 1.0
                return -min_triangle_area_cart(pts)

            current_best_val = nm_objective(best_pts.flatten())
            
            # Run Nelder-Mead with limited iterations
            result_nm = minimize(
                nm_objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if result_nm.fun < current_best_val:
                new_pts = result_nm.x.reshape(N, 2)
                # Final validity check
                valid = True
                for i in range(N):
                    x, y = new_pts[i]
                    if x < -1e-9 or x > 1.0 + 1e-9 or y < -1e-9 or y > SQRT3/2 + 1e-9:
                        valid = False
                        break
                    max_y = SQRT3/2 * min(max(x,0), 1.0-max(x,0))
                    if y > max_y + 1e-9:
                        valid = False
                        break
                if valid:
                    best_pts = new_pts
        except Exception:
            pass # If NM fails, keep previous best

    return best_pts


# Test
if __name__ == '__main__':
    pts = construct()
    print("Points:")
    for p in pts:
        print(f"  ({p[0]:.6f}, {p[1]:.6f})")
    print(f"Min triangle area: {min_triangle_area_cart(pts):.6f}")
    print(f"Normalized (div by tri area): {min_triangle_area_cart(pts)/TRI_AREA:.6f}")
    print(f"Time: {time.time()-T_START:.2f}s")
