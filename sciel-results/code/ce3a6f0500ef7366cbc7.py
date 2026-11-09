"""Strategy: L-BFGS-B on smooth objective with symmetry reduction."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    SQRT3 = math.sqrt(3)
    R = 1.0  # Circumradius of unit hexagon
    INR = SQRT3 / 2.0  # Inradius

    def hex_verts(cx, cy, ang_deg):
        a = math.radians(ang_deg)
        return [(cx + R * math.cos(a + i * math.pi / 3),
                 cy + R * math.sin(a + i * math.pi / 3)) for i in range(6)]

    def hex_normals(ang_deg):
        a = math.radians(ang_deg)
        return [(math.cos(a + i * math.pi / 3 + math.pi / 6),
                 math.sin(a + i * math.pi / 3 + math.pi / 6)) for i in range(6)]

    def is_disjoint(v1, v2, n1, n2):
        # Separating Axis Theorem
        for nx, ny in n1 + n2:
            p1 = [vx * nx + vy * ny for vx, vy in v1]
            p2 = [vx * nx + vy * ny for vx, vy in v2]
            if max(p1) < min(p2) or max(p2) < min(p1):
                return True
        return False

    def outer_side(hex_list, oa_deg):
        oa = math.radians(oa_deg)
        max_p = 0.0
        for verts in hex_list:
            for vx, vy in verts:
                for i in range(6):
                    nx = math.cos(oa + i * math.pi / 3 + math.pi / 6)
                    ny = math.sin(oa + i * math.pi / 3 + math.pi / 6)
                    p = abs(vx * nx + vy * ny)
                    if p > max_p:
                        max_p = p
        return 2.0 * max_p / SQRT3

    def axial_to_cart(q, r, s=1.0):
        return (SQRT3 * s * (q + r / 2.0), 1.5 * s * r)

    def objective(params):
        """
        Smooth objective function for L-BFGS-B.
        Returns (value, gradient)
        """
        p = params
        n = len(p) // 3
        grad = np.zeros_like(p)
        
        # 1. Compute outer side length (approximate with soft max for smoothness)
        # We optimize for the side length of a hexagon centered at origin with rotation 0
        # side = 2 * max_proj / sqrt(3)
        # To make it differentiable, we use a soft-max approximation or just compute max and use subgradient.
        # For L-BFGS-B, we can compute the max projection and its gradient.
        
        max_proj = 0.0
        max_proj_idx = None # (hex_idx, vert_idx, wall_idx)
        
        verts_all = []
        for i in range(n):
            cx = p[i*3]
            cy = p[i*3+1]
            ang = p[i*3+2]
            v = hex_verts(cx, cy, ang)
            verts_all.append(v)
            
            for vx, vy in v:
                for k in range(6):
                    nx = math.cos(k * math.pi / 3 + math.pi / 6)
                    ny = math.sin(k * math.pi / 3 + math.pi / 6)
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
                        max_proj_idx = (i, vx, vy, nx, ny)
                        
        side_len = 2.0 * max_proj / SQRT3
        
        # Gradient of side_len w.r.t. params
        if max_proj_idx:
            i, vx, vy, nx, ny = max_proj_idx
            cx = p[i*3]
            cy = p[i*3+1]
            ang = p[i*3+2]
            # d(proj)/d(cx) = nx * sign(vx*nx+vy*ny)
            # d(proj)/d(cy) = ny * sign(vx*nx+vy*ny)
            # d(proj)/d(ang) = ... (complex, but we can use finite difference or analytic)
            
            # Let's use a simpler approach: 
            # The objective is dominated by the max projection.
            # We can just push the hexagon that is furthest out inwards.
            
            sign = 1.0 if (vx * nx + vy * ny) > 0 else -1.0
            coeff = 2.0 / SQRT3
            
            grad[i*3] += coeff * sign * nx
            grad[i*3+1] += coeff * sign * ny
            # Angle gradient is tricky. We can approximate or ignore for now if position dominates.
            # Actually, rotating a hexagon can reduce its max projection.
            # Let's compute d(proj)/d(ang) analytically.
            # vx = cx + cos(ang + i*pi/3)
            # d(vx)/d(ang) = -sin(ang + i*pi/3)
            # We need to know which vertex i was used.
            # This is getting complicated. Let's use a penalty-based approach instead which is smoother.

        # 2. Overlap Penalty
        # For each pair, if they overlap, add penalty.
        # Smooth penalty: max(0, overlap_depth)^2
        # We need a smooth measure of overlap depth.
        # Approximation: center distance - 2*R. If < 0, they overlap.
        # This is a lower bound on actual overlap depth, but good enough for driving separation.
        
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = p[i*3] - p[j*3]
                dy = p[i*3+1] - p[j*3+1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = 2.0 * R # Approximate minimum distance for non-overlap (actually depends on rotation)
                # A better min_dist for hexagons:
                # If aligned, min dist is 2*INR*sqrt(3) = 3.0? No.
                # Closest distance between two unit hexagons can be 0 if interlocked? No, they are disjoint.
                # The minimum distance between centers of two disjoint unit hexagons is at least 2*INR = sqrt(3) ~ 1.732 if flat-to-flat.
                # But they can rotate.
                # Let's use a conservative min_dist = 1.8
                min_dist = 1.8
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    penalty += overlap * overlap
                    
                    # Gradient of penalty w.r.t. positions
                    # d(overlap^2)/d(dx) = 2*overlap * d(overlap)/d(dx)
                    # d(overlap)/d(dx) = -dx/dist
                    if dist > 1e-6:
                        grad[i*3] += 2.0 * overlap * (-dx / dist)
                        grad[i*3+1] += 2.0 * overlap * (-dy / dist)
                        grad[j*3] += 2.0 * overlap * (dx / dist)
                        grad[j*3+1] += 2.0 * overlap * (dy / dist)
                    else:
                        # Random push
                        rx = random.uniform(-1, 1)
                        ry = random.uniform(-1, 1)
                        grad[i*3] += 2.0 * overlap * rx
                        grad[i*3+1] += 2.0 * overlap * ry
                        grad[j*3] -= 2.0 * overlap * rx
                        grad[j*3+1] -= 2.0 * overlap * ry

        # Weighting: We want to minimize side_len, but keep penalty low.
        # If penalty > 0, it should dominate.
        value = side_len + 10.0 * penalty
        
        # Add small regularization to angle to encourage alignment? No, let it rotate.
        
        return value, grad

    def objective_value_only(params):
        # For Nelder-Mead or if we don't need gradient
        p = params
        n = len(p) // 3
        max_proj = 0.0
        for i in range(n):
            cx = p[i*3]
            cy = p[i*3+1]
            ang = p[i*3+2]
            v = hex_verts(cx, cy, ang)
            for vx, vy in v:
                for k in range(6):
                    nx = math.cos(k * math.pi / 3 + math.pi / 6)
                    ny = math.sin(k * math.pi / 3 + math.pi / 6)
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        side_len = 2.0 * max_proj / SQRT3
        
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = p[i*3] - p[j*3]
                dy = p[i*3+1] - p[j*3+1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = 1.8
                if dist < min_dist:
                    overlap = min_dist - dist
                    penalty += overlap * overlap
        
        return side_len + 10.0 * penalty

    def refine_with_lbfgs(params):
        """Run L-BFGS-B optimization."""
        p = params.copy()
        try:
            res = minimize(objective, p, jac=True, method='L-BFGS-B', 
                           options={'maxiter': 200, 'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < objective_value_only(p):
                return res.x
        except Exception:
            pass
        return p

    def refine_with_nm(params):
        """Run Nelder-Mead as fallback or secondary refinement."""
        p = params.copy()
        try:
            res = minimize(objective_value_only, p, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success or res.fun < objective_value_only(p):
                return res.x
        except Exception:
            pass
        return p

    def validate(params):
        """Check if the configuration is valid (disjoint)."""
        n = len(params) // 3
        hex_list = []
        norms_list = []
        for i in range(n):
            cx = params[i*3]
            cy = params[i*3+1]
            ang = params[i*3+2]
            hex_list.append(hex_verts(cx, cy, ang))
            norms_list.append(hex_normals(ang))
            
        for i in range(n):
            for j in range(i+1, n):
                if not is_disjoint(hex_list[i], hex_list[j], norms_list[i], norms_list[j]):
                    return False
        return True

    def score(params):
        """Calculate the actual outer side length for a valid config."""
        n = len(params) // 3
        hex_list = []
        for i in range(n):
            cx = params[i*3]
            cy = params[i*3+1]
            ang = params[i*3+2]
            hex_list.append(hex_verts(cx, cy, ang))
            
        # Sweep outer rotation
        best_oa = 0.0
        best_s = float('inf')
        for oa in np.arange(0, 60, 1.0):
            s = outer_side(hex_list, oa)
            if s < best_s:
                best_s = s
                best_oa = oa
        # Fine sweep
        for oa in np.arange(best_oa - 1, best_oa + 1, 0.1):
            s = outer_side(hex_list, oa)
            if s < best_s:
                best_s = s
                best_oa = oa
        return best_s, best_oa

    topologies = [
        [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)],
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)],
        [(0,0),(1,0),(-1,0),(0,1),(1,1),(-1,1),(0,-1),(1,-1),(-1,-1),(2,0),(-2,0)],
    ]

    def init_params(coords):
        params = []
        for q, r in coords:
            x, y = axial_to_cart(q, r)
            params.extend([x, y, 0.0])
        return np.array(params)

    def random_init():
        params = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            params.extend([x, y, a])
        return np.array(params)

    best_params = None
    best_side = float('inf')
    best_oa = 0.0

    inits = [init_params(t) for t in topologies]
    for _ in range(8):
        inits.append(random_init())

    for init in inits:
        if time.time() > deadline - 20.0:
            break

        # Center the init
        cx_avg = np.mean(init[0::3])
        cy_avg = np.mean(init[1::3])
        for i in range(N):
            init[i*3] -= cx_avg
            init[i*3+1] -= cy_avg

        # 1. L-BFGS-B Refinement
        p1 = refine_with_lbfgs(init)
        
        # 2. Nelder-Mead Refinement (for local minima escape)
        p2 = refine_with_nm(p1)

        # Center again
        cx_avg = np.mean(p2[0::3])
        cy_avg = np.mean(p2[1::3])
        for i in range(N):
            p2[i*3] -= cx_avg
            p2[i*3+1] -= cy_avg

        # Validate
        if validate(p2):
            s, oa = score(p2)
            if s < best_side:
                best_side = s
                best_params = p2.copy()
                best_oa = oa

    if best_params is None:
        # Fallback
        p = init_params(topologies[0])
        cx_avg = np.mean(p[0::3])
        cy_avg = np.mean(p[1::3])
        for i in range(N):
            p[i*3] -= cx_avg
            p[i*3+1] -= cy_avg
        s, oa = score(p)
        best_side = s
        best_params = p
        best_oa = oa

    inner_list = []
    for i in range(N):
        inner_list.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])

    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
