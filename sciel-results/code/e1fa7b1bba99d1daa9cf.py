import math
import time
import random
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

N = 12
DEADLINE = 140.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    def hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        return [(cx + side * math.cos(angle + i * math.pi / 3),
                 cy + side * math.sin(angle + i * math.pi / 3)) for i in range(6)]
    
    def support_function(cx, cy, angle_deg, nx, ny):
        """Maximum of (v·n) over vertices of unit hexagon at (cx,cy) with angle angle_deg."""
        angle = math.radians(angle_deg)
        max_val = -float('inf')
        for i in range(6):
            theta = angle + i * math.pi / 3
            vx = cx + math.cos(theta)
            vy = cy + math.sin(theta)
            proj = vx * nx + vy * ny
            if proj > max_val:
                max_val = proj
        return max_val
    
    def compute_side(params, ocx, ocy, otheta_deg):
        """Compute outer hexagon side length given all parameters.
        params: [x1,y1,a1, x2,y2,a2, ..., x12,y2,a12] (degrees)
        """
        n = len(params) // 3
        otheta = math.radians(otheta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(n):
                cx = params[3*i]
                cy = params[3*i+1]
                ai = params[3*i+2]
                sf = support_function(cx, cy, ai, nx, ny)
                proj = sf - (ocx * nx + ocy * ny)
                if proj > max_proj:
                    max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)
    
    def min_pairwise_dist(params):
        """Minimum pairwise center distance."""
        n = len(params) // 3
        min_d = float('inf')
        for i in range(n):
            xi, yi = params[3*i], params[3*i+1]
            for j in range(i+1, n):
                xj, yj = params[3*j], params[3*j+1]
                d = math.hypot(xi - xj, yi - yj)
                if d < min_d:
                    min_d = d
        return min_d
    
    def overlap_constraints(params):
        """Return vector of constraint values. All must be >= 0.
        For each pair (i,j): dist(centers) - (sqrt(3) - eps) >= 0
        """
        n = len(params) // 3
        eps = 1e-4
        min_d = math.sqrt(3) - eps
        cons = []
        for i in range(n):
            xi, yi = params[3*i], params[3*i+1]
            for j in range(i+1, n):
                xj, yj = params[3*j], params[3*j+1]
                d = math.hypot(xi - xj, yi - yj)
                cons.append(d - min_d)
        return np.array(cons)
    
    def objective(params):
        """Objective: outer side length with outer hexagon centered at origin, angle 0."""
        return compute_side(params, 0.0, 0.0, 0.0)
    
    # Initial configuration: hexagonal lattice
    eps = 0.001
    d = math.sqrt(3) + eps
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d * math.sqrt(3) / 2.0])
    
    centers = [np.array([0.0, 0.0])]
    ring1 = [u, v, v-u, -u, -v, u-v]
    for vec in ring1:
        centers.append(vec.copy())
    
    ring2_cands = []
    basis = [u, v, v-u, -u, -v, u-v]
    for i in range(6):
        p1 = ring1[i]
        for b in basis:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.zeros(2), atol=0.01)
            is_r1 = any(np.allclose(p2, r, atol=0.01) for r in ring1)
            if not is_r0 and not is_r1:
                if not any(np.allclose(p2, pc, atol=0.01) for pc in ring2_cands):
                    ring2_cands.append(p2.copy())
    
    ring2_cands.sort(key=lambda p: np.linalg.norm(p))
    for i in range(5):
        centers.append(ring2_cands[i].copy())
    
    # Build initial params
    init_params = np.zeros(36)
    for i in range(12):
        init_params[3*i] = centers[i][0]
        init_params[3*i+1] = centers[i][1]
        init_params[3*i+2] = 30.0  # angle in degrees
    
    best_params = init_params.copy()
    best_side = compute_side(init_params, 0.0, 0.0, 0.0)
    
    # Phase 1: Optimize inner positions with SLSQP, fixing outer at origin/0deg
    if time.time() < deadline - 60:
        constraints = [
            {'type': 'ineq', 'fun': lambda p: overlap_constraints(p)}
        ]
        bounds = [(-5, 5)] * 12 + [(-5, 5)] * 12 + [(0, 60)] * 12
        
        result = minimize(
            objective, init_params, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        if result.success or result.fun < best_side:
            # Check constraints
            cons = overlap_constraints(result.x)
            if np.all(cons >= -1e-3):
                best_params = result.x
                best_side = result.fun
    
    # Phase 2: Joint optimization including outer hexagon parameters
    # Extend params: [inner_params (36), ocx, ocy, otheta_deg]
    if time.time() < deadline - 30:
        def extended_objective(p):
            inner_p = p[:36]
            ocx, ocy, oth = p[36], p[37], p[38]
            return compute_side(inner_p, ocx, ocy, oth)
        
        def extended_constraints(p):
            return overlap_constraints(p[:36])
        
        ext_init = np.concatenate([best_params, [0.0, 0.0, 0.0]])
        ext_bounds = [(-5, 5)] * 12 + [(-5, 5)] * 12 + [(0, 60)] * 12 + [(-3, 3), (-3, 3), (0, 60)]
        
        result2 = minimize(
            extended_objective, ext_init, method='SLSQP',
            bounds=ext_bounds,
            constraints=[{'type': 'ineq', 'fun': extended_constraints}],
            options={'maxiter': 300, 'ftol': 1e-9}
        )
        if result2.success or result2.fun < best_side:
            cons = overlap_constraints(result2.x[:36])
            if np.all(cons >= -1e-3):
                best_params = result2.x[:36]
                best_side = result2.fun
    
    # Phase 3: Fine-tune outer hexagon
    best_ocx, best_ocy, best_otheta = 0.0, 0.0, 0.0
    if time.time() < deadline - 10:
        for oth_deg in np.linspace(0, 60, 25):
            for ocx in np.linspace(-1.0, 1.0, 7):
                for ocy in np.linspace(-1.0, 1.0, 7):
                    s = compute_side(best_params, float(ocx), float(ocy), float(oth_deg))
                    if s < best_side:
                        best_side = s
                        best_ocx, best_ocy, best_otheta = float(ocx), float(ocy), float(oth_deg)
    
    # Final polish: local Nelder-Mead on all parameters
    if time.time() < deadline - 5:
        def full_obj(p):
            inner_p = p[:36]
            ocx, ocy, oth = p[36], p[37], p[38]
            side = compute_side(inner_p, ocx, ocy, oth)
            # Penalty for overlaps
            cons = overlap_constraints(inner_p)
            if np.any(cons < 0):
                side += 10.0 * np.sum(cons[cons < 0]**2)
            return side
        
        final_init = np.concatenate([best_params, [best_ocx, best_ocy, best_otheta]])
        result3 = minimize(
            full_obj, final_init, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if result3.fun < best_side:
            cons = overlap_constraints(result3.x[:36])
            if np.all(cons >= -1e-3):
                best_params = result3.x[:36]
                best_ocx, best_ocy, best_otheta = result3.x[36], result3.x[37], result3.x[38]
                best_side = compute_side(best_params, best_ocx, best_ocy, best_otheta)
    
    # Build output
    inner = []
    for i in range(12):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    outer_center = (float(best_ocx), float(best_ocy))
    outer_side = float(best_side)
    outer_angle = float(best_otheta)
    
    return inner, outer_center, outer_side, outer_angle
