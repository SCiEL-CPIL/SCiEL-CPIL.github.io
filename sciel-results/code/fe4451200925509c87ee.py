"""Optimize 12 unit hexagons inside a regular hexagon using SDF-based smooth penalties."""
import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
TIME_LIMIT = 145.0

def _hex_support(cx, cy, angle_deg, nx, ny, side=1.0):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    max_dot = 0.0
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        rx = ca * x - sa * y
        ry = sa * x + ca * y
        d = nx * rx + ny * ry
        if d > max_dot:
            max_dot = d
    return max_dot + nx * cx + ny * cy

def _hex_min_support(cx, cy, angle_deg, nx, ny, side=1.0):
    return -_hex_support(cx, cy, angle_deg, -nx, -ny, side)

def _outer_hex_side(cx_arr, cy_arr, angle_arr, ocx, ocy, oangle_deg):
    apothem_unit = math.sqrt(3) / 2.0
    oa = math.radians(oangle_deg)
    max_need = 0.0
    for k in range(6):
        na = oa + math.radians(90 + 60 * k)
        nx, ny = math.cos(na), math.sin(na)
        for i in range(N):
            s = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], nx, ny)
            val = (s - nx * ocx - ny * ocy) / apothem_unit
            if val > max_need:
                max_need = val
    return max_need

def _is_valid(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq > (2 * math.sqrt(3)) ** 2:
                continue
            if dist_sq < 4.0 - 1e-9:
                return False
            for ai in range(6):
                aia = math.radians(angle_arr[i] + 90 + 60 * ai)
                anx, any_ = math.cos(aia), math.sin(aia)
                minA = _hex_min_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                maxA = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                minB = _hex_min_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                maxB = _hex_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    break
            else:
                for aj in range(6):
                    aja = math.radians(angle_arr[j] + 90 + 60 * aj)
                    anx, any_ = math.cos(aja), math.sin(aja)
                    minA = _hex_min_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                    maxA = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                    minB = _hex_min_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                    maxB = _hex_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                    if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                        break
                else:
                    return False
    return True

def _sdf_collision_penalty(cx_arr, cy_arr, angle_arr, eps=0.1):
    """Smooth penalty based on approximate gap between hexagons."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx_arr[j] - cx_arr[i]
            dy = cy_arr[j] - cy_arr[i]
            dist = math.hypot(dx, dy)
            if dist < 1e-12:
                penalty += 100.0
                continue
            nx, ny = dx / dist, dy / dist
            sup_i = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], nx, ny)
            sup_j = _hex_support(cx_arr[j], cy_arr[j], angle_arr[j], -nx, -ny)
            gap_i = sup_i - (nx * cx_arr[i] + ny * cy_arr[i])
            gap_j = sup_j - (-nx * cx_arr[j] - ny * cy_arr[j])
            sep = dist - gap_i - gap_j
            if sep < eps:
                penalty += (eps - sep) ** 2
    return penalty

def _objective(params, ocx, ocy, oangle_deg, beta=100.0):
    cx = params[0::3]
    cy = params[1::3]
    ang = params[2::3]
    side = _outer_hex_side(cx, cy, ang, ocx, ocy, oangle_deg)
    penalty = _sdf_collision_penalty(cx, cy, ang, eps=0.1)
    return side + beta * penalty

def _make_initial_configs():
    configs = []
    # Flower: center + 6 around + 5 more
    pts = [[0.0, 0.0, 0.0]]
    r1 = 2 * math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        pts.append([r1 * math.cos(angle), r1 * math.sin(angle), 0.0])
    r2 = r1 * 1.732
    for deg in [30, 90, 150, 210, 270]:
        rad = math.radians(deg)
        pts.append([r2 * math.cos(rad), r2 * math.sin(rad), 0.0])
    configs.append(pts[:12])
    
    # Compact 3x4 grid
    pts = []
    rh = math.sqrt(3)
    cw = 2.0
    for row in range(3):
        for col in range(4):
            x = col * cw - 1.5 * cw
            y = row * 2 * rh - rh
            if row % 2 == 1:
                x += cw / 2
            pts.append([x, y, 0.0])
    configs.append(pts[:12])
    
    # Ring of 12
    pts = []
    r = 2.5 * math.sqrt(3)
    for i in range(12):
        angle = math.radians(30 * i)
        pts.append([r * math.cos(angle), r * math.sin(angle), 0.0])
    configs.append(pts)
    
    # Two rings
    pts = []
    r1 = 2 * math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        pts.append([r1 * math.cos(angle), r1 * math.sin(angle), 0.0])
    r2 = 3.5 * math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i + 30)
        pts.append([r2 * math.cos(angle), r2 * math.sin(angle), 0.0])
    configs.append(pts)
    
    # Random
    for _ in range(4):
        pts = []
        for _ in range(N):
            pts.append([random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0, 60)])
        configs.append(pts)
    
    return configs

def _optimize_outer(cx_arr, cy_arr, angle_arr):
    """Find best outer hexagon parameters for given inner config."""
    best_side = float('inf')
    best_ocx, best_ocy, best_oa = 0.0, 0.0, 0.0
    for oa in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]:
        for ocx in np.linspace(-1, 1, 5):
            for ocy in np.linspace(-1, 1, 5):
                s = _outer_hex_side(cx_arr, cy_arr, angle_arr, float(ocx), float(ocy), float(oa))
                if s < best_side:
                    best_side = s
                    best_ocx, best_ocy, best_oa = float(ocx), float(ocy), float(oa)
    return best_side, best_ocx, best_ocy, best_oa

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    configs = _make_initial_configs()
    
    # Phase 1: Evaluate initial configs
    for cfg in configs:
        if time.time() - t0 > TIME_LIMIT - 80:
            break
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid(cx_arr, cy_arr, angle_arr):
            side, ocx, ocy, oa = _optimize_outer(cx_arr, cy_arr, angle_arr)
            if side < best_side:
                best_side = side
                best_cfg = [row[:] for row in cfg]
                best_center = (ocx, ocy)
                best_angle = oa
    
    if best_cfg is None:
        best_cfg = configs[0]
        cx_arr = np.array([p[0] for p in best_cfg])
        cy_arr = np.array([p[1] for p in best_cfg])
        angle_arr = np.array([p[2] for p in best_cfg])
        best_side, best_center[0], best_center[1], best_angle = _optimize_outer(cx_arr, cy_arr, angle_arr)
    
    # Phase 2: Differential Evolution with smooth penalty
    bounds = []
    for i in range(N):
        bounds.append((-4.0, 4.0))
        bounds.append((-4.0, 4.0))
        bounds.append((0.0, 60.0))
    
    def de_obj(params):
        if time.time() - t0 > TIME_LIMIT - 20:
            return 1e6
        return _objective(params, best_center[0], best_center[1], best_angle, beta=50.0)
    
    for start_idx in range(min(4, len(configs))):
        if time.time() - t0 > TIME_LIMIT - 30:
            break
        cfg = configs[start_idx]
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = cfg[i][0]
            x0[3*i+1] = cfg[i][1]
            x0[3*i+2] = cfg[i][2]
        
        try:
            result = differential_evolution(
                de_obj, bounds, x0=x0, maxiter=25, popsize=8,
                tol=1e-4, seed=100 + start_idx, polish=False
            )
            x_opt = result.x
            cx_opt = x_opt[0::3]
            cy_opt = x_opt[1::3]
            ang_opt = x_opt[2::3]
            
            if _is_valid(cx_opt, cy_opt, ang_opt):
                side, ocx, ocy, oa = _optimize_outer(cx_opt, cy_opt, ang_opt)
                if side < best_side:
                    best_side = side
                    best_cfg = [[float(cx_opt[i]), float(cy_opt[i]), float(ang_opt[i])] for i in range(N)]
                    best_center = (ocx, ocy)
                    best_angle = oa
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B with smooth SDF penalty (key improvement)
    if best_cfg is not None and time.time() - t0 < TIME_LIMIT - 15:
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = best_cfg[i][0]
            x0[3*i+1] = best_cfg[i][1]
            x0[3*i+2] = best_cfg[i][2]
        
        try:
            result = minimize(
                lambda p: _objective(p, best_center[0], best_center[1], best_angle, beta=100.0),
                x0, method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            x_opt = result.x
            cx_opt = x_opt[0::3]
            cy_opt = x_opt[1::3]
            ang_opt = x_opt[2::3]
            
            if _is_valid(cx_opt, cy_opt, ang_opt):
                side, ocx, ocy, oa = _optimize_outer(cx_opt, cy_opt, ang_opt)
                if side < best_side:
                    best_side = side
                    best_cfg = [[float(cx_opt[i]), float(cy_opt[i]), float(ang_opt[i])] for i in range(N)]
                    best_center = (ocx, ocy)
                    best_angle = oa
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead polish
    if best_cfg is not None and time.time() - t0 < TIME_LIMIT - 10:
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = best_cfg[i][0]
            x0[3*i+1] = best_cfg[i][1]
            x0[3*i+2] = best_cfg[i][2]
        
        try:
            result = minimize(
                lambda p: _objective(p, best_center[0], best_center[1], best_angle, beta=200.0),
                x0, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-8}
            )
            x_opt = result.x
            cx_opt = x_opt[0::3]
            cy_opt = x_opt[1::3]
            ang_opt = x_opt[2::3]
            
            if _is_valid(cx_opt, cy_opt, ang_opt):
                side, ocx, ocy, oa = _optimize_outer(cx_opt, cy_opt, ang_opt)
                if side < best_side:
                    best_side = side
                    best_cfg = [[float(cx_opt[i]), float(cy_opt[i]), float(ang_opt[i])] for i in range(N)]
                    best_center = (ocx, ocy)
                    best_angle = oa
        except Exception:
            pass
    
    # Final fallback
    if best_cfg is None:
        best_cfg = configs[0]
        cx_arr = np.array([p[0] for p in best_cfg])
        cy_arr = np.array([p[1] for p in best_cfg])
        angle_arr = np.array([p[2] for p in best_cfg])
        best_side, best_center[0], best_center[1], best_angle = _optimize_outer(cx_arr, cy_arr, angle_arr)
    
    return best_cfg, best_center, best_side, best_angle
