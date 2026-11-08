"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses differential evolution with a vectorized constraint penalty.
"""
N = 12

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy)."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _support_unit_hex(angle_deg, dx, dy):
    """Support function of a unit regular hexagon in direction (dx, dy)."""
    a = np.radians(angle_deg)
    angles_v = a + np.radians(60 * np.arange(6))
    dots = np.cos(angles_v - np.arctan2(dy, dx))
    return np.max(dots)

def _required_side(centers, angles, outer_angle_deg):
    """Compute minimum outer side length to contain all unit hexagons."""
    outer_a = np.radians(outer_angle_deg)
    n_out = np.array([
        [np.cos(outer_a + np.radians(60 * i)),
         np.sin(outer_a + np.radians(60 * i))]
        for i in range(6)
    ])
    
    max_ext = 0.0
    for k in range(N):
        cx, cy = centers[k]
        ang = angles[k]
        for ni in range(6):
            dx, dy = n_out[ni]
            h = _support_unit_hex(ang, dx, dy)
            needed = cx * dx + cy * dy + h
            if needed > max_ext:
                max_ext = needed
    
    return 2.0 * max_ext / np.sqrt(3.0)

def _pairwise_overlap_penalty(centers):
    """Penalty for overlapping hexagons based on center distance."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
            if dist < 2.0:
                penalty += (2.0 - dist) ** 2
    return penalty

def _objective(params, outer_angle_deg):
    """Objective: required outer side + overlap penalty."""
    centers = params[:2*N].reshape(N, 2)
    angles = params[2*N:]
    S = _required_side(centers, angles, outer_angle_deg)
    overlap_pen = _pairwise_overlap_penalty(centers)
    return S + 100.0 * overlap_pen

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    best_S = float('inf')
    best_params = None
    best_outer_angle = 0.0
    
    for outer_angle in [0.0, 30.0]:
        if time.time() > deadline - 40.0:
            break
        
        # Tighter bounds based on expected packing size
        # 12 hexagons in a hexagonal arrangement: roughly 4x3 grid
        # Max extent ~ 4*2 + 2 = 10, so [-5, 5] is safe
        bounds = []
        for i in range(N):
            bounds.append((-5.0, 5.0))  # x
            bounds.append((-5.0, 5.0))  # y
        for i in range(N):
            bounds.append((0.0, 60.0))  # angle
        
        # Initial population: hex lattice
        hex_positions = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                hex_positions.append((x, y, 0.0))
        
        cx_mean = np.mean([p[0] for p in hex_positions])
        cy_mean = np.mean([p[1] for p in hex_positions])
        
        init_params = []
        for (x, y, a) in hex_positions:
            init_params.extend([x - cx_mean, y - cy_mean, a])
        init_params = np.array(init_params)
        
        try:
            result = differential_evolution(
                _objective,
                bounds,
                args=(outer_angle,),
                seed=42,
                maxiter=100,
                tol=1e-10,
                popsize=20,
                polish=True,
                init='latinhypercube',
                mutation=(0.3, 0.9),
                recombination=0.8
            )
            opt_params = result.x
            S_val = _objective(opt_params, outer_angle)
            centers_check = opt_params[:2*N].reshape(N, 2)
            overlap_pen = _pairwise_overlap_penalty(centers_check)
            
            if overlap_pen < 1e-6 and S_val < best_S:
                best_S = S_val
                best_params = opt_params.copy()
                best_outer_angle = outer_angle
        except Exception:
            continue
        
        # Polish with L-BFGS-B
        if time.time() < deadline - 15.0 and best_params is not None:
            try:
                result2 = minimize(
                    _objective,
                    best_params,
                    args=(best_outer_angle,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                opt2 = result2.x
                S2 = _objective(opt2, best_outer_angle)
                centers2 = opt2[:2*N].reshape(N, 2)
                op2 = _pairwise_overlap_penalty(centers2)
                if op2 < 1e-6 and S2 < best_S:
                    best_S = S2
                    best_params = opt2.copy()
            except Exception:
                pass
    
    # Fallback
    if best_params is None:
        inner = []
        for r in range(3):
            for c in range(4):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 6.0, 0.0
    
    # Extract inner hexagons
    inner = []
    for i in range(N):
        x = float(best_params[2*i])
        y = float(best_params[2*i+1])
        ang = float(best_params[2*N+i])
        inner.append([x, y, ang])
    
    # Final validation
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(0.0, 0.0, best_S, best_outer_angle)
    
    def point_in_hex(pt, outer_verts):
        for i in range(6):
            v1 = outer_verts[i]
            v2 = outer_verts[(i+1) % 6]
            edge = v2 - v1
            rel = pt - v1
            cross = edge[0]*rel[1] - edge[1]*rel[0]
            if cross < -1e-10:
                return False
        return True
    
    valid = True
    for iv in inner_verts_list:
        for pt in iv:
            if not point_in_hex(pt, outer_verts):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        best_S *= 1.05
    
    return inner, (0.0, 0.0), float(best_S), float(best_outer_angle)
