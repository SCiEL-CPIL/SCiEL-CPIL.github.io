"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
TIME_LIMIT = 140.0  # safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _overlap_depth(vertsA, vertsB):
    """Compute the minimum overlap depth between two polygons using SAT."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    min_overlap = float('inf')
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        overlap = min(maxA, maxB) - max(minA, minB)
        if overlap <= 0:
            return 0.0
        min_overlap = min(min_overlap, overlap)
    return min_overlap if min_overlap < float('inf') else 0.0

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j]):
                return False
    return True

def _smooth_collision_penalty(cfg):
    """Smooth collision penalty using overlap depth squared."""
    penalty = 0.0
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in cfg]
    for i in range(N):
        for j in range(i + 1, N):
            depth = _overlap_depth(polys[i], polys[j])
            if depth > 0:
                penalty += depth * depth
    return penalty

def _smooth_objective(v):
    """
    Smooth objective: outer side length + collision penalty.
    v = [x1,y1,a1, ..., x12,y12,a12, cx,cy, outer_angle]
    """
    cfg = []
    for i in range(N):
        cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
    center = (v[3*N], v[3*N+1])
    outer_angle = v[3*N+2]
    
    # Compute outer hex side
    side = _outer_hex_side(cfg, center, outer_angle)
    
    # Collision penalty
    penalty = _smooth_collision_penalty(cfg)
    
    return side + 10.0 * penalty

def _full_score(cfg, center, angle):
    """Return outer side if valid, else None."""
    if not _is_valid(cfg):
        return None
    return _outer_hex_side(cfg, center, angle)

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def make_hex_lattice():
        """3x4 hexagonal lattice with offset."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts
    
    def make_ring():
        """12 hexagons in a ring, oriented outward."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts
    
    def make_grid_initial():
        """Grid-like arrangement with some rotation variety."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                ang = random.uniform(-10, 10)
                pts.append([x - 3.0, y - row_height, ang])
        return pts
    
    def make_center_ring_outer():
        """1 center + 6 ring + 5 outer arrangement."""
        pts = []
        sqrt3 = math.sqrt(3)
        d1 = sqrt3 + 0.05
        d2 = 2 * sqrt3 + 0.05
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = math.radians(60 * k)
            pts.append([d1 * math.cos(ang), d1 * math.sin(ang), 0.0])
        for k in range(5):
            ang = math.radians(60 * k + 30)
            pts.append([d2 * math.cos(ang), d2 * math.sin(ang), 0.0])
        # Center the configuration
        cx = sum(p[0] for p in pts) / N
        cy = sum(p[1] for p in pts) / N
        pts = [[p[0] - cx, p[1] - cy, p[2]] for p in pts]
        return pts
    
    def make_dense_cluster():
        """Dense cluster arrangement."""
        pts = []
        sqrt3 = math.sqrt(3)
        # Central hexagon
        pts.append([0, 0, 0.0])
        # First ring: 6 hexagons
        d1 = sqrt3 + 0.02
        for k in range(6):
            ang = math.radians(60 * k)
            pts.append([d1 * math.cos(ang), d1 * math.sin(ang), 0.0])
        # Second ring: 5 more hexagons
        d2 = 2 * sqrt3 + 0.02
        for k in range(5):
            ang = math.radians(60 * k + 30)
            pts.append([d2 * math.cos(ang), d2 * math.sin(ang), 0.0])
        # Center
        cx = sum(p[0] for p in pts) / N
        cy = sum(p[1] for p in pts) / N
        pts = [[p[0] - cx, p[1] - cy, p[2]] for p in pts]
        return pts
    
    def flatten(cfg, center, angle):
        v = []
        for p in cfg:
            v.extend([p[0], p[1], p[2]])
        v.extend([center[0], center[1], angle])
        return v
    
    def unflatten(v):
        cfg = []
        for i in range(N):
            cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
        center = (v[3*N], v[3*N+1])
        angle = v[3*N+2]
        return cfg, center, angle
    
    # Try multiple initial configurations
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    inits = [make_hex_lattice(), make_ring(), make_grid_initial(), make_center_ring_outer(), make_dense_cluster()]
    outer_angles = [0.0, 15.0, 30.0, -15.0, 45.0]
    for init in inits:
        for oa in outer_angles:
            s = _full_score(init, (0.0, 0.0), oa)
            if s is not None and s < best_score:
                best_score = s
                best_cfg = [row[:] for row in init]
                best_center = [0.0, 0.0]
                best_angle = oa
    
    if best_cfg is None:
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0
    
    # Phase 1: Differential Evolution for global optimization
    v0 = flatten(best_cfg, best_center, best_angle)
    n_params = len(v0)
    
    # Define bounds
    bounds = [(-5, 5)] * (3 * N) + [(-3, 3), (-3, 3), (-90, 90)]
    
    # Time budget for DE
    de_time_budget = 30.0
    
    def de_objective(v):
        return _smooth_objective(v)
    
    try:
        de_result = differential_evolution(
            de_objective, bounds,
            seed=42, maxiter=100, tol=1e-8,
            popsize=15, mutation=(0.3, 1.0), recombination=0.8,
            polish=False, init='latinhypercube'
        )
        if de_result.success and de_result.fun < best_score:
            v_de = de_result.x
            cfg_de, center_de, angle_de = unflatten(v_de)
            score_de = _full_score(cfg_de, center_de, angle_de)
            if score_de is not None and score_de < best_score:
                best_score = score_de
                best_cfg = cfg_de
                best_center = [center_de[0], center_de[1]]
                best_angle = angle_de
    except Exception:
        pass
    
    if time.time() - t0 > TIME_LIMIT - 40:
        pass  # Skip to final output
    
    # Phase 2: L-BFGS-B local optimization from multiple starting points
    remaining = TIME_LIMIT - (time.time() - t0) - 25.0  # Reserve 25s for final polish
    
    if remaining > 10.0:
        # Generate multiple starting points near the best found
        start_points = [v0[:]]
        for _ in range(5):
            v_pert = v0[:]
            for i in range(n_params):
                if i < 3 * N:
                    v_pert[i] += random.gauss(0, 0.5)
                else:
                    v_pert[i] += random.gauss(0, 0.2)
            start_points.append(v_pert)
        
        for sp in start_points:
            if time.time() - t0 > TIME_LIMIT - 25:
                break
            try:
                lbfgs_result = minimize(
                    _smooth_objective, sp,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if lbfgs_result.fun < best_score:
                    v_lbfgs = lbfgs_result.x
                    cfg_lbfgs, center_lbfgs, angle_lbfgs = unflatten(v_lbfgs)
                    score_lbfgs = _full_score(cfg_lbfgs, center_lbfgs, angle_lbfgs)
                    if score_lbfgs is not None and score_lbfgs < best_score:
                        best_score = score_lbfgs
                        best_cfg = cfg_lbfgs
                        best_center = [center_lbfgs[0], center_lbfgs[1]]
                        best_angle = angle_lbfgs
            except Exception:
                pass
    
    # Phase 3: High-precision coordinate descent polish
    remaining = TIME_LIMIT - (time.time() - t0) - 5.0
    
    if remaining > 5.0:
        cfg = [row[:] for row in best_cfg]
        center = best_center[:]
        angle = best_angle
        
        cur_score = best_score
        
        n_params_cd = 3 * N + 3
        step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        for iteration in range(300):
            if time.time() - t0 > TIME_LIMIT - 5:
                break
            
            improved = False
            for p in range(n_params_cd):
                if time.time() - t0 > TIME_LIMIT - 5:
                    break
                
                if p < 3 * N:
                    idx = p // 3
                    param_type = p % 3
                    old_val = cfg[idx][param_type]
                elif p < 3 * N + 2:
                    idx = p - 3 * N
                    old_val = center[idx]
                else:
                    old_val = angle
                
                best_local_val = old_val
                best_local_score = cur_score
                
                for step_size in step_sizes:
                    for direction in [-1, 1]:
                        new_val = old_val + direction * step_size
                        
                        if p < 3 * N:
                            idx = p // 3
                            param_type = p % 3
                            cfg[idx][param_type] = new_val
                        elif p < 3 * N + 2:
                            idx = p - 3 * N
                            center[idx] = new_val
                        else:
                            angle = new_val
                        
                        new_score = _full_score(cfg, tuple(center), angle)
                        if new_score is not None and new_score < best_local_score:
                            best_local_score = new_score
                            best_local_val = new_val
                        
                        if p < 3 * N:
                            idx = p // 3
                            param_type = p % 3
                            cfg[idx][param_type] = old_val
                        elif p < 3 * N + 2:
                            idx = p - 3 * N
                            center[idx] = old_val
                        else:
                            angle = old_val
                
                if best_local_score < cur_score - 1e-12:
                    if p < 3 * N:
                        idx = p // 3
                        param_type = p % 3
                        cfg[idx][param_type] = best_local_val
                    elif p < 3 * N + 2:
                        idx = p - 3 * N
                        center[idx] = best_local_val
                    else:
                        angle = best_local_val
                    
                    cur_score = best_local_score
                    improved = True
                    
                    if cur_score < best_score:
                        best_score = cur_score
                        best_cfg = [row[:] for row in cfg]
                        best_center = center[:]
                        best_angle = angle
            
            if not improved:
                break
    
    # Final validation
    if not _is_valid(best_cfg):
        # Fallback to a known valid configuration
        best_cfg = make_hex_lattice()
        best_center = [0.0, 0.0]
        best_angle = 0.0
        best_score = _outer_hex_side(best_cfg, tuple(best_center), best_angle)
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
