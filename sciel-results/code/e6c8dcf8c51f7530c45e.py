"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 112.0  # safety margin below 120s

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

def construct():
    t0 = time.time()
    random.seed(42)
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, center, angle)
    
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
    
    def smooth_objective(v):
        """
        Smooth objective using log-sum-exp for max projection and collision penalty.
        v = [x1,y1,a1, ..., x12,y12,a12, cx,cy, outer_angle]
        """
        cfg = []
        for i in range(N):
            cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
        center = (v[3*N], v[3*N+1])
        outer_angle = v[3*N+2]
        
        # Compute all vertices for all hexagons
        all_verts = []
        for p in cfg:
            verts = _hex_vertices(p[0], p[1], p[2])
            all_verts.append(verts)
        
        # Outer hexagon normals
        a = math.radians(outer_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        cx, cy = center
        
        # Compute max projection (smoothed via log-sum-exp)
        beta = 10.0  # sharpness parameter
        max_proj_logsumexp = 0.0
        max_val = 0.0
        for verts in all_verts:
            for (vx, vy) in verts:
                for (nx, ny) in normals:
                    val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                    if val > max_val:
                        max_val = val
                    max_proj_logsumexp += math.exp(beta * val)
        
        if max_proj_logsumexp > 0:
            smooth_max = (1.0 / beta) * math.log(max_proj_logsumexp) + max_val - (1.0/beta)*math.log(6*N*6)
            # Actually, let's use a simpler approach: just compute the max directly but add a small smooth term
            # For now, use the actual max since log-sum-exp can be tricky
            smooth_max = max_val
        else:
            smooth_max = 0.0
        
        # Collision penalty: for each pair, compute minimum distance and penalize if overlapping
        collision_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Check if polygons overlap using SAT
                if _separating_axis_check(all_verts[i], all_verts[j]):
                    # They overlap; add penalty based on how much they overlap
                    # Approximate overlap depth by checking projections
                    overlap_depth = 0.0
                    axes = _axes_from_polygon(all_verts[i]) + _axes_from_polygon(all_verts[j])
                    for axis in axes:
                        minA, maxA = _project_polygon(all_verts[i], axis)
                        minB, maxB = _project_polygon(all_verts[j], axis)
                        overlap = min(maxA, maxB) - max(minA, minB)
                        if overlap > overlap_depth:
                            overlap_depth = overlap
                    collision_penalty += overlap_depth ** 2
        
        return smooth_max + 10.0 * collision_penalty
    
    def gradient_descent_polish(cfg_init, center_init, angle_init, time_budget):
        """Gradient descent with smooth objective to polish the configuration."""
        def flatten(cfg, center, angle):
            v = []
            for p in cfg:
                v.extend([p[0], p[1], p[2]])
            v.extend([center[0], center[1], angle])
            return np.array(v, dtype=float)
        
        def unflatten(v):
            cfg = []
            for i in range(N):
                cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
            center = [v[3*N], v[3*N+1]]
            angle = v[3*N+2]
            return cfg, center, angle
        
        v = flatten(cfg_init, center_init, angle_init)
        n_params = len(v)
        
        # Numerical gradient
        def compute_gradient(v, h=1e-5):
            grad = np.zeros(n_params)
            f0 = smooth_objective(v)
            for i in range(n_params):
                v_plus = v.copy()
                v_plus[i] += h
                f_plus = smooth_objective(v_plus)
                grad[i] = (f_plus - f0) / h
            return grad
        
        lr = 0.05
        best_v = v.copy()
        best_f = smooth_objective(v)
        
        max_iters = 500
        iter_count = 0
        while iter_count < max_iters and time.time() - t0 < time_budget:
            grad = compute_gradient(v)
            
            # Check for convergence
            if np.linalg.norm(grad) < 1e-6:
                break
            
            # Update
            v_new = v - lr * grad
            
            # Line search: reduce step if objective increases
            f_new = smooth_objective(v_new)
            step_scale = 1.0
            while f_new > best_f and step_scale > 0.01:
                step_scale *= 0.5
                v_new = v - lr * step_scale * grad
                f_new = smooth_objective(v_new)
            
            if f_new < best_f:
                best_f = f_new
                best_v = v_new.copy()
            v = v_new
            
            # Adaptive learning rate
            if f_new > best_f:
                lr *= 0.9
            else:
                lr = min(lr * 1.05, 0.1)
            
            iter_count += 1
        
        return best_v, best_f
    
    # Try multiple initial configurations
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    inits = [make_hex_lattice(), make_ring()]
    for init in inits:
        s = full_score(init, (0.0, 0.0), 0.0)
        if s is not None and s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0
    
    if best_cfg is None:
        # Fallback: spread out widely
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0
    
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score
    
    T = 0.8
    T_end = 0.003
    alpha = 0.9997
    
    # Reserve time for gradient descent polish
    gd_time_budget = 15.0  # seconds reserved for polishing
    sa_time_limit = TIME_LIMIT - gd_time_budget
    
    while time.time() - t0 < sa_time_limit:
        r = random.random()
        
        if r < 0.50:
            # Move one hexagon
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.2 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.70:
            # Rotate one hexagon
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 5.0 + 3.0 * T)
        elif r < 0.80:
            # Move center
            old_cx, old_cy = center[0], center[1]
            step = 0.1 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.90:
            # Rotate outer
            old_angle = angle
            angle += random.gauss(0, 2.0 + 2.0 * T)
        else:
            # Perturb two hexagons
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.15 * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
        
        new_score = full_score(cfg, tuple(center), angle)
        
        if new_score is not None:
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                # Revert
                if r < 0.50:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.70:
                    cfg[idx][2] = old_ang
                elif r < 0.80:
                    center[0], center[1] = old_cx, old_cy
                elif r < 0.90:
                    angle = old_angle
                else:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        else:
            # Invalid: revert
            if r < 0.50:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.70:
                cfg[idx][2] = old_ang
            elif r < 0.80:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.90:
                angle = old_angle
            else:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
        
        T = max(T_end, T * alpha)
    
    # Gradient descent polish on the best configuration found
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 5.0:  # Only polish if we have enough time
        try:
            polished_v, polished_f = gradient_descent_polish(
                best_cfg, best_center, best_angle, TIME_LIMIT - 2.0
            )
            # Unflatten the polished result
            cfg_polished = []
            for i in range(N):
                cfg_polished.append([polished_v[3*i], polished_v[3*i+1], polished_v[3*i+2]])
            center_polished = [polished_v[3*N], polished_v[3*N+1]]
            angle_polished = polished_v[3*N+2]
            
            # Verify the polished result
            polished_score = full_score(cfg_polished, tuple(center_polished), angle_polished)
            if polished_score is not None and polished_score < best_score:
                best_score = polished_score
                best_cfg = cfg_polished
                best_center = center_polished
                best_angle = angle_polished
        except Exception:
            pass  # Keep the SA result if polish fails
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
