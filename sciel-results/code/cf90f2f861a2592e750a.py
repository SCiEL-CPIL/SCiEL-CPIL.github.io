import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
TIME_LIMIT = 146.0  # Safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
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
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
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

def _is_valid_fast(cx_arr, cy_arr, angle_arr):
    n = len(cx_arr)
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i]))
    
    axes_list = []
    for i in range(n):
        axes_list.append(_axes_from_polygon(verts_list[i]))
    
    for i in range(n):
        vi = verts_list[i]
        ai = axes_list[i]
        for j in range(i + 1, n):
            vj = verts_list[j]
            aj = axes_list[j]
            
            # Quick distance check
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx * dx + dy * dy
            
            # Min distance between centers for non-overlap is sqrt(3) ~ 1.732
            # If dist > 2.0, they definitely don't overlap? No, max radius is 1.0.
            # If dist > 2.0, they don't overlap.
            if dist_sq > 4.0:
                continue
            
            # Separating Axis Theorem
            found_sep = False
            for axis in ai:
                ax, ay = axis
                minA, maxA = _project_polygon(vi, axis)
                minB, maxB = _project_polygon(vj, axis)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    found_sep = True
                    break
            
            if found_sep:
                continue
                
            for axis in aj:
                ax, ay = axis
                minA, maxA = _project_polygon(vi, axis)
                minB, maxB = _project_polygon(vj, axis)
                if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
                    found_sep = True
                    break
            
            if not found_sep:
                return False
    
    return True

def _outer_hex_side(cx_arr, cy_arr, angle_arr, center, angle_deg):
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for i in range(len(cx_arr)):
        verts = _hex_vertices(cx_arr[i], cy_arr[i], angle_arr[i])
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def construct():
    t0 = time.time()
    random.seed(42)
    
    # Initial configuration: Flower
    def make_flower_initial():
        pts = []
        pts.append([0.0, 0.0])
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            angle = math.radians(60 * i)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            pts.append([x, y])
        r2 = r1 * 1.732
        angles = [30, 90, 150, 210, 270]
        for deg in angles:
            rad = math.radians(deg)
            x = r2 * math.cos(rad)
            y = r2 * math.sin(rad)
            pts.append([x, y])
        return pts[:12]
    
    init_pts = make_flower_initial()
    
    # State: 12 x-coords, 12 y-coords, 1 global angle (degrees)
    # Total 25 variables
    # We will optimize positions and global angle.
    # However, allowing individual angles might yield better results.
    # Let's stick to global angle first for robustness, then polish with individual angles if time permits?
    # Or just use individual angles in DE? 36 variables is fine for DE.
    
    def objective(params):
        if time.time() - t0 > TIME_LIMIT - 5:
            return 1e9
        
        xs = params[0:N]
        ys = params[N:2*N]
        angles = params[2*N:3*N]
        
        # Check validity
        if not _is_valid_fast(xs, ys, angles):
            # Penalty for overlap
            # Calculate min penetration depth roughly
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d = math.sqrt(dx*dx + dy*dy)
                    # Approximate min distance as sqrt(3)
                    if d < math.sqrt(3):
                        pen += (math.sqrt(3) - d) ** 2
            return 100.0 + 10.0 * pen
        
        # Compute enclosing hexagon side
        # Try a few outer angles to find the best fit?
        # Or just fix outer angle to 0 and center to centroid?
        # Better: compute min enclosing hexagon for the set of points.
        
        # Centroid
        cx_c = np.mean(xs)
        cy_c = np.mean(ys)
        
        best_side = 1e9
        # Try outer angles 0 to 30
        for outer_angle in range(0, 30, 5):
            side = _outer_hex_side(xs, ys, angles, (cx_c, cy_c), float(outer_angle))
            if side < best_side:
                best_side = side
                
        return best_side

    # Bounds
    # x, y in [-5, 5]
    # angle in [0, 60]
    bounds = [(-5, 5)] * (2 * N) + [(0, 60)] * N
    
    # Initial population guess
    x0 = [p[0] for p in init_pts] + [p[1] for p in init_pts] + [0.0] * N
    
    try:
        result = differential_evolution(
            objective,
            bounds,
            x0=x0,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-4,
            atol=1e-4,
            polish=False, # We will polish manually
            init='latinhypercube'
        )
        
        if result.success or result.fun < 100:
            best_params = result.x
        else:
            best_params = x0
            
        xs = best_params[0:N]
        ys = best_params[N:2*N]
        angles = best_params[2*N:3*N]
        
        # Polish with Nelder-Mead on individual positions and angles
        # Objective for polish: valid config -> side length, invalid -> large penalty
        
        def polish_obj(params):
            xs_p = params[0:N]
            ys_p = params[N:2*N]
            angles_p = params[2*N:3*N]
            
            if not _is_valid_fast(xs_p, ys_p, angles_p):
                # Calculate penalty
                pen = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        dx = xs_p[i] - xs_p[j]
                        dy = ys_p[i] - ys_p[j]
                        d = math.sqrt(dx*dx + dy*dy)
                        if d < math.sqrt(3):
                            pen += (math.sqrt(3) - d) ** 2
                return 100.0 + 10.0 * pen
            
            cx_c = np.mean(xs_p)
            cy_c = np.mean(ys_p)
            
            best_side = 1e9
            for outer_angle in range(0, 30, 5):
                side = _outer_hex_side(xs_p, ys_p, angles_p, (cx_c, cy_c), float(outer_angle))
                if side < best_side:
                    best_side = side
            return best_side

        res_polish = minimize(
            polish_obj,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4}
        )
        
        if res_polish.fun < result.fun:
            best_params = res_polish.x
            
        xs_final = best_params[0:N]
        ys_final = best_params[N:2*N]
        angles_final = best_params[2*N:3*N]
        
        # Final optimization of outer hexagon center and angle
        cx_c = np.mean(xs_final)
        cy_c = np.mean(ys_final)
        
        best_side = 1e9
        best_center = (cx_c, cy_c)
        best_outer_angle = 0.0
        
        for outer_angle in range(0, 30, 1):
            # Local search for center
            cx_cur = cx_c
            cy_cur = cy_c
            for _ in range(10):
                side_cur = _outer_hex_side(xs_final, ys_final, angles_final, (cx_cur, cy_cur), float(outer_angle))
                best_side_local = side_cur
                best_cx = cx_cur
                best_cy = cy_cur
                
                for dx in [-0.1, 0, 0.1]:
                    for dy in [-0.1, 0, 0.1]:
                        ncx = cx_cur + dx
                        ncy = cy_cur + dy
                        side_test = _outer_hex_side(xs_final, ys_final, angles_final, (ncx, ncy), float(outer_angle))
                        if side_test < best_side_local:
                            best_side_local = side_test
                            best_cx = ncx
                            best_cy = ncy
                            
                if abs(best_cx - cx_cur) < 1e-5 and abs(best_cy - cy_cur) < 1e-5:
                    break
                cx_cur = best_cx
                cy_cur = best_cy
                
            if best_side_local < best_side:
                best_side = best_side_local
                best_center = (best_cx, best_cy)
                best_outer_angle = float(outer_angle)
                
        # Verify validity
        if _is_valid_fast(xs_final, ys_final, angles_final):
            inner = [[float(xs_final[i]), float(ys_final[i]), float(angles_final[i])] for i in range(N)]
            return inner, best_center, best_side, best_outer_angle
            
    except Exception:
        pass

    # Fallback: Return initial flower configuration
    cfg = make_flower_initial()
    xs = [p[0] for p in cfg]
    ys = [p[1] for p in cfg]
    angles = [0.0] * N
    
    cx_c = np.mean(xs)
    cy_c = np.mean(ys)
    side = _outer_hex_side(xs, ys, angles, (cx_c, cy_c), 0.0)
    
    inner = [[float(xs[i]), float(ys[i]), float(angles[i])] for i in range(N)]
    return inner, (cx_c, cy_c), side, 0.0

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
