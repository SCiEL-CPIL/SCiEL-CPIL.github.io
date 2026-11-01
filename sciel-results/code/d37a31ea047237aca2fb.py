"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as (x, y) tuples."""
    angles = angle_rad + np.pi / 3.0 * np.arange(6)
    return [(cx + side * math.cos(a), cy + side * math.sin(a)) for a in angles]

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_vec(inner_rad, outer_center, outer_angle_rad):
    """
    Vectorized computation of required outer side length.
    inner_rad: list of (x, y, angle_rad)
    """
    if not inner_rad:
        return 0.0
    
    ox, oy = outer_center
    cos_a = math.cos(outer_angle_rad)
    sin_a = math.sin(outer_angle_rad)
    
    # Precompute normals
    normals = np.array([
        [math.cos(math.radians(30 + i * 60)), math.sin(math.radians(30 + i * 60))]
        for i in range(6)
    ])
    
    max_proj = 0.0
    for (ix, iy, iangle) in inner_rad:
        # 6 vertices
        for k in range(6):
            a = iangle + k * math.pi / 3.0
            vx = ix + math.cos(a)
            vy = iy + math.sin(a)
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            # Project onto all 6 normals
            projs = lx * normals[:, 0] + ly * normals[:, 1]
            if np.max(projs) > max_proj:
                max_proj = np.max(projs)
    
    return 2.0 * max_proj / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using SAT.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    verts_list = []
    for (ix, iy, iangle) in inner:
        verts_list.append(get_hex_vertices(ix, iy, 1.0, math.radians(iangle)))
    
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_list[i], verts_list[j], eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for (vx, vy) in verts:
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                px = vx - p1[0]
                py = vy - p1[1]
                cross = ex * py - ey * px
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_deg):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def objective_rigid(params):
    """
    Rigid composite objective: all 11 hexagons move as one rigid body.
    params: [ox, oy, outer_angle_rad] (3 DOF for outer) + shared translation/rotation for inner
    Actually, let's use: the inner cluster has a fixed shape (axial coords), 
    and we optimize: (tx, ty, theta_inner, ox, oy, theta_outer)
    But since inner is rigid relative to each other, and outer is separate,
    we have 6 DOF: (tx, ty, theta_i, ox, oy, theta_o)
    """
    tx, ty, theta_i, ox, oy, theta_o = params
    
    # Axial coords for 11 hexagons
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    def axial_to_cartesian(q, r, scale=1.0):
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        return (x, y)
    
    # Compute inner positions with rigid transformation
    inner_rad = []
    for q, r in axial_coords:
        px, py = axial_to_cartesian(q, r, 1.0)
        # Apply rotation theta_i then translation (tx, ty)
        rx = px * math.cos(theta_i) - py * math.sin(theta_i)
        ry = px * math.sin(theta_i) + py * math.cos(theta_i)
        inner_rad.append((tx + rx, ty + ry, theta_i))
    
    # Compute required side
    S = compute_required_side_vec(inner_rad, (ox, oy), theta_o)
    
    # Overlap penalty: for rigid hexagonal lattice, check pairwise distances
    # In the lattice, minimum center distance is sqrt(3) for touching hexagons
    # With scale 1.0, adjacent centers are exactly sqrt(3) apart
    # But we need to check actual distances after transformation (which preserves distances)
    # So overlaps only occur if scale < 1.0, which it isn't here
    # Actually, the rigid body preserves all pairwise distances, so no overlaps possible
    # as long as the base configuration has no overlaps (which it doesn't at scale 1.0)
    
    return S

def objective_semi_rigid(params):
    """
    Semi-rigid: inner hexagons can have individual angle adjustments.
    params: for each inner hex: (dx, dy, dangle) relative to lattice position, 
    plus outer: (ox, oy, theta_o)
    Total: 3*N + 3
    """
    n = N
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    def axial_to_cartesian(q, r, scale=1.0):
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        return (x, y)
    
    inner_rad = []
    for i in range(n):
        base_x, base_y = axial_to_cartesian(axial_coords[i][0], axial_coords[i][1], 1.0)
        dx = params[3*i]
        dy = params[3*i+1]
        dangle = params[3*i+2]
        inner_rad.append((base_x + dx, base_y + dy, dangle))
    
    ox = params[3*n]
    oy = params[3*n+1]
    theta_o = params[3*n+2]
    
    S = compute_required_side_vec(inner_rad, (ox, oy), theta_o)
    
    # Overlap penalty
    penalty = 0.0
    for i in range(n):
        cx1, cy1, _ = inner_rad[i]
        for j in range(i+1, n):
            cx2, cy2, _ = inner_rad[j]
            d = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            min_d = SQRT3 - 1e-4  # touching distance
            if d < min_d:
                penalty += (min_d - d) ** 2 * 100.0
    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    def axial_to_cartesian(q, r, scale=1.0):
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        return (x, y)
    
    best_result = None
    best_score = float('inf')
    
    # Phase 1: Rigid composite optimization (6 DOF)
    # Try different initial conditions for the rigid body
    rigid_inits = []
    for theta_i in [0, np.pi/6, np.pi/3, np.pi/4, np.pi/12]:
        for theta_o in [0, np.pi/6, np.pi/3]:
            for ox, oy in [(0,0), (0.5,0), (-0.5,0), (0,0.5), (0,-0.5)]:
                rigid_inits.append([0, 0, theta_i, ox, oy, theta_o])
    
    for init in rigid_inits:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(objective_rigid, init, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            tx, ty, theta_i, ox, oy, theta_o = res.x
            
            # Build inner configuration
            inner_rad = []
            for q, r in axial_coords:
                px, py = axial_to_cartesian(q, r, 1.0)
                rx = px * math.cos(theta_i) - py * math.sin(theta_i)
                ry = px * math.sin(theta_i) + py * math.cos(theta_i)
                inner_rad.append((tx + rx, ty + ry, theta_i))
            
            S = compute_required_side_vec(inner_rad, (ox, oy), theta_o)
            
            # Convert to degrees for validity check
            inner_deg = [(x, y, np.degrees(a) % 360) for (x, y, a) in inner_rad]
            outer_angle_deg = np.degrees(theta_o) % 360
            
            if check_validity(inner_deg, (ox, oy), S, outer_angle_deg):
                if S < best_score:
                    best_score = S
                    best_result = (inner_deg, (ox, oy), S, outer_angle_deg)
        except Exception:
            pass
    
    # Phase 2: Semi-rigid refinement (33 DOF)
    if best_result is not None and time.time() < deadline - 20:
        try:
            # Start from best rigid result
            inner_deg, (ox, oy), S, outer_angle_deg = best_result
            inner_rad = [(x, y, np.radians(a)) for (x, y, a) in inner_deg]
            
            # Build semi-rigid params
            params = np.zeros(3*N + 3)
            for i in range(N):
                base_x, base_y = axial_to_cartesian(axial_coords[i][0], axial_coords[i][1], 1.0)
                # The rigid result has inner at rotated positions
                # We need to find the offsets from the base lattice positions
                # Actually, let's just use the current positions as base
                params[3*i] = inner_rad[i][0] - base_x
                params[3*i+1] = inner_rad[i][1] - base_y
                params[3*i+2] = inner_rad[i][2]
            params[3*N] = ox
            params[3*N+1] = oy
            params[3*N+2] = np.radians(outer_angle_deg)
            
            res = minimize(objective_semi_rigid, params, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_score:
                inner_rad_new = []
                for i in range(N):
                    base_x, base_y = axial_to_cartesian(axial_coords[i][0], axial_coords[i][1], 1.0)
                    inner_rad_new.append((base_x + res.x[3*i], base_y + res.x[3*i+1], res.x[3*i+2]))
                ox_new = res.x[3*N]
                oy_new = res.x[3*N+1]
                theta_o_new = res.x[3*N+2]
                
                S_new = compute_required_side_vec(inner_rad_new, (ox_new, oy_new), theta_o_new)
                inner_deg_new = [(x, y, np.degrees(a) % 360) for (x, y, a) in inner_rad_new]
                outer_angle_deg_new = np.degrees(theta_o_new) % 360
                
                if check_validity(inner_deg_new, (ox_new, oy_new), S_new, outer_angle_deg_new):
                    if S_new < best_score:
                        best_score = S_new
                        best_result = (inner_deg_new, (ox_new, oy_new), S_new, outer_angle_deg_new)
        except Exception:
            pass
    
    # Phase 3: Full individual optimization (36 DOF) if time permits
    if best_result is not None and time.time() < deadline - 15:
        try:
            inner_deg, (ox, oy), S, outer_angle_deg = best_result
            params = np.zeros(3*N + 3)
            for i in range(N):
                params[3*i] = inner_deg[i][0]
                params[3*i+1] = inner_deg[i][1]
                params[3*i+2] = np.radians(inner_deg[i][2])
            params[3*N] = ox
            params[3*N+1] = oy
            params[3*N+2] = np.radians(outer_angle_deg)
            
            def objective_full(params):
                n = N
                inner_rad = []
                for i in range(n):
                    inner_rad.append((params[3*i], params[3*i+1], params[3*i+2]))
                ox = params[3*n]
                oy = params[3*n+1]
                theta_o = params[3*n+2]
                
                S = compute_required_side_vec(inner_rad, (ox, oy), theta_o)
                
                penalty = 0.0
                for i in range(n):
                    cx1, cy1, _ = inner_rad[i]
                    for j in range(i+1, n):
                        cx2, cy2, _ = inner_rad[j]
                        d = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                        min_d = SQRT3 - 1e-4
                        if d < min_d:
                            penalty += (min_d - d) ** 2 * 100.0
                
                return S + penalty
            
            res = minimize(objective_full, params, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_score:
                inner_rad_new = []
                for i in range(N):
                    inner_rad_new.append((res.x[3*i], res.x[3*i+1], res.x[3*i+2]))
                ox_new = res.x[3*N]
                oy_new = res.x[3*N+1]
                theta_o_new = res.x[3*N+2]
                
                S_new = compute_required_side_vec(inner_rad_new, (ox_new, oy_new), theta_o_new)
                inner_deg_new = [(x, y, np.degrees(a) % 360) for (x, y, a) in inner_rad_new]
                outer_angle_deg_new = np.degrees(theta_o_new) % 360
                
                if check_validity(inner_deg_new, (ox_new, oy_new), S_new, outer_angle_deg_new):
                    if S_new < best_score:
                        best_score = S_new
                        best_result = (inner_deg_new, (ox_new, oy_new), S_new, outer_angle_deg_new)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        centers_fb = [axial_to_cartesian(q, r, 1.0) for q, r in axial_coords]
        inner = [(px, py, 0.0) for px, py in centers_fb]
        S = compute_required_side(inner, (0.0, 0.0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    return best_result

# EVOLVE-BLOCK-END
