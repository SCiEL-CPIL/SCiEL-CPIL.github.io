"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    angles = angle_rad + np.pi / 3.0 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using SAT.
    v1, v2: Nx2 arrays of vertices.
    Returns True if they overlap (including touching).
    """
    axes = []
    for v in (v1, v2):
        n = len(v)
        for i in range(n):
            edge = v[(i + 1) % n] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
            return False
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    verts = []
    for i in range(n):
        cx, cy, ang_deg = inner[i]
        verts.append(get_hex_vertices(cx, cy, 1.0, math.radians(ang_deg)))
    
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts[i], verts[j], eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                # Edge vector
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                # Vector from p1 to point
                px = v[0] - p1[0]
                py = v[1] - p1[1]
                # Cross product
                cross = ex * py - ey * px
                # For CCW polygon, interior is left (positive cross)
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_deg):
    """Check if the configuration is valid."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=100.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    # Collect all projections
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    # Log-sum-exp for smooth max
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

def objective(params):
    """Objective function: outer side length + penalty for overlaps."""
    n = N
    inner = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = np.degrees(params[3*i+2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3*n]
    oy = params[3*n+1]
    outer_angle = np.degrees(params[3*n+2]) % 360
    
    S = compute_required_side_smooth(inner, (ox, oy), outer_angle, beta=50.0)
    
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, a1 = inner[i]
            cx2, cy2, a2 = inner[j]
            d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.0:
                v1 = get_hex_vertices(cx1, cy1, 1.0, a1)
                v2 = get_hex_vertices(cx2, cy2, 1.0, a2)
                if sat_overlap_check(v1, v2, eps=1e-8):
                    penalty += (2.0 - d) ** 2 * 100
    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate diverse structural arrangements (discrete lattice search)
    # This avoids local minima of continuous optimizers by exploring globally distinct configurations
    
    # Helper to generate hexagonal lattice positions
    def hex_lattice_positions(n_hex, scale=1.0, rotation_deg=0):
        """Generate positions for n_hex hexagons on a hexagonal lattice."""
        positions = [(0.0, 0.0)]
        rot = math.radians(rotation_deg)
        
        # Ring 1: 6 neighbors
        for k in range(6):
            angle = math.radians(60 * k) + rot
            x = SQRT3 * scale * math.cos(angle)
            y = SQRT3 * scale * math.sin(angle)
            positions.append((x, y))
            if len(positions) >= n_hex:
                break
        
        # Ring 2: 12 positions
        if len(positions) < n_hex:
            for k in range(6):
                angle = math.radians(60 * k) + rot
                x = 2 * SQRT3 * scale * math.cos(angle)
                y = 2 * SQRT3 * scale * math.sin(angle)
                positions.append((x, y))
                if len(positions) >= n_hex:
                    break
            for k in range(6):
                angle = math.radians(30 + 60 * k) + rot
                x = 2 * SQRT3 * scale * math.cos(angle)
                y = 2 * SQRT3 * scale * math.sin(angle)
                positions.append((x, y))
                if len(positions) >= n_hex:
                    break
        
        return positions[:n_hex]
    
    # Generate diverse initial conditions using different structural arrangements
    initial_conditions = []
    
    # Arrangement 1: Standard hexagonal lattice
    positions1 = hex_lattice_positions(N, scale=1.0, rotation_deg=0)
    ic1 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(positions1):
        ic1[3*i] = px
        ic1[3*i+1] = py
        ic1[3*i+2] = 0.0
    ic1[3*N] = 0.0
    ic1[3*N+1] = 0.0
    ic1[3*N+2] = 0.0
    initial_conditions.append(ic1)
    
    # Arrangement 2: Rotated lattice (30 degrees)
    positions2 = hex_lattice_positions(N, scale=1.0, rotation_deg=30)
    ic2 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(positions2):
        ic2[3*i] = px
        ic2[3*i+1] = py
        ic2[3*i+2] = 0.0
    ic2[3*N] = 0.0
    ic2[3*N+1] = 0.0
    ic2[3*N+2] = math.radians(30)
    initial_conditions.append(ic2)
    
    # Arrangement 3: Tighter lattice (scale 0.98)
    positions3 = hex_lattice_positions(N, scale=0.98, rotation_deg=0)
    ic3 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(positions3):
        ic3[3*i] = px
        ic3[3*i+1] = py
        ic3[3*i+2] = 0.0
    ic3[3*N] = 0.0
    ic3[3*N+1] = 0.0
    ic3[3*N+2] = 0.0
    initial_conditions.append(ic3)
    
    # Arrangement 4: Different outer angle
    ic4 = ic1.copy()
    ic4[3*N+2] = math.radians(15)
    initial_conditions.append(ic4)
    
    # Arrangement 5: Perturbed standard lattice
    np.random.seed(42)
    ic5 = ic1 + np.random.randn(3*N+3) * 0.05
    initial_conditions.append(ic5)
    
    # Arrangement 6: Clustered arrangement (center + tight ring)
    positions6 = [(0.0, 0.0)]
    for k in range(6):
        angle = math.radians(60 * k)
        x = 1.7 * math.cos(angle)
        y = 1.7 * math.sin(angle)
        positions6.append((x, y))
    for k in range(4):
        angle = math.radians(30 + 60 * k)
        x = 3.0 * math.cos(angle)
        y = 3.0 * math.sin(angle)
        positions6.append((x, y))
    ic6 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(positions6[:N]):
        ic6[3*i] = px
        ic6[3*i+1] = py
        ic6[3*i+2] = 0.0
    ic6[3*N] = 0.0
    ic6[3*N+1] = 0.0
    ic6[3*N+2] = 0.0
    initial_conditions.append(ic6)
    
    best_result = None
    best_score = float('inf')
    
    # Optimize each initial condition
    for ic in initial_conditions:
        if time.time() > deadline - 20:
            break
        
        try:
            res = minimize(objective, ic, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    if S < best_score:
                        best_score = S
                        best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        positions_fb = hex_lattice_positions(N, scale=1.0, rotation_deg=0)
        inner = [[px, py, 0.0] for px, py in positions_fb]
        S = compute_required_side(inner, (0.0, 0.0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    # Refine if time permits
    if time.time() < deadline - 50:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = math.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = math.radians(best_result[3])
            
            res = minimize(objective, params_best, method='L-BFGS-B',
                          options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
