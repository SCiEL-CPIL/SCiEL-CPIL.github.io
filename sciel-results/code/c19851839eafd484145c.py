"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi/3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for v in (v1, v2):
        n = len(v)
        for i in range(n):
            edge = v[(i+1) % n] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False, axis
    return True, None

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length that contains all inner hexagons."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    
    normals = []
    for k in range(6):
        theta = oa + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    min_S = 0.0
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for n in normals:
                d = np.dot(v - np.array([ox, oy]), n)
                if d > 0:
                    S_needed = 2.0 * d / np.sqrt(3.0)
                    if S_needed > min_S:
                        min_S = S_needed
    return min_S

def check_validity(inner_hexagons, outer_center, outer_side, outer_angle):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    ox, oy = outer_center
    
    # Check no overlaps between inner hexagons
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            overlap, _ = polygons_overlap(v1, v2, tol=1e-8)
            if overlap:
                return False
    
    # Check all inner hexagons are inside outer hexagon
    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
    
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            n = len(outer_verts)
            inside = True
            for i in range(n):
                v1 = outer_verts[i]
                v2 = outer_verts[(i+1) % n]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                d1 = np.dot(normal, np.array([px, py]) - v1)
                d2 = np.dot(normal, outer_verts[(i+2) % n] - v1)
                if d1 * d2 < 0:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

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
    
    S = compute_outer_side_length(inner, (ox, oy), outer_angle)
    
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, a1 = inner[i]
            cx2, cy2, a2 = inner[j]
            d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.0:
                v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
                v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
                overlap, _ = polygons_overlap(v1, v2, tol=1e-8)
                if overlap:
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
        rot = np.radians(rotation_deg)
        
        # Ring 1: 6 neighbors
        for k in range(6):
            angle = np.radians(60 * k) + rot
            x = np.sqrt(3) * scale * np.cos(angle)
            y = np.sqrt(3) * scale * np.sin(angle)
            positions.append((x, y))
            if len(positions) >= n_hex:
                break
        
        # Ring 2: 12 positions
        if len(positions) < n_hex:
            for k in range(6):
                angle = np.radians(60 * k) + rot
                x = 2 * np.sqrt(3) * scale * np.cos(angle)
                y = 2 * np.sqrt(3) * scale * np.sin(angle)
                positions.append((x, y))
                if len(positions) >= n_hex:
                    break
            for k in range(6):
                angle = np.radians(30 + 60 * k) + rot
                x = 2 * np.sqrt(3) * scale * np.cos(angle)
                y = 2 * np.sqrt(3) * scale * np.sin(angle)
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
    ic2[3*N+2] = np.radians(30)
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
    ic4[3*N+2] = np.radians(15)
    initial_conditions.append(ic4)
    
    # Arrangement 5: Perturbed standard lattice
    np.random.seed(42)
    ic5 = ic1 + np.random.randn(3*N+3) * 0.05
    initial_conditions.append(ic5)
    
    # Arrangement 6: Clustered arrangement (center + tight ring)
    positions6 = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = 1.7 * np.cos(angle)
        y = 1.7 * np.sin(angle)
        positions6.append((x, y))
    for k in range(4):
        angle = np.radians(30 + 60 * k)
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
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
                
                S = compute_outer_side_length(inner, (ox, oy), outer_angle)
                
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
        S = compute_outer_side_length(inner, (0.0, 0.0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    # Refine if time permits
    if time.time() < deadline - 50:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
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
                S = compute_outer_side_length(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
