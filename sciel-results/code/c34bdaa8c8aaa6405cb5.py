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
            return False
    return True

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length containing all inner hexagons."""
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
    """Check no overlaps and all inside outer hexagon."""
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            if polygons_overlap(v1, v2, tol=1e-8):
                return False
    ox, oy = outer_center
    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            inside = True
            for i in range(6):
                v1 = outer_verts[i]
                v2 = outer_verts[(i+1) % 6]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                d1 = np.dot(normal, np.array([px, py]) - v1)
                d2 = np.dot(normal, outer_verts[(i+2) % 6] - v1)
                if d1 * d2 < 0:
                    inside = False
                    break
            if not inside:
                return False
    return True

def objective(params):
    """Objective: outer side length + overlap penalty."""
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
                if polygons_overlap(v1, v2, tol=1e-8):
                    penalty += (2.0 - d) ** 2 * 100
    return S + penalty

def construct():
    """Main entry point: find optimal packing using multi-start local optimization."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    best_result = None
    best_score = float('inf')
    
    # Generate diverse initial conditions based on hexagonal lattice structures
    initial_conditions = []
    
    # 1. Standard Hexagonal Lattice (Center + Ring 1 + Part of Ring 2)
    # Distance between centers for touching hexagons is sqrt(3)
    d = np.sqrt(3)
    hex_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        hex_positions.append((x, y))
    for k in range(6):
        angle = np.radians(60 * k)
        x = 2 * d * np.cos(angle)
        y = 2 * d * np.sin(angle)
        hex_positions.append((x, y))
        if len(hex_positions) >= N:
            break
    hex_positions = hex_positions[:N]
    
    ic1 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(hex_positions):
        ic1[3*i] = px
        ic1[3*i+1] = py
        ic1[3*i+2] = 0.0
    ic1[3*N] = 0.0
    ic1[3*N+1] = 0.0
    ic1[3*N+2] = 0.0
    initial_conditions.append(ic1)
    
    # 2. Perturbed Hexagonal Lattice
    np.random.seed(42)
    ic2 = ic1 + np.random.randn(3*N+3) * 0.1
    initial_conditions.append(ic2)
    
    # 3. Rotated Inner Hexagons (30 deg)
    ic3 = ic1.copy()
    for i in range(N):
        ic3[3*i+2] = np.radians(30)
    initial_conditions.append(ic3)
    
    # 4. Different Outer Angle (30 deg)
    ic4 = ic1.copy()
    ic4[3*N+2] = np.radians(30)
    initial_conditions.append(ic4)
    
    # 5. Tighter Lattice (Scale 0.95) to force overlap resolution
    tight_scale = 0.95
    tight_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = d * tight_scale * np.cos(angle)
        y = d * tight_scale * np.sin(angle)
        tight_positions.append((x, y))
    for k in range(6):
        angle = np.radians(60 * k)
        x = 2 * d * tight_scale * np.cos(angle)
        y = 2 * d * tight_scale * np.sin(angle)
        tight_positions.append((x, y))
        if len(tight_positions) >= N:
            break
    tight_positions = tight_positions[:N]
    ic5 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(tight_positions):
        ic5[3*i] = px
        ic5[3*i+1] = py
        ic5[3*i+2] = 0.0
    ic5[3*N] = 0.0
    ic5[3*N+1] = 0.0
    ic5[3*N+2] = 0.0
    initial_conditions.append(ic5)
    
    # 6. Random Cluster Start (for diversity)
    np.random.seed(123)
    ic6 = np.random.randn(3*N+3) * 2.0
    # Keep angles small initially
    for i in range(N):
        ic6[3*i+2] = np.random.uniform(-np.pi/6, np.pi/6)
    ic6[3*N] = 0.0
    ic6[3*N+1] = 0.0
    ic6[3*N+2] = 0.0
    initial_conditions.append(ic6)
    
    # 7. Shifted Center
    ic7 = ic1.copy()
    ic7[3*N] = 0.5
    ic7[3*N+1] = 0.5
    initial_conditions.append(ic7)

    # Run L-BFGS-B from each initial condition
    for idx, ic in enumerate(initial_conditions):
        if time.time() > deadline - 15:
            break
        
        try:
            res = minimize(objective, ic, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
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
                
                # Verify validity
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_score = S
                    best_result = (inner, (ox, oy), S, outer_angle)
                    
        except Exception as e:
            pass

    # Final refinement if we have a result and time remains
    if best_result is not None and time.time() < deadline - 5:
        try:
            params_base = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_base[3*i] = cx
                params_base[3*i+1] = cy
                params_base[3*i+2] = np.radians(ang)
            params_base[3*N] = best_result[1][0]
            params_base[3*N+1] = best_result[1][1]
            params_base[3*N+2] = np.radians(best_result[3])
            
            res = minimize(objective, params_base, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
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
                    best_score = S
                    best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass

    # Fallback if nothing valid found
    if best_result is None:
        inner = []
        for i, (px, py) in enumerate(hex_positions):
            inner.append([px, py, 0.0])
        ox, oy = 0.0, 0.0
        outer_angle = 0.0
        S = compute_outer_side_length(inner, (ox, oy), outer_angle)
        S += 0.1
        best_result = (inner, (ox, oy), S, outer_angle)

    return best_result

# EVOLVE-BLOCK-END
