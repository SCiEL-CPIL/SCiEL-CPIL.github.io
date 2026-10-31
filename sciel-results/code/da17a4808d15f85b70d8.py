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

def objective_inner(params):
    """Objective: outer side length with outer fixed at origin/angle=0, plus overlap penalty."""
    n = N
    inner = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = np.degrees(params[3*i+2]) % 360
        inner.append((cx, cy, ang))
    
    S = compute_outer_side_length(inner, (0.0, 0.0), 0.0)
    
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
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Hexagonal lattice positions for N=11
    scale = 1.001
    hex_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * scale * np.cos(angle)
        y = np.sqrt(3) * scale * np.sin(angle)
        hex_positions.append((x, y))
    for k in range(6):
        angle = np.radians(60 * k)
        x = 2 * np.sqrt(3) * scale * np.cos(angle)
        y = 2 * np.sqrt(3) * scale * np.sin(angle)
        hex_positions.append((x, y))
        if len(hex_positions) >= N:
            break
    
    hex_positions = hex_positions[:N]
    
    best_result = None
    best_score = float('inf')
    
    # Build initial conditions for inner-only optimization (3*N params)
    initial_conditions = []
    
    # IC1: Standard hexagonal lattice
    ic1 = np.zeros(3 * N)
    for i, (px, py) in enumerate(hex_positions):
        ic1[3*i] = px
        ic1[3*i+1] = py
        ic1[3*i+2] = 0.0
    initial_conditions.append(ic1)
    
    # IC2: Perturbed lattice
    np.random.seed(42)
    ic2 = ic1 + np.random.randn(3*N) * 0.15
    initial_conditions.append(ic2)
    
    # IC3: Tighter lattice
    tight_scale = 0.99
    tight_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * tight_scale * np.cos(angle)
        y = np.sqrt(3) * tight_scale * np.sin(angle)
        tight_positions.append((x, y))
    for k in range(6):
        angle = np.radians(60 * k)
        x = 2 * np.sqrt(3) * tight_scale * np.cos(angle)
        y = 2 * np.sqrt(3) * tight_scale * np.sin(angle)
        tight_positions.append((x, y))
        if len(tight_positions) >= N:
            break
    tight_positions = tight_positions[:N]
    ic3 = np.zeros(3 * N)
    for i, (px, py) in enumerate(tight_positions):
        ic3[3*i] = px
        ic3[3*i+1] = py
        ic3[3*i+2] = 0.0
    initial_conditions.append(ic3)
    
    # IC4: Rotated inner hexagons by 30 deg
    ic4 = ic1.copy()
    for i in range(N):
        ic4[3*i+2] = np.radians(30)
    initial_conditions.append(ic4)
    
    # IC5: Perturbed rotated
    ic5 = ic4 + np.random.randn(3*N) * 0.1
    initial_conditions.append(ic5)
    
    # IC6: Different perturbation seed
    np.random.seed(123)
    ic6 = ic1 + np.random.randn(3*N) * 0.2
    initial_conditions.append(ic6)
    
    # IC7: Compact cluster - shift ring-2 hexagons inward
    ic7 = np.zeros(3 * N)
    compact_scale = 1.0
    compact_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * compact_scale * np.cos(angle)
        y = np.sqrt(3) * compact_scale * np.sin(angle)
        compact_positions.append((x, y))
    # Ring 2: use 4 positions but pull them inward
    for k in range(4):
        angle = np.radians(60 * k)
        r = 2 * np.sqrt(3) * 0.95
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        compact_positions.append((x, y))
    compact_positions = compact_positions[:N]
    for i, (px, py) in enumerate(compact_positions):
        ic7[3*i] = px
        ic7[3*i+1] = py
        ic7[3*i+2] = 0.0
    initial_conditions.append(ic7)
    
    for ic in initial_conditions:
        if time.time() > deadline - 25:
            break
        
        try:
            res = minimize(objective_inner, ic, method='L-BFGS-B',
                          options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                
                # Compute optimal outer hexagon for this inner configuration
                S = compute_outer_side_length(inner, (0.0, 0.0), 0.0)
                
                # Verify validity
                if check_validity(inner, (0.0, 0.0), S, 0.0):
                    best_score = S
                    best_result = (inner, (0.0, 0.0), S, 0.0)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        inner = [[px, py, 0.0] for px, py in hex_positions]
        S = compute_outer_side_length(inner, (0.0, 0.0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    # Phase 2: Joint optimization with outer parameters for refinement
    if time.time() < deadline - 40:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            def objective_full(params):
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
            
            res = minimize(objective_full, params_best, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            
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
                    best_score = S
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead refinement on best result
    if time.time() < deadline - 20:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            def objective_full2(params):
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
            
            res = minimize(objective_full2, params_best, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            
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
                    best_score = S
        except Exception:
            pass
    
    # Final time check
    if time.time() > deadline:
        return best_result
    
    return best_result

# EVOLVE-BLOCK-END
