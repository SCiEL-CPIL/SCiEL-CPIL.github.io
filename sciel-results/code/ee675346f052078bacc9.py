"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import random

N = 12
DEADLINE = 138.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = np.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()

    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS

    def get_all_vertices(params):
        # params: [x1, y1, a1, x2, y2, a2, ...]
        verts_list = []
        for i in range(N):
            x, y, a = params[i*3], params[i*3+1], params[i*3+2]
            rad = np.deg2rad(a)
            c, s = np.cos(rad), np.sin(rad)
            rot = np.array([[c, -s], [s, c]])
            verts = unit_hex_verts @ rot.T + np.array([x, y])
            verts_list.append(verts)
        return verts_list

    def sat_check(v1, v2, eps=1e-9):
        """Check if two convex hexagons are disjoint using SAT. Returns True if disjoint."""
        # Quick reject based on center distance
        c1 = np.mean(v1, axis=0)
        c2 = np.mean(v2, axis=0)
        if np.linalg.norm(c1 - c2) > 2.0 + eps:
            return True
        
        # Check all 12 edge normals (6 from each polygon)
        for k in range(6):
            # Normal from v1
            e1 = v1[(k+1)%6] - v1[k]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 < 1e-12:
                continue
            n1 /= norm1
            proj1 = v1 @ n1
            proj2 = v2 @ n1
            if np.max(proj1) <= np.min(proj2) + eps or np.max(proj2) <= np.min(proj1) + eps:
                return True
        
        for k in range(6):
            # Normal from v2
            e2 = v2[(k+1)%6] - v2[k]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 < 1e-12:
                continue
            n2 /= norm2
            proj1 = v1 @ n2
            proj2 = v2 @ n2
            if np.max(proj1) <= np.min(proj2) + eps or np.max(proj2) <= np.min(proj1) + eps:
                return True
        
        return False

    def hexagons_disjoint(verts_list, eps=1e-9):
        for i in range(N):
            for j in range(i+1, N):
                if not sat_check(verts_list[i], verts_list[j], eps):
                    return False
        return True

    def compute_min_outer_side(verts_list, outer_angle_deg=0.0):
        """Compute minimum side length S of outer hexagon containing all inner hexagons."""
        sqrt3 = np.sqrt(3)
        # Outer hexagon has 6 edge normals at angles outer_angle_deg + 30 + 60*k
        # For containment, we need max projection on each normal <= apothem
        # Use 3 unique directions (opposite pairs give same |dot|)
        base_angle = np.radians(outer_angle_deg + 30)
        dirs = np.array([
            [np.cos(base_angle + np.radians(60*k)), np.sin(base_angle + np.radians(60*k))]
            for k in range(3)
        ])
        
        max_apothem = 0.0
        for verts in verts_list:
            dots = verts @ dirs.T
            abs_dots = np.abs(dots)
            max_apothem = max(max_apothem, np.max(abs_dots))
            
        # Apothem a = S * sqrt(3)/2, so S = 2*a/sqrt(3)
        S = 2.0 * max_apothem / sqrt3
        return S

    def objective(params, outer_angle_deg=0.0):
        verts_list = get_all_vertices(params)
        
        # Check disjointness
        if not hexagons_disjoint(verts_list, eps=1e-9):
            # Penalty for overlap - use a large constant
            return 1000.0 + compute_min_outer_side(verts_list, outer_angle_deg)
        
        return compute_min_outer_side(verts_list, outer_angle_deg)

    def time_check():
        return (time.time() - t0) < DEADLINE

    # Generate initial hexagonal lattice configuration
    def generate_hex_lattice(scale=1.0):
        """Generate 12 hexagons in a hexagonal lattice pattern."""
        points = []
        # Hex lattice: centers at (1.5*q, sqrt(3)*(r + 0.5*q))
        for q in range(-3, 4):
            for r in range(-3, 4):
                x = 1.5 * q * scale
                y = np.sqrt(3) * (r + 0.5 * q) * scale
                points.append((x, y))
        
        # Sort by distance from origin and take 12 closest
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_points = points[:N]
        
        params = []
        for x, y in init_points:
            params.extend([x, y, 0.0])
        return np.array(params)

    # Try multiple initial configurations and optimize each
    best_result = None
    best_score = float('inf')
    
    # Phase 1: Try DE from multiple starting points
    initial_configs = []
    
    # Hexagonal lattice with different scales
    for scale in [1.0, 1.05, 1.1]:
        initial_configs.append(generate_hex_lattice(scale))
    
    # Random configurations
    random.seed(42)
    for _ in range(3):
        params = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            params.extend([x, y, a])
        initial_configs.append(np.array(params))
    
    # Run DE on each initial config
    bounds = [(-4, 4), (-4, 4), (0, 60)] * N
    
    for idx, init_params in enumerate(initial_configs):
        if not time_check():
            break
        
        try:
            de_result = differential_evolution(
                lambda p: objective(p, 0.0), 
                bounds, 
                x0=init_params,
                maxiter=50, 
                popsize=15, 
                tol=1e-4, 
                seed=42+idx,
                polish=False,
                update_interval=5,
                callback=lambda xk, convergence: False if not time_check() else None
            )
            
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_result = de_result.x.copy()
                
        except Exception as e:
            pass
    
    # Phase 2: Polish with Nelder-Mead
    if best_result is not None and time_check():
        try:
            nm_result = minimize(
                lambda p: objective(p, 0.0), 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False}
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_result = nm_result.x.copy()
        except:
            pass

    # Phase 3: If still no valid result, use fallback
    if best_result is None or best_score > 100.0:
        # Fallback: hexagonal lattice with scale 1.02
        best_result = generate_hex_lattice(1.02)
        best_score = objective(best_result, 0.0)

    # Final validation
    verts_list = get_all_vertices(best_result)
    if not hexagons_disjoint(verts_list, eps=1e-8):
        # Try to fix by increasing scale
        for scale in [1.03, 1.05, 1.08, 1.10]:
            best_result = generate_hex_lattice(scale)
            verts_list = get_all_vertices(best_result)
            if hexagons_disjoint(verts_list, eps=1e-8):
                best_score = objective(best_result, 0.0)
                break

    inner = []
    for i in range(N):
        x = best_result[i*3]
        y = best_result[i*3+1]
        a = best_result[i*3+2]
        inner.append([float(x), float(y), float(a)])
        
    # Try multiple outer angles to find the optimal one
    verts_list = get_all_vertices(best_result)
    best_outer_angle = 0.0
    best_S = float('inf')
    
    for oa in range(0, 30, 5):  # Try 0, 5, 10, 15, 20, 25 degrees
        S = compute_min_outer_side(verts_list, oa)
        if S < best_S:
            best_S = S
            best_outer_angle = oa
    
    center = (0.0, 0.0)
    side = float(best_S)
    angle = float(best_outer_angle)
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
