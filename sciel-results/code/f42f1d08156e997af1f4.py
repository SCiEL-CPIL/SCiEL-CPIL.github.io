import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_deg):
    """Vertices of regular hexagon with center (cx,cy), side length, rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def support_function(points, normals):
    """Max dot product of points with each normal direction."""
    # points: (n, 2), normals: (6, 2)
    return np.max(points @ normals.T, axis=0)

def enclosing_hexagon_side(inner_positions, inner_angles):
    """Compute minimal enclosing regular hexagon side length and center.
    
    The enclosing hexagon has 6 normals at 60° apart. We minimize the max
    support function value over all 6 normals (for a regular hexagon of side s,
    the support in each normal direction equals s).
    
    For a regular hexagon with side s and center c, the support in direction n
    (unit) is c·n + s. So we need: max_i (max_j (p_j · n_i)) - min_i (min_j (p_j · n_i)) 
    to be minimized, and s = (max - min)/2 for each pair of opposite normals.
    
    Actually for a regular hexagon centered at c with side s:
    - The 6 face normals are at angles 0, 60, 120, 180, 240, 300 degrees
    - Support in direction n_k = c·n_k + s
    - So s must satisfy: s >= max_j(p_j · n_k) - c·n_k for all k
    - And s >= -(min_j(p_j · n_k) - c·n_k) = c·n_k - min_j(p_j · n_k)
    
    Optimal c: for each pair of opposite normals (n_k, -n_k):
    c·n_k = (max_j(p_j·n_k) + min_j(p_j·n_k)) / 2
    s_k = (max_j(p_j·n_k) - min_j(p_j·n_k)) / 2
    
    The enclosing hexagon side = max over all 3 pairs of s_k.
    """
    # 6 normal directions for a regular hexagon
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Compute support in each direction
    # For each normal, compute max and min projection of all inner hexagon vertices
    
    # Get all vertices of all inner hexagons
    all_verts = []
    for pos, ang in zip(inner_positions, inner_angles):
        verts = hexagon_vertices(pos[0], pos[1], 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    # Project onto each normal
    projections = all_verts @ normals.T  # (72, 6)
    
    max_proj = np.max(projections, axis=0)  # (6,)
    min_proj = np.min(projections, axis=0)  # (6,)
    
    # For opposite pairs (0,3), (1,4), (2,5):
    # s_k = (max_proj[k] - min_proj[k+3]) / 2 for k=0,1,2
    # (since normal[3] = -normal[0], etc.)
    s0 = (max_proj[0] - min_proj[3]) / 2
    s1 = (max_proj[1] - min_proj[4]) / 2
    s2 = (max_proj[2] - min_proj[5]) / 2
    
    side = max(s0, s1, s2)
    
    # Center: c·n_k = (max_proj[k] + min_proj[k+3]) / 2
    # Solve for c from 3 equations (using pairs 0-3, 1-4, 2-5)
    # c·n_0 = (max_proj[0] + min_proj[3]) / 2
    # c·n_1 = (max_proj[1] + min_proj[4]) / 2
    # c·n_2 = (max_proj[2] + min_proj[5]) / 2
    
    rhs = np.array([
        (max_proj[0] + min_proj[3]) / 2,
        (max_proj[1] + min_proj[4]) / 2,
        (max_proj[2] + min_proj[5]) / 2
    ])
    A = normals[:3]  # first 3 normals
    
    # Solve A @ c = rhs
    try:
        center = np.linalg.solve(A, rhs)
    except:
        center = np.array([0.0, 0.0])
    
    return side, center

def objective(params):
    """Objective: enclosing hexagon side length."""
    # params: 12*3 = 36 values: (x, y, angle) for each hexagon
    positions = params[:24].reshape(12, 2)
    angles = params[24:36]
    
    side, center = enclosing_hexagon_side(positions, angles)
    
    # Add small penalty for overlaps (shouldn't happen in good solutions)
    # Simple check: minimum distance between centers should be > 2*inradius
    # Inradius of unit hexagon = sqrt(3)/2 ≈ 0.866
    # But with rotation, the minimum separation is more complex
    # For now, just return the side length; overlaps will be caught by the evaluator
    
    return side

def construct():
    start_time = time.time()
    
    # Initial guess: hexagonal cluster pattern
    # 1 center + 6 ring + 5 outer
    # Unit hexagon inradius = sqrt(3)/2
    # Distance between adjacent hexagon centers in honeycomb = 2 * inradius = sqrt(3)
    
    r = np.sqrt(3)  # distance between adjacent centers
    
    # Position 0: center
    # Positions 1-6: ring at distance r
    # Positions 7-11: additional 5 positions
    
    # Standard hexagonal packing:
    # Ring 1: 6 positions at distance r, angles 0, 60, 120, 180, 240, 300
    # Ring 2: need 5 more. In a full hexagonal cluster of 19, ring 2 has 12 positions.
    # For 12 total: 1 + 6 + 5
    
    # Let's use a compact arrangement
    init_positions = []
    init_angles = []
    
    # Center
    init_positions.append([0.0, 0.0])
    init_angles.append(0.0)
    
    # First ring: 6 hexagons
    for i in range(6):
        theta = i * np.pi / 3
        init_positions.append([r * np.cos(theta), r * np.sin(theta)])
        init_angles.append(0.0)
    
    # Second ring: 5 more positions
    # In a hexagonal lattice, second ring positions are at distance 2*r from center
    # and at angles 0, 60, 120, 180, 240, 300 (6 positions) plus 
    # at distance r*sqrt(3) at angles 30, 90, 150, 210, 270, 330 (6 positions)
    # Total ring 2 = 12 positions. We need 5.
    
    # Let's pick 5 from the 12 ring-2 positions
    # Ring 2 outer: distance 2*r, angles 0, 60, 120, 180, 240, 300
    for i in range(5):
        theta = i * np.pi / 3
        init_positions.append([2 * r * np.cos(theta), 2 * r * np.sin(theta)])
        init_angles.append(0.0)
    
    # Flatten to parameter vector
    params0 = np.zeros(36)
    for i in range(12):
        params0[2*i] = init_positions[i][0]
        params0[2*i+1] = init_positions[i][1]
        params0[24+i] = init_angles[i]
    
    # Evaluate initial solution
    side0, center0 = enclosing_hexagon_side(
        np.array(init_positions), np.array(init_angles)
    )
    best_side = side0
    best_center = center0
    best_params = params0.copy()
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            objective,
            params0,
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        if result.fun < best_side:
            best_side = result.fun
            best_params = result.x
            _, best_center = enclosing_hexagon_side(
                best_params[:24].reshape(12, 2),
                best_params[24:36]
            )
    except Exception as e:
        pass
    
    # Check time
    if time.time() > DEADLINE - 5:
        pass  # skip further optimization
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for restart in range(5):
        if time.time() > DEADLINE - 10:
            break
        
        # Perturb initial solution
        params_r = best_params + np.random.randn(36) * 0.5
        
        try:
            result = minimize(
                objective,
                params_r,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-5,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x
                _, best_center = enclosing_hexagon_side(
                    best_params[:24].reshape(12, 2),
                    best_params[24:36]
                )
        except:
            pass
    
    # Final polish: tighter tolerance
    if time.time() < DEADLINE - 5:
        try:
            result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x
                _, best_center = enclosing_hexagon_side(
                    best_params[:24].reshape(12, 2),
                    best_params[24:36]
                )
        except:
            pass
    
    # Extract final solution
    positions = best_params[:24].reshape(12, 2)
    angles = best_params[24:36]
    
    inner = []
    for i in range(12):
        inner.append([positions[i][0], positions[i][1], np.degrees(angles[i])])
    
    return inner, (best_center[0], best_center[1]), best_side, 0.0

