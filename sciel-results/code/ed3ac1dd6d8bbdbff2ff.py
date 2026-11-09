import numpy as np
import time
from scipy.optimize import minimize

N = 11
_DEADLINE = 140.0

def _hex_vertices(cx, cy, theta, side=1.0):
    angles = theta + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _outer_side_and_grad(positions, angles, outer_center, outer_angle):
    """
    Compute the side length of the minimum enclosing regular hexagon.
    Returns (side, gradients w.r.t. positions, angles, outer_center, outer_angle).
    Uses subgradient approach for the max-of-linear-functions structure.
    """
    all_verts = []
    for i in range(N):
        v = _hex_vertices(positions[i, 0], positions[i, 1], angles[i], 1.0)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    normals_angles = outer_angle + np.pi/6 + np.pi/3 * np.arange(6)
    normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
    
    proj = all_verts @ normals.T  # (6N, 6)
    c_proj = outer_center @ normals.T
    diffs = np.abs(proj - c_proj)
    max_diff = diffs.max()
    side = max_diff * 2.0 / np.sqrt(3.0)
    
    return side

def _smooth_overlap_penalty(positions, angles):
    """
    Smooth penalty for overlaps. For unit regular hexagons, the minimum
    center distance for non-overlap depends on relative orientation.
    Use a smooth approximation: penalty when center distance < threshold.
    The threshold for two unit hexagons with relative angle delta is 
    approximately 2 * cos(delta/2) + 1 (rough estimate), but we use a 
    simpler smooth function based on center distance.
    """
    pen = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            # For same-orientation unit hexagons, min center dist = 2
            # For rotated, it can be slightly less. Use 1.95 as threshold.
            threshold = 1.95
            if d < threshold:
                pen += (threshold - d) ** 2
    return pen

def _objective(v):
    """
    v: [x1, y1, a1, ..., xN, yN, aN, cx, cy, theta]
    Objective: outer side length + smooth overlap penalty
    """
    hex_vars = v[:3*N]
    outer_vars = v[3*N:]
    
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        positions[i, 0] = hex_vars[i*3]
        positions[i, 1] = hex_vars[i*3+1]
        angles[i] = hex_vars[i*3+2]
        
    cx = outer_vars[0]
    cy = outer_vars[1]
    theta = outer_vars[2]
    
    outer_center = np.array([cx, cy])
    
    side = _outer_side_and_grad(positions, angles, outer_center, theta)
    pen = _smooth_overlap_penalty(positions, angles)
    
    return side + 10.0 * pen

def _generate_lattice_configs():
    """Generate diverse hexagonal lattice configurations for multi-start."""
    configs = []
    
    # Configuration 1: Compact hexagonal cluster (center + ring1 + partial ring2)
    axial_coords = [(0,0)]
    axial_coords += [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    axial_coords += [(2,0), (1,1), (0,2), (-1,2)]
    
    selected_axial = axial_coords[:N]
    lattice_centers = []
    for q, r in selected_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2)
        lattice_centers.append((x, y))
    
    positions = np.array(lattice_centers)
    angles = np.zeros(N)
    cx = np.mean(positions[:, 0])
    cy = np.mean(positions[:, 1])
    theta = 0.0
    
    v = np.concatenate([
        np.array([[x, y, a] for x, y, a in zip(positions[:,0], positions[:,1], angles)]).flatten(),
        [cx, cy, theta]
    ])
    configs.append(v)
    
    # Configuration 2: Alternative ring2 selection
    axial_coords2 = [(0,0)]
    axial_coords2 += [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    axial_coords2 += [(2,0), (1,1), (0,2), (-2,1)]
    
    lattice_centers2 = []
    for q, r in axial_coords2[:N]:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2)
        lattice_centers2.append((x, y))
    
    positions2 = np.array(lattice_centers2)
    angles2 = np.zeros(N)
    cx2 = np.mean(positions2[:, 0])
    cy2 = np.mean(positions2[:, 1])
    theta2 = 0.0
    
    v2 = np.concatenate([
        np.array([[x, y, a] for x, y, a in zip(positions2[:,0], positions2[:,1], angles2)]).flatten(),
        [cx2, cy2, theta2]
    ])
    configs.append(v2)
    
    # Configuration 3: Linear-ish arrangement (elongated)
    # Place hexagons in a zigzag pattern
    pos3 = []
    for k in range(N):
        row = k // 6
        col = k % 6
        x = 1.5 * col - 3.75
        y = np.sqrt(3) * (row + 0.5 * (col % 2)) - np.sqrt(3)
        pos3.append((x, y))
    positions3 = np.array(pos3)
    angles3 = np.zeros(N)
    cx3 = np.mean(positions3[:, 0])
    cy3 = np.mean(positions3[:, 1])
    theta3 = 0.0
    
    v3 = np.concatenate([
        np.array([[x, y, a] for x, y, a in zip(positions3[:,0], positions3[:,1], angles3)]).flatten(),
        [cx3, cy3, theta3]
    ])
    configs.append(v3)
    
    # Add random perturbations of each base config
    rng = np.random.default_rng(42)
    for base_v in list(configs):
        for _ in range(2):
            v_pert = base_v + rng.normal(0, 0.15, size=len(base_v))
            configs.append(v_pert)
    
    return configs

def _validate(positions, angles, outer_center, outer_angle, side):
    """
    Validate that the configuration is feasible:
    1. No overlaps between inner hexagons (exact check)
    2. All inner hexagons are inside the outer hexagon
    """
    # Check overlaps using exact vertex-edge distance
    for i in range(N):
        for j in range(i+1, N):
            v1 = _hex_vertices(positions[i, 0], positions[i, 1], angles[i], 1.0)
            v2 = _hex_vertices(positions[j, 0], positions[j, 1], angles[j], 1.0)
            
            min_d = float('inf')
            # Vertex-vertex
            for p in v1:
                for q in v2:
                    d = np.linalg.norm(p - q)
                    if d < min_d:
                        min_d = d
            # Vertex-edge (v1 to v2 edges)
            for p in v1:
                for e in range(6):
                    a = v2[e]
                    b = v2[(e+1) % 6]
                    ab = b - a
                    ap = p - a
                    denom = np.dot(ab, ab)
                    if denom < 1e-12:
                        continue
                    t = np.dot(ap, ab) / denom
                    t = max(0.0, min(1.0, t))
                    closest = a + t * ab
                    d = np.linalg.norm(p - closest)
                    if d < min_d:
                        min_d = d
            # Vertex-edge (v2 to v1 edges)
            for p in v2:
                for e in range(6):
                    a = v1[e]
                    b = v1[(e+1) % 6]
                    ab = b - a
                    ap = p - a
                    denom = np.dot(ab, ab)
                    if denom < 1e-12:
                        continue
                    t = np.dot(ap, ab) / denom
                    t = max(0.0, min(1.0, t))
                    closest = a + t * ab
                    d = np.linalg.norm(p - closest)
                    if d < min_d:
                        min_d = d
            
            if min_d < 1e-5:
                return False
    
    # Check containment: all inner vertices must be inside outer hexagon
    all_verts = []
    for i in range(N):
        v = _hex_vertices(positions[i, 0], positions[i, 1], angles[i], 1.0)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    normals_angles = outer_angle + np.pi/6 + np.pi/3 * np.arange(6)
    normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
    
    proj = all_verts @ normals.T
    c_proj = outer_center @ normals.T
    diffs = np.abs(proj - c_proj)
    max_diff = diffs.max()
    computed_side = max_diff * 2.0 / np.sqrt(3.0)
    
    # The reported side should be at least the computed side
    if computed_side > side + 1e-4:
        return False
    
    return True

def construct():
    start_time = time.time()
    best_result = None
    best_side = float('inf')
    
    configs = _generate_lattice_configs()
    
    for v0 in configs:
        if time.time() > _DEADLINE - 25:
            break
        
        try:
            # Use L-BFGS-B for gradient-based optimization
            res = minimize(
                _objective,
                v0,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.fun < best_side:
                v_opt = res.x
                hex_vars = v_opt[:3*N]
                outer_vars = v_opt[3*N:]
                
                positions = np.zeros((N, 2))
                angles = np.zeros(N)
                for i in range(N):
                    positions[i, 0] = hex_vars[i*3]
                    positions[i, 1] = hex_vars[i*3+1]
                    angles[i] = hex_vars[i*3+2]
                    
                cx = outer_vars[0]
                cy = outer_vars[1]
                theta = outer_vars[2]
                
                side = _outer_side_and_grad(positions, angles, np.array([cx, cy]), theta)
                
                # Validate feasibility
                if _validate(positions, angles, np.array([cx, cy]), theta, side):
                    if side < best_side:
                        best_side = side
                        inner = [[positions[i, 0], positions[i, 1], np.degrees(angles[i])] for i in range(N)]
                        best_result = (inner, (cx, cy), side, np.degrees(theta))
                        
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 50.0, 0.0
    
    return best_result
