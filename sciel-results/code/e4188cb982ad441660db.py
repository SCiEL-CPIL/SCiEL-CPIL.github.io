import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0
SQRT3 = math.sqrt(3)

def _hex_vertices(angle_rad):
    """Vertices of a unit regular hexagon rotated by angle_rad."""
    angles = angle_rad + np.pi / 3 * np.arange(6)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _minkowski_support_distance(p1, a1, p2, a2):
    """
    Compute the minimum center-to-center distance for two unit hexagons
    with positions p1, p2 and orientations a1, a2 (radians) to be non-overlapping.
    
    Uses Minkowski sum: the Minkowski sum H1 ⊕ (-H2) is a convex polygon.
    The minimum distance is the minimum over all edge normals of the support function.
    """
    # Vertices of hexagon 1 and hexagon 2 (negated for Minkowski with -H2)
    v1 = _hex_vertices(a1)  # shape (6, 2)
    v2 = -_hex_vertices(a2)  # shape (6, 2), negated for Minkowski sum H1 ⊕ (-H2)
    
    # Compute all 36 edge normals of the Minkowski sum
    # The Minkowski sum's edges come from edges of H1 and edges of -H2
    # For each edge, compute its outward normal
    
    # Edges of H1: v1[i+1] - v1[i]
    edges1 = np.roll(v1, -1, axis=0) - v1
    # Edges of -H2: v2[i+1] - v2[i]
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    # Outward normals (perpendicular, pointing outward)
    # For a convex polygon with CCW vertices, outward normal of edge (dx, dy) is (dy, -dx)
    # But we need to be careful about orientation
    def outward_normals(edges):
        normals = np.zeros_like(edges)
        for i in range(len(edges)):
            dx, dy = edges[i]
            # Outward normal: rotate edge by -90 degrees
            normals[i] = [dy, -dx]
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return normals / norms
    
    n1 = outward_normals(edges1)
    n2 = outward_normals(edges2)
    
    # All normals of the Minkowski sum
    all_normals = np.vstack([n1, n2])  # shape (12, 2)
    
    # Support function: h(n) = max_i (n · v1[i]) + max_j (n · v2[j])
    # The minimum distance is min over all normals n of h(n)
    # But we need the actual minimum center distance, which is:
    # min_{n in normals} [max_i(n·v1[i]) + max_j(n·v2[j])]
    # This is the support function of the Minkowski sum evaluated at n
    
    # Compute for each normal
    min_dist = float('inf')
    for n in all_normals:
        # Support of H1 in direction n
        s1 = np.max(v1 @ n)
        # Support of -H2 in direction n (which is -min of H2 support in direction n)
        s2 = np.max(v2 @ n)
        h = s1 + s2
        if h < min_dist:
            min_dist = h
    
    return min_dist

def _collision_penalty(p1, a1, p2, a2, margin=0.0):
    """
    Compute a smooth collision penalty. If the hexagons overlap,
    return a positive penalty; otherwise return 0.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dist = math.sqrt(dx*dx + dy*dy)
    
    min_dist = _minkowski_support_distance(p1, a1, p2, a2)
    
    # Required distance
    required = min_dist + margin
    
    if dist >= required:
        return 0.0
    else:
        # Smooth penalty: quadratic
        violation = required - dist
        return violation * violation

def _outer_hex_side(positions, angles_deg):
    """
    Compute the minimum side length of the outer regular hexagon
    that contains all inner hexagons, and the optimal rotation.
    """
    angles_rad = np.radians(angles_deg)
    
    # Get all vertices of all hexagons
    all_verts = []
    for i in range(N):
        verts = _hex_vertices(angles_rad[i]) + positions[i]
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # shape (72, 2)
    
    # Center of mass
    center = np.mean(all_verts, axis=0)
    offsets = all_verts - center
    
    def eval_side(phi_deg):
        phi = np.radians(phi_deg)
        # 6 directions for the outer hexagon edges
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / SQRT3) * max_projs
        return np.max(side_vals)
    
    # Coarse scan
    best_side = float('inf')
    best_phi = 0.0
    for phi_deg in range(0, 60, 2):
        s = eval_side(phi_deg)
        if s < best_side:
            best_side = s
            best_phi = phi_deg
    
    # Fine tune
    try:
        lo = max(0.0, best_phi - 2.0)
        hi = min(60.0, best_phi + 2.0)
        res = minimize_scalar_wrapper(eval_side, lo, hi)
        if res < best_side:
            best_side = res
    except Exception:
        pass
    
    return best_side, best_phi, center[0], center[1]

def minimize_scalar_wrapper(func, lo, hi):
    """Simple bounded minimization using golden section search."""
    gr = (math.sqrt(5) - 1) / 2  # golden ratio
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = func(c)
    fd = func(d)
    
    for _ in range(50):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = func(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = func(d)
    
    return min(fc, fd)

def _is_valid(positions, angles_deg, tolerance=1e-6):
    """Check if configuration is valid (no overlaps)."""
    angles_rad = np.radians(angles_deg)
    for i in range(N):
        for j in range(i + 1, N):
            p1 = positions[i]
            p2 = positions[j]
            a1 = angles_rad[i]
            a2 = angles_rad[j]
            
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            min_dist = _minkowski_support_distance(p1, a1, p2, a2)
            
            if dist < min_dist - tolerance:
                return False
    return True

def construct():
    t0 = time.time()
    
    # Honeycomb initial configuration
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    base_angles = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2
        base_angles[idx] = 0.0
    
    best_side = float('inf')
    best_positions = base_positions.copy()
    best_angles = base_angles.copy()
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Evaluate base
    base_side, base_phi, base_cx, base_cy = _outer_hex_side(base_positions, base_angles)
    if base_side < best_side:
        best_side = base_side
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_phi = base_phi
        best_cx = base_cx
        best_cy = base_cy
    
    # L-BFGS-B optimization with multiple random initializations
    def objective(params):
        """
        Objective: outer hexagon side length + collision penalty.
        params: 3*N array [x0, y0, a0, x1, y1, a1, ...]
        """
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3*i]
            positions[i, 1] = params[3*i+1]
            angles[i] = params[3*i+2]
        
        # Collision penalty
        penalty = 0.0
        angles_rad = np.radians(angles)
        for i in range(N):
            for j in range(i + 1, N):
                p1 = positions[i]
                p2 = positions[j]
                a1 = angles_rad[i]
                a2 = angles_rad[j]
                
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                min_dist = _minkowski_support_distance(p1, a1, p2, a2)
                required = min_dist + 1e-8
                
                if dist < required:
                    violation = required - dist
                    penalty += violation * violation
        
        # Outer hex side
        side, _, _, _ = _outer_hex_side(positions, angles)
        
        return side + 100.0 * penalty
    
    # Run L-BFGS-B from multiple initializations
    n_initializations = 8
    time_per_init = (DEADLINE - (time.time() - t0)) / n_initializations
    
    for init_idx in range(n_initializations):
        if time.time() - t0 > DEADLINE - 5.0:
            break
        
        # Generate random initialization
        if init_idx == 0:
            init_positions = base_positions.copy()
            init_angles = base_angles.copy()
        else:
            init_positions = base_positions + np.random.randn(N, 2) * 0.5
            init_angles = base_angles + np.random.randn(N) * 10.0
        
        init_params = np.zeros(3 * N)
        for i in range(N):
            init_params[3*i] = init_positions[i, 0]
            init_params[3*i+1] = init_positions[i, 1]
            init_params[3*i+2] = init_angles[i]
        
        try:
            res = minimize(objective, init_params, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if res.fun < best_side:
                new_positions = np.zeros((N, 2))
                new_angles = np.zeros(N)
                for i in range(N):
                    new_positions[i, 0] = res.x[3*i]
                    new_positions[i, 1] = res.x[3*i+1]
                    new_angles[i] = res.x[3*i+2]
                
                if _is_valid(new_positions, new_angles):
                    new_side, new_phi, new_cx, new_cy = _outer_hex_side(new_positions, new_angles)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_positions
                        best_angles = new_angles
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
        except Exception:
            pass
    
    # Final verification
    if not _is_valid(best_positions, best_angles):
        best_positions = base_positions.copy()
        best_angles = base_angles.copy()
        best_side, best_phi, best_cx, best_cy = _outer_hex_side(best_positions, best_angles)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
    
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
