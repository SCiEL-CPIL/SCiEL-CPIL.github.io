import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 145.0  # Safety margin
SQRT3 = math.sqrt(3.0)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def _hex_edge_normals(angle_deg):
    """Return 6 outward normal vectors of a regular hexagon."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals

def _hex_min_distance(c1, a1, c2, a2):
    """
    Compute the minimum distance between two unit hexagons (0 if overlapping).
    Uses SAT to find the minimum separation.
    """
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = math.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    min_sep = float('inf')
    
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep > min_sep:
            min_sep = sep
    
    return max(0.0, min_sep)

def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check containment and non-overlap exactly."""
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * SQRT3 / 2
    
    # Check containment
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    # Check overlap
    for i in range(N):
        for j in range(i + 1, N):
            dist = _hex_min_distance(centers[i], inner_angles[i], centers[j], inner_angles[j])
            if dist < -tol: # dist is 0 if overlapping, so < -tol means deep overlap
                # Actually _hex_min_distance returns 0 if overlapping. 
                # So if dist == 0, they touch or overlap.
                # We need to distinguish touching from overlapping?
                # The task says "disjoint". Usually touching is allowed or not?
                # "Place disjoint regular hexagons". Disjoint usually means no common interior points.
                # Touching boundaries is typically allowed in packing problems unless "strictly disjoint" is specified.
                # However, _hex_min_distance returns 0 for both touching and overlapping.
                # To be safe, we should check if they actually overlap interiors.
                # If dist == 0, check if vertices are inside each other?
                # Or just trust the optimizer to keep them separated by a tiny epsilon.
                pass
    
    # A more robust overlap check:
    # If min_distance is 0, check if any vertex of one is strictly inside the other.
    # But for the purpose of the objective function, we penalize dist < 0? 
    # No, _hex_min_distance returns 0 for overlap. It doesn't return negative.
    # So we need a function that returns negative for overlap?
    # Let's modify _hex_min_distance to return the actual signed distance?
    # SAT gives the maximum separation. If max separation < 0, they overlap.
    # My current _hex_min_distance returns max(0, min_sep).
    # So if they overlap, min_sep is negative, and it returns 0.
    # This is not differentiable at 0.
    # For the smooth objective, I want a penalty that is 0 when separated and positive when overlapping.
    # I can use a soft penalty: penalty = max(0, -min_sep_signed)^2.
    # Let's create a signed distance function.
    
    def _signed_min_distance(c1, a1, c2, a2):
        v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
        v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
        
        def get_axes(verts):
            axes = []
            n = len(verts)
            for i in range(n):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % n]
                ex = x2 - x1
                ey = y2 - y1
                nx, ny = -ey, ex
                norm = math.sqrt(nx**2 + ny**2)
                if norm > 1e-12:
                    axes.append((nx / norm, ny / norm))
            return axes
        
        axes = get_axes(v1) + get_axes(v2)
        min_sep = float('inf')
        
        for ax, ay in axes:
            proj1 = v1[:, 0] * ax + v1[:, 1] * ay
            proj2 = v2[:, 0] * ax + v2[:, 1] * ay
            sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
            if sep > min_sep:
                min_sep = sep
        
        return min_sep # Negative if overlapping, 0 if touching, positive if separated

    for i in range(N):
        for j in range(i + 1, N):
            dist = _signed_min_distance(centers[i], inner_angles[i], centers[j], inner_angles[j])
            if dist < -tol:
                return False
                
    return True

def _smooth_objective(x, beta=100.0, lambda_overlap=100.0):
    """
    Smooth objective function for L-BFGS-B.
    x: [c1x, c1y, a1, ..., cNx, cNy, aN, Cx, Cy, theta]
    """
    # Unpack
    c_flat = x[:2*N]
    angles = x[2*N:3*N]
    Cx = x[3*N]
    Cy = x[3*N+1]
    theta_deg = x[3*N+2]
    
    # Compute inner vertices
    # Shape: (N, 6, 2)
    all_verts = np.zeros((N, 6, 2))
    for i in range(N):
        all_verts[i] = _hex_vertices(c_flat[2*i], c_flat[2*i+1], 1.0, angles[i])
        
    # Flatten vertices for matrix ops: (N*6, 2)
    verts_flat = all_verts.reshape(-1, 2)
    
    # Outer normals
    outer_normals = _hex_edge_normals(theta_deg) # (6, 2)
    
    # Compute projections d_ijk
    # d_ijk = n_k . (V_ij - C)
    # V_ij is (N*6, 2), n_k is (6, 2)
    # We want (N*6, 6) matrix of projections
    # proj[i, k] = sum_j V[i, j] * n[k, j] - (Cx*n[k,0] + Cy*n[k,1])
    
    # V @ n.T -> (N*6, 6)
    proj = verts_flat @ outer_normals.T
    
    # Subtract center projection
    center_proj = outer_normals @ np.array([Cx, Cy]) # (6,)
    proj -= center_proj
    
    # Now proj[i, k] is the projection of vertex i onto normal k.
    # We need max over k for each vertex i.
    # M_i = max_k proj[i, k]
    
    # Smooth max using Log-Sum-Exp
    # M_i = (1/beta) * log( sum_k exp(beta * proj[i, k]) )
    
    # To avoid overflow, subtract max
    proj_max = np.max(proj, axis=1)
    exp_proj = np.exp(beta * (proj - proj_max[:, np.newaxis]))
    sum_exp = np.sum(exp_proj, axis=1)
    M = (1.0 / beta) * np.log(sum_exp) + proj_max
    
    # Now M is (N*6,). We need max over all vertices.
    # R = max_i M_i
    # Smooth max again
    M_max = np.max(M)
    exp_M = np.exp(beta * (M - M_max))
    sum_exp_M = np.sum(exp_M)
    R = (1.0 / beta) * np.log(sum_exp_M) + M_max
    
    side = 2.0 * R / SQRT3
    
    # Overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            c1 = np.array([c_flat[2*i], c_flat[2*i+1]])
            c2 = np.array([c_flat[2*j], c_flat[2*j+1]])
            # Use signed distance
            # We need a vectorized or fast signed distance
            # For now, call the python function. It's 55 calls, which is okay.
            dist = _signed_min_distance(c1, angles[i], c2, angles[j])
            if dist < 0:
                penalty += lambda_overlap * (dist ** 2)
                
    return side + penalty

def _signed_min_distance(c1, a1, c2, a2):
    """
    Compute the signed minimum distance between two unit hexagons.
    Positive if separated, 0 if touching, negative if overlapping.
    """
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = math.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    min_sep = float('inf')
    
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep > min_sep:
            min_sep = sep
            
    return min_sep

def _generate_initial_configs():
    configs = []
    
    # 1. Hexagonal Lattice
    d1 = 2 * SQRT3
    first_ring = [(d1 * np.cos(np.pi/3*i), d1 * np.sin(np.pi/3*i)) for i in range(6)]
    centers = [(0.0, 0.0)] + first_ring
    # Add 4 more from second ring
    d2 = 4 * SQRT3
    for i in [0, 2, 4, 6]:
        a = np.pi/3*i
        centers.append((d2 * np.cos(a), d2 * np.sin(a)))
        
    angles = [0.0] * N
    Cx, Cy = 0.0, 0.0
    theta = 0.0
    x0 = np.concatenate([np.array(centers).flatten(), angles, [Cx, Cy, theta]])
    configs.append(x0)
    
    # 2. Random perturbations around lattice
    np.random.seed(42)
    for _ in range(5):
        perturbed_centers = []
        for cx, cy in centers:
            perturbed_centers.append([cx + np.random.normal(0, 0.5), cy + np.random.normal(0, 0.5)])
        angles_pert = [np.random.normal(0, 10.0) for _ in range(N)]
        x0 = np.concatenate([np.array(perturbed_centers).flatten(), angles_pert, [0.0, 0.0, 0.0]])
        configs.append(x0)
        
    # 3. Different outer angles
    for theta in [15, 30, 45]:
        x0 = np.concatenate([np.array(centers).flatten(), [0.0]*N, [0.0, 0.0, theta]])
        configs.append(x0)
        
    return configs

def construct():
    start_time = time.time()
    best_side = float('inf')
    best_result = None
    
    initial_configs = _generate_initial_configs()
    
    for i, x0 in enumerate(initial_configs):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        try:
            res = minimize(_smooth_objective, x0, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 2000})
            
            if res.success or np.isfinite(res.fun):
                x_opt = res.x
                c_flat = x_opt[:2*N]
                angles = x_opt[2*N:3*N]
                Cx = x_opt[3*N]
                Cy = x_opt[3*N+1]
                theta = x_opt[3*N+2]
                
                centers = [(c_flat[2*i], c_flat[2*i+1]) for i in range(N)]
                
                # Calculate exact side for this configuration
                # We need to find the min enclosing side exactly for these positions
                # We can use the LP method or just compute the max projection
                outer_normals = _hex_edge_normals(theta)
                all_verts = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(N)])
                proj = all_verts @ outer_normals.T
                center_proj = outer_normals @ np.array([Cx, Cy])
                proj -= center_proj
                R = np.max(proj)
                side = 2.0 * R / SQRT3
                
                # Check validity
                if _check_validity(centers, angles, (Cx, Cy), side, theta):
                    if side < best_side:
                        best_side = side
                        best_result = (centers, angles, (Cx, Cy), side, theta)
                        
        except Exception as e:
            pass
            
    # Fallback if no valid result found
    if best_result is None:
        # Simple linear arrangement
        centers = [(i * 4.0, 0.0) for i in range(N)]
        angles = [0.0] * N
        Cx = (N-1) * 2.0
        Cy = 0.0
        theta = 0.0
        
        outer_normals = _hex_edge_normals(theta)
        all_verts = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(N)])
        proj = all_verts @ outer_normals.T
        center_proj = outer_normals @ np.array([Cx, Cy])
        proj -= center_proj
        R = np.max(proj)
        side = 2.0 * R / SQRT3
        
        best_result = (centers, angles, (Cx, Cy), side, theta)
        
    centers, angles, outer_center, side, theta = best_result
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(side), float(theta)

