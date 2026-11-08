import numpy as np
from scipy.optimize import linprog
import time

N = 11

def _get_hex_vertices(center, angle_deg):
    """Get vertices of a unit hexagon."""
    cx, cy = center
    angles = np.deg2rad(np.arange(0, 360, 60) + angle_deg)
    xs = cx + np.cos(angles)
    ys = cy + np.sin(angles)
    return np.column_stack((xs, ys))

def _solve_lp_for_theta(vertices, theta):
    """
    Solve LP for minimal S given theta.
    Returns S, Cx, Cy.
    """
    # Normals for the 6 sides of the outer hexagon (rotated by theta)
    # The sides are perpendicular to the normals.
    # Normal angles: theta, theta+60, ..., theta+300
    normal_angles = np.deg2rad(np.arange(0, 360, 60) + np.degrees(theta))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    # a = S * sqrt(3) / 2
    # Constraint: |(v_i - C) . n_k| <= S * sqrt(3) / 2
    # Let a = sqrt(3)/2 * S. We minimize S, so we minimize 2*a/sqrt(3).
    # Variables: Cx, Cy, a
    # Minimize: 2/sqrt(3) * a
    c = np.array([0, 0, 2.0/np.sqrt(3)])
    
    # Inequalities: A_ub x <= b_ub
    # (v_i - C) . n_k <= a
    # v_i . n_k - C . n_k <= a
    # -n_kx Cx - n_ky Cy - a <= -v_i . n_k
    # (v_i - C) . n_k >= -a
    # v_i . n_k - C . n_k >= -a
    # n_kx Cx + n_ky Cy - a <= v_i . n_k
    
    # Total constraints: 2 * N_vertices * 6_normals? No.
    # For each vertex i and each normal k, we have 2 constraints.
    # Total vertices = 6 * N
    # Total constraints = 2 * 6 * N * 6 = 72 * N
    
    A_rows = []
    b_rows = []
    
    for i in range(vertices.shape[0]):
        vi = vertices[i]
        for k in range(6):
            nk = normals[k]
            dot_vn = np.dot(vi, nk)
            
            # 1. dot_vn - C.nk <= a  =>  -nkx Cx - nky Cy - a <= -dot_vn
            A_rows.append([-nk[0], -nk[1], -1.0])
            b_rows.append(-dot_vn)
            
            # 2. dot_vn - C.nk >= -a =>  nkx Cx + nky Cy - a <= dot_vn
            A_rows.append([nk[0], nk[1], -1.0])
            b_rows.append(dot_vn)
            
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: Cx, Cy can be anything. a >= 0.
    bounds = [(None, None), (None, None), (0, None)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        Cx, Cy, a = res.x
        S = 2.0 * a / np.sqrt(3)
        return S, Cx, Cy
    else:
        return float('inf'), 0, 0

def construct():
    start_time = time.time()
    
    # Lattice vectors
    e1 = np.array([np.sqrt(3), 0.0])
    e2 = np.array([np.sqrt(3)/2, 1.5])
    
    def lattice_to_cartesian(q, r):
        return q * e1 + r * e2

    # Valid cluster of 11 hexagons
    # Center + Ring 1 + 4 selected from Ring 2
    cluster = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,-2), (-1,-2), (-2,0)
    ]
    
    # Scale to ensure strict disjointness (gap)
    scale = 1.01
    centers = [lattice_to_cartesian(q, r) * scale for q, r in cluster]
    angles = [0.0] * 11
    
    # Generate all vertices
    all_vertices = []
    for c, a in zip(centers, angles):
        verts = _get_hex_vertices(c, a)
        all_vertices.append(verts)
    vertices = np.vstack(all_vertices)
    
    best_side = float('inf')
    best_C = (0,0)
    best_theta = 0.0
    
    # Search for optimal theta
    # The optimal theta is likely aligned with the cluster symmetry or 0.
    # We scan 0 to 60 degrees.
    # Step size 1 degree is ~60 LPs. Fast.
    for theta_deg in range(0, 61, 1):
        if time.time() - start_time > 140:
            break
        theta_rad = np.radians(theta_deg)
        S, Cx, Cy = _solve_lp_for_theta(vertices, theta_rad)
        if S < best_side:
            best_side = S
            best_C = (Cx, Cy)
            best_theta = theta_rad
            
    # Refine around best theta
    if best_theta > 0 and best_theta < np.pi/3:
        for delta in np.arange(-1, 1.01, 0.5):
             if time.time() - start_time > 145:
                 break
             theta_rad = best_theta + np.radians(delta)
             S, Cx, Cy = _solve_lp_for_theta(vertices, theta_rad)
             if S < best_side:
                 best_side = S
                 best_C = (Cx, Cy)
                 best_theta = theta_rad

    if best_side == float('inf'):
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for c, a in zip(centers, angles):
        inner.append([c[0], c[1], a])
        
    return inner, best_C, best_side, np.degrees(best_theta)
