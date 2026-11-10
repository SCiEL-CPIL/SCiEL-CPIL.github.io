# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _compute_all_vertices(centers, angles_deg, side=1.0):
    n = len(centers)
    verts = np.empty((n * 6, 2))
    for i in range(n):
        verts[i*6:(i+1)*6] = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
    return verts

def _min_outer_side(vertices_all):
    # Normals to the sides of the outer hexagon
    # Outer hexagon vertices at angles 0, 60, ...
    # Sides are perpendicular to directions 30, 90, ...
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = vertices_all @ m.T
    # s must satisfy |dot| <= sqrt(3)/2 * s
    # So s >= 2/sqrt(3) * max(|dot|)
    max_abs_dot = np.max(np.abs(dots))
    s_min = (2.0 / np.sqrt(3.0)) * max_abs_dot
    return s_min

def _min_outer_side_smooth(vertices_all, beta=100.0):
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = vertices_all @ m.T
    abs_dots = np.abs(dots)
    # Soft max of abs_dots
    # max(abs_dots) approx log(sum(exp(beta * abs_dots)) / N) / beta
    max_val = np.max(abs_dots)
    if max_val == 0:
        return 0.0
    shifted = abs_dots - max_val
    log_sum_exp = max_val + np.log(np.sum(np.exp(beta * shifted)) / abs_dots.size) / beta
    s_min = (2.0 / np.sqrt(3.0)) * log_sum_exp
    return s_min

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    n = len(centers)
    vertices_list = []
    normals_list = []
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        vertices_list.append(verts)
        normals = []
        for e in range(6):
            p1 = verts[e]
            p2 = verts[(e+1) % 6]
            edge = p2 - p1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append(np.array([nx/norm, ny/norm]))
        normals_list.append(normals)
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            overlap = True
            for hex_idx in range(2):
                verts = [v1, v2][hex_idx]
                normals = normals_list[hex_idx]
                for normal in normals:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 + tol < min2 or max2 + tol < min1:
                        overlap = False
                        break
                if not overlap:
                    break
            if overlap:
                return False
    return True

def _check_inside(centers, angles_deg, s, side=1.0, tol=1e-6):
    # Check if all vertices are inside the outer hexagon of side s
    # Outer hexagon centered at (0,0) with angle 0
    vertices_all = _compute_all_vertices(centers, angles_deg, side)
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = vertices_all @ m.T
    max_dot = np.max(np.abs(dots))
    required_s = (2.0 / np.sqrt(3.0)) * max_dot
    return required_s <= s + tol

def _initial_honeycomb_12():
    # Center
    positions = np.array([[0.0, 0.0]])
    angles = np.array([0.0])
    
    d = np.sqrt(3.0)
    # 6 neighbors
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions = np.vstack([positions, [[x, y]]])
        angles = np.append(angles, 0.0)
    
    # 5 more from the next ring
    # The next ring has 12 positions.
    # We pick 5 to form a compact shape.
    # Let's pick the ones at angles 0, 60, 120, 180, 240? No, that's symmetric.
    # Let's pick 5 consecutive ones? Or spread out?
    # A known good cluster for 12 is 1-6-5 where the 5 are arranged to minimize bounding box.
    # Let's try 5 of the 12 in the 3rd ring.
    # The 12 positions in the 3rd ring are at distance 2*sqrt(3) from center?
    # No, the distance from center to center of hex in layer 2 is 2*sqrt(3) only for the "corners" of the hexagonal cluster?
    # In a hexagonal lattice, the distance from center to a hex at (i,j) in axial coordinates is sqrt(3)*sqrt(i^2 + i*j + j^2).
    # Layer 1: (1,0), (0,1), etc. dist sqrt(3).
    # Layer 2: (2,0), (1,1), (0,2), etc.
    # (2,0) dist 2*sqrt(3).
    # (1,1) dist sqrt(3)*sqrt(1+1+1) = 3.
    # So the 3rd ring (layer 2) has 6 "corners" at dist 2*sqrt(3) and 6 "edges" at dist 3.
    # To minimize bounding box, we should probably use the "corners" or a mix.
    # Let's just take 5 of the 6 "corners" (dist 2*sqrt(3)) to maintain symmetry-ish?
    # Or 5 of the 12.
    # Let's try 5 of the 6 corners of the hexagon of side 2.
    # Corners are at angles 0, 60, 120, 180, 240, 300.
    # Let's pick 0, 60, 120, 180, 240.
    for k in range(5):
        angle = k * np.pi / 3
        x = 2.0 * d * np.cos(angle)
        y = 2.0 * d * np.sin(angle)
        positions = np.vstack([positions, [[x, y]]])
        angles = np.append(angles, 0.0)
        
    return positions, angles

def _objective_smooth_24(params, beta=100.0):
    # params is 24 coordinates (12 x 2)
    centers = params.reshape(12, 2)
    angles_deg = np.zeros(12)
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side_smooth(all_verts, beta)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 50.0
    
    return s

def _objective_smooth_36(params, beta=100.0):
    # params is 36 coordinates (12 x 3)
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    s = _min_outer_side_smooth(all_verts, beta)
    
    if not _check_nonoverlap(centers, angles_deg):
        return s + 50.0
    
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590.0
    
    np.random.seed(42)
    random.seed(42)
    
    best_s = float('inf')
    best_params = None
    
    # Phase 1: Optimize 24 coordinates (fixed angles)
    # Generate a few initial guesses
    inits = []
    
    # 1. Standard honeycomb 1-6-5 (corners)
    c1, a1 = _initial_honeycomb_12()
    inits.append(np.hstack([c1, np.zeros((12,1))]).flatten())
    
    # 2. Perturbed honeycomb
    c2, a2 = _initial_honeycomb_12()
    c2 += np.random.randn(12, 2) * 0.1
    inits.append(np.hstack([c2, np.zeros((12,1))]).flatten())
    
    # 3. Another variant: 5 edge hexes instead of corners?
    c3 = np.array([[0.0, 0.0]])
    a3 = np.array([0.0])
    d = np.sqrt(3.0)
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        c3 = np.vstack([c3, [[x, y]]])
        a3 = np.append(a3, 0.0)
    # 5 edge hexes (dist 3)
    for k in range(5):
        angle = (k * 60 + 30) * np.pi / 180
        x = 3.0 * np.cos(angle)
        y = 3.0 * np.sin(angle)
        c3 = np.vstack([c3, [[x, y]]])
        a3 = np.append(a3, 0.0)
    inits.append(np.hstack([c3, np.zeros((12,1))]).flatten())

    for init in inits:
        if time.time() > deadline - 100.0:
            break
        
        try:
            result = minimize(
                _objective_smooth_36, # Use 36 to allow angle optimization from start, but init has 0 angles
                init,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'maxls': 100, 'ftol': 1e-14, 'gtol': 1e-12},
                tol=1e-14
            )
            
            c = result.x.reshape(12, 3)[:, :2]
            a = result.x.reshape(12, 3)[:, 2]
            
            if _check_nonoverlap(c, a):
                s_exact = _min_outer_side(_compute_all_vertices(c, a))
                if s_exact < best_s:
                    best_s = s_exact
                    best_params = result.x.copy()
        except Exception:
            pass

    # Phase 2: Polish with 36 variables if we have a good candidate
    if best_params is not None:
        # Run a few more optimizations from the best point with different perturbations
        for _ in range(5):
            if time.time() > deadline - 50.0:
                break
            x0 = best_params + np.random.randn(36) * 0.01
            try:
                result = minimize(
                    _objective_smooth_36,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'maxls': 100, 'ftol': 1e-14, 'gtol': 1e-12},
                    tol=1e-14
                )
                c = result.x.reshape(12, 3)[:, :2]
                a = result.x.reshape(12, 3)[:, 2]
                if _check_nonoverlap(c, a):
                    s_exact = _min_outer_side(_compute_all_vertices(c, a))
                    if s_exact < best_s:
                        best_s = s_exact
                        best_params = result.x.copy()
            except Exception:
                pass

    if best_params is None:
        # Fallback to initial honeycomb
        centers, angles_deg = _initial_honeycomb_12()
        best_params = np.column_stack([centers, angles_deg]).flatten()
        all_verts = _compute_all_vertices(centers, angles_deg)
        best_s = _min_outer_side(all_verts)

    centers_final = best_params.reshape(12, 3)[:, :2]
    angles_final = best_params.reshape(12, 3)[:, 2]
    
    # Center the cluster
    centroid = np.mean(centers_final, axis=0)
    centers_final = centers_final - centroid
    
    all_verts_final = _compute_all_vertices(centers_final, angles_final)
    s_final = _min_outer_side(all_verts_final)
    
    if not _check_nonoverlap(centers_final, angles_final):
        # Should not happen, but fallback
        centers_final, angles_final = _initial_honeycomb_12()
        all_verts_final = _compute_all_vertices(centers_final, angles_final)
        s_final = _min_outer_side(all_verts_final)

    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
