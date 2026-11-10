# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _unit_hex_vertices():
    """Vertices of a unit regular hexagon centered at origin, angle 0."""
    angles = np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _get_all_inner_vertices(centers, angles_deg):
    """Get all vertices of all inner hexagons. Returns (72, 2) array."""
    n = len(centers)
    all_verts = np.zeros((n * 6, 2))
    for i in range(n):
        a = np.deg2rad(angles_deg[i])
        angles = a + np.deg2rad(60.0 * np.arange(6))
        all_verts[i*6:(i+1)*6, 0] = centers[i, 0] + np.cos(angles)
        all_verts[i*6:(i+1)*6, 1] = centers[i, 1] + np.sin(angles)
    return all_verts


def _check_separation_fast(centers, angles_deg, tol=1e-6):
    """Fast SAT check for non-overlap of all pairs of unit hexagons."""
    n = len(centers)
    all_verts = _get_all_inner_vertices(centers, angles_deg)
    
    for i in range(n):
        vi = all_verts[i*6:(i+1)*6]
        a_i = np.deg2rad(angles_deg[i])
        for j in range(i + 1, n):
            vj = all_verts[j*6:(j+1)*6]
            a_j = np.deg2rad(angles_deg[j])
            
            overlap = True
            # Check normals of hex i
            for k in range(6):
                n_angle = a_i + np.deg2rad(30.0 + 60.0 * k)
                n = np.array([np.cos(n_angle), np.sin(n_angle)])
                proj_i = vi @ n
                proj_j = vj @ n
                if np.max(proj_i) < np.min(proj_j) - tol or np.max(proj_j) < np.min(proj_i) - tol:
                    overlap = False
                    break
            if not overlap:
                continue
            # Check normals of hex j
            for k in range(6):
                n_angle = a_j + np.deg2rad(30.0 + 60.0 * k)
                n = np.array([np.cos(n_angle), np.sin(n_angle)])
                proj_i = vi @ n
                proj_j = vj @ n
                if np.max(proj_i) < np.min(proj_j) - tol or np.max(proj_j) < np.min(proj_i) - tol:
                    overlap = False
                    break
            if overlap:
                return False
    return True


def _min_containing_hex_side(all_verts, outer_angle_deg):
    """
    Compute minimum side length of a regular hexagon (centered at origin)
    at the given angle that contains all given vertices.
    """
    outer_angle = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Project all vertices onto each normal
    # all_verts: (72, 2), normals: (6, 2)
    projections = all_verts @ normals.T  # (72, 6)
    max_proj = np.max(np.abs(projections), axis=0)  # (6,)
    max_support = np.max(max_proj)
    
    # apothem = s * sqrt(3) / 2, so s = 2 * apothem / sqrt(3)
    s = 2.0 * max_support / np.sqrt(3.0)
    return s


def _compute_s(centers, angles_deg, outer_angle_deg):
    """Compute the minimum containing hexagon side length."""
    all_verts = _get_all_inner_vertices(centers, angles_deg)
    return _min_containing_hex_side(all_verts, outer_angle_deg)


def _hex_lattice_positions():
    """Generate hexagonal lattice positions for 12 unit hexagons."""
    sqrt3 = np.sqrt(3.0)
    # Use proper hexagonal lattice: neighbors at distance 2*sqrt(3)
    positions = []
    # Center
    positions.append([0.0, 0.0])
    # Ring 1: 6 hexagons
    for i in range(6):
        ang = np.deg2rad(60.0 * i)
        positions.append([2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)])
    # Ring 2: 5 hexagons at the "corner" positions (distance 2*sqrt(3) at 30+60k)
    for i in range(5):
        ang = np.deg2rad(30.0 + 60.0 * i)
        positions.append([2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)])
    
    positions = np.array(positions)
    centroid = np.mean(positions, axis=0)
    return positions - centroid


def _circular_positions(n=12, radius=3.0):
    """Generate circular arrangement."""
    angles = np.deg2rad(360.0 * np.arange(n) / n)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _ring_positions():
    """Generate two-ring arrangement."""
    sqrt3 = np.sqrt(3.0)
    positions = [[0.0, 0.0]]
    for i in range(6):
        ang = np.deg2rad(60.0 * i)
        positions.append([2.0 * sqrt3 * np.cos(ang), 2.0 * sqrt3 * np.sin(ang)])
    for i in range(5):
        ang = np.deg2rad(30.0 + 60.0 * i)
        positions.append([4.0 * sqrt3 * np.cos(ang), 4.0 * sqrt3 * np.sin(ang)])
    positions = np.array(positions)
    centroid = np.mean(positions, axis=0)
    return positions - centroid


def _compact_positions():
    """Generate a compact 4x3 grid-like arrangement."""
    sqrt3 = np.sqrt(3.0)
    dx = 2.0 * sqrt3
    dy = 3.0
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * dx
            y = row * dy
            if row % 2 == 1:
                x += dx / 2
            positions.append([x, y])
    positions = np.array(positions[:12])
    centroid = np.mean(positions, axis=0)
    return positions - centroid


def _optimized_lattice():
    """
    Optimized hexagonal packing: 1 center + 6 ring-1 + 5 ring-2.
    Using correct hexagonal lattice spacing.
    """
    sqrt3 = np.sqrt(3.0)
    # For unit hexagons, the center-to-center distance for non-overlapping
    # hexagons on a hexagonal lattice is 2*sqrt(3) ≈ 3.464
    d = 2.0 * sqrt3
    
    positions = [[0.0, 0.0]]
    # Ring 1
    for i in range(6):
        ang = np.deg2rad(60.0 * i)
        positions.append([d * np.cos(ang), d * np.sin(ang)])
    # Ring 2: pick 5 positions that form a compact cluster
    # The 12 ring-2 positions are at:
    #   d at angles 0, 60, 120, 180, 240, 300 (6 positions, same as ring 1 angles)
    #   2*d at angles 30, 90, 150, 210, 270, 330 (6 positions)
    # We use the 6 at 2*d (the "between" positions) and drop one
    for i in range(5):
        ang = np.deg2rad(30.0 + 60.0 * i)
        positions.append([2.0 * d * np.cos(ang), 2.0 * d * np.sin(ang)])
    
    positions = np.array(positions)
    centroid = np.mean(positions, axis=0)
    return positions - centroid


def _objective_de(params):
    """
    Objective function for differential evolution.
    params layout: [x1, y1, ..., x12, y12, outer_angle_deg, outer_cx, outer_cy]
    Total: 24 + 3 = 27 parameters.
    """
    n = 12
    centers = params[:2*n].reshape(n, 2)
    outer_angle_deg = params[2*n]
    outer_cx = params[2*n + 1]
    outer_cy = params[2*n + 2]
    
    angles_deg = np.zeros(n)
    
    # Check separation
    if not _check_separation_fast(centers, angles_deg):
        return 1e6
    
    # Shift centers so outer hexagon is centered at (outer_cx, outer_cy)
    # The inner hexagons should be inside the outer hexagon
    # Outer hexagon vertices are relative to its center
    shifted_centers = centers - np.array([outer_cx, outer_cy])
    all_verts = _get_all_inner_vertices(shifted_centers, angles_deg)
    
    s = _min_containing_hex_side(all_verts, outer_angle_deg)
    
    # Also check that all vertices are inside (the min containing hex should handle this)
    # But let's verify: for the outer hexagon centered at origin with angle outer_angle_deg,
    # all vertices must satisfy the half-plane constraints
    
    outer_angle = np.deg2rad(outer_angle_deg)
    normal_angles = outer_angle + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For each normal, the max projection must be <= apothem = s*sqrt(3)/2
    apothem = s * np.sqrt(3.0) / 2.0
    projections = all_verts @ normals.T  # (72, 6)
    max_proj = np.max(projections, axis=0)  # (6,)
    
    if np.any(max_proj > apothem + 1e-6):
        return 1e6
    
    return s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    n = 12
    best_s = np.inf
    best_centers = None
    best_outer_angle = 0.0
    best_outer_cx = 0.0
    best_outer_cy = 0.0
    
    # Phase 1: Try multiple initial configurations and find best rigid arrangement
    initial_configs = [
        _hex_lattice_positions(),
        _circular_positions(n=12, radius=3.0),
        _ring_positions(),
        _compact_positions(),
        _optimized_lattice(),
    ]
    
    for config in initial_configs:
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # For each config, try different outer angles
        for angle in np.linspace(0.0, 60.0, 61):
            s = _compute_s(config, np.zeros(12), angle)
            if s < best_s:
                best_s = s
                best_centers = config.copy()
                best_outer_angle = angle
                best_outer_cx = 0.0
                best_outer_cy = 0.0
    
    # Phase 2: Differential evolution global optimization
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN:
        try:
            # Bounds for DE
            # Inner centers: roughly within [-10, 10] for each coordinate
            # Outer angle: [0, 60]
            # Outer center: [-5, 5] for each coordinate
            bounds = [(-10.0, 10.0)] * (2 * n) + [(0.0, 60.0), (-5.0, 5.0), (-5.0, 5.0)]
            
            # Use the best initial configuration as a starting point
            # But DE doesn't take a single start; it uses a population
            # We'll run DE with a moderate population size
            
            de_result = differential_evolution(
                _objective_de,
                bounds=bounds,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 0.8),
                recombination=0.8,
                seed=42,
                init='latinhypercube',
                polish=True,
                maxfev=5000,
                update_interval=10,
                callback=lambda x, convergence: (time.time() - _START_TIME > _DEADLINE - _SAFETY_MARGIN)
            )
            
            if de_result.success or de_result.fun < best_s:
                params = de_result.x
                centers = params[:2*n].reshape(n, 2)
                outer_angle_deg = params[2*n]
                outer_cx = params[2*n + 1]
                outer_cy = params[2*n + 2]
                
                # Verify validity
                angles_deg = np.zeros(n)
                if _check_separation_fast(centers, angles_deg):
                    s_cand = de_result.fun
                    if s_cand < best_s and s_cand < 1e5:
                        best_s = s_cand
                        best_centers = centers
                        best_outer_angle = outer_angle_deg
                        best_outer_cx = outer_cx
                        best_outer_cy = outer_cy
        except Exception:
            pass
    
    # Phase 3: Local refinement using Nelder-Mead on the best solution found
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN and best_centers is not None:
        try:
            def _obj_local(params):
                centers = params[:2*n].reshape(n, 2)
                outer_angle_deg = params[2*n]
                outer_cx = params[2*n + 1]
                outer_cy = params[2*n + 2]
                
                angles_deg = np.zeros(n)
                if not _check_separation_fast(centers, angles_deg):
                    return 1e6
                
                shifted_centers = centers - np.array([outer_cx, outer_cy])
                all_verts = _get_all_inner_vertices(shifted_centers, angles_deg)
                s = _min_containing_hex_side(all_verts, outer_angle_deg)
                
                outer_angle = np.deg2rad(outer_angle_deg)
                normal_angles = outer_angle + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
                normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
                apothem = s * np.sqrt(3.0) / 2.0
                projections = all_verts @ normals.T
                max_proj = np.max(projections, axis=0)
                
                if np.any(max_proj > apothem + 1e-6):
                    return 1e6
                
                return s
            
            initial_params = np.concatenate([
                best_centers.flatten(),
                [best_outer_angle, best_outer_cx, best_outer_cy]
            ])
            
            result = minimize(
                _obj_local,
                initial_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 10000,
                    'maxfev': 50000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
            
            if result.fun < best_s and result.fun < 1e5:
                params = result.x
                centers = params[:2*n].reshape(n, 2)
                outer_angle_deg = params[2*n]
                outer_cx = params[2*n + 1]
                outer_cy = params[2*n + 2]
                
                angles_deg = np.zeros(n)
                if _check_separation_fast(centers, angles_deg):
                    best_s = result.fun
                    best_centers = centers
                    best_outer_angle = outer_angle_deg
                    best_outer_cx = outer_cx
                    best_outer_cy = outer_cy
        except Exception:
            pass
    
    # Phase 4: Final verification and fallback
    if best_centers is None:
        # Fallback to a known valid configuration
        best_centers = _hex_lattice_positions()
        best_s = _compute_s(best_centers, np.zeros(12), 0.0)
        best_outer_angle = 0.0
        best_outer_cx = 0.0
        best_outer_cy = 0.0
    
    # Final verification
    angles_deg = np.zeros(12)
    if not _check_separation_fast(best_centers, angles_deg):
        # Use a very spread out configuration as ultimate fallback
        sqrt3 = np.sqrt(3.0)
        best_centers = np.array([
            [0.0, 0.0],
            [2.5, 0.0],
            [-2.5, 0.0],
            [1.25, 2.165],
            [-1.25, 2.165],
            [1.25, -2.165],
            [-1.25, -2.165],
            [3.75, 2.165],
            [-3.75, 2.165],
            [3.75, -2.165],
            [-3.75, -2.165],
            [0.0, -4.33],
        ])
        best_s = _compute_s(best_centers, np.zeros(12), 0.0)
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([best_outer_cx, best_outer_cy, best_outer_angle])
    outer_hex_side_length = best_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
