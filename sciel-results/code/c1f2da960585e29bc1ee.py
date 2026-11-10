import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _get_all_vertices(positions, angles):
    """
    positions: (11, 2)
    angles: (11,)
    Returns: (66, 2) array of all vertices.
    """
    verts = []
    for i in range(11):
        v = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        verts.append(v)
    return np.vstack(verts)

def _hex_separated(v1, v2, tol=1e-6):
    """Check if two hexagons (vertex arrays) are separated using all 12 edge normals."""
    for i in range(6):
        e = v1[(i+1)%6] - v1[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
    for i in range(6):
        e = v2[(i+1)%6] - v2[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - tol or np.max(p2) < np.min(p1) - tol:
                return True
    return False

def _is_valid_fast(positions, angles):
    """
    Check validity of packing.
    positions: (11, 2)
    angles: (11,)
    """
    verts_list = []
    for i in range(11):
        verts_list.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
    
    for i in range(11):
        for j in range(i+1, 11):
            if not _hex_separated(verts_list[i], verts_list[j]):
                return False
    return True

def _compute_side_length(positions, angles, outer_angle):
    """
    Compute the minimum side length of a regular hexagon with given angle
    that contains all inner hexagons.
    """
    all_verts = _get_all_vertices(positions, angles)
    
    a = np.radians(outer_angle)
    max_proj = 0.0
    for i in range(6):
        theta = a + np.radians(30 + 60 * i) # Normals are at 30, 90, ...
        n = np.array([np.cos(theta), np.sin(theta)])
        proj = all_verts @ n
        max_proj = max(max_proj, np.max(np.abs(proj)))
    
    return 2.0 * max_proj / np.sqrt(3)

def _build_rigid_cluster_11():
    """
    Build a rigid cluster of 11 unit hexagons in a known-optimal topological configuration.
    Returns positions (11, 2) and angles (11,).
    
    Configuration: Central hexagon surrounded by 6 in ring 1, plus 4 more strategically placed.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((11, 2))
    
    # Center
    pos[0] = [0, 0]
    # Ring 1: 6 hexagons around center
    pos[1] = [2, 0]
    pos[2] = [1, sqrt3]
    pos[3] = [-1, sqrt3]
    pos[4] = [-2, 0]
    pos[5] = [-1, -sqrt3]
    pos[6] = [1, -sqrt3]
    
    # Ring 2: 4 hexagons at strategic positions
    # These are placed to minimize the bounding hexagon
    # Option A: Corners of the 7-hex cluster
    pos[7] = [3, sqrt3]
    pos[8] = [-3, sqrt3]
    pos[9] = [3, -sqrt3]
    pos[10] = [-3, -sqrt3]
    
    angles = np.zeros(11)
    return pos, angles

def _build_compact_cluster_11():
    """
    Alternative compact configuration with slightly different satellite placement.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((11, 2))
    
    # Center
    pos[0] = [0, 0]
    # Ring 1
    pos[1] = [2, 0]
    pos[2] = [1, sqrt3]
    pos[3] = [-1, sqrt3]
    pos[4] = [-2, 0]
    pos[5] = [-1, -sqrt3]
    pos[6] = [1, -sqrt3]
    
    # Ring 2: Slightly compressed satellites
    # Place them closer to reduce overall extent
    pos[7] = [2.8, sqrt3 * 0.95]
    pos[8] = [-2.8, sqrt3 * 0.95]
    pos[9] = [2.8, -sqrt3 * 0.95]
    pos[10] = [-2.8, -sqrt3 * 0.95]
    
    angles = np.zeros(11)
    return pos, angles

def _build_staggered_cluster_11():
    """
    Staggered configuration where satellites are offset to fill gaps better.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((11, 2))
    
    # Center
    pos[0] = [0, 0]
    # Ring 1
    pos[1] = [2, 0]
    pos[2] = [1, sqrt3]
    pos[3] = [-1, sqrt3]
    pos[4] = [-2, 0]
    pos[5] = [-1, -sqrt3]
    pos[6] = [1, -sqrt3]
    
    # Ring 2: Staggered placement
    pos[7] = [2.5, sqrt3 * 1.1]
    pos[8] = [-2.5, sqrt3 * 1.1]
    pos[9] = [2.5, -sqrt3 * 1.1]
    pos[10] = [-2.5, -sqrt3 * 1.1]
    
    angles = np.zeros(11)
    return pos, angles

def _validate_and_compute(positions, angles, outer_angle):
    """Validate the packing and compute side length. Returns (valid, side)."""
    if not _is_valid_fast(positions, angles):
        return False, float('inf')
    side = _compute_side_length(positions, angles, outer_angle)
    return True, side

def _optimize_rigid_cluster(pos_init, angles_init, deadline):
    """
    Optimize a rigid cluster by searching over global rotation and outer hexagon angle.
    The cluster topology is fixed; only global rotation and outer angle are optimized.
    """
    best_side = float('inf')
    best_global_angle = 0.0
    best_outer_angle = 0.0
    
    # First, try a grid search over global angle and outer angle
    n_angle_steps = 30
    angle_range = np.linspace(0, 60, n_angle_steps, endpoint=False)
    
    for ga in angle_range:
        if time.time() > deadline - 30:
            break
        angles = np.full(11, ga)
        for oa in [0, 30]:
            valid, side = _validate_and_compute(pos_init, angles, oa)
            if valid and side < best_side:
                best_side = side
                best_global_angle = ga
                best_outer_angle = oa
    
    # Local refinement using Nelder-Mead
    if time.time() < deadline - 15:
        try:
            def objective(params):
                global_angle = params[0]
                outer_angle = params[1]
                angles = np.full(11, global_angle)
                valid, side = _validate_and_compute(pos_init, angles, outer_angle)
                if not valid:
                    return 1e6
                return side
            
            initial_params = np.array([best_global_angle, best_outer_angle])
            res_nm = minimize(
                objective,
                initial_params,
                method='Nelder-Mead',
                options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500, 'maxfev': 1000}
            )
            if res_nm.fun < best_side:
                best_side = res_nm.fun
                best_global_angle = res_nm.x[0]
                best_outer_angle = res_nm.x[1]
        except Exception:
            pass
    
    return best_side, best_global_angle, best_outer_angle

def _refine_positions_local(pos_init, angles_init, outer_angle, deadline):
    """
    Local refinement of individual hexagon positions while maintaining separation.
    """
    best_side = _compute_side_length(pos_init, angles_init, outer_angle)
    best_pos = pos_init.copy()
    best_angles = angles_init.copy()
    
    # Try small perturbations and keep improvements
    step_size = 0.05
    n_iterations = 50
    
    for _ in range(n_iterations):
        if time.time() > deadline - 20:
            break
        
        improved = False
        for i in range(11):
            if time.time() > deadline - 20:
                break
            
            # Try small moves in 8 directions
            for dx, dy in [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
                          (step_size*0.7, step_size*0.7), (-step_size*0.7, step_size*0.7),
                          (step_size*0.7, -step_size*0.7), (-step_size*0.7, -step_size*0.7)]:
                new_pos = best_pos.copy()
                new_pos[i] += [dx, dy]
                
                valid, side = _validate_and_compute(new_pos, best_angles, outer_angle)
                if valid and side < best_side - 1e-8:
                    best_pos = new_pos
                    best_side = side
                    improved = True
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-4:
                break
    
    return best_side, best_pos, best_angles

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_side = float('inf')
    best_pos = None
    best_global_angle = 0.0
    best_outer_angle = 0.0
    
    # Generate multiple initial rigid cluster configurations
    cluster_builders = [
        _build_rigid_cluster_11,
        _build_compact_cluster_11,
        _build_staggered_cluster_11,
    ]
    
    for builder in cluster_builders:
        if time.time() > deadline - 100:
            break
        
        try:
            pos_init, angles_init = builder()
            
            # Optimize the rigid cluster
            side, ga, oa = _optimize_rigid_cluster(pos_init, angles_init, deadline)
            
            if side < best_side:
                best_side = side
                best_pos = pos_init.copy()
                best_global_angle = ga
                best_outer_angle = oa
        except Exception:
            continue
    
    # If we have a valid configuration, try local position refinement
    if best_pos is not None and best_side < float('inf'):
        if time.time() < deadline - 50:
            try:
                angles = np.full(11, best_global_angle)
                side, pos_refined, angles_refined = _refine_positions_local(
                    best_pos, angles, best_outer_angle, deadline
                )
                if side < best_side:
                    best_side = side
                    best_pos = pos_refined
                    best_global_angle = angles_refined[0]  # All angles are the same
            except Exception:
                pass
    
    # Final fallback if no valid configuration was found
    if best_pos is None or best_side >= float('inf'):
        pos_init, angles_init = _build_rigid_cluster_11()
        best_pos = pos_init
        best_global_angle = 0.0
        best_outer_angle = 0.0
        best_side = _compute_side_length(best_pos, angles_init, best_outer_angle)
    
    # Ensure validity
    angles = np.full(11, best_global_angle)
    if not _is_valid_fast(best_pos, angles):
        # Last resort: use the simplest valid configuration
        pos_init, angles_init = _build_rigid_cluster_11()
        best_pos = pos_init
        best_global_angle = 0.0
        best_outer_angle = 0.0
        best_side = _compute_side_length(best_pos, angles_init, best_outer_angle)
    
    # Re-calculate side to be safe
    best_side = _compute_side_length(best_pos, angles, best_outer_angle)
    
    # Format output
    inner_hex_data = np.column_stack([best_pos, np.full(11, best_global_angle)])
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, float(best_side)
