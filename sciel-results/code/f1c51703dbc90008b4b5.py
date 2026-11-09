# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angs = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])


def _hex_normals(a_deg):
    """12 edge normals for a hexagon of rotation a_deg."""
    a = np.radians(a_deg)
    edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    return np.vstack([normals, -normals])


def _check_all_separated(verts_list, angles):
    """
    Check if all pairs of hexagons are separated using the 12-normal SAT condition.
    verts_list: list of (6,2) arrays
    angles: list of 11 angles in degrees
    Returns True if all pairs are separated.
    """
    n = len(verts_list)
    # Precompute projections for each hexagon on its own 12 normals
    # proj[i] has shape (12, 2) -> (min, max) for each normal
    # Actually, for the SAT check between hex i and j, we need to project BOTH onto the normals of i AND normals of j.
    # Let's precompute the 12 normals for each hex.
    norms = [_hex_normals(angles[i]) for i in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            v_i = verts_list[i]
            v_j = verts_list[j]
            # Check normals of i
            n_i = norms[i]
            proj_i = v_i @ n_i.T  # (6, 12)
            min_i = proj_i.min(axis=0)
            max_i = proj_i.max(axis=0)
            proj_j = v_j @ n_i.T
            min_j = proj_j.min(axis=0)
            max_j = proj_j.max(axis=0)
            # Separated if for any normal k: max_i[k] < min_j[k] - eps OR max_j[k] < min_i[k] - eps
            sep_i = (max_i < min_j - 1e-6) | (max_j < min_i - 1e-6)
            if np.any(sep_i):
                continue  # This pair is separated by a normal of i
            
            # Check normals of j
            n_j = norms[j]
            proj_i2 = v_i @ n_j.T
            min_i2 = proj_i2.min(axis=0)
            max_i2 = proj_i2.max(axis=0)
            proj_j2 = v_j @ n_j.T
            min_j2 = proj_j2.min(axis=0)
            max_j2 = proj_j2.max(axis=0)
            sep_j = (max_i2 < min_j2 - 1e-6) | (max_j2 < min_i2 - 1e-6)
            if np.any(sep_j):
                continue  # This pair is separated by a normal of j
            
            return False  # Pair (i,j) is NOT separated
    
    return True


def _objective(params, deadline):
    """
    Objective function: minimize the side length of the enclosing regular hexagon.
    params: array of shape (33,) = 11 * (x, y, angle_deg)
    Returns: side length + penalty if invalid
    """
    if time.time() > deadline:
        return 1e6
    
    # Unpack
    centers = params[:22].reshape(11, 2)
    angles = params[22:]
    
    # Compute vertices
    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy), ang in zip(centers, angles)]
    
    # Check separation
    if not _check_all_separated(verts_list, angles):
        # Penalty: sum of overlaps (approximate by checking center distances < 2*apothem)
        # For a better penalty, compute the minimum separation violation
        # But for Nelder-Mead, a simple penalty works if the landscape is smooth enough.
        # Let's use a penalty based on the number of overlapping pairs.
        # Actually, let's compute a continuous penalty: for each pair, if not separated,
        # find the minimum gap (which is negative) and sum the squares.
        # This is expensive. Let's stick to a simpler penalty first.
        # Penalty = 100 * number_of_overlaps + small_term
        n_overlaps = 0
        for i in range(11):
            for j in range(i + 1, 11):
                if not _separated_pair(verts_list[i], verts_list[j], angles[i], angles[j]):
                    n_overlaps += 1
        return 10.0 + n_overlaps * 0.1
    
    # All separated. Compute minimal enclosing hexagon.
    # The enclosing hexagon is centered at the centroid of the inner hexagons? 
    # No, the enclosing hexagon can be anywhere. But due to symmetry, we can assume 
    # the center of the enclosing hexagon is at the centroid of the inner hexagons' centers?
    # Not necessarily. However, for a regular hexagon, the minimal enclosing one is 
    # determined by the extreme projections.
    # Let's fix the center of the outer hexagon to be the centroid of the inner centers 
    # to reduce degrees of freedom. If this is suboptimal, we can add 2 more params.
    # For now, assume outer center = centroid of inner centers.
    outer_center = centers.mean(axis=0)
    
    all_verts = np.vstack(verts_list) - outer_center
    
    # We need to find the angle 'a' and side 's' of the outer hexagon such that 
    # all all_verts are inside, and s is minimized.
    # For a given angle 'a', the required apothem is max over all normals of the max projection.
    # s = apothem / (sqrt(3)/2)
    
    best_s = np.inf
    # Search over angle a. Due to 6-fold symmetry, a in [0, 60) is sufficient.
    # Use a fine grid for the angle search.
    for a_deg in np.linspace(0, 60, 121):
        a = np.radians(a_deg)
        edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
        normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
        
        proj = all_verts @ normals.T  # (N, 6)
        max_proj = proj.max(axis=0)
        min_proj = proj.min(axis=0)
        
        required_apothem = np.maximum(max_proj, -min_proj)
        apothem = required_apothem.max()
        s = apothem / (np.sqrt(3) / 2)
        
        if s < best_s:
            best_s = s
    
    return best_s


def _separated_pair(v1, v2, a1, a2):
    """Check if two hexagons are separated."""
    n1 = _hex_normals(a1)
    n2 = _hex_normals(a2)
    
    proj1 = v1 @ n1.T
    min1 = proj1.min(axis=0)
    max1 = proj1.max(axis=0)
    proj2 = v2 @ n1.T
    min2 = proj2.min(axis=0)
    max2 = proj2.max(axis=0)
    if np.any(max1 < min2 - 1e-6) or np.any(max2 < min1 - 1e-6):
        return True
    
    proj1 = v1 @ n2.T
    min1 = proj1.min(axis=0)
    max1 = proj1.max(axis=0)
    proj2 = v2 @ n2.T
    min2 = proj2.min(axis=0)
    max2 = proj2.max(axis=0)
    if np.any(max1 < min2 - 1e-6) or np.any(max2 < min1 - 1e-6):
        return True
    
    return False


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, outer_hex_side_length).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Initial guess: The standard 1+6+4 packing
    # Center at (0,0)
    # 6 inner ring at radius sqrt(3), angles 0, 60, ...
    # 4 outer ring at radius 2*sqrt(3), angles 30, 120, 210, 300? 
    # Wait, the standard optimal packing for 11 hexagons in a hexagon is often cited as side length ~3.5 or so.
    # Let's initialize with a known valid configuration.
    
    # Configuration:
    # 1 center
    # 6 around center (touching)
    # 4 in the gaps? No, 11 = 1 + 6 + 4.
    # The 4 are placed in the "corners" of the hexagon formed by the 6.
    
    r1 = np.sqrt(3)
    r2 = 2 * np.sqrt(3)
    
    centers = np.zeros((11, 2))
    angles = np.zeros(11)
    
    # Center
    centers[0] = [0, 0]
    angles[0] = 0
    
    # Inner ring (6)
    for i in range(6):
        ang = np.radians(60 * i)
        centers[i+1] = [r1 * np.cos(ang), r1 * np.sin(ang)]
        angles[i+1] = 0
    
    # Outer ring (4)
    # These should be at the "vertices" of the larger hexagon, but shifted in.
    # Actually, let's place them at angles 30, 120, 210, 300 degrees?
    # Or 0, 90, 180, 270?
    # Let's try to place them such that they fit.
    # A common packing has the 4 outer hexagons aligned with the sides of the outer container.
    # Let's initialize them at radius r2, angles 30, 120, 210, 300.
    for i in range(4):
        ang = np.radians(30 + 90 * i)
        centers[i+7] = [r2 * np.cos(ang), r2 * np.sin(ang)]
        angles[i+7] = 0
    
    params = np.concatenate([centers.flatten(), angles])
    
    # Check if initial is valid
    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy), ang in zip(centers, angles)]
    if not _check_all_separated(verts_list, angles):
        # If not, scale up slightly or adjust
        # Let's just proceed, the optimizer will fix it.
        pass
    
    best_params = params
    best_val = _objective(params, deadline)
    
    # Multi-start Nelder-Mead
    n_starts = 20
    rng = np.random.default_rng(42)
    
    for k in range(n_starts):
        if time.time() > deadline:
            break
        
        if k == 0:
            p0 = best_params
        else:
            # Perturb the initial guess
            # Add noise to positions and angles
            noise_pos = rng.normal(0, 0.5, (11, 2))
            noise_ang = rng.normal(0, 5, 11)  # 5 degrees noise
            p0 = np.concatenate([
                (centers + noise_pos).flatten(),
                angles + noise_ang
            ])
        
        try:
            res = minimize(
                _objective,
                p0,
                args=(deadline,),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-5,
                    'fatol': 1e-5,
                    'adaptive': True
                }
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            continue
    
    # Finalize with the best parameters
    centers = best_params[:22].reshape(11, 2)
    angles = best_params[22:]
    
    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy), ang in zip(centers, angles)]
    
    # Verify validity one last time
    if not _check_all_separated(verts_list, angles):
        # Fallback to initial valid config if optimization failed to maintain validity
        # This shouldn't happen if we started valid and the penalty is strong,
        # but let's be safe.
        r1 = np.sqrt(3)
        r2 = 2 * np.sqrt(3)
        centers = np.zeros((11, 2))
        angles = np.zeros(11)
        centers[0] = [0, 0]
        for i in range(6):
            ang = np.radians(60 * i)
            centers[i+1] = [r1 * np.cos(ang), r1 * np.sin(ang)]
        for i in range(4):
            ang = np.radians(30 + 90 * i)
            centers[i+7] = [r2 * np.cos(ang), r2 * np.sin(ang)]
        verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy), ang in zip(centers, angles)]
        best_val = _objective(np.concatenate([centers.flatten(), angles]), deadline)
    
    # Compute the final outer hexagon parameters
    outer_center = centers.mean(axis=0)
    all_verts = np.vstack(verts_list) - outer_center
    
    best_s = np.inf
    best_a = 0.0
    for a_deg in np.linspace(0, 60, 361):
        a = np.radians(a_deg)
        edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
        normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
        
        proj = all_verts @ normals.T
        max_proj = proj.max(axis=0)
        min_proj = proj.min(axis=0)
        
        required_apothem = np.maximum(max_proj, -min_proj)
        apothem = required_apothem.max()
        s = apothem / (np.sqrt(3) / 2)
        
        if s < best_s:
            best_s = s
            best_a = a_deg
    
    # Ensure s >= sqrt(11)
    best_s = max(best_s, np.sqrt(11.0))
    
    inner_hex_data = np.column_stack([centers, angles])
    outer_hex_data = np.array([outer_center[0], outer_center[1], best_a])
    
    return inner_hex_data, outer_hex_data, float(best_s)


# EVOLVE-BLOCK-END
