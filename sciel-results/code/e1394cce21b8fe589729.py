# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_s_for_outer_rotation(verts, rot_deg):
    """
    Given all inner hexagon vertices (72,2) and outer hexagon rotation,
    compute minimum side length s and center (ox,oy) of outer hexagon.
    Uses LP via scipy.linprog.
    """
    from scipy.optimize import linprog
    rot = np.radians(rot_deg)
    n = len(verts)
    # For each of 6 edge directions, the outer hexagon edge i has outward normal
    # at angle rot + 30 + 60*i degrees (pointing outward from center)
    # Constraint: for each vertex v and each edge direction i:
    #   n_i . v <= n_i . (center + s * n_i_perp)  ... actually let's think more carefully
    #
    # A regular hexagon with center (ox,oy), side s, angle rot has edges.
    # The 6 edge normals (outward) are at angles: rot + 30 + 60*i for i=0..5
    # Each edge is at distance s*sqrt(3)/2 from center along its normal.
    # So constraint: n_i . (v - center) <= s * sqrt(3)/2 for all i, v
    # i.e., n_i . v - n_i . center <= s * sqrt(3)/2
    # i.e., -n_i . center - (sqrt(3)/2) * s <= -n_i . v
    #
    # Variables: x = [ox, oy, s]
    # A_ub @ x <= b_ub
    # For each (i, j): -ci*ox - si*oy - (sqrt(3)/2)*s <= -(ci*vx + si*vy)
    # where (ci, si) = (cos(theta_i), sin(theta_i)), theta_i = rot + 30 + 60*i
    
    n_constraints = 6 * n
    A = np.zeros((n_constraints, 3))
    b = np.zeros(n_constraints)
    
    idx = 0
    for i in range(6):
        theta = rot + np.radians(30 + 60 * i)
        ci = np.cos(theta)
        si = np.sin(theta)
        for j in range(n):
            A[idx, 0] = -ci
            A[idx, 1] = -si
            A[idx, 2] = -np.sqrt(3) / 2
            b[idx] = -(ci * verts[j, 0] + si * verts[j, 1])
            idx += 1
    
    c_obj = np.array([0, 0, 1])
    bounds = [(None, None), (None, None), (0, None)]
    
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.status == 0:
        return res.x[2], (res.x[0], res.x[1])
    else:
        return np.inf, None


def _compute_all_inner_vertices(params):
    """
    params: (36,) array = [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    Returns (72,2) array of all vertices.
    """
    verts = np.zeros((72, 2))
    for i in range(12):
        cx = params[3*i]
        cy = params[3*i+1]
        a = params[3*i+2]
        v = _hex_vertices(cx, cy, 1, a)
        verts[6*i:6*i+6] = v
    return verts


def _objective(params):
    """
    Objective: minimize s (outer hexagon side length).
    params: 36-dim array of 12 hexagon (cx, cy, angle)
    We try several outer rotations and take the min s.
    For smoothness in DE, we use a fixed set of rotations and take max of smooth approximations.
    Actually, let's just pick the best rotation from a fine grid and use that s.
    """
    verts = _compute_all_inner_vertices(params)
    
    # Try a few rotations to find best s
    best_s = np.inf
    for rot in [0, 15, 30, 45, 60]:
        s, _ = _min_s_for_outer_rotation(verts, rot)
        if s < best_s:
            best_s = s
    
    if not np.isfinite(best_s):
        return 1e6
    
    # Add separation penalty: check if any two inner hexagons overlap
    # Use a soft penalty based on minimum distance between hexagon boundaries
    # For speed, just check center distances - if too close, add penalty
    centers = params.reshape(12, 3)[:, :2]
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(centers[i] - centers[j])
            # Minimum center distance for non-overlapping unit hexagons is 2
            # (when aligned). With rotation, it can be less, but 2 is a safe lower bound.
            if d < 1.99:
                best_s += (1.99 - d) * 100
    
    return best_s


def _validate_packing(inner_hex_data, outer_hex_data, s, eps=1e-6):
    """Check if the packing is valid. Returns True if valid."""
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    # Get outer hexagon vertices
    ox, oy, oang = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, s, oang)
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False
    
    # Shift all centers by minus outer center
    # Check unit edges of inner hexagons
    for i in range(12):
        cx, cy, aang = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1, aang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1) % 6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1) > 1e-8:
                return False
    
    # Check separation between inner hexagons using separating axis test
    for i in range(12):
        for j in range(i+1, 12):
            vi = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1, inner_hex_data[i, 2])
            vj = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1, inner_hex_data[j, 2])
            
            # Get all edge normals from both hexagons
            normals = []
            for v in [vi, vj]:
                for k in range(6):
                    e = v[(k+1) % 6] - v[k]
                    n = np.array([-e[1], e[0]])
                    n = n / np.linalg.norm(n)
                    normals.append(n)
            
            # Check if any normal separates the two hexagons
            separated = False
            for n in normals:
                proj_i = np.dot(vi, n)
                proj_j = np.dot(vj, n)
                min_i, max_i = proj_i.min(), proj_i.max()
                min_j, max_j = proj_j.min(), proj_j.max()
                if max_i + eps < min_j or max_j + eps < min_i:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check all inner vertices are inside outer hexagon
    for i in range(12):
        cx, cy, aang = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1, aang)
        for v in verts:
            # For each outer edge, check cross product
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1) % 6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True


def _refine_with_local(params, deadline):
    """
    Local refinement using Nelder-Mead on the objective.
    """
    def obj_local(p):
        verts = _compute_all_inner_vertices(p)
        best_s = np.inf
        for rot in np.linspace(0, 60, 7):
            s, _ = _min_s_for_outer_rotation(verts, rot)
            if s < best_s:
                best_s = s
        if not np.isfinite(best_s):
            return 1e6
        # Separation penalty
        centers = p.reshape(12, 3)[:, :2]
        penalty = 0
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 1.99:
                    penalty += (1.99 - d) * 100
        return best_s + penalty
    
    result = minimize(obj_local, params, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
    return result.x, result.fun


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Uses differential evolution to optimize positions and angles of all 12 hexagons.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Leave 20s safety margin
    
    sqrt3 = np.sqrt(3)
    
    # Initial guess: hexagonal lattice arrangement
    lattice_points = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (-1, 2), (-2, 1), (1, -2), (2, -1), (0, 2),
    ]
    e1 = np.array([2.0, 0.0])
    e2 = np.array([1.0, sqrt3])
    centers_init = np.array([a * e1 + b * e2 for a, b in lattice_points])
    angles_init = np.zeros(12)
    params_init = np.column_stack([centers_init, angles_init]).flatten()
    
    # Bounds for DE: centers in [-10, 10], angles in [0, 60]
    bounds = []
    for i in range(12):
        bounds.append((-10, 10))  # x
        bounds.append((-10, 10))  # y
        bounds.append((0, 60))    # angle
    
    best_s = np.inf
    best_params = params_init.copy()
    best_outer_rot = 0
    best_outer_center = (0, 0)
    
    # Strategy 1: Differential Evolution
    if time.time() < deadline - 120:
        try:
            de_result = differential_evolution(
                _objective, bounds, 
                maxiter=50, popsize=15, seed=42,
                tol=1e-6, polish=False,
                updating='deferred',
                mutation=(0.5, 1.0), recombination=0.7
            )
            if de_result.fun < best_s:
                best_params = de_result.x
                # Find best rotation
                verts = _compute_all_inner_vertices(best_params)
                for rot in np.linspace(0, 60, 25):
                    s, center = _min_s_for_outer_rotation(verts, rot)
                    if s < best_s:
                        best_s = s
                        best_outer_rot = rot
                        best_outer_center = center
        except Exception as e:
            pass
    
    # Strategy 2: Local refinement from multiple starting points
    if time.time() < deadline - 60:
        # Starting points: initial lattice, DE result, and some perturbations
        starting_points = [params_init.copy(), best_params.copy()]
        # Add a few random perturbations
        rng = np.random.default_rng(123)
        for _ in range(4):
            perturbed = params_init + rng.normal(0, 0.5, size=36)
            perturbed[2::3] = np.mod(perturbed[2::3], 60)  # angles in [0,60]
            starting_points.append(perturbed)
        
        for sp in starting_points:
            if time.time() > deadline - 30:
                break
            try:
                refined_params, refined_fun = _refine_with_local(sp, deadline)
                if refined_fun < best_s:
                    best_params = refined_params
                    verts = _compute_all_inner_vertices(best_params)
                    for rot in np.linspace(0, 60, 25):
                        s, center = _min_s_for_outer_rotation(verts, rot)
                        if s < best_s:
                            best_s = s
                            best_outer_rot = rot
                            best_outer_center = center
            except Exception as e:
                pass
    
    # Strategy 3: Fine rotation search around best found
    if time.time() < deadline - 10:
        verts = _compute_all_inner_vertices(best_params)
        # Fine scan around best rotation
        for rot in np.linspace(best_outer_rot - 10, best_outer_rot + 10, 41):
            s, center = _min_s_for_outer_rotation(verts, rot)
            if s < best_s:
                best_s = s
                best_outer_rot = rot
                best_outer_center = center
    
    # Build final result
    inner_hex_data = best_params.reshape(12, 3)
    outer_hex_data = np.array([best_outer_center[0], best_outer_center[1], best_outer_rot])
    
    # Validate
    if not _validate_packing(inner_hex_data, outer_hex_data, best_s):
        # Fallback: use standard lattice packing
        centers_std = np.array([a * e1 + b * e2 for a, b in lattice_points])
        inner_hex_data = np.column_stack([centers_std, np.zeros(12)])
        verts_std = _compute_all_inner_vertices(inner_hex_data.flatten())
        s_std, center_std = _min_s_for_outer_rotation(verts_std, 0)
        outer_hex_data = np.array([center_std[0], center_std[1], 0])
        best_s = s_std
    
    return inner_hex_data, outer_hex_data, best_s


# EVOLVE-BLOCK-END
