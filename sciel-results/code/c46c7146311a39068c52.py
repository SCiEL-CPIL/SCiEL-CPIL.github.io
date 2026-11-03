import numpy as np
import time
from scipy.optimize import minimize_scalar, minimize

N = 12
DEADLINE = 145.0  # Safety margin for 150s limit
_TIME_START = time.time()

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """Get vertices of a single regular hexagon."""
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def get_hex_vertices_batch(xys_rots):
    """
    Batch compute vertices for all hexagons. 
    xys_rots: (N, 3) array of [x, y, angle_degrees]
    Returns: (N, 6, 2) array of vertices
    """
    N_pts = len(xys_rots)
    base_angles = np.radians(np.array([0, 60, 120, 180, 240, 300]))
    rots = np.radians(xys_rots[:, 2])
    # angles shape: (N, 6)
    angles = base_angles[None, :] + rots[:, None]
    cos_a = np.cos(angles)  # (N, 6)
    sin_a = np.sin(angles)  # (N, 6)
    cx = xys_rots[:, 0]
    cy = xys_rots[:, 1]
    x = cx[:, None] + cos_a
    y = cy[:, None] + sin_a
    # Return shape (N, 6, 2)
    return np.stack((x, y), axis=-1)

def sat_overlap_batch(vertices_list):
    """
    Vectorized SAT overlap check for all pairs.
    Returns (has_overlap, penalty) where penalty is sum of squared penetration depths.
    """
    N_pts = len(vertices_list)
    if N_pts == 0:
        return False, 0.0
    
    # Precompute axes for all hexagons
    axes_all = []
    for v in vertices_list:
        hex_axes = []
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                hex_axes.append(n)
        axes_all.append(hex_axes)
    
    has_overlap = False
    penalty = 0.0
    
    for i in range(N_pts):
        for j in range(i+1, N_pts):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            
            # Combine axes from both polygons
            all_axes = axes_all[i] + axes_all[j]
            
            min_overlap = float('inf')
            separated = False
            for n in all_axes:
                proj1 = v1 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                proj2 = v2 @ n
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 or max2 < min1:
                    separated = True
                    break
                
                overlap = min(max1, max2) - max(min1, min2)
                min_overlap = min(min_overlap, overlap)
            
            if not separated:
                if min_overlap > 1e-9:
                    has_overlap = True
                    penalty += min_overlap ** 2
    
    return has_overlap, penalty

def outer_hex_side(points, outer_rot_deg):
    """
    Minimum side length of regular hexagon centered at origin with given rotation
    that encloses all points.
    """
    # Normals of the outer hexagon sides
    # A regular hexagon with side S has apothem a = S * sqrt(3)/2
    # Distance from center to vertex is S
    # The condition for a point p to be inside is |p . n_k| <= a for all 6 normals
    # So S >= 2/sqrt(3) * max(|p . n_k|)
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def outer_hex_side_batch(vertices_list, outer_rot_deg):
    """
    Compute outer hex side from a list of vertex arrays.
    """
    if not vertices_list:
        return 0.0
    all_points = np.vstack(vertices_list)
    return outer_hex_side(all_points, outer_rot_deg)

def generate_lattice_candidates():
    """
    Generate a set of candidate configurations based on triangular lattice structures.
    We assume all inner hexagons have the same orientation (0 degrees relative to each other)
    initially, and we will optimize the global rotation later.
    """
    candidates = []
    sqrt3 = np.sqrt(3)
    
    # Helper to create a grid of lattice points
    def make_grid(rows, cols, offset_x=0, offset_y=0):
        pts = []
        for r in range(rows):
            for c in range(cols):
                x = c * sqrt3
                if r % 2 == 1:
                    x += sqrt3 / 2.0
                y = r * 1.5
                pts.append([x + offset_x, y + offset_y, 0.0])
        return np.array(pts)

    # 1. Rectangular grids (2x6, 3x4, 4x3, 6x2)
    # Note: These are "tight" packings. We add a small epsilon to ensure they are valid initially?
    # Actually, for the search, we want them to be valid. The lattice distance is exactly sqrt(3) for horizontal neighbors.
    # For diagonal neighbors, distance is sqrt( (sqrt(3)/2)^2 + 1.5^2 ) = sqrt(3/4 + 9/4) = sqrt(3).
    # So the lattice points are exactly on the boundary of non-overlap.
    # To ensure validity in floating point, we might need to scale slightly or rely on the optimizer to push them apart if needed.
    # However, for the *search* phase, we just want the topology.
    
    # 2x6 Grid
    g_2_6 = make_grid(2, 6)
    # Center it
    g_2_6[:, :2] -= np.mean(g_2_6[:, :2], axis=0)
    candidates.append(g_2_6)
    
    # 3x4 Grid
    g_3_4 = make_grid(3, 4)
    g_3_4[:, :2] -= np.mean(g_3_4[:, :2], axis=0)
    candidates.append(g_3_4)
    
    # 4x3 Grid
    g_4_3 = make_grid(4, 3)
    g_4_3[:, :2] -= np.mean(g_4_3[:, :2], axis=0)
    candidates.append(g_4_3)
    
    # 6x2 Grid
    g_6_2 = make_grid(6, 2)
    g_6_2[:, :2] -= np.mean(g_6_2[:, :2], axis=0)
    candidates.append(g_6_2)
    
    # 2. Ring Structures
    # 1 center + 6 ring + 5 outer
    # Center at (0,0)
    # Ring 1: 6 hexes at distance sqrt(3)
    # Ring 2: 6 hexes at distance 2*sqrt(3)
    # We need 12. 1 + 6 + 5 = 12.
    
    def make_ring_partial(n_outer=5):
        pts = [[0, 0, 0.0]]
        # Inner ring
        for k in range(6):
            ang = np.radians(60 * k)
            x = sqrt3 * np.cos(ang)
            y = sqrt3 * np.sin(ang)
            pts.append([x, y, 0.0])
        # Outer ring (partial)
        # The outer ring hexes are at distance 2*sqrt(3) from center, at angles 30, 90, 150...
        # But wait, the lattice structure for the second ring is different.
        # Let's stick to lattice points for the outer ring too.
        # Lattice points at "radius" 2:
        # (2*sqrt(3), 0), (sqrt(3), 1.5), (0, 3), (-sqrt(3), 1.5)... etc
        # Let's just generate all lattice points within a certain radius and pick the outermost 5.
        lattice_pts = []
        for r in range(-3, 4):
            for c in range(-3, 4):
                x = c * sqrt3
                if r % 2 == 1:
                    x += sqrt3 / 2.0
                y = r * 1.5
                lattice_pts.append([x, y])
        
        lattice_pts = np.array(lattice_pts)
        dists = np.linalg.norm(lattice_pts, axis=1)
        # Exclude origin and first ring
        # First ring dist is sqrt(3) ~ 1.732
        # Second ring dist is 2*sqrt(3) ~ 3.464 (for some) or sqrt(7) ~ 2.645 (for others?)
        # Let's check distances of lattice points.
        # (sqrt(3), 1.5) -> sqrt(3+2.25)=sqrt(5.25) ~ 2.29
        # (2*sqrt(3), 0) -> 3.464
        # (0, 3) -> 3.0
        # It's easier to just define specific topologies manually if needed, but let's try a "hexagon of hexagons" shape.
        
        # A common optimal packing for 12 is a 2x6 rectangle or a 3x4 rectangle.
        # Let's add a "diamond" shape: 1-3-5-3-1? No, that's 13.
        # 1-2-3-3-2-1 = 12?
        
        # Let's just add variations of the grids with slight rotations of the lattice itself.
        for rot_deg in [0, 30]:
            rot_rad = np.radians(rot_deg)
            R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]])
            
            g_2_6_rot = g_2_6.copy()
            g_2_6_rot[:, :2] = g_2_6_rot[:, :2] @ R.T
            candidates.append(g_2_6_rot)
            
            g_3_4_rot = g_3_4.copy()
            g_3_4_rot[:, :2] = g_3_4_rot[:, :2] @ R.T
            candidates.append(g_3_4_rot)
            
            g_4_3_rot = g_4_3.copy()
            g_4_3_rot[:, :2] = g_4_3_rot[:, :2] @ R.T
            candidates.append(g_4_3_rot)
            
            g_6_2_rot = g_6_2.copy()
            g_6_2_rot[:, :2] = g_6_2_rot[:, :2] @ R.T
            candidates.append(g_6_2_rot)
            
    return candidates

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    best_L = float('inf')
    best_params = None
    
    # Generate candidates
    candidates = generate_lattice_candidates()
    
    # For each candidate, optimize the global rotation of the inner hexagons
    # and the outer hexagon rotation.
    # Since the inner hexagons are identical and arranged symmetrically, 
    # rotating the whole cluster by theta is equivalent to rotating the outer hexagon by -theta (relative frame).
    # So we can fix the outer hexagon rotation to 0 and just rotate the inner cluster?
    # No, the inner hexagons themselves have orientation. 
    # In the lattice candidates, all inner hexagons have angle 0.
    # We can rotate all inner hexagons by alpha, and the outer hexagon by beta.
    # Due to symmetry, alpha and beta are coupled.
    # Let's just search over alpha (inner rotation) and beta (outer rotation) in a coarse grid first.
    
    # Actually, a better approach:
    # Fix the relative positions of the centers.
    # Optimize: 
    # 1. Global rotation of the entire cluster (inner centers + inner hex orientations).
    # 2. Rotation of the outer hexagon.
    # These two are independent variables in the objective function.
    
    # Let's define a function that takes a candidate (positions, initial angles) 
    # and returns the best (L, params) by optimizing the global rotation.
    
    def optimize_candidate(candidate_pts):
        # candidate_pts: (N, 3) [x, y, angle]
        # We will rotate all points and angles by a global angle theta.
        # And we will rotate the outer hexagon by phi.
        
        # Initial guess: theta = 0, phi = 0
        
        def objective_2d(vars):
            theta = vars[0]
            phi = vars[1]
            
            # Rotate points
            rot_rad = np.radians(theta)
            R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]])
            
            new_pts = candidate_pts.copy()
            new_pts[:, :2] = new_pts[:, :2] @ R.T
            new_pts[:, 2] = new_pts[:, 2] + theta # Rotate the hexagons themselves
            
            # Check validity
            vertices_list = get_hex_vertices_batch(new_pts)
            has_overlap, penalty = sat_overlap_batch(vertices_list)
            
            if has_overlap:
                return 1e6 + penalty
            
            L = outer_hex_side_batch(vertices_list, phi)
            return L
        
        # Coarse search first
        best_val = float('inf')
        best_vars = None
        
        # Search theta in [0, 60) and phi in [0, 60)
        # Step size 1 degree
        for theta_deg in range(0, 60, 1):
            if time_left() < 5:
                return None, None
            for phi_deg in range(0, 60, 1):
                val = objective_2d([theta_deg, phi_deg])
                if val < best_val:
                    best_val = val
                    best_vars = [theta_deg, phi_deg]
        
        if best_vars is None:
            return None, None
            
        # Fine search around best_vars
        try:
            res = minimize(
                objective_2d,
                best_vars,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_vars = res.x
        except Exception:
            pass
            
        return best_val, best_vars

    # Run optimization on candidates
    for cand in candidates:
        if time_left() < 10:
            break
            
        L_val, vars_opt = optimize_candidate(cand)
        
        if L_val is not None and L_val < best_L:
            best_L = L_val
            # Reconstruct full params
            theta = vars_opt[0]
            phi = vars_opt[1]
            
            rot_rad = np.radians(theta)
            R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], [np.sin(rot_rad), np.cos(rot_rad)]])
            
            final_pts = cand.copy()
            final_pts[:, :2] = final_pts[:, :2] @ R.T
            final_pts[:, 2] = final_pts[:, 2] + theta
            
            best_params = np.concatenate((final_pts.flatten(), [phi]))
    
    # If we found a good candidate, do a final high-precision refinement on the full parameter set
    if best_params is not None and time_left() > 20:
        # best_params: (N*3 + 1)
        # We can refine this using Nelder-Mead on the full 37D space
        # But this is risky. Let's try a few iterations.
        
        def objective_full(params):
            xys_rots = params[:3*N].reshape(N, 3)
            outer_rot = params[3*N]
            
            vertices_list = get_hex_vertices_batch(xys_rots)
            has_overlap, penalty = sat_overlap_batch(vertices_list)
            
            if has_overlap:
                return 1e6 + penalty
            
            L = outer_hex_side_batch(vertices_list, outer_rot)
            return L
        
        try:
            res = minimize(
                objective_full,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if res.fun < best_L:
                # Verify validity
                xys_rots = res.x[:3*N].reshape(N, 3)
                outer_rot = res.x[3*N]
                vertices_list = get_hex_vertices_batch(xys_rots)
                has_overlap, _ = sat_overlap_batch(vertices_list)
                
                if not has_overlap:
                    best_L = res.fun
                    best_params = res.x
        except Exception:
            pass

    # Fallback
    if best_params is None:
        # Create a simple valid grid with large gaps
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
        pts = np.array([[x, y] for x, y, r in inner])
        L_val = outer_hex_side(pts, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0

    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle

if __name__ == "__main__":
    # Simple test
    inner, center, side, angle = construct()
    print(f"Side length: {side}")
