import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0
_TIME_START = time.time()

def get_hex_vertices_batch(xys_rots):
    """
    Batch compute vertices for all hexagons. xys_rots is (N, 3) array: [x, y, angle_deg].
    Returns (N, 6, 2) array of vertices.
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

def compute_sat_penalties_vectorized(vertices):
    """
    Vectorized computation of penetration depths for all pairs of hexagons using SAT.
    vertices: (N, 6, 2) array of vertices for N hexagons.
    Returns: (N, N) matrix of penetration depths (0 if no overlap, positive if overlapping).
    """
    N_pts = vertices.shape[0]
    if N_pts == 0:
        return np.zeros((0, 0))
    
    # Precompute edge normals for each hexagon
    # For hexagon i, edge j goes from vertex j to vertex (j+1)%6
    # Normal is perpendicular to edge, pointing outward
    edges = np.zeros((N_pts, 6, 2))
    for i in range(N_pts):
        for j in range(6):
            p1 = vertices[i, j]
            p2 = vertices[i, (j+1) % 6]
            e = p2 - p1
            # Outward normal: rotate edge 90 degrees clockwise
            edges[i, j] = np.array([e[1], -e[0]])
            norm = np.linalg.norm(edges[i, j])
            if norm > 1e-12:
                edges[i, j] /= norm
    
    # For each pair (i, j), compute min overlap across all axes
    # Axes come from both hexagon i and hexagon j
    penetrations = np.zeros((N_pts, N_pts))
    
    for i in range(N_pts):
        for j in range(i+1, N_pts):
            # Get all axes for this pair
            axes_i = edges[i]  # (6, 2)
            axes_j = edges[j]  # (6, 2)
            all_axes = np.vstack([axes_i, axes_j])  # (12, 2)
            
            # Project vertices of hexagon i onto all axes
            proj_i = vertices[i] @ all_axes.T  # (6, 12)
            min_i = np.min(proj_i, axis=0)  # (12,)
            max_i = np.max(proj_i, axis=0)  # (12,)
            
            # Project vertices of hexagon j onto all axes
            proj_j = vertices[j] @ all_axes.T  # (6, 12)
            min_j = np.min(proj_j, axis=0)  # (12,)
            max_j = np.max(proj_j, axis=0)  # (12,)
            
            # Check for separation on any axis
            separated = (max_i < min_j) | (max_j < min_i)
            
            if np.any(separated):
                penetrations[i, j] = 0.0
                penetrations[j, i] = 0.0
            else:
                # Compute overlap on each axis
                overlaps = np.minimum(max_i, max_j) - np.maximum(min_i, min_j)
                # Min overlap is the penetration depth
                min_overlap = np.min(overlaps)
                if min_overlap > 1e-9:
                    penetrations[i, j] = min_overlap
                    penetrations[j, i] = min_overlap
                else:
                    penetrations[i, j] = 0.0
                    penetrations[j, i] = 0.0
    
    return penetrations

def outer_hex_side_batch(vertices_list, outer_rot_deg):
    """
    Compute outer hex side from a list of vertex arrays.
    """
    all_points = np.vstack(vertices_list)
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = all_points @ normals.T
    max_proj = np.max(np.abs(proj))
    return (2.0 / np.sqrt(3.0)) * max_proj

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        # 1. Compute vertices
        vertices = get_hex_vertices_batch(xys_rots)
        vertices_list = [vertices[i] for i in range(N)]
        
        # 2. Compute smooth overlap penalty using vectorized SAT
        # Use soft penalty: sum of (max(0, penetration))^2
        pen_matrix = compute_sat_penalties_vectorized(vertices)
        # Only count upper triangle to avoid double counting
        iu = np.triu_indices(N, k=1)
        penetrations = pen_matrix[iu]
        penalty = np.sum(np.maximum(0, penetrations) ** 2)
        
        # 3. Compute enclosing Hexagon Side Length
        L = outer_hex_side_batch(vertices_list, outer_rot)
        
        return L + 100.0 * penalty
    
    def is_valid(params, tol=1e-6):
        """Check if configuration has no overlaps."""
        xys_rots = params[:3*N].reshape(N, 3)
        vertices = get_hex_vertices_batch(xys_rots)
        pen_matrix = compute_sat_penalties_vectorized(vertices)
        iu = np.triu_indices(N, k=1)
        penetrations = pen_matrix[iu]
        return np.all(penetrations < tol)
    
    def get_L(params):
        """Compute outer side length for a valid configuration."""
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        vertices = get_hex_vertices_batch(xys_rots)
        vertices_list = [vertices[i] for i in range(N)]
        return outer_hex_side_batch(vertices_list, outer_rot)
    
    def make_hex_grid(rows, cols, rot_offset=0.0, gap=0.0):
        """Create a hex grid of unit hexagons with optional gap."""
        pts = []
        sqrt3 = np.sqrt(3)
        spacing_x = sqrt3 + gap
        spacing_y = 1.5 + gap * 0.5 
        for r in range(rows):
            y = r * spacing_y
            for c in range(cols):
                x = c * spacing_x
                if r % 2 == 1:
                    x += spacing_x / 2.0
                pts.append([x, y, rot_offset])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts
    
    def make_ring(gap=0.0):
        """1 center + 6 ring + 5 outer with gap."""
        pts = []
        sqrt3 = np.sqrt(3)
        d1 = sqrt3 + gap
        d2 = 2 * sqrt3 + gap
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([d1 * np.cos(ang), d1 * np.sin(ang), 0.0])
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([d2 * np.cos(ang), d2 * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts

    best_L = float('inf')
    best_params = None
    
    # Generate diverse initial configurations
    configs = []
    gap = 0.05
    configs.append(make_hex_grid(4, 3, gap=gap))
    configs.append(make_hex_grid(3, 4, gap=gap))
    configs.append(make_hex_grid(2, 6, gap=gap))
    configs.append(make_hex_grid(6, 2, gap=gap))
    configs.append(make_ring(gap=gap))
    
    # Add rotated variants
    for base_cfg in configs[:3]:
        rotated = base_cfg.copy()
        rotated[:, 2] = 30.0
        configs.append(rotated)
    
    outer_rots = [0.0, 30.0, 15.0, 45.0]
    
    for cfg in configs:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        for orot in outer_rots:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            init_params = np.concatenate((cfg.flatten(), [orot]))
            
            # Check if initial config is valid and better
            if is_valid(init_params):
                L_val = get_L(init_params)
                if L_val < best_L:
                    best_L = L_val
                    best_params = init_params.copy()
            
            try:
                # Use Nelder-Mead with adaptive options
                res = minimize(
                    objective, 
                    init_params, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 800,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                
                # Evaluate the result
                xys_rots = res.x[:3*N].reshape(N, 3)
                outer_rot = res.x[3*N]
                
                vertices = get_hex_vertices_batch(xys_rots)
                
                # Check for overlaps
                pen_matrix = compute_sat_penalties_vectorized(vertices)
                iu = np.triu_indices(N, k=1)
                penetrations = pen_matrix[iu]
                
                if np.all(penetrations < 1e-6):
                    vertices_list = [vertices[i] for i in range(N)]
                    L_val = outer_hex_side_batch(vertices_list, outer_rot)
                    
                    if L_val < best_L:
                        best_L = L_val
                        best_params = res.x.copy()
                        
            except Exception:
                pass
    
    # Final refinement with tighter tolerances
    if best_params is not None and time.time() - start_time < DEADLINE - 20:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 1500,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            xys_rots = res.x[:3*N].reshape(N, 3)
            outer_rot = res.x[3*N]
            
            vertices = get_hex_vertices_batch(xys_rots)
            
            pen_matrix = compute_sat_penalties_vectorized(vertices)
            iu = np.triu_indices(N, k=1)
            penetrations = pen_matrix[iu]
            
            if np.all(penetrations < 1e-6):
                vertices_list = [vertices[i] for i in range(N)]
                L_val = outer_hex_side_batch(vertices_list, outer_rot)
                
                if L_val < best_L:
                    best_L = L_val
                    best_params = res.x.copy()
                    
        except Exception:
            pass
    
    # Fallback: Use the best valid initial config or a simple valid grid
    if best_params is None:
        # Create a simple valid grid with large gaps
        inner = []
        for k in range(N):
            inner.append([4.0 * k, 0.0, 0.0])
        # Calculate side for this fallback
        vertices = get_hex_vertices_batch(np.array(inner))
        vertices_list = [vertices[i] for i in range(N)]
        L_val = outer_hex_side_batch(vertices_list, 0.0)
        return inner, (0.0, 0.0), L_val, 0.0
    
    # Extract final result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = [[float(x), float(y), float(r)] for x, y, r in xys_rots]
    center = [0.0, 0.0]
    side = float(best_L)
    angle = float(outer_rot)
    
    return inner, center, side, angle
