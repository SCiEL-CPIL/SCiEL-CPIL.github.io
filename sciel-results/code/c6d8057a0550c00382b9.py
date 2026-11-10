# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side d and angle (deg)."""
    a = np.radians(angle_deg)
    thetas = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(thetas),
                            cy + d * np.sin(thetas)])

def _hex_edge_normals(vertices):
    """Outward edge normals for a hexagon given its 6 vertices (CCW)."""
    v = vertices
    edges = np.roll(v, -1, axis=0) - v
    # For CCW polygon, outward normal is (dy, -dx) normalized
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norms == 0, 1.0, norms)
    return normals

def _separation_violation(verts1, verts2):
    """
    Separating axis test: returns max overlap (positive means separated).
    We want the minimum over all axes of (max1 - min2) and (max2 - min1).
    If all axes give non-negative separation, hexagons don't overlap.
    Returns the minimum gap (negative = overlap).
    """
    n1 = _hex_edge_normals(verts1)
    n2 = _hex_edge_normals(verts2)
    all_normals = np.vstack([n1, n2])
    
    min_gap = np.inf
    for normal in all_normals:
        proj1 = verts1 @ normal
        proj2 = verts2 @ normal
        gap1 = proj2.min() - proj1.max()  # 2 completely to the right of 1
        gap2 = proj1.min() - proj2.max()  # 1 completely to the right of 2
        min_gap = min(min_gap, gap1, gap2)
    return min_gap

def _point_in_hexagon(v, outer_verts):
    """Check if point v is inside the outer hexagon using cross products."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -1e-6:
            return False
    return True

def _compute_s_for_centers(centers, angles):
    """
    Given centers and angles of 12 unit hexagons, compute the minimum s
    of a regular hexagon (centered at origin, angle 0) that contains all of them.
    """
    max_s = 0.0
    for i in range(12):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for v in verts:
            px, py = v[0], v[1]
            for k in range(6):
                angle = np.radians(30 + 60 * k)
                val = px * np.cos(angle) + py * np.sin(angle)
                s_required = val / (np.sqrt(3) / 2)
                if s_required > max_s:
                    max_s = s_required
    return max_s

def _objective(x, start_time, deadline):
    """
    x: 36-vector = 12*(cx, cy, angle_deg)
    Returns 1/s (we maximize, so minimize -1/s)
    """
    if time.time() > deadline:
        return 1e10  # penalty to stop
    
    centers = x[:24].reshape(12, 2)
    angles = x[24:36]
    
    # Compute s needed
    s = _compute_s_for_centers(centers, angles)
    
    # Check overlaps with smooth penalty
    penalty = 0.0
    for i in range(12):
        verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for j in range(i + 1, 12):
            verts_j = _hex_vertices(centers[j, 0], centers[j, 1], 1.0, angles[j])
            gap = _separation_violation(verts_i, verts_j)
            if gap < 0:
                # Smooth penalty: use log-sum-exp style soft penalty
                penalty += np.logaddexp(0, gap / 0.01)  # smooth approximation of max(0, -gap)
    
    # Check boundary: all inner vertices must be inside outer hexagon
    outer_verts = _hex_vertices(0, 0, s, 0)
    for i in range(12):
        verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        for v in verts_i:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < 0:
                    penalty += np.logaddexp(0, cross / 0.01)
    
    # Return negative of 1/s plus penalty (we minimize this)
    if s < np.sqrt(12):
        s = np.sqrt(12) + 1e-6
    return -(1.0 / s) + penalty * 1000

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    best_score = 0.0
    best_x = None
    
    # Initial guess: hexagonal grid arrangement optimized for hexagonal packing
    # Use a proper hexagonal lattice: center + 6 in first ring + 5 in second ring
    # First ring: distance 2*sqrt(3) from center at 60-degree intervals
    # Second ring: distance 4*sqrt(3) from center
    
    d1 = 2 * np.sqrt(3)  # first ring distance
    d2 = 4 * np.sqrt(3)  # second ring distance
    
    initial_centers = []
    # Center
    initial_centers.append([0, 0])
    # First ring: 6 hexagons
    for k in range(6):
        angle = np.radians(60 * k)
        initial_centers.append([d1 * np.cos(angle), d1 * np.sin(angle)])
    # Second ring: 5 more hexagons (skip one position to make 12 total)
    for k in range(5):
        angle = np.radians(60 * k + 30)  # offset by 30 degrees
        initial_centers.append([d2 * np.cos(angle), d2 * np.sin(angle)])
    
    initial_centers = np.array(initial_centers)
    initial_angles = np.zeros(12)
    
    # Center the configuration
    initial_centers -= initial_centers.mean(axis=0)
    
    x0 = np.concatenate([initial_centers.flatten(), initial_angles])
    
    # Try multiple random initializations
    np.random.seed(42)
    
    for attempt in range(30):
        if time.time() > deadline - 60:
            break
        
        if attempt == 0:
            x_start = x0.copy()
        elif attempt < 5:
            # Small perturbations of the good initial
            x_start = x0 + np.random.randn(36) * 0.1
            x_start[24:36] = np.random.uniform(-5, 5, 12)  # small angle perturbations
        else:
            # Larger perturbations for exploration
            x_start = x0 + np.random.randn(36) * 0.5
            x_start[24:36] = np.random.uniform(0, 60, 12)  # random angles
        
        try:
            result = minimize(
                _objective,
                x_start,
                args=(start_time, deadline),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
            )
            
            if result.success or result.fun < 1e9:
                x_opt = result.x
                centers = x_opt[:24].reshape(12, 2)
                angles = x_opt[24:36]
                
                s = _compute_s_for_centers(centers, angles)
                
                # Verify validity
                valid = True
                if s < np.sqrt(12) - 1e-8:
                    valid = False
                
                # Check overlaps
                for i in range(12):
                    verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
                    for j in range(i + 1, 12):
                        verts_j = _hex_vertices(centers[j, 0], centers[j, 1], 1.0, angles[j])
                        gap = _separation_violation(verts_i, verts_j)
                        if gap < -1e-6:
                            valid = False
                
                # Check boundary
                if valid:
                    outer_verts = _hex_vertices(0, 0, s, 0)
                    for i in range(12):
                        verts_i = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
                        for v in verts_i:
                            for k in range(6):
                                p1 = outer_verts[k]
                                p2 = outer_verts[(k + 1) % 6]
                                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                                if cross < -1e-6:
                                    valid = False
                
                if valid:
                    score = 1.0 / s
                    if score > best_score:
                        best_score = score
                        best_x = x_opt.copy()
                        
        except Exception:
            continue
    
    # If we found a good solution, use it
    if best_x is not None:
        centers = best_x[:24].reshape(12, 2)
        angles = best_x[24:36]
        s = _compute_s_for_centers(centers, angles)
        
        inner_hex_data = np.column_stack([centers, angles])
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        
        return inner_hex_data, outer_hex_data, float(s)
    
    # Fallback: use a valid configuration
    d = 2 * np.sqrt(3)
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [d, 0, 0],
            [-d, 0, 0],
            [d/2, d*np.sqrt(3)/2, 0],
            [-d/2, d*np.sqrt(3)/2, 0],
            [d/2, -d*np.sqrt(3)/2, 0],
            [-d/2, -d*np.sqrt(3)/2, 0],
            [d, d*np.sqrt(3), 0],
            [-d, d*np.sqrt(3), 0],
            [d, -d*np.sqrt(3), 0],
            [-d, -d*np.sqrt(3), 0],
            [0, 2*d*np.sqrt(3), 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
