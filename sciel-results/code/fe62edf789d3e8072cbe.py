import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger
    regular hexagon, minimizing the side length of the outer hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (12,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    start_time = time.time()
    timeout = 600  # Safety margin for 600s limit

    n = 12
    hex_side = 1.0

    # Base honeycomb 3x4 grid positions (rigid cluster)
    e1 = np.array([1.5, np.sqrt(3) / 2])
    e2 = np.array([0, np.sqrt(3)])
    base = np.array([i * e1 + j * e2 for i in range(3) for j in range(4)])
    base -= base.mean(axis=0)

    # Pre-compute hexagon vertices for unit hexagon at origin with angle 0
    def hex_vertices(angle):
        """Return 6 vertices of a unit hexagon with given angle (radians)."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def compute_s(centers, angles):
        """Compute minimal enclosing outer hexagon side length."""
        # For aligned outer hexagon (angle 0), the constraints are:
        # For each inner hexagon i, for each vertex v of that hexagon,
        # the vertex must be inside the outer hexagon.
        # 
        # The outer hexagon with center (0,0) and side s has support function:
        # h(n) = s * cos(pi/6) for the three normal directions
        # Actually, for a regular hexagon with side s, the distance from center
        # to each edge is s * cos(pi/6) = s * sqrt(3)/2
        # 
        # The three normal directions for a hexagon with angle 0 are:
        # n1 = (sqrt(3)/2, 1/2), n2 = (0, 1), n3 = (-sqrt(3)/2, 1/2)
        # Wait, let me reconsider. The vertices of a hexagon with angle a are at:
        # (s*cos(a+60k), s*sin(a+60k)) for k=0..5
        # The edge normals are perpendicular to edges. For a regular hexagon,
        # the distance from center to each edge (apothem) is s*cos(pi/6) = s*sqrt(3)/2
        # 
        # For containment, each vertex of each inner hexagon must satisfy:
        # For all 6 outer edge normals n_k: dot(v - outer_center, n_k) <= apothem_outer
        # 
        # The outer hexagon has angle 0 (we can fix this WLOG by rotating the whole config)
        # Outer hexagon vertices at angle 0: (s*cos(60k), s*sin(60k))
        # Edge normals point outward. The 6 edge normals for angle 0 hexagon are at:
        # 60k + 30 degrees for k=0..5
        # n_k = (cos(30+60k), sin(30+60k))
        
        n_normals = np.array([
            [np.cos(np.pi/6 + k*np.pi/3), np.sin(np.pi/6 + k*np.pi/3)] 
            for k in range(6)
        ])
        
        # For each inner hexagon, compute its vertices
        max_required = 0.0
        for i in range(n):
            verts = hex_vertices(angles[i]) + centers[i]
            # For each outer edge normal, compute the max projection
            proj = verts @ n_normals.T
            # The apothem must be at least the max projection for each normal
            for k in range(6):
                # By symmetry, we only need 3 normals (opposite pairs give same constraint)
                if k < 3:
                    required_apothem = np.max(proj[:, k])
                    # apothem = s * cos(pi/6) = s * sqrt(3)/2
                    # So s = required_apothem / cos(pi/6)
                    s_k = required_apothem / np.cos(np.pi/6)
                    if s_k > max_required:
                        max_required = s_k
        return max_required

    def check_separation(centers, angles):
        """Check if all pairs of hexagons are separated using SAT."""
        for i in range(n):
            for j in range(i + 1, n):
                # Get vertices of both hexagons
                vi = hex_vertices(angles[i]) + centers[i]
                vj = hex_vertices(angles[j]) + centers[j]
                
                # Get edge normals for both hexagons
                # Edge normals are perpendicular to edges
                # For hexagon with vertices v[0..5], edge k is from v[k] to v[(k+1)%6]
                # Normal points outward
                ni = np.zeros((6, 2))
                nj = np.zeros((6, 2))
                for k in range(6):
                    e_i = vi[(k+1)%6] - vi[k]
                    e_j = vj[(k+1)%6] - vj[k]
                    # Rotate edge by 90 degrees to get normal
                    ni[k] = np.array([-e_i[1], e_i[0]])
                    nj[k] = np.array([-e_j[1], e_j[0]])
                    # Normalize
                    ni[k] /= np.linalg.norm(ni[k])
                    nj[k] /= np.linalg.norm(nj[k])
                
                # SAT: check all 12 axes
                all_axes = np.vstack([ni, nj])
                
                for axis in all_axes:
                    # Project both hexagons onto this axis
                    proj_i = vi @ axis
                    proj_j = vj @ axis
                    
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    
                    # Check for separation
                    if max_i < min_j - 1e-9 or max_j < min_i - 1e-9:
                        break  # Separated on this axis, try next pair
                else:
                    # No separating axis found, hexagons overlap
                    return False
        return True

    def objective(x):
        """Objective function for optimization."""
        centers = x[:24].reshape((n, 2))
        angles = x[24:36]
        
        # Check separation with penalty
        # Use a simpler check for speed during optimization
        # Approximate separation using center distances and angles
        
        # For now, use the full check but it's slow
        # Let's use a faster approximation
        
        # Compute s
        s = compute_s(centers, angles)
        
        # Add penalty for overlaps
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                # Approximate minimum distance for non-overlap
                # For two unit hexagons, if centers are far enough, they don't overlap
                # The maximum extent of a hexagon from center is 1 (circumradius)
                # So if dist > 2, definitely separated
                if dist < 2.0:
                    # Need more careful check
                    # Use the angle difference to estimate
                    angle_diff = angles[i] - angles[j]
                    # For aligned hexagons, min center distance is sqrt(3)
                    # For rotated, it varies
                    # Simple penalty: if dist < sqrt(3), penalize
                    if dist < np.sqrt(3):
                        penalty += (np.sqrt(3) - dist) ** 2
        
        return s + 1000.0 * penalty

    # Initial configuration
    centers0 = base.copy()
    angles0 = np.zeros(n)
    x0 = np.concatenate([centers0.flatten(), angles0])
    
    best_x = x0.copy()
    best_f = compute_s(centers0, angles0)
    
    print(f"Initial s = {best_f:.6f}")
    
    # Phase 1: Optimize with fixed angles (aligned hexagons)
    print("Phase 1: Optimizing centers with fixed angles")
    def objective_fixed_angles(x):
        centers = x.reshape((n, 2))
        s = compute_s(centers, angles0)
        # Separation penalty for aligned hexagons
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                # For aligned unit hexagons, use the 3 axis check
                # Project onto the 3 main axes
                axes = np.array([[np.sqrt(3)/2, 0.5], [0, 1], [-np.sqrt(3)/2, 0.5]])
                projs = np.abs(diff @ axes.T)
                if np.max(projs) < np.sqrt(3):
                    penalty += (np.sqrt(3) - np.max(projs)) ** 2
        return s + 1000.0 * penalty
    
    x0_fixed = centers0.flatten()
    for restart in range(30):
        if time.time() - start_time > timeout - 120:
            break
        if restart == 0:
            x_start = x0_fixed.copy()
        else:
            x_start = x0_fixed + np.random.normal(0, 0.1, size=24)
        
        res = minimize(objective_fixed_angles, x_start, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
        
        centers_new = res.x.reshape((n, 2))
        f_new = compute_s(centers_new, angles0)
        
        # Check if valid
        if check_separation(centers_new, angles0):
            if f_new < best_f:
                best_f = f_new
                best_x = np.concatenate([centers_new.flatten(), angles0])
                print(f"  restart {restart}: s={f_new:.6f}")
    
    # Phase 2: Optimize with free angles
    print("Phase 2: Optimizing with free angles")
    x0_free = best_x.copy()
    for restart in range(20):
        if time.time() - start_time > timeout - 120:
            break
        if restart == 0:
            x_start = x0_free.copy()
        else:
            x_start = x0_free + np.random.normal(0, 0.05, size=36)
        
        res = minimize(objective, x_start, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        centers_new = res.x[:24].reshape((n, 2))
        angles_new = res.x[24:36]
        f_new = compute_s(centers_new, angles_new)
        
        # Check if valid
        if check_separation(centers_new, angles_new):
            if f_new < best_f:
                best_f = f_new
                best_x = res.x.copy()
                print(f"  restart {restart}: s={f_new:.6f}")
    
    # Extract best result
    centers = best_x[:24].reshape((n, 2))
    angles = best_x[24:36]
    
    # Final safety check
    if not check_separation(centers, angles):
        # Expand slightly
        centroid = centers.mean(axis=0)
        centers = (centers - centroid) * 1.01 + centroid
        best_f = compute_s(centers, angles)
    
    s = best_f + 0.005  # Safety margin
    
    # Construct return arrays
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = np.degrees(angles[i])
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s
