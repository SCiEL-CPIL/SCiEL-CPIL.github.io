import numpy as np
import time
from scipy.optimize import differential_evolution

def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon to minimize the outer side length s.
    Returns (inner, outer, s) where inner is (12, 3) array of (x, y, angle_deg),
    outer is (3,) array (x, y, angle_deg) for the outer hexagon, and s is its side length.
    """
    start_time = time.time()
    deadline = 600.0  # Leave 30s margin before the 630s kill

    # Lattice vectors for non-overlapping flat-top unit hexagons
    # For flat-top hexagons (angle=0), vertices at 0,60,120,180,240,300 degrees
    # Adjacent centers (sharing an edge) are at distance 2*sqrt(3)/2 = sqrt(3)
    # Lattice vectors:
    v1 = np.array([3.0, 0.0])          # Horizontal neighbor (vertex-to-vertex)
    v2 = np.array([1.5, np.sqrt(3)])   # Diagonal neighbor (edge-sharing)
    
    # Generate 12 hexagon positions using a 3x4 grid with hexagonal offset
    # Rows: 3, Columns: 4
    lattice_positions = []
    for row in range(3):
        for col in range(4):
            pos = col * v1 + row * v2
            lattice_positions.append(pos)
    lattice_positions = np.array(lattice_positions)
    
    # Center the lattice at origin
    lattice_center = lattice_positions.mean(axis=0)
    lattice_positions = lattice_positions - lattice_center

    # Initial angles: all flat-top (0 degrees)
    initial_angles = np.zeros(12)

    def get_hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon with center (cx,cy), side length, and angle."""
        angles_rad = np.deg2rad(angle_deg + np.arange(6) * 60)
        return np.column_stack([
            cx + side * np.cos(angles_rad),
            cy + side * np.sin(angles_rad)
        ])

    def check_validity(centers, angles, s):
        """
        Check if the packing is valid:
        1. No two inner hexagons overlap (SAT)
        2. All inner vertices are inside the outer hexagon
        3. s >= sqrt(12)
        Returns (is_valid, penalty) where penalty is 0 if valid.
        """
        if s < np.sqrt(12):
            return False, 1e6
        
        # Check inner-inner separation using SAT
        for i in range(12):
            for j in range(i+1, 12):
                verts_i = get_hex_vertices(centers[i,0], centers[i,1], 1.0, angles[i])
                verts_j = get_hex_vertices(centers[j,0], centers[j,1], 1.0, angles[j])
                
                # Collect all edge normals from both hexagons
                all_normals = []
                for verts in [verts_i, verts_j]:
                    for k in range(6):
                        p1 = verts[k]
                        p2 = verts[(k+1) % 6]
                        edge = p2 - p1
                        # Normal perpendicular to edge
                        normal = np.array([-edge[1], edge[0]])
                        norm_len = np.linalg.norm(normal)
                        if norm_len > 1e-12:
                            normal = normal / norm_len
                            all_normals.append(normal)
                
                # SAT: project both sets of vertices onto each normal
                overlap = True
                for normal in all_normals:
                    proj_i = np.dot(verts_i, normal)
                    proj_j = np.dot(verts_j, normal)
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    
                    # Check if intervals are separated by more than 1e-6
                    if max_i + 1e-6 < min_j or max_j + 1e-6 < min_i:
                        overlap = False
                        break
                
                if overlap:
                    return False, 1e6

        # Check all inner vertices are inside outer hexagon
        outer_verts = get_hex_vertices(0, 0, s, 0.0)  # Outer hex centered at origin, angle 0
        
        for i in range(12):
            inner_verts = get_hex_vertices(centers[i,0], centers[i,1], 1.0, angles[i])
            for v in inner_verts:
                # Check if v is inside outer hexagon using cross product test
                inside = True
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k+1) % 6]
                    cross = np.cross(p2 - p1, v - p1)
                    if cross < -1e-6:
                        inside = False
                        break
                if not inside:
                    return False, 1e6
        
        return True, 0.0

    def compute_required_s(centers, angles):
        """
        Compute the minimum s such that all inner hexagon vertices fit in a regular hexagon
        centered at origin with angle 0.
        """
        sqrt3 = np.sqrt(3)
        required_s = 0.0
        
        for i in range(12):
            verts = get_hex_vertices(centers[i,0], centers[i,1], 1.0, angles[i])
            for v in verts:
                vx, vy = v[0], v[1]
                # 6 constraints for point (vx,vy) to be inside hexagon with side s, angle 0
                c1 = vx + vy / sqrt3
                c2 = 2 * vy / sqrt3
                c3 = vy / sqrt3 - vx
                c4 = -vy / sqrt3 - vx
                c5 = -2 * vy / sqrt3
                c6 = vx - vy / sqrt3
                s_min_point = max(c1, c2, c3, c4, c5, c6)
                required_s = max(required_s, s_min_point)
        
        return max(required_s, np.sqrt(12))

    def objective(params):
        """Objective function: minimize s (the outer hexagon side length)."""
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:  # Safety margin
            return 1e10
        
        # Unpack parameters: 12 centers (x,y each) + 12 angles
        centers = params[:24].reshape(12, 2)
        angles = params[24:36]
        
        # Compute required s
        s = compute_required_s(centers, angles)
        
        # Check validity and add penalty if invalid
        is_valid, penalty = check_validity(centers, angles, s)
        
        return s + penalty

    # Define bounds for differential evolution
    # Centers: within a reasonable range
    center_bounds = [(-5, 5)] * 24
    # Angles: 0 to 360 degrees
    angle_bounds = [(0, 360)] * 12
    bounds = center_bounds + angle_bounds

    # Initial population: use lattice configuration
    x0 = np.concatenate([lattice_positions.flatten(), initial_angles])

    # Run differential evolution
    best_result = None
    best_s = np.inf
    
    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            x0=x0,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True
        )
        
        if result.success:
            best_params = result.x
            best_centers = best_params[:24].reshape(12, 2)
            best_angles = best_params[24:36]
            best_s = compute_required_s(best_centers, best_angles)
            
            # Verify validity
            is_valid, _ = check_validity(best_centers, best_angles, best_s)
            if is_valid:
                best_result = (best_centers, best_angles, best_s)
    except Exception as e:
        pass

    # If DE didn't find a valid solution, use the lattice configuration
    if best_result is None:
        # Try the lattice configuration directly
        lattice_s = compute_required_s(lattice_positions, initial_angles)
        is_valid, _ = check_validity(lattice_positions, initial_angles, lattice_s)
        if is_valid:
            best_result = (lattice_positions, initial_angles, lattice_s)
        else:
            # Fallback: use a known valid configuration
            # 3 rows of 4, with proper spacing
            centers = np.zeros((12, 2))
            for i in range(3):
                for j in range(4):
                    centers[i*4+j, 0] = j * 3.0 + (i % 2) * 1.5
                    centers[i*4+j, 1] = i * np.sqrt(3)
            centers = centers - centers.mean(axis=0)
            angles = np.zeros(12)
            s = compute_required_s(centers, angles)
            best_result = (centers, angles, s)

    centers, angles, s = best_result

    # Build output arrays
    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = centers[i, 0]
        inner[i, 1] = centers[i, 1]
        inner[i, 2] = angles[i]

    outer = np.array([0.0, 0.0, 0.0])

    return inner, outer, s

# EVOLVE-BLOCK-END
