import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    sqrt3 = np.sqrt(3.0)
    
    # Precompute constants
    # Angles for vertices: 0, 60, 120, 180, 240, 300 degrees
    vertex_angles = np.deg2rad(np.arange(0, 360, 60))
    # Normals for edges: 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.deg2rad(np.arange(30, 390, 60))
    normals_base = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    def get_hex_vertices(cx, cy, angle_deg, s=1.0):
        """
        Compute vertices of a hexagon.
        Vertices are at angles angle + 0, 60, ..., 300.
        """
        a = np.deg2rad(angle_deg)
        angles = a + vertex_angles
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def get_hex_normals(angle_deg):
        """
        Compute edge normals for a hexagon with given angle.
        Normals are perpendicular to edges, pointing outwards.
        Edge i connects vertex i and i+1.
        Normal angle is angle + 30, 90, ...
        """
        a = np.deg2rad(angle_deg)
        angles = a + np.deg2rad(np.arange(30, 390, 60))
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def compute_min_side(vertices_flat, outer_angle_deg):
        """
        Compute minimum side length S such that outer hex (centered at origin)
        contains all vertices in vertices_flat (N, 2).
        """
        normals = get_hex_normals(outer_angle_deg)
        # projections (N, 6)
        projections = vertices_flat @ normals.T
        max_proj = np.max(projections, axis=0)
        min_proj = np.min(projections, axis=0)
        # Distance from center to edge must be >= max_proj and >= -min_proj
        # Apothem = S * sqrt(3)/2
        needed = np.maximum(max_proj, -min_proj)
        return 2.0 * np.max(needed) / sqrt3

    def check_separation_pair(v1, v2, tol=1e-6):
        """
        Check if two hexagons v1, v2 (6,2) are separated.
        Returns True if separated (no intersection).
        """
        n1 = get_hex_normals(0) # Normals are relative, we need to rotate them? 
        # Wait, v1 and v2 are already in Cartesian coords. 
        # The normals must be aligned with the edges of the hexagons themselves.
        # We need the normals of v1 and v2 in global frame.
        
        # Re-calculate normals from the vertices directly to be safe and precise
        def get_normals_from_verts(verts):
            normals = np.zeros((6, 2))
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i + 1) % 6]
                edge = p2 - p1
                # Normal is perpendicular to edge
                # For a regular hexagon, the outward normal points away from center.
                # Let's check direction: cross(edge, normal) should be positive? 
                # Or just check distance from origin?
                # Simpler: 
                n = np.array([-edge[1], edge[0]])
                # Normalize
                n /= np.linalg.norm(n)
                # Ensure it points outward
                # Midpoint of edge
                mid = (p1 + p2) / 2.0
                if np.dot(n, mid) < 0:
                    n = -n
                normals[i] = n
            return normals

        n1 = get_normals_from_verts(v1)
        n2 = get_normals_from_verts(v2)
        
        all_normals = np.vstack([n1, n2])
        
        for n in all_normals:
            proj1 = v1 @ n
            proj2 = v2 @ n
            if np.min(proj2) - np.max(proj1) > tol or np.min(proj1) - np.max(proj2) > tol:
                return True # Separated on this normal
        
        return False # Not separated on any normal -> Intersecting

    def evaluate_packing(positions, angles_deg, outer_angle_deg):
        """
        positions: (11, 2)
        angles_deg: (11,)
        outer_angle_deg: scalar
        Returns (is_valid, side_length, penalty)
        """
        verts_list = []
        all_verts = []
        for i in range(11):
            v = get_hex_vertices(positions[i, 0], positions[i, 1], angles_deg[i], 1.0)
            verts_list.append(v)
            all_verts.append(v)
        all_verts = np.concatenate(all_verts, axis=0)  # (66, 2)

        S = compute_min_side(all_verts, outer_angle_deg)
        
        # Check containment
        outer_verts = get_hex_vertices(0, 0, outer_angle_deg, S)
        # Check if all inner vertices are inside outer hex
        # For each edge of outer hex, check cross product
        n_outer = 6
        containment_ok = True
        for i in range(n_outer):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % n_outer]
            edge = p2 - p1
            # Normal pointing inward? 
            # Let's use the cross product test: (p2-p1) x (p-p1) should be >= 0 for CCW polygon
            # Outer hex is CCW?
            # Vertices: 0, 60, 120... -> CCW.
            # So cross > 0 means inside.
            cross_vals = (p2[0] - p1[0]) * (all_verts[:, 1] - p1[1]) - (p2[1] - p1[1]) * (all_verts[:, 0] - p1[0])
            if np.min(cross_vals) < -1e-6:
                containment_ok = False
                break
        
        if not containment_ok:
            # Calculate how much it violates to create a penalty
            # We can just return a large penalty
            return False, S, 1e10

        # Check separation
        separation_ok = True
        max_overlap_penalty = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                if not check_separation_pair(verts_list[i], verts_list[j], tol=1e-6):
                    separation_ok = False
                    # Estimate overlap depth for penalty
                    # Use the normal with smallest gap
                    v1 = verts_list[i]
                    v2 = verts_list[j]
                    n1 = get_normals_from_verts(v1)
                    n2 = get_normals_from_verts(v2)
                    all_normals = np.vstack([n1, n2])
                    min_gap = np.inf
                    for n in all_normals:
                        proj1 = v1 @ n
                        proj2 = v2 @ n
                        gap1 = np.min(proj2) - np.max(proj1)
                        gap2 = np.min(proj1) - np.max(proj2)
                        # If gap is negative, they overlap on this axis
                        # The "overlap" is the amount by which they fail to separate
                        # We want to maximize the minimum separation.
                        # If min_gap < 0, they overlap.
                        # Penalty should be proportional to -min_gap
                        min_gap = min(min_gap, gap1, gap2)
                    # min_gap is the best (largest) separation found. If < 0, it's the overlap depth.
                    overlap = max(0, -min_gap)
                    max_overlap_penalty += overlap * 100.0 # Scale penalty
        
        if not separation_ok:
            return False, S, max_overlap_penalty

        # Check edge lengths (should be 1.0)
        edge_ok = True
        for v in verts_list:
            for k in range(6):
                p1 = v[k]
                p2 = v[(k + 1) % 6]
                if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                    edge_ok = False
        if not edge_ok:
            return False, S, 1e10

        if S < np.sqrt(11.0) - 1e-6:
            return False, S, 1e10

        return True, S, 0.0

    def objective(params):
        """
        Objective function for optimization.
        params: (34,) -> 11*(x,y,angle) + outer_angle
        """
        if time.time() > deadline - 5:
            return 1e10
            
        positions = params[:33].reshape(11, 3)
        outer_angle = params[33]
        
        # Unwrap angles to [0, 360) for consistency? Not strictly necessary for cos/sin
        x = positions[:, 0]
        y = positions[:, 1]
        angles = positions[:, 2]
        
        valid, S, penalty = evaluate_packing(x, y, angles, outer_angle)
        
        # The objective is S + penalty
        # If invalid due to overlap, penalty is large.
        # If invalid due to containment, penalty is huge.
        return S + penalty

    # Initial configurations
    # We will generate a few good starting points based on lattice structures
    
    def lattice_to_cartesian(q, r):
        x = sqrt3 * (q + 0.5 * r)
        y = 1.5 * r
        return x, y

    # Set 1: 1 + 6 + 4 (Compact)
    set1_axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (0,2), (-2,2), (2,-2)]
    # Set 2: 1 + 6 + 4 (Different)
    set2_axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,-1), (-2,1), (-1,2), (1,-2)]
    # Set 3: Row-based
    set3_axial = [(-2,-1), (-1,-1), (0,-1), (1,-1), (2,-1), (-1,0), (0,0), (1,0), (-2,1), (0,1), (2,1)]
    # Set 4: Diamond
    set4_axial = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1), (2,0), (0,2)]

    initial_sets = [set1_axial, set2_axial, set3_axial, set4_axial]
    
    best_S = np.inf
    best_params = None

    for set_idx, axial_set in enumerate(initial_sets):
        if time.time() > deadline - 30:
            break
            
        # Convert to Cartesian
        positions_cart = np.array([lattice_to_cartesian(q, r) for q, r in axial_set])
        
        # Initial angles: all 0
        angles_init = np.zeros(11)
        
        # Initial outer angle: 0
        outer_init = 0.0
        
        # Build initial params
        init_params = np.zeros(34)
        init_params[0:11] = positions_cart[:, 0]
        init_params[11:22] = positions_cart[:, 1]
        init_params[22:33] = angles_init
        init_params[33] = outer_init
        
        # Try a few variations of initial outer angle
        for outer_start in [0, 30, 60]:
            if time.time() > deadline - 10:
                break
                
            init_params[33] = outer_start
            
            # Run Nelder-Mead
            try:
                res = minimize(objective, init_params, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
                
                if res.fun < best_S:
                    # Validate the result strictly
                    pos = res.x[:33].reshape(11, 3)
                    outer_a = res.x[33]
                    valid, S, pen = evaluate_packing(pos[:, 0], pos[:, 1], pos[:, 2], outer_a)
                    if valid and S < best_S:
                        best_S = S
                        best_params = res.x.copy()
                        
            except Exception:
                pass

    # If no valid result found via optimization, fallback to a known valid configuration
    if best_params is None:
        # Use a simple valid configuration, e.g., spread out
        # We need to ensure it's valid.
        # Let's construct one manually or use the first lattice set and scale it up?
        # No, side length is fixed by the packing.
        # If optimization failed, we return a large valid packing.
        
        # Fallback: Use Set 1 but ensure it's valid by checking.
        # Actually, the lattice sets are valid packings (no overlap) but might not fit in the minimal hex.
        # We just need *a* valid packing.
        # Let's take Set 1, compute its bounding hex side S, and return that.
        axial_set = set1_axial
        positions_cart = np.array([lattice_to_cartesian(q, r) for q, r in axial_set])
        angles_init = np.zeros(11)
        outer_init = 0.0
        
        valid, S, pen = evaluate_packing(positions_cart[:, 0], positions_cart[:, 1], angles_init, outer_init)
        
        if not valid:
            # If even this fails (shouldn't happen for lattice), spread them more
            # Create a grid
            positions_cart = np.array([
                [0, 0], [-3, 0], [3, 0], [-1.5, 2.6], [1.5, 2.6],
                [-1.5, -2.6], [1.5, -2.6], [-4.5, 2.6], [4.5, 2.6],
                [-4.5, -2.6], [4.5, -2.6]
            ])
            angles_init = np.zeros(11)
            outer_init = 0.0
            valid, S, pen = evaluate_packing(positions_cart[:, 0], positions_cart[:, 1], angles_init, outer_init)
        
        if valid:
            inner_hex_data = np.column_stack([positions_cart[:, 0], positions_cart[:, 1], angles_init])
            outer_hex_data = np.array([0.0, 0.0, outer_init])
            return inner_hex_data, outer_hex_data, float(S)
        else:
            # Emergency fallback: return something that might be invalid but better than crashing?
            # No, must be valid.
            # This case should be extremely rare.
            raise ValueError("Failed to find a valid packing")

    # Format best result
    pos = best_params[:33].reshape(11, 3)
    outer_a = best_params[33]
    
    inner_hex_data = np.column_stack([pos[:, 0], pos[:, 1], pos[:, 2]])
    outer_hex_data = np.array([0.0, 0.0, float(outer_a)])
    
    return inner_hex_data, outer_hex_data, float(best_S)

if __name__ == "__main__":
    inner, outer, side = hexagon_packing_11()
    print(f"Side: {side}")
    print(f"Inner: {inner.shape}")
    print(f"Outer: {outer}")
