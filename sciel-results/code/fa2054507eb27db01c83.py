import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import itertools

def hexagon_packing_11():
    start_time = time.time()
    deadline = 630.0
    
    # Unit hexagon vertices (angle 0)
    hex_verts_base = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])
    
    def get_vertices(center, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rot = np.array([[c, -s], [s, c]])
        return center + hex_verts_base @ rot.T

    def get_normals(vertices):
        # Returns unit normals for the 6 edges
        normals = []
        for i in range(6):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % 6]
            edge_vec = v2 - v1
            n = np.array([edge_vec[1], -edge_vec[0]])
            n = n / np.linalg.norm(n)
            normals.append(n)
        return np.array(normals)

    def max_penetration(vertices1, vertices2):
        # SAT for two convex polygons
        # Returns max penetration depth (0 if disjoint)
        max_pen = 0.0
        
        normals1 = get_normals(vertices1)
        normals2 = get_normals(vertices2)
        
        all_normals = np.vstack([normals1, normals2])
        
        for n in all_normals:
            # Project vertices1
            proj1 = vertices1 @ n
            min1 = np.min(proj1)
            max1 = np.max(proj1)
            
            # Project vertices2
            proj2 = vertices2 @ n
            min2 = np.min(proj2)
            max2 = np.max(proj2)
            
            # Check for gap
            if max1 < min2 or max2 < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap
            overlap = np.minimum(max1, max2) - np.maximum(min1, min2)
            if overlap > max_pen:
                max_pen = overlap
                
        return max_pen

    def compute_side_length(centers, angles, theta_outer_deg):
        """Compute the outer hexagon side length needed to contain all inner hexagons."""
        all_vertices = []
        for i in range(11):
            verts = get_vertices(centers[i], angles[i])
            all_vertices.append(verts)
        
        theta_outer_rad = np.deg2rad(theta_outer_deg)
        outer_normals = []
        for i in range(6):
            a = theta_outer_rad + np.deg2rad(60 * i)
            outer_normals.append([np.cos(a), np.sin(a)])
        outer_normals = np.array(outer_normals)
        
        all_verts_flat = np.vstack(all_vertices)
        proj = all_verts_flat @ outer_normals.T
        max_proj = np.max(proj, axis=0)
        
        # Apothem of outer hexagon = max projection
        # Side length S = 2 * apothem / sqrt(3)
        side = (2.0 / np.sqrt(3.0)) * np.max(max_proj)
        return side, all_vertices

    def check_validity(centers, angles, theta_outer_deg, tol=1e-6):
        """Check if the configuration is valid (no overlaps, all inside outer hex)."""
        side, all_vertices = compute_side_length(centers, angles, theta_outer_deg)
        
        if not np.isfinite(side) or side < np.sqrt(11):
            return False, None
        
        # Check no overlaps between inner hexagons
        for i in range(11):
            for j in range(i + 1, 11):
                pen = max_penetration(all_vertices[i], all_vertices[j])
                if pen > tol:
                    return False, None
        
        # Check all inner vertices are inside outer hexagon
        theta_outer_rad = np.deg2rad(theta_outer_deg)
        outer_center = np.array([0.0, 0.0])
        outer_verts = get_vertices(outer_center, theta_outer_deg)
        
        # For each outer edge, check all inner vertices
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            edge_vec = p2 - p1
            # Cross product test: all inner vertices should be on the same side
            for verts in all_vertices:
                for p in verts:
                    cp = np.cross(edge_vec, p - p1)
                    if cp < -tol:
                        return False, None
        
        return True, side

    def objective(params):
        """Objective function for optimization. params: [x1,y1,a1,...,x11,y11,a11,theta_outer]"""
        if time.time() > deadline - 10:
            return 1e10
            
        data = params[:33].reshape(11, 3)
        theta_outer_deg = params[33]
        
        centers = data[:, :2]
        angles = data[:, 2]
        
        # Compute vertices
        all_vertices = []
        for i in range(11):
            verts = get_vertices(centers[i], angles[i])
            all_vertices.append(verts)
            
        # Check overlaps
        pen_sum = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                pen = max_penetration(all_vertices[i], all_vertices[j])
                if pen > 0:
                    pen_sum += pen * pen
        
        # Calculate side length
        side, _ = compute_side_length(centers, angles, theta_outer_deg)
        
        # Penalty for overlaps
        weight = 1000.0
        return side + weight * pen_sum

    # Strategy 1: Differential Evolution with cluster-based initialization
    # The cluster has a central hexagon, 6 surrounding, and 4 more
    d = np.sqrt(3)
    
    def make_cluster_init(seed_idx):
        """Create initial cluster configuration with some randomness."""
        rng = np.random.RandomState(seed_idx)
        
        # Center hexagon
        centers = [[0.0, 0.0]]
        angles = [0.0]
        
        # Ring 1: 6 hexagons around center
        for i in range(6):
            angle_pos = np.deg2rad(30 + 60 * i + rng.uniform(-10, 10))
            dist = d * rng.uniform(0.95, 1.05)
            cx = dist * np.cos(angle_pos)
            cy = dist * np.sin(angle_pos)
            centers.append([cx, cy])
            angles.append(rng.uniform(-15, 15))
        
        # Ring 2: 4 hexagons in gaps
        gap_angles = [0, 60, 120, 180, 240, 300]
        # Choose 4 of 6 gaps
        chosen_gaps = rng.choice(6, size=4, replace=False)
        for idx in sorted(chosen_gaps):
            angle_pos = np.deg2rad(gap_angles[idx] + rng.uniform(-10, 10))
            dist = rng.uniform(2.8, 3.5)
            cx = dist * np.cos(angle_pos)
            cy = dist * np.sin(angle_pos)
            centers.append([cx, cy])
            angles.append(rng.uniform(-30, 30))
        
        centers = np.array(centers)
        angles = np.array(angles)
        
        # Flatten
        params = []
        for i in range(11):
            params.extend([centers[i][0], centers[i][1], angles[i]])
        params.append(rng.uniform(-30, 30))  # Outer angle
        
        return np.array(params)

    # Strategy 2: Rigid body cluster optimization
    # Represent the cluster as: center position, center angle, 
    # then relative positions and angles of 10 other hexagons
    def rigid_body_objective(params):
        """
        Rigid body representation:
        params[0:2] = center position
        params[2] = center angle
        params[3:35] = 10 * (dx, dy, dangle) for other hexagons relative to center
        params[35] = outer hexagon angle
        """
        if time.time() > deadline - 10:
            return 1e10
            
        cx, cy = params[0:2]
        ca = params[2]
        
        centers = [[cx, cy]]
        angles = [ca]
        
        for i in range(10):
            base = 3 + i * 3
            dx, dy, da = params[base:base+3]
            centers.append([cx + dx, cy + dy])
            angles.append(ca + da)
        
        theta_outer = params[35]
        
        centers = np.array(centers)
        angles = np.array(angles)
        
        # Compute vertices
        all_vertices = []
        for i in range(11):
            verts = get_vertices(centers[i], angles[i])
            all_vertices.append(verts)
            
        # Check overlaps
        pen_sum = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                pen = max_penetration(all_vertices[i], all_vertices[j])
                if pen > 0:
                    pen_sum += pen * pen
        
        # Calculate side length
        side, _ = compute_side_length(centers, angles, theta_outer)
        
        weight = 1000.0
        return side + weight * pen_sum

    best_result = None
    best_side = float('inf')
    
    # Phase 1: Differential Evolution on rigid body representation
    # Bounds: center position in [-5,5]^2, center angle in [-30,30],
    # relative positions in [-4,4]^2, relative angles in [-60,60], outer angle in [-30,30]
    
    if time.time() < deadline - 120:
        bounds = [(-5, 5), (-5, 5), (-30, 30)]
        for _ in range(10):
            bounds.append((-4, 4))
            bounds.append((-4, 4))
            bounds.append((-60, 60))
        bounds.append((-30, 30))
        
        try:
            de_result = differential_evolution(
                rigid_body_objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=True
            )
            
            if de_result.fun < best_side:
                # Convert rigid body params to standard format
                params = de_result.x
                cx, cy = params[0:2]
                ca = params[2]
                
                centers = [[cx, cy]]
                angles = [ca]
                
                for i in range(10):
                    base = 3 + i * 3
                    dx, dy, da = params[base:base+3]
                    centers.append([cx + dx, cy + dy])
                    angles.append(ca + da)
                
                theta_outer = params[35]
                
                centers = np.array(centers)
                angles = np.array(angles)
                
                valid, side = check_validity(centers, angles, theta_outer)
                if valid and side < best_side:
                    best_side = side
                    inner_hex_data = np.column_stack([centers, angles])
                    outer_hex_data = np.array([0, 0, theta_outer])
                    best_result = (inner_hex_data, outer_hex_data, best_side)
                    
        except Exception as e:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead from best DE result
    if best_result is not None:
        if time.time() < deadline - 30:
            inner_hex_data, outer_hex_data, _ = best_result
            centers = inner_hex_data[:, :2]
            angles = inner_hex_data[:, 2]
            theta_outer = outer_hex_data[2]
            
            init_params = []
            for i in range(11):
                init_params.extend([centers[i][0], centers[i][1], angles[i]])
            init_params.append(theta_outer)
            init_params = np.array(init_params)
            
            try:
                res = minimize(objective, init_params, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
                
                if res.fun < best_side:
                    data = res.x[:33].reshape(11, 3)
                    theta_outer_deg = res.x[33]
                    centers = data[:, :2]
                    angles = data[:, 2]
                    
                    valid, side = check_validity(centers, angles, theta_outer_deg)
                    if valid and side < best_side:
                        best_side = side
                        inner_hex_data = np.column_stack([centers, angles])
                        outer_hex_data = np.array([0, 0, theta_outer_deg])
                        best_result = (inner_hex_data, outer_hex_data, best_side)
                        
            except Exception as e:
                pass
    
    # Phase 3: Try multiple random cluster initializations with local search
    np.random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 30:
            break
            
        try:
            init = make_cluster_init(100 + trial)
            
            res = minimize(objective, init, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7, 'disp': False})
            
            if res.fun < best_side:
                data = res.x[:33].reshape(11, 3)
                theta_outer_deg = res.x[33]
                centers = data[:, :2]
                angles = data[:, 2]
                
                valid, side = check_validity(centers, angles, theta_outer_deg)
                if valid and side < best_side:
                    best_side = side
                    inner_hex_data = np.column_stack([centers, angles])
                    outer_hex_data = np.array([0, 0, theta_outer_deg])
                    best_result = (inner_hex_data, outer_hex_data, best_side)
                    
        except Exception as e:
            pass
    
    if best_result is None:
        # Fallback: simple grid arrangement
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    return best_result
