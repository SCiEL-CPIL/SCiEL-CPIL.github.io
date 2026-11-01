import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_edge_normals(verts):
        """Returns 6x2 array of outward unit normals for a convex hexagon."""
        edges = np.roll(verts, -1, axis=0) - verts
        # For a counter-clockwise polygon, the outward normal to edge (dx, dy) is (dy, -dx)
        # Wait, let's check orientation.
        # vertices are generated with increasing angle, so CCW.
        # Edge vector e = v[i+1] - v[i].
        # Normal pointing outward: rotate e by -90 degrees?
        # If e = (1, 0), normal should be (0, -1) for bottom edge? No, (0,1) is up.
        # Let's just compute and verify with a known case.
        # Center (0,0), side 1, angle 0. Vertices: (1,0), (0.5, 0.866), ...
        # Edge 0: (1,0) to (0.5, 0.866). e = (-0.5, 0.866).
        # Outward normal should point roughly in direction (1, 0.5) or similar?
        # The face is at x=0.75? No.
        # Let's use the cross product rule.
        # For CCW polygon, outward normal of edge (dx, dy) is (dy, -dx).
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / norms[:, np.newaxis]
        return normals

    def sat_penetration(v1, v2):
        """
        Compute penetration depth between two convex polygons v1, v2 (6x2).
        Returns 0 if disjoint or separated, otherwise the minimum translation distance to separate.
        """
        normals1 = get_edge_normals(v1)
        normals2 = get_edge_normals(v2)
        normals = np.vstack([normals1, normals2])
        
        # Project v1 and v2 onto all normals
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        # Overlap on each axis
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        
        # If any overlap is <= 0, they are separated
        if np.min(overlaps) <= 1e-9:
            return 0.0
            
        # The minimum overlap is the penetration depth
        return np.min(overlaps)

    def containment_violation(inner_verts, outer_verts):
        """
        Compute how much inner_verts sticks out of outer_verts.
        outer_verts is a convex hexagon.
        Returns max violation (positive if sticking out).
        """
        outer_normals = get_edge_normals(outer_verts)
        
        # Project inner vertices onto outer normals
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        
        p_inner_max = p_inner.max(axis=0)
        
        viol = p_inner_max - p_outer_max
        return np.max(viol)

    def evaluate(params, alpha_rad, theta_rad):
        """
        params: [x1, y1, ..., x12, y12, s]
        Returns (side_length, penalty)
        """
        positions = params[:2 * N].reshape(N, 2)
        s = params[2 * N]
        
        if s < 2.0: # Absolute minimum for 12 hexagons is roughly > 3
            return s, 1e6

        outer_verts = hex_vertices(0, 0, s, theta_rad)
        
        penalty = 0.0
        inner_verts_list = []
        
        for i in range(N):
            x, y = positions[i]
            verts = hex_vertices(x, y, 1.0, alpha_rad)
            inner_verts_list.append(verts)
            
            # Containment
            viol = containment_violation(verts, outer_verts)
            if viol > 0:
                penalty += viol * 100.0  # High weight for containment
        
        # Pairwise disjointness
        for i in range(N):
            for j in range(i + 1, N):
                pd = sat_penetration(inner_verts_list[i], inner_verts_list[j])
                if pd > 0:
                    penalty += pd * 100.0  # High weight for overlap

        return s, penalty

    def objective(params, alpha_rad, theta_rad):
        if time.time() > deadline:
            return 1e10
        s, penalty = evaluate(params, alpha_rad, theta_rad)
        return s + penalty

    # Initial configuration: Honeycomb cluster
    # 1 center, 6 ring, 5 outer
    d1 = np.sqrt(3)  # Distance for touching hexagons in honeycomb
    d2 = 2 * np.sqrt(3)
    
    init_pos = []
    # Center
    init_pos.append([0.0, 0.0])
    # Ring 1
    for k in range(6):
        a = np.radians(60 * k)
        init_pos.append([d1 * np.cos(a), d1 * np.sin(a)])
    # Ring 2 (5 hexagons)
    # In a honeycomb, the next layer is at distance 2*sqrt(3) from center, at angles 0, 60, ...
    # But we only need 5. Let's pick 5 symmetric ones.
    # Actually, the full ring 2 has 12 spots? No, hexagonal packing layers are 1, 6, 12, 18...
    # Layer 2 (distance 2*sqrt(3)) has 12 spots?
    # Let's check.
    # Center (0,0).
    # Layer 1: (sqrt(3), 0), (sqrt(3)/2, 1.5), ...
    # Layer 2: (2*sqrt(3), 0), ...
    # The spots at distance 2*sqrt(3) are at angles 0, 30, 60, 90, ...?
    # No, the hexagonal lattice points are at:
    # (0,0)
    # (sqrt(3), 0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), (-sqrt(3), 0), ...
    # (2*sqrt(3), 0), (3*sqrt(3)/2, 3), ...
    # Let's just use the 6 "corner" spots of the second layer ring which are at distance 2*sqrt(3) from center, aligned with the first layer.
    # Plus maybe some others?
    # For 12 hexagons, a compact shape is 1+6+5.
    # The 5 outer ones can be placed at 5 of the 6 positions of the next "ring" of the honeycomb?
    # The next ring in honeycomb (distance 2*sqrt(3) from center? No, that's not a ring of touching hexagons to the previous ring in all directions).
    # Actually, the hexagons at distance 2*sqrt(3) from the center touch the ring 1 hexagons.
    # There are 12 such positions?
    # Let's just place them at angles 0, 60, 120, 180, 240 with radius 2*sqrt(3).
    for k in range(5):
        a = np.radians(60 * k)
        init_pos.append([d2 * np.cos(a), d2 * np.sin(a)])
        
    init_pos = np.array(init_pos)
    init_s = 4.5 # Guess
    
    best_s = np.inf
    best_params = None
    best_alpha = 0.0
    best_theta = 0.0
    best_penalty = np.inf

    # Grid search over rotations
    # Inner rotation alpha, Outer rotation theta
    # Due to symmetry, we can limit search range.
    # Hexagon has 60 degree rotational symmetry.
    alphas = np.radians([0, 30]) # Try 0 and 30
    thetas = np.radians([0, 30]) # Try 0 and 30
    
    # We can also try random restarts if time permits
    # But first, let's run the main grid
    
    for alpha in alphas:
        if time.time() > deadline - 20:
            break
        for theta in thetas:
            if time.time() > deadline - 20:
                break
                
            x0 = np.concatenate([init_pos.flatten(), [init_s]])
            
            try:
                res = minimize(
                    objective,
                    x0,
                    args=(alpha, theta),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                s, penalty = evaluate(res.x, alpha, theta)
                
                if penalty < 1e-4 and s < best_s:
                    best_s = s
                    best_params = res.x.copy()
                    best_alpha = alpha
                    best_theta = theta
                    best_penalty = penalty
                    
            except Exception:
                pass

    # If we have time, try some random perturbations of the initial position
    # to escape local minima
    if best_params is not None and time.time() < deadline - 30:
        # Try a few random restarts
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            
            # Perturb initial positions
            perturbation = np.random.normal(0, 0.2, init_pos.shape)
            x0_perturbed = np.concatenate([(init_pos + perturbation).flatten(), [init_s + 0.5]])
            
            for alpha in alphas:
                for theta in thetas:
                    try:
                        res = minimize(
                            objective,
                            x0_perturbed,
                            args=(alpha, theta),
                            method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                        )
                        
                        s, penalty = evaluate(res.x, alpha, theta)
                        
                        if penalty < 1e-4 and s < best_s:
                            best_s = s
                            best_params = res.x.copy()
                            best_alpha = alpha
                            best_theta = theta
                            best_penalty = penalty
                    except Exception:
                        pass

    # If no valid solution found, use a larger fallback
    if best_params is None:
        # Fallback: Just return a large enough container
        # The minimal side length for 12 unit hexagons is at least sqrt(3) * 3 / 2?
        # Let's just use 5.0 to be safe
        best_s = 5.0
        best_params = np.concatenate([init_pos.flatten(), [best_s]])
        best_alpha = 0
        best_theta = 0

    # Extract final positions
    final_pos = best_params[:2 * N].reshape(N, 2)
    final_s = best_params[2 * N]
    
    # The outer center is (0,0) in our optimization frame.
    inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(best_alpha)] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = final_s
    outer_angle_deg = np.degrees(best_theta)
    
    return inner, outer_center, outer_side, outer_angle_deg
