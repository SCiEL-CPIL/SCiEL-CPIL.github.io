import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def smooth_distance(p1, p2, eps=1e-6):
        """
        Approximate the minimum distance between two convex polygons (hexagons)
        using a smooth function based on SAT.
        Returns 0 if overlapping, >0 if separated.
        """
        # We check all edges of both polygons.
        # For each edge normal, compute the signed distance.
        # If the minimum signed distance over all axes is > 0, they are disjoint.
        # We want a smooth measure of "penetration depth" or "separation".
        
        min_dist = np.inf
        
        # Axes from p1
        for i in range(len(p1)):
            j = (i + 1) % len(p1)
            edge = p1[j] - p1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10: continue
            normal /= norm
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap is min(max1 - min2, max2 - min1)
            # If gap > 0, separated.
            gap = min(max1 - min2, max2 - min1)
            if gap < min_dist:
                min_dist = gap
                
        # Axes from p2
        for i in range(len(p2)):
            j = (i + 1) % len(p2)
            edge = p2[j] - p2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10: continue
            normal /= norm
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = min(max1 - min2, max2 - min1)
            if gap < min_dist:
                min_dist = gap
                
        return min_dist

    def objective(params):
        """
        params: (36,) array of x, y, angle_deg for 12 hexagons.
        Returns outer hexagon side length + penalty if invalid.
        """
        if time.time() > deadline:
            return 1e10
            
        positions = params.reshape((N, 3))
        
        # Generate vertices for all hexagons
        vertices_list = []
        for i in range(N):
            x, y, ang = positions[i]
            verts = hexagon_vertices(x, y, 1.0, ang)
            vertices_list.append(verts)
        
        # Check disjointness with smooth penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = smooth_distance(vertices_list[i], vertices_list[j])
                if dist < 0:
                    # Overlapping. Penalty is quadratic in depth.
                    penalty += (-dist) ** 2
                # else dist >= 0, no penalty (or small reward for separation? no, just 0)
        
        if penalty > 1e-4:
            return 1e10 + penalty # Strong penalty
            
        # Calculate required outer side length
        # We assume the outer hexagon is centered at (0,0) and has rotation 0.
        # We can optimize the outer rotation later, but for now, let's fix it or optimize it.
        # Actually, the problem allows the outer hexagon to rotate.
        # Let's add the outer rotation as an optimization variable?
        # The current interface returns (inner, center, side, angle).
        # The score is the side length.
        # To minimize side length, we should also optimize the outer hexagon's rotation and center.
        # However, translating the whole system is equivalent to translating the inner hexagons.
        # So we can fix the outer center at (0,0) by centering the inner hexagons?
        # No, the inner hexagons can be anywhere.
        # But relative to the outer hexagon, we can fix the outer center at (0,0) and optimize inner positions.
        # And we should optimize the outer rotation.
        
        # Let's add outer rotation to the parameters?
        # The function signature is fixed: construct() -> ...
        # But inside, I can define the optimization.
        # Let's stick to 36 params for inner hexagons for now.
        # We can try a few fixed outer rotations or optimize it separately.
        
        # For a regular hexagon with side S and rotation theta, the condition for a point (x,y) to be inside is:
        # max_{k=0..5} dot(rotate(p, -theta), n_k) <= S * sqrt(3) / 2
        # where n_k are normals to the sides.
        # Actually, simpler:
        # The distance from center to boundary in direction u is R(u) = S * sqrt(3)/2 / cos(angle(u, nearest_normal))
        # This is complicated to minimize over S and theta.
        
        # Alternative:
        # S(theta) = max_{i, v in vertices_i} ( distance from origin to v in metric defined by hexagon(theta) )
        # Let's just compute the required S for a given theta.
        # S(theta) = max_{i} max_{v in vertices_i} ( hex_dist(v, theta) )
        # where hex_dist(v, theta) is the minimal S such that v is inside hexagon(center=0, side=S, angle=theta).
        
        # Let's optimize theta as well?
        # That would be 37 parameters.
        # Let's try to optimize 36 inner params + 1 outer angle.
        # Total 37 params.
        
        # But wait, the current `params` in this function is 36.
        # I need to modify the objective to take 37 params, or handle theta separately.
        # Let's assume for now we fix outer angle to 0, and see if we can improve.
        # Or, better: optimize outer angle inside a nested loop? No, too slow.
        # Let's add outer angle to the optimization vector.
        
        # REVISION: Let's change the optimization to 37 parameters: 12*(x,y,angle) + 1 (outer_angle).
        # But the `objective` function defined above takes `params` which I reshape to (N,3).
        # I need to be careful.
        
        # Let's redefine the strategy:
        # Use `minimize` with 37 variables.
        # The `objective` function below will be replaced.
        
        # For now, just return a large value to trigger the redefinition.
        return 1e10

    # Initial guess: Honeycomb cluster
    hex_centers = []
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    init_positions = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init_positions.append([x, y, 0.0])
        
    init_positions = np.array(init_positions)
    init_outer_angle = 0.0
    init_params = np.append(init_positions.flatten(), init_outer_angle)
    
    def objective_full(params):
        if time.time() > deadline:
            return 1e10
            
        positions = params[:36].reshape((N, 3))
        outer_angle = params[36]
        
        vertices_list = []
        for i in range(N):
            x, y, ang = positions[i]
            verts = hexagon_vertices(x, y, 1.0, ang)
            vertices_list.append(verts)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = smooth_distance(vertices_list[i], vertices_list[j])
                if dist < 0:
                    penalty += (-dist) ** 2
        
        if penalty > 1e-4:
            return 1e10 + penalty
            
        # Calculate S required for outer hexagon with rotation outer_angle
        # The outer hexagon is centered at (0,0).
        # We need to find S such that all vertices of all inner hexagons are inside.
        # A point p is inside a regular hexagon (center 0, side S, angle theta) iff:
        # max_{k=0..5} dot(R(-theta) p, n_k) <= S * sqrt(3) / 2
        # where n_k are the outward normals of the sides of the hexagon at angle 0.
        # For a hexagon with vertex at (1,0), the sides are at angles 30, 90, 150, 210, 270, 330 degrees?
        # Vertices at 0, 60, 120, 180, 240, 300.
        # Midpoints of sides (normals) are at 30, 90, 150, 210, 270, 330.
        
        normals = []
        for k in range(6):
            ang = np.radians(30 + k * 60)
            normals.append(np.array([np.cos(ang), np.sin(ang)]))
        normals = np.array(normals)
        
        max_s = 0.0
        for verts in vertices_list:
            # Rotate vertices by -outer_angle
            rot_verts = verts @ np.array([
                [np.cos(-np.radians(outer_angle)), -np.sin(-np.radians(outer_angle))],
                [np.sin(-np.radians(outer_angle)), np.cos(-np.radians(outer_angle))]
            ]).T
            
            # Project onto normals
            projections = rot_verts.dot(normals.T)
            
            # For each point, the required S is max_k proj_k / (sqrt(3)/2)
            # Actually, the condition is proj_k <= S * sqrt(3) / 2
            # So S >= proj_k / (sqrt(3)/2)
            # We need max over all points and all k.
            
            # Note: projections can be negative. We only care about the maximum projection in the direction of the normal.
            # Wait, the hexagon is the intersection of half-planes dot(p, n_k) <= S * sqrt(3) / 2.
            # So for each point p, we need dot(p, n_k) <= S * sqrt(3) / 2 for all k.
            # So S >= max_k dot(p, n_k) / (sqrt(3) / 2).
            
            for p in rot_verts:
                projs = p.dot(normals.T)
                s_req = np.max(projs) / (np.sqrt(3) / 2)
                if s_req > max_s:
                    max_s = s_req
                    
        return max_s

    # Bounds for inner hexagons: -10 to 10 for x, y. 0 to 360 for angle.
    # Bounds for outer angle: 0 to 60 (symmetry).
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 360))
    bounds.append((0, 60))
    
    # Use L-BFGS-B or Powell?
    # L-BFGS-B is good for smooth functions.
    # Powell is derivative-free but can be slow.
    # Let's try L-BFGS-B first.
    
    try:
        result = minimize(
            objective_full,
            init_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if time.time() > deadline:
            pass # Use result if available, else init
            
        best_params = result.x
        best_val = result.fun
        
        if best_val > 1e9:
            best_params = init_params
            best_val = objective_full(init_params)
            
    except Exception as e:
        print(f"Optimization error: {e}")
        best_params = init_params
        best_val = objective_full(init_params)
        
    # Extract results
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i + 1]
        ang = best_params[3*i + 2]
        inner.append([float(x), float(y), float(ang)])
        
    outer_angle = best_params[36]
    
    return inner, (0.0, 0.0), float(best_val), float(outer_angle)

# EVOLVE-BLOCK-END
