import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep margin
    
    # Hexagon vertices for side d, angle a (degrees)
    def get_hex_vertices(cx, cy, d, a_deg):
        a_rad = np.deg2rad(a_deg)
        angles = a_rad + np.deg2rad(60 * np.arange(6))
        xs = cx + d * np.cos(angles)
        ys = cy + d * np.sin(angles)
        return np.column_stack((xs, ys))

    # Check if point p is inside regular hexagon defined by center, side, angle
    # Using cross products for edges
    def is_inside_hex(p, center, side, angle_deg):
        verts = get_hex_vertices(center[0], center[1], side, angle_deg)
        # verts is (6, 2)
        # For each edge (v_i, v_{i+1}), check if p is on the left (or right)
        # Assuming CCW order, p should be to the left of each edge
        # cross = (v_{i+1} - v_i) x (p - v_i)
        # If all cross >= -eps, it's inside.
        for i in range(6):
            v1 = verts[i]
            v2 = verts[(i + 1) % 6]
            edge = v2 - v1
            vec = p - v1
            cross = edge[0] * vec[1] - edge[1] * vec[0]
            if cross < -1e-8:
                return False
        return True

    # Calculate penalty for a configuration
    # x: [x1, y1, ..., x12, y12, alpha_deg, s]
    def cost(x):
        if time.time() > deadline:
            return 1e10
        centers = x[:24].reshape(12, 2)
        alpha = x[24]
        s = x[25]
        
        # Penalty for s being too small (must be >= sqrt(12) approx 3.46, but realistically larger)
        if s < 3.5:
            return 1e10 + (3.5 - s) * 100

        penalty = 0.0
        
        # 1. Non-overlap constraint
        # For same orientation hexagons, min center distance is sqrt(3)
        min_dist = np.sqrt(3) - 1e-6
        for i in range(12):
            for j in range(i + 1, 12):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        # 2. Containment constraint
        # Outer hexagon: center (0,0), side s, angle 0
        outer_center = np.array([0.0, 0.0])
        outer_angle = 0.0
        outer_verts = get_hex_vertices(0, 0, s, outer_angle)
        
        for i in range(12):
            cx, cy = centers[i]
            # Get vertices of inner hexagon i
            inner_verts = get_hex_vertices(cx, cy, 1.0, alpha)
            for v in inner_verts:
                # Check if v is inside outer hexagon
                if not is_inside_hex(v, outer_center, s, outer_angle):
                    # Calculate distance to boundary for penalty?
                    # Simple penalty: count violations
                    penalty += 100.0
                    # More precise: distance to nearest edge
                    # We can approximate by checking cross product magnitude
                    for k in range(6):
                        v1 = outer_verts[k]
                        v2 = outer_verts[(k + 1) % 6]
                        edge = v2 - v1
                        vec = v - v1
                        cross = edge[0] * vec[1] - edge[1] * vec[0]
                        if cross < 0:
                            # Project v onto line of edge to find distance?
                            # Or just use cross product magnitude as a proxy for depth
                            # Normalize edge
                            edge_len = np.linalg.norm(edge)
                            if edge_len > 1e-8:
                                depth = -cross / edge_len
                                penalty += depth ** 2 * 10.0
                            break # Only count once per vertex per hex? Or sum all? Summing is safer.
                            # Actually, let's just sum the negative cross products scaled
                            penalty += (-cross) ** 2 * 0.1
                            
        return s + penalty

    # Initial guess: Honeycomb arrangement
    # Axial to Cartesian for side 1
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])

    # 12 positions in a compact cluster
    # Center (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2 partial: (2,0), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,0)
    # Let's pick 12 that form a nice shape.
    # (0,0)
    # (1,0), (-1,0)
    # (0,1), (0,-1)
    # (1,1), (-1,-1)
    # (1,-1), (-1,1)
    # (2,0), (-2,0)
    # This is 1+2+2+2+2+2 = 11. Add (0,2) or (0,-2)?
    # Let's use (0,2). Total 12.
    coords = [
        (0,0),
        (1,0), (-1,0),
        (0,1), (0,-1),
        (1,1), (-1,-1),
        (1,-1), (-1,1),
        (2,0), (-2,0),
        (0,2)
    ]
    
    initial_centers = np.array([axial_to_cartesian(q, r) for q, r in coords])
    # Center the cluster
    initial_centers -= np.mean(initial_centers, axis=0)
    
    alpha_init = 0.0
    # Estimate initial s
    max_r = np.max(np.linalg.norm(initial_centers, axis=1))
    s_init = max_r + 2.0 # Buffer
    
    x0 = np.concatenate([initial_centers.flatten(), [alpha_init, s_init]])
    
    # Optimization
    best_x = x0
    best_cost = cost(x0)
    
    # Use Nelder-Mead
    # Max iterations limited by time
    options = {
        'maxiter': 2000,
        'xatol': 1e-8,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    try:
        res = minimize(cost, x0, method='Nelder-Mead', options=options)
        if res.fun < best_cost:
            best_cost = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # If time remains, try a few random restarts or different initial angles
    if time.time() < deadline - 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            # Perturb initial angle
            alpha_try = np.random.uniform(0, 60)
            x_try = np.concatenate([initial_centers.flatten(), [alpha_try, s_init]])
            try:
                res_try = minimize(cost, x_try, method='Nelder-Mead', options={'maxiter': 500, 'adaptive': True})
                if res_try.fun < best_cost:
                    best_cost = res_try.fun
                    best_x = res_try.x
            except Exception:
                pass

    # Extract final result
    final_centers = best_x[:24].reshape(12, 2)
    final_alpha = best_x[24]
    final_s = best_x[25]
    
    # Ensure s is not too small
    if final_s < np.sqrt(12):
        final_s = np.sqrt(12) * 1.1

    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = final_centers[i, 0]
        inner_hex_data[i, 1] = final_centers[i, 1]
        inner_hex_data[i, 2] = final_alpha

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, final_s

