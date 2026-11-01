import numpy as np
import time
from scipy.optimize import minimize

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

    def sat_disjoint(p1, p2, eps=1e-5):
        """
        Returns True if two convex polygons are disjoint.
        """
        for i in range(len(p1)):
            j = (i + 1) % len(p1)
            edge = p1[j] - p1[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal /= np.linalg.norm(normal)
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        for i in range(len(p2)):
            j = (i + 1) % len(p2)
            edge = p2[j] - p2[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal /= np.linalg.norm(normal)
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        return False

    def check_validity(positions, angles):
        """
        positions: (12, 2) array
        angles: (12,) array in radians
        Returns True if all are pairwise disjoint.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            ang_deg = np.degrees(angles[i])
            verts = hexagon_vertices(x, y, 1.0, ang_deg)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(vertices_list[i], vertices_list[j]):
                    return False
        return True

    def get_min_outer_side(positions, angles, outer_angle_rad):
        """
        Calculate the minimum side length S of a regular hexagon 
        (centered at origin, rotated by outer_angle_rad) 
        that contains all vertices of the inner hexagons.
        """
        s_max = 0.0
        # The condition for a point (x,y) to be inside a regular hexagon 
        # with side S, center (0,0), and rotation alpha is:
        # For each of the 6 side normals n_k (rotated by alpha), 
        # dot(p, n_k) <= S * sqrt(3) / 2
        
        # Normals for a hexagon with vertex at angle alpha:
        # The sides are perpendicular to the directions alpha + k*60 + 30 ? 
        # Let's use the vertex definition to derive the half-width in normal directions.
        # A regular hexagon with side S has apothem a = S * sqrt(3) / 2.
        # The distance from center to side is a.
        # So if we define the hexagon by its side normals n_k, 
        # the condition is |dot(p, n_k)| <= a for all k.
        # Thus S = max_k ( |dot(p, n_k)| ) * 2 / sqrt(3).
        
        # What are the side normals?
        # If a vertex is at angle alpha, the adjacent sides are at angles alpha +/- 30.
        # The normal to the side between vertex(alpha) and vertex(alpha+60) 
        # is at angle alpha + 30.
        # So the 6 normal directions are alpha + 30 + k * 60 for k=0..5.
        
        normal_angles = outer_angle_rad + np.radians(30) + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        for i in range(N):
            x, y = positions[i]
            ang_deg = np.degrees(angles[i])
            verts = hexagon_vertices(x, y, 1.0, ang_deg)
            # verts shape (6, 2)
            # Project all vertices onto all 6 normals
            # proj[i, k] = dot(verts[i], normals[k])
            proj = verts.dot(normals.T) # (6, 6)
            max_proj = np.max(np.abs(proj))
            s_req = max_proj * 2.0 / np.sqrt(3.0)
            if s_req > s_max:
                s_max = s_req
        return s_max

    def objective(params):
        """
        params: (36,) array: x0, y0, a0, x1, y1, a1, ...
        Returns outer side length + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        positions = params[:2*N].reshape((N, 2))
        angles = params[2*N:]
        
        # 1. Calculate outer side length
        # We can fix outer angle to 0 for simplicity in the objective, 
        # or optimize it. Let's fix it to 0 initially to reduce dimensions, 
        # but the problem allows rotating outer hexagon. 
        # However, rotating the outer hexagon is equivalent to rotating the whole system.
        # Since we can rotate inner hexagons, fixing outer to 0 is WLOG for the *set* of solutions,
        # but might be restrictive if inner rotations don't align.
        # Actually, we can just assume outer is at angle 0.
        s_val = get_min_outer_side(positions, angles, 0.0)
        
        # 2. Calculate penalty for overlaps
        # For hexagons with side 1, the minimum distance between centers 
        # for them to be disjoint is sqrt(3) if they are aligned (flat-to-flat).
        # If rotated, the minimum distance can be smaller? 
        # No, the minimum distance between centers of two disjoint unit hexagons is always >= sqrt(3).
        # Wait, is it?
        # If one is rotated 30 degrees relative to the other, can they be closer?
        # The "width" of a hexagon is 2 (vertex to vertex) or sqrt(3) (side to side).
        # The minimum distance between centers of two disjoint convex bodies is the minimum 
        # over all directions of the sum of their support functions in opposite directions.
        # For two unit hexagons, the minimum center distance is indeed sqrt(3).
        # This occurs when they touch side-to-side.
        
        # So, a simple proxy for overlap is:
        # If dist(center_i, center_j) < sqrt(3) - epsilon, they MUST overlap?
        # Yes, because the maximum distance from center to boundary is 1, but the "radius" 
        # in the direction of the other center might be less.
        # Actually, the condition d >= sqrt(3) is necessary for ANY two unit hexagons to be disjoint.
        # Proof: The area of a hexagon is (3*sqrt(3)/2) * S^2. 
        # If d < sqrt(3), the circles of radius sqrt(3)/2 around centers overlap? No.
        # Let's just use the SAT check for the penalty? 
        # SAT is non-differentiable.
        
        # Better proxy: 
        # For each pair, compute the minimum distance between their boundaries.
        # If dist < 0, they overlap.
        # We can approximate this by checking the distance between centers 
        # and subtracting the sum of their "radii" in the direction of the connecting vector.
        
        # However, a simpler robust penalty for optimization:
        # Use the fact that if d < sqrt(3), they overlap.
        # Penalty = sum( max(0, sqrt(3) - d_ij)^2 )
        
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                min_d = np.sqrt(3.0)
                if d < min_d:
                    penalty += (min_d - d) ** 2
        
        # Scale penalty to be significant compared to s_val
        # s_val is typically around 3-4.
        # Penalty should be large enough to force separation.
        return s_val + 10.0 * penalty

    # Initial guess: Honeycomb cluster
    # Center at (0,0)
    # 1 at (0,0)
    # 6 around it at distance sqrt(3)
    # 5 more to make 12
    
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    init_positions = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init_positions.append([x, y])
        
    init_positions = np.array(init_positions)
    
    # Check if initial is valid
    init_angles = np.zeros(N)
    if check_validity(init_positions, init_angles):
        init_params = np.concatenate([init_positions.flatten(), init_angles])
    else:
        # Fallback: spread them out more
        init_params = np.zeros(3 * N)
        for i in range(N):
            init_params[3*i] = i * 2.0
            init_params[3*i+1] = 0.0
            init_params[3*i+2] = 0.0

    # Optimization
    best_params = init_params
    best_val = objective(init_params)
    
    # Try multiple random restarts with small perturbations
    np.random.seed(42)
    
    methods = ['L-BFGS-B', 'BFGS']
    
    for method in methods:
        # Perturb initial position
        perturbation = np.random.normal(0, 0.1, 3 * N)
        x0 = init_params + perturbation
        
        try:
            result = minimize(
                objective, 
                x0, 
                method=method, 
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < best_val:
                # Verify validity of the result
                pos = result.x[:2*N].reshape((N, 2))
                ang = result.x[2*N:]
                
                if check_validity(pos, ang):
                    val = get_min_outer_side(pos, ang, 0.0)
                    if val < best_val:
                        best_val = val
                        best_params = result.x
                else:
                    # If invalid, we can try to "fix" it by pushing apart?
                    # Or just discard.
                    pass
                    
        except Exception:
            pass
            
        if time.time() > deadline - 10:
            break

    # Final check and construction
    final_pos = best_params[:2*N].reshape((N, 2))
    final_ang = best_params[2*N:]
    
    # One last polish: if invalid, try to separate them manually? 
    # Or just return the best valid one found.
    # If the best one is invalid, we need a valid one.
    # The initial honeycomb is valid.
    if not check_validity(final_pos, final_ang):
        final_pos = init_positions
        final_ang = init_angles
        best_val = get_min_outer_side(final_pos, final_ang, 0.0)

    inner = []
    for i in range(N):
        x = float(final_pos[i][0])
        y = float(final_pos[i][1])
        angle_deg = float(np.degrees(final_ang[i]))
        inner.append([x, y, angle_deg])
        
    # Outer hexagon
    # Center (0,0)
    # Side length best_val
    # Angle 0.0
    
    return inner, (0.0, 0.0), float(best_val), 0.0

# EVOLVE-BLOCK-END
