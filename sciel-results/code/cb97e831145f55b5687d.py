import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Unit hexagon vertices (flat-top orientation, angle=0)
    def hex_vertices(cx, cy, angle_deg, side=1.0):
        a = np.radians(angle_deg)
        angles = a + np.arange(6) * np.pi / 3
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])
    
    # Check if two convex polygons overlap using SAT (Separating Axis Theorem)
    def polygons_overlap(p1, p2):
        n1 = len(p1)
        n2 = len(p2)
        # Get all edge normals
        axes = []
        for i in range(n1):
            edge = p1[(i+1) % n1] - p1[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-10:
                normal /= np.linalg.norm(normal)
                axes.append(normal)
        for i in range(n2):
            edge = p2[(i+1) % n2] - p2[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-10:
                normal /= np.linalg.norm(normal)
                axes.append(normal)
        
        for axis in axes:
            proj1 = np.dot(p1, axis)
            proj2 = np.dot(p2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return False, 0.0
        # Overlap - compute minimum penetration depth
        min_depth = np.inf
        for axis in axes:
            proj1 = np.dot(p1, axis)
            proj2 = np.dot(p2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 >= min2 and max2 >= min1:
                depth = min(max1, max2) - max(min1, min2)
                if depth < min_depth:
                    min_depth = depth
        return True, min_depth
    
    # Compute minimum container side length needed for given hexagon placements
    def compute_container_size(positions, angles_deg, outer_center, outer_angle_deg):
        # All hexagon vertices in world coordinates
        all_vertices = []
        for i in range(N):
            verts = hex_vertices(positions[i, 0], positions[i, 1], angles_deg[i])
            all_vertices.append(verts)
        
        # Transform to outer hexagon's local frame
        outer_a = np.radians(outer_angle_deg)
        rot = np.array([[np.cos(outer_a), -np.sin(outer_a)],
                        [np.sin(outer_a), np.cos(outer_a)]])
        
        # In the local frame, the container is a regular hexagon centered at origin
        # with side length s. A point (x,y) is inside if:
        # For a flat-top hexagon with side s, the constraints are:
        # |x| <= s, |y| <= s*sqrt(3)/2, and |x| + |y|/sqrt(3) <= s
        # Actually for a regular hexagon with circumradius R = s:
        # The hexagon has vertices at angles 0, 60, 120, 180, 240, 300
        # The half-width in x direction is R, half-height is R*sqrt(3)/2
        # But we need to check if a point is inside.
        
        # Better: use the fact that a regular hexagon with side s has inradius r = s*sqrt(3)/2
        # and circumradius R = s.
        # A point is inside if its distance to each edge is >= 0.
        # The edges are at distance r from center, with normals at angles 30, 90, 150, 210, 270, 330 degrees
        # (for a flat-top hexagon)
        
        # Let's just compute the required side length by checking all vertices
        # The container hexagon (in its local frame) must contain all vertices.
        # For a regular hexagon with circumradius R, the vertices of the container are at
        # distance R from center. A point at distance d from center is inside if d <= R.
        # But that's only true for the center. Actually, the maximum distance from center
        # to any point in the hexagon is R (at the vertices), but points near edges are closer.
        
        # The correct check: a point (x,y) in the hexagon's local frame is inside the hexagon
        # with circumradius R if:
        # For a hexagon with vertices at angles 0,60,120,180,240,300:
        # The constraint is that the point must be on the correct side of each edge.
        # Edge normals point outward at angles 30, 90, 150, 210, 270, 330
        # The distance from center to each edge is R * cos(30°) = R * sqrt(3)/2
        # So the constraint is: dot(point, normal_k) <= R * sqrt(3)/2 for each k
        
        # Let me compute the minimum R such that all vertices are inside
        all_verts_local = []
        for verts in all_vertices:
            shifted = verts - np.array(outer_center)
            local = (rot.T @ shifted.T).T  # Rotate by -outer_angle
            all_verts_local.append(local)
        
        all_verts_local = np.vstack(all_verts_local)
        
        # Normals for flat-top hexagon edges (outward pointing)
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # For each vertex, compute the required R
        # dot(v, n_k) <= R * sqrt(3)/2  =>  R >= dot(v, n_k) / (sqrt(3)/2)
        # Also need R >= 0 and the hexagon must contain the point, so we also need
        # the point to be within the hexagonal region, which means all 6 constraints hold.
        
        min_R = 0.0
        for v in all_verts_local:
            for n in normals:
                val = np.dot(v, n)
                if val > 0:
                    R_needed = val / (np.sqrt(3) / 2)
                    if R_needed > min_R:
                        min_R = R_needed
        
        return min_R
    
    # Objective: minimize container size + penalty for overlaps
    def objective(params):
        if time.time() > deadline:
            return np.inf
        positions = params[:2*N].reshape(N, 2)
        angles_deg = params[2*N:3*N]
        outer_center = params[3*N:3*N+2]
        outer_angle_deg = params[3*N+2]
        
        # Compute container size
        container_size = compute_container_size(positions, angles_deg, outer_center, outer_angle_deg)
        
        # Compute overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            vi = hex_vertices(positions[i, 0], positions[i, 1], angles_deg[i])
            for j in range(i+1, N):
                vj = hex_vertices(positions[j, 0], positions[j, 1], angles_deg[j])
                overlap, depth = polygons_overlap(vi, vj)
                if overlap:
                    overlap_penalty += depth ** 2
        
        # Also add penalty if outer center is far from origin (to keep solution centered)
        center_penalty = 0.1 * (outer_center[0]**2 + outer_center[1]**2)
        
        return container_size + 10.0 * overlap_penalty + center_penalty
    
    # Initial guess: hexagonal lattice arrangement
    def initial_guess():
        # Place 11 hexagons in a compact arrangement
        # Center at origin, 6 around it, 4 more
        # Hexagonal lattice spacing: 2*sqrt(3) for touching hexagons
        s = 2 * np.sqrt(3)
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        
        # Central hexagon
        positions[0] = [0, 0]
        
        # 6 neighbors in hexagonal pattern
        for k in range(6):
            angle = k * np.pi / 3
            positions[k+1] = [s * np.cos(angle), s * np.sin(angle)]
        
        # 4 more: place them at the "corners" of the hexagonal cluster
        # Between the 6 neighbors, add hexagons at the outer positions
        # For a fuller cluster, add at distances that fill gaps
        # Let's try adding at angles 30, 90, 150, 210 (offset by 30 degrees)
        # at a larger radius
        for k in range(4):
            angle = (k * 90 + 30) * np.pi / 180
            positions[k+7] = [s * 1.8 * np.cos(angle), s * 1.8 * np.sin(angle)]
        
        outer_center = [0, 0]
        outer_angle = 0.0
        
        params = np.concatenate([positions.flatten(), angles_deg, outer_center, [outer_angle]])
        return params
    
    x0 = initial_guess()
    
    # Try multiple restarts with different initial configurations
    best_obj = np.inf
    best_params = x0.copy()
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            x_try = x0.copy()
        else:
            # Perturb the initial guess
            x_try = x0 + np.random.randn(len(x0)) * 0.5
        
        try:
            result = minimize(objective, x_try, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract best solution
    positions = best_params[:2*N].reshape(N, 2)
    angles_deg = best_params[2*N:3*N]
    outer_center = best_params[3*N:3*N+2]
    outer_angle_deg = best_params[3*N+2]
    
    # Verify no overlaps in final solution
    final_overlap = 0.0
    for i in range(N):
        vi = hex_vertices(positions[i, 0], positions[i, 1], angles_deg[i])
        for j in range(i+1, N):
            vj = hex_vertices(positions[j, 0], positions[j, 1], angles_deg[j])
            overlap, depth = polygons_overlap(vi, vj)
            if overlap:
                final_overlap += depth
    
    if final_overlap > 0.01:
        # Solution has overlaps, use a safe fallback
        # Use a simple grid arrangement that's guaranteed valid
        s = 2 * np.sqrt(3) + 0.1  # Small gap
        inner = []
        # Place in a 3x4 grid pattern
        layout = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2),
            (1, 3)
        ]
        for i, (gx, gy) in enumerate(layout[:N]):
            x = (gx - 1.5) * s
            y = (gy - 1.5) * s * np.sqrt(3) / 2
            inner.append([x, y, 0.0])
        # Compute container size for this arrangement
        all_x = [p[0] for p in inner]
        all_y = [p[1] for p in inner]
        max_extent = max(max(abs(x) for x in all_x), max(abs(y) for y in all_y) / (np.sqrt(3)/2))
        side = max_extent / (np.sqrt(3)/2) + 1.0
        return inner, [0.0, 0.0], side, 0.0
    
    # Build inner list
    inner = [[positions[i, 0], positions[i, 1], angles_deg[i]] for i in range(N)]
    
    # Compute actual container side length
    side = compute_container_size(positions, angles_deg, outer_center, outer_angle_deg)
    
    return inner, [outer_center[0], outer_center[1]], side, outer_angle_deg
# EVOLVE-BLOCK-END
