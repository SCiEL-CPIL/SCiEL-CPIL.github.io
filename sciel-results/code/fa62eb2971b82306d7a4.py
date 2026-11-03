import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        Returns side length.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        # For a regular hexagon of side s, the distance from center to side is s * sqrt(3)/2.
        # The projection of a vertex onto the normal of a side is the distance to that side.
        # So max_proj is the required inradius.
        # s = inradius / (sqrt(3)/2) = max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint_fast(poly1, poly2):
        """
        Fast check for disjointness using SAT.
        Returns True if disjoint (or touching), False if overlapping.
        """
        def project_polygon(poly, axis):
            proj = poly @ axis
            return np.min(proj), np.max(proj)
            
        axes = []
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
            
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            min1, max1 = project_polygon(poly1, axis)
            min2, max2 = project_polygon(poly2, axis)
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
                
        return False

    def sdf_hexagon(point, center, angle_deg, side=1.0):
        """
        Signed distance field for a regular hexagon.
        Returns positive if outside, negative if inside.
        """
        # Rotate point into hexagon's local frame
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        
        # Rotate by -angle
        lx = cos_a * dx + sin_a * dy
        ly = -sin_a * dx + cos_a * dy
        
        # In local frame, hexagon has vertices at (1,0), (0.5, sqrt(3)/2), etc.
        # Sides are at distance sqrt(3)/2 from center along normals at 0, 60, ...
        # Actually, for pointy-top (vertex at 0), normals are at 30, 90, ...
        # Let's verify unit_verts_0: (1,0), (0.5, 0.866)...
        # Edge from (1,0) to (0.5, 0.866) is x + 0.5y = 1? No.
        # Normal to edge connecting (1,0) and (0.5, sqrt(3)/2) is at angle 30 deg.
        # Distance to side is sqrt(3)/2.
        
        # The SDF of a regular polygon can be computed by checking the 6 half-planes.
        # For a convex polygon defined by half-planes n_i . p <= d_i,
        # SDF is min_i (d_i - n_i . p) if inside, but strictly speaking SDF is distance to boundary.
        # For a regular hexagon, the distance to the boundary is the minimum of the distances to the 6 sides.
        # If the point is inside, distance is min(d_i - n_i . p).
        # If outside, it's more complex (distance to vertices).
        # However, for optimization purposes, a "soft" distance or just the half-plane constraint is often sufficient 
        # if we stay in the regime where points are close to the sides.
        # A better SDF for a regular hexagon:
        
        # Let's use the fact that a regular hexagon is the intersection of 6 strips.
        # The distance to the boundary of the intersection is min over sides of (distance to side) if inside.
        # If outside, it's the distance to the nearest point on the polygon.
        
        # For simplicity and differentiability, we can approximate the collision constraint 
        # using the distance between centers and the "radius" of the hexagons in the direction of separation.
        # But the prompt asks for SDF.
        
        # Let's implement a robust SDF for a regular hexagon.
        # Reference: https://www.shadertoy.com/view/MdX3zR
        # Adapted for hexagon.
        
        # Local coords lx, ly
        # Hexagon with side 1.
        # Vertices: (1,0), (0.5, h), (-0.5, h), (-1,0), (-0.5, -h), (0.5, -h) where h = sqrt(3)/2.
        
        # The hexagon is symmetric. We can map to a fundamental domain.
        # Or simply calculate distance to the 6 edges and 6 vertices.
        
        h = SQRT3 / 2.0
        verts_local = np.array([
            [1.0, 0.0],
            [0.5, h],
            [-0.5, h],
            [-1.0, 0.0],
            [-0.5, -h],
            [0.5, -h]
        ]) * side
        
        # Edges
        edges = []
        for i in range(6):
            p1 = verts_local[i]
            p2 = verts_local[(i+1)%6]
            edges.append((p1, p2))
            
        min_dist = float('inf')
        for p1, p2 in edges:
            # Distance from point (lx, ly) to segment p1-p2
            d = p2 - p1
            L2 = np.dot(d, d)
            if L2 < 1e-9: continue
            t = np.dot((lx - p1[0], ly - p1[1]), d) / L2
            t = max(0.0, min(1.0, t))
            closest = p1 + t * d
            dist = np.sqrt((lx - closest[0])**2 + (ly - closest[1])**2)
            if dist < min_dist:
                min_dist = dist
        
        # Determine if inside or outside
        # A point is inside if it satisfies all half-plane inequalities.
        # For edge i from v_i to v_{i+1}, the normal pointing inward is n_i.
        # Let's just use the sign of the area or winding number?
        # Easier: Check if point is to the "left" of all directed edges (counter-clockwise).
        # Our vertices are CCW? 
        # (1,0) -> (0.5, 0.866) -> ... Yes, CCW.
        
        inside = True
        for i in range(6):
            p1 = verts_local[i]
            p2 = verts_local[(i+1)%6]
            # Cross product (p2-p1) x (p-p1)
            # If > 0, point is to the left (inside)
            cross = (p2[0]-p1[0])*(ly-p1[1]) - (p2[1]-p1[1])*(lx-p1[0])
            if cross < -1e-9: # Tolerance
                inside = False
                break
        
        if inside:
            return -min_dist
        else:
            return min_dist

    def get_all_sdfs(params):
        """
        Calculate SDFs for all pairs and for container.
        Returns:
        - max_container_sdf: Max SDF of any inner hex vertex relative to container (should be <= 0 for inside, but we want to minimize the required container size).
          Actually, we want to minimize the container size. 
          Let's define the objective as the required container side length.
          And the constraint as min_pair_sdf >= 0.
        """
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # 1. Calculate required container side
        # We can optimize the container size directly as part of the objective?
        # No, the interface returns the side. We should minimize the side.
        # So objective = side_length.
        # Constraint = min distance between pairs >= 0.
        
        # To make it differentiable, we need the gradient of the side length.
        # side = max_proj * 2 / sqrt(3).
        # max_proj = max over all vertices of dot(v, n_k) for k in 6 normals.
        # This is a max of linear functions, which is convex but not smooth at the max.
        # We can use a soft-max (log-sum-exp) to smooth it.
        
        theta_rad = 0.0 # Container rotation fixed at 0 for now
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        # Calculate projections for all vertices of all hexagons
        # Total vertices = N * 6
        all_projs = []
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            all_projs.append(projs)
            
        all_projs = np.concatenate(all_projs, axis=0) # (6N, 6)
        
        # For each normal direction k, find the max projection.
        max_projs_per_normal = np.max(all_projs, axis=0) # (6,)
        
        # The required inradius is the max of these 6 values.
        # To make it smooth, use soft-max with a small temperature.
        # Or just use the max, and rely on the optimizer to handle the kinks?
        # L-BFGS-B might struggle with non-smooth max.
        # Let's use a soft-max: M = T * log(sum(exp(x/T))).
        T = 0.1
        max_proj_smooth = T * np.log(np.sum(np.exp(max_projs_per_normal / T)))
        
        side = max_proj_smooth * 2.0 / SQRT3
        
        # 2. Calculate min SDF between pairs
        # This is expensive: N*(N-1)/2 pairs.
        # For each pair, we need the min SDF.
        # SDF is not differentiable at the boundary, but we can use a smooth approximation.
        # Or, we can use the center-to-center distance minus the sum of radii?
        # The "radius" of a hexagon in direction u is the support function h(u).
        # Distance between two convex shapes A and B is min_{x in A, y in B} ||x-y||.
        # This is equivalent to min_{u} (h_A(u) + h_B(-u) - u . (c_A - c_B))? No.
        
        # Let's stick to the SDF of the vertices of one hexagon relative to the other.
        # But that's not the true distance if the closest points are on edges.
        
        # Alternative: Use the support function approach for the objective?
        # No, the prompt suggests SDF.
        
        # Let's compute the SDF for the center of hex j relative to hex i? No, that's not enough.
        
        # Let's compute the distance between the two polygons using SAT in a differentiable way?
        # SAT gives the minimum gap. The gap is 0 if they intersect.
        # The function "minimum gap" is continuous but not differentiable when the separating axis changes.
        
        # For this implementation, I will use the SDF of the vertices of hex j evaluated on the surface of hex i?
        # No, let's use a simpler differentiable metric:
        # Minimize the maximum "penetration depth" or just maximize the minimum distance.
        
        # Let's calculate the distance between centers and subtract the maximum possible extent?
        # That's too conservative.
        
        # Let's try to compute the SDF of all vertices of hex j relative to hex i.
        # If all SDFs are > 0, they are disjoint? No, one could be inside the other (impossible for same size) or edge crossing.
        # If all vertices of j are outside i, and all vertices of i are outside j, they are disjoint.
        
        # So, for each pair (i, j):
        # min_sdf_ij = min( min_{v in verts_j} SDF_i(v), min_{v in verts_i} SDF_j(v) )
        # If min_sdf_ij < 0, they overlap.
        
        min_pair_sdf = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                ci = positions[i]
                ai = angles[i]
                cj = positions[j]
                aj = angles[j]
                
                verts_i = get_hex_verts(ci[0], ci[1], 1.0, ai)
                verts_j = get_hex_verts(cj[0], cj[1], 1.0, aj)
                
                # SDF of verts_j relative to hex_i
                sdfs_j_on_i = [sdf_hexagon(v, ci, ai) for v in verts_j]
                # SDF of verts_i relative to hex_j
                sdfs_i_on_j = [sdf_hexagon(v, cj, aj) for v in verts_i]
                
                min_sdf = min(min(sdfs_j_on_i), min(sdfs_i_on_j))
                
                if min_sdf < min_pair_sdf:
                    min_pair_sdf = min_sdf
        
        return side, min_pair_sdf

    def objective(params):
        if time.time() > deadline - 5:
            return 1e6
        side, min_sdf = get_all_sdfs(params)
        # Penalty for overlap
        # If min_sdf < 0, add penalty
        penalty = 0.0
        if min_sdf < 0:
            penalty = -min_sdf * 100.0
        return side + penalty

    # Initial guess: Lattice
    sqrt3 = math.sqrt(3)
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
            
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    
    best_params = x0.copy()
    best_score = objective(x0)
    
    # Optimize
    try:
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if res.fun < best_score:
            # Verify validity
            positions = res.x[:2*N].reshape(N, 2)
            angles = res.x[2*N:]
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not check_disjoint_fast(polys[i], polys[j]):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                best_score = res.fun
                best_params = res.x
    except Exception:
        pass

    # Final calculation
    final_positions = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:]
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

