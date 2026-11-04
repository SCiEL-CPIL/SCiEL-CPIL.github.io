import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    # Vertices at 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Precompute normals for the 6 sides of a unit hexagon
    # Edge 0: (1,0) to (0.5, sqrt(3)/2). Normal points outward: (sqrt(3)/2, 0.5) -> 30 deg
    unit_normals_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 360, 60)])

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
        """
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon sides: 6 directions
        normal_angles = theta_rad + math.pi/6 + np.arange(6) * math.pi/3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj_all = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang_deg = angles[i]
            ang_rad = math.radians(ang_deg)
            c, s = math.cos(ang_rad), math.sin(ang_rad)
            rot = np.array([[c, -s], [s, c]])
            verts = unit_verts_0 @ rot.T + np.array([cx, cy])
            projs = verts @ normals.T  # (6, 6)
            max_proj_i = np.max(projs, axis=0)  # (6,)
            m = np.max(max_proj_i)
            if m > max_proj_all:
                max_proj_all = m
        
        return max_proj_all * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separated, returns the positive distance.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap on this axis
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                # Overlapping on this axis
                min_gap = min(min_gap, gap)
            else:
                # Separated on this axis -> disjoint
                return gap
                
        # Axes from poly2 edges
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                # Overlapping on this axis
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    # --- STRATEGY: RIGID-BODY LATTICE ASSEMBLY ---
    # We fix the topology to a 3x4 grid of hexagons on a triangular lattice.
    # The lattice basis vectors are:
    # v1 = (1.5 * s, sqrt(3)/2 * s)
    # v2 = (0, sqrt(3) * s)
    # where s is a scale factor.
    # We also allow a global rotation of the entire cluster and a uniform rotation of the hexagons.
    # Actually, rotating the lattice is equivalent to rotating the cluster.
    # Let's parameterize by:
    # 1. s (lattice scale)
    # 2. theta (global rotation of the lattice/cluster)
    # 3. phi (rotation of individual hexagons relative to lattice? Or absolute?)
    # If we rotate the lattice by theta and keep hexagons aligned with lattice (phi=theta),
    # the relative positions are fixed.
    # To allow "shear" or deformation, we can vary the lattice vectors independently?
    # The prompt suggests "uniform shear deformation parameter".
    # Let's define lattice vectors:
    # v1 = (a, b)
    # v2 = (c, d)
    # For a regular triangular lattice, v1=(1.5, sqrt(3)/2), v2=(0, sqrt(3)).
    # A shear could modify v1 or v2 while keeping the hexagons rigid.
    # However, if we shear the lattice, the distance between neighbors changes.
    # If we shear too much, they might overlap.
    # Let's try a simpler structural change:
    # Fix the relative positions of the 12 hexagons to a 3x4 grid on a *deformed* lattice.
    # Let the lattice be defined by two vectors u and v.
    # u = (u_x, u_y), v = (v_x, v_y).
    # We can parameterize u and v using 4 parameters (or 2 if we constrain them).
    # Let's use:
    # u = s1 * (cos(alpha), sin(alpha))
    # v = s2 * (cos(beta), sin(beta))
    # This is a general affine transformation of the grid.
    # But we must ensure non-overlap.
    # The closest pairs in a 3x4 grid are the nearest neighbors.
    # In a 3x4 grid (indices (i,j) for i in 0..2, j in 0..3), the neighbors are:
    # (i, j) and (i+1, j) -> distance |u|
    # (i, j) and (i, j+1) -> distance |v|
    # (i, j) and (i+1, j+1) -> distance |u+v|
    # (i, j) and (i+1, j-1) -> distance |u-v|
    # For hexagons of side 1, the minimum center distance to avoid overlap is 2 * inradius = sqrt(3) if aligned.
    # If rotated, the minimum distance can be smaller.
    # The Minkowski sum of two unit hexagons is a hexagon of side 2.
    # The support function of the Minkowski sum determines the minimum distance.
    # To guarantee non-overlap for ANY relative rotation, the center distance must be at least the max width of the Minkowski sum.
    # But we can optimize the rotations too.
    
    # Let's stick to the prompt's suggestion: "fixed 3x4 hexagonal lattice topology... global rotation angle and a uniform shear deformation parameter".
    # This implies the relative topology is fixed (3x4 grid), but the lattice can be sheared.
    # Let's define the lattice vectors as:
    # v1 = (1.5 * s, sqrt(3)/2 * s + shear)
    # v2 = (0, sqrt(3) * s)
    # Or better, use a general 2D affine transform on the standard lattice.
    # Let standard lattice points be P_ij = i * e1 + j * e2, where e1=(1.5, sqrt(3)/2), e2=(0, sqrt(3)).
    # Apply a linear transform M to all points.
    # M = [[a, b], [c, d]]
    # To preserve some structure, let's constrain M to be a rotation + uniform scale + shear.
    # Let M = R(theta) * S(s) * H(k)
    # Where R is rotation, S is scale, H is shear.
    # This gives 4 parameters: theta, s, k, and maybe a second shear?
    # Let's use 3 parameters: theta (rotation), s (scale), k (shear).
    # H(k) = [[1, k], [0, 1]]
    # Then the lattice vectors become:
    # v1 = M @ e1
    # v2 = M @ e2
    
    # The positions are P_ij = M @ (i * e1 + j * e2) + offset.
    # The hexagons can also be rotated. Let's assume all hexagons have the same rotation phi.
    # So variables are: theta, s, k, phi, and offset (x, y) - but offset is irrelevant for size.
    # We want to minimize the outer side length.
    # The outer hexagon is also rotatable. Let's add outer_angle.
    # Total variables: theta, s, k, phi, outer_angle. (5 parameters)
    # This is a huge reduction from 36.
    
    # Let's implement this.
    
    e1 = np.array([1.5, math.sqrt(3)/2])
    e2 = np.array([0, math.sqrt(3)])
    
    # Generate 3x4 grid indices
    # i in 0..2, j in 0..3
    # Center the grid
    indices = []
    for i in range(3):
        for j in range(4):
            indices.append((i, j))
    
    # To center, subtract the mean of the indices
    idx_arr = np.array(indices)
    idx_mean = np.mean(idx_arr, axis=0)
    idx_centered = idx_arr - idx_mean
    
    def get_positions(theta, s, k, phi):
        """
        Generate positions and angles for the 12 hexagons.
        theta: global rotation of the lattice
        s: scale factor
        k: shear factor
        phi: rotation of individual hexagons
        """
        # Construct the transform matrix M
        # R(theta)
        ct, st = math.cos(theta), math.sin(theta)
        R = np.array([[ct, -st], [st, ct]])
        
        # S(s)
        S = s * np.eye(2)
        
        # H(k)
        H = np.array([[1, k], [0, 1]])
        
        M = R @ S @ H
        
        # Calculate positions
        positions = []
        for i, j in idx_centered:
            base_point = i * e1 + j * e2
            pos = M @ base_point
            positions.append(pos)
        
        positions = np.array(positions)
        angles = np.full(N, math.degrees(phi))
        
        return positions, angles

    def objective(params):
        if time.time() > deadline:
            return float('inf')
        
        theta, s, k, phi, outer_angle = params
        
        # s must be positive
        if s <= 0:
            return float('inf')
        
        positions, angles = get_positions(theta, s, k, phi)
        
        # Check validity
        if not is_valid(positions, angles, tol=1e-5):
            # Penalize overlap
            # Calculate min gap
            min_gap = float('inf')
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            for i in range(N):
                for j in range(i+1, N):
                    gap = sat_gap(polys[i], polys[j])
                    if gap < min_gap:
                        min_gap = gap
            penalty = max(0, -min_gap) * 1000
            return calc_min_outer_side(positions, angles, outer_angle) + penalty
        
        return calc_min_outer_side(positions, angles, outer_angle)

    # Bounds for parameters
    # theta: 0 to 2*pi
    # s: 0.8 to 1.2 (scale around 1)
    # k: -0.5 to 0.5 (shear)
    # phi: 0 to pi (hexagon rotation, periodicity pi/3 but let's allow full range for safety)
    # outer_angle: 0 to pi/3
    
    bounds = [
        (0, 2*math.pi),
        (0.8, 1.2),
        (-0.5, 0.5),
        (0, math.pi),
        (0, math.pi/3)
    ]
    
    # Initial guess
    x0 = [0, 1.0, 0, 0, 0]
    
    # Use Differential Evolution for global search, then SLSQP for local refinement
    best_score = float('inf')
    best_result = x0
    
    # Phase 1: Differential Evolution
    if time.time() < deadline - 60:
        try:
            res_de = differential_evolution(
                objective, 
                bounds, 
                seed=42,
                maxiter=50,
                tol=1e-8,
                polish=False,
                init='sobol'
            )
            
            if res_de.fun < best_score:
                best_score = res_de.fun
                best_result = res_de.x
        except Exception as e:
            print(f"DE error: {e}")
            
    # Phase 2: SLSQP Refinement
    if time.time() < deadline - 20:
        try:
            res_slqp = minimize(
                objective, 
                best_result, 
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
            )
            
            if res_slqp.fun < best_score:
                best_score = res_slqp.fun
                best_result = res_slqp.x
        except Exception as e:
            print(f"SLSQP error: {e}")
            
    # Finalize
    theta, s, k, phi, outer_angle = best_result
    final_positions, final_angles = get_positions(theta, s, k, phi)
    final_side = calc_min_outer_side(final_positions, final_angles, outer_angle)
    
    # Verify validity one last time
    if not is_valid(final_positions, final_angles, tol=1e-5):
        # Fallback to a valid configuration if the optimized one is invalid
        # Use a standard honeycomb subset
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * math.sqrt(3) / 2.0 + j * math.sqrt(3)
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_positions = np.array(lattice_points[:12])
        init_positions -= np.mean(init_positions, axis=0)
        init_angles = np.zeros(N)
        final_positions = init_positions
        final_angles = init_angles
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
        outer_angle = 0.0

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, outer_angle

if __name__ == "__main__":
    construct()
