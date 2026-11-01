"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        """Check if two convex polygons are disjoint using SAT."""
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-9:
                axes.append(normal / np.linalg.norm(normal))
        
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-9:
                axes.append(normal / np.linalg.norm(normal))
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def hexagons_disjoint(v1, s1, a1, v2, s2, a2):
        verts1 = hexagon_vertices(v1[0], v1[1], s1, a1)
        verts2 = hexagon_vertices(v2[0], v2[1], s2, a2)
        return polygons_disjoint_sat(verts1, verts2)

    def all_inside_outer(inner_positions, outer_center, outer_side, outer_angle):
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        return True

    def pairwise_disjoint(inner_positions):
        for i in range(len(inner_positions)):
            for j in range(i+1, len(inner_positions)):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                if not hexagons_disjoint(ci, 1.0, ai, cj, 1.0, aj):
                    return False
        return True

    def compute_min_outer_side(inner_positions):
        """Compute the minimum outer hexagon side length that contains all inner hexagons."""
        # Try multiple outer angles and find the tightest fit
        best_side = float('inf')
        best_center = (0.0, 0.0)
        best_angle = 30.0
        
        # Collect all vertices from all inner hexagons
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Try different outer angles
        for outer_angle in [0, 30, 60, 90, 120, 150]:
            # For each angle, find the minimum enclosing hexagon
            # The hexagon is defined by 6 half-planes
            # We need to find center (ox, oy) and side s such that all points are inside
            
            # For a regular hexagon with side s and angle theta, the constraints are:
            # For each of 6 normal directions (at angles 30+60k + theta):
            #   dot(p - c, n_k) <= s * sqrt(3) / 2
            # This gives us linear constraints on c and s.
            
            # Let's use a simpler approach: try centers near the centroid
            centroid = np.mean(all_verts, axis=0)
            
            # Try a grid of centers around centroid
            for dx in np.linspace(-2, 2, 5):
                for dy in np.linspace(-2, 2, 5):
                    ox, oy = centroid[0] + dx, centroid[1] + dy
                    
                    # Compute required side length
                    max_proj = 0
                    for k in range(6):
                        na = np.radians(30 + 60*k + outer_angle)
                        nx, ny = np.cos(na), np.sin(na)
                        projs = (all_verts[:, 0] - ox) * nx + (all_verts[:, 1] - oy) * ny
                        max_proj = max(max_proj, np.max(projs))
                    
                    side = max_proj / (np.sqrt(3) / 2)
                    if side < best_side:
                        best_side = side
                        best_center = (ox, oy)
                        best_angle = outer_angle
        
        return best_side, best_center, best_angle

    def physics_relaxation(initial_params, num_steps=2000, dt=0.01):
        """
        Physics-based relaxation using repulsive forces.
        params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
        Only optimizes inner positions (not outer).
        """
        params = np.array(initial_params, dtype=float).copy()
        n = N
        
        # Precompute hexagon vertices for each inner hexagon
        def get_verts(i):
            x, y, a = params[3*i], params[3*i+1], params[3*i+2]
            return hexagon_vertices(x, y, 1.0, a)
        
        # Lennard-Jones-like repulsion between hexagons
        # We'll use a simpler model: repel if center distance < threshold
        min_center_dist = 2.0  # Minimum center distance for non-overlap
        
        for step in range(num_steps):
            if time.time() > deadline - 10:
                break
                
            forces = np.zeros_like(params)
            
            # Compute pairwise repulsive forces
            for i in range(n):
                for j in range(i+1, n):
                    xi, yi = params[3*i], params[3*i+1]
                    xj, yj = params[3*j], params[3*j+1]
                    
                    dx = xj - xi
                    dy = yj - yi
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < min_center_dist and dist > 1e-6:
                        # Repulsive force proportional to overlap
                        overlap = min_center_dist - dist
                        # Use a strong repulsion
                        force_mag = 10.0 * overlap
                        fx = force_mag * dx / dist
                        fy = force_mag * dy / dist
                        
                        forces[3*i] -= fx
                        forces[3*i+1] -= fy
                        forces[3*j] += fx
                        forces[3*j+1] += fy
            
            # Add a weak centering force to keep configuration compact
            center_of_mass = np.mean(params[:3*n].reshape(n, 3)[:, :2], axis=0)
            for i in range(n):
                forces[3*i] += -0.01 * (params[3*i] - center_of_mass[0])
                forces[3*i+1] += -0.01 * (params[3*i+1] - center_of_mass[1])
            
            # Update positions with damping
            params[:3*n:3] += forces[:3*n:3] * dt
            params[1:3*n:3] += forces[1:3*n:3] * dt
            
            # Keep angles fixed (they're not critical for packing)
        
        return params

    # Initial configuration: hexagonal lattice pattern
    # For 12 hexagons, use a 4x3 grid with hexagonal offset
    initial_positions = []
    # Row 0: 4 hexagons
    for j in range(4):
        x = (j - 1.5) * 2.05
        y = 0.0
        initial_positions.append([x, y, 0.0])
    # Row 1: 4 hexagons (offset)
    for j in range(4):
        x = (j - 1.5) * 2.05 + 1.025
        y = 1.78
        initial_positions.append([x, y, 0.0])
    # Row 2: 4 hexagons
    for j in range(4):
        x = (j - 1.5) * 2.05
        y = 3.56
        initial_positions.append([x, y, 0.0])
    
    # Compute initial outer hexagon
    inner_init = [(p[0], p[1], p[2]) for p in initial_positions]
    os_init, oc_init, oa_init = compute_min_outer_side(inner_init)
    
    # Create params vector (only inner positions, no outer)
    params = []
    for pos in initial_positions:
        params.extend([pos[0], pos[1], pos[2]])
    
    # Run physics relaxation
    if time.time() < deadline - 20:
        relaxed_params = physics_relaxation(params, num_steps=3000, dt=0.005)
    else:
        relaxed_params = np.array(params)
    
    # Convert back to positions
    inner_relaxed = []
    for i in range(N):
        x = relaxed_params[3*i]
        y = relaxed_params[3*i+1]
        a = relaxed_params[3*i+2]
        inner_relaxed.append((x, y, a))
    
    # Compute optimal outer hexagon for relaxed configuration
    os_relaxed, oc_relaxed, oa_relaxed = compute_min_outer_side(inner_relaxed)
    
    # Validate the relaxed configuration
    if not all_inside_outer(inner_relaxed, oc_relaxed, os_relaxed, oa_relaxed):
        # Expand outer if needed
        while not all_inside_outer(inner_relaxed, oc_relaxed, os_relaxed, oa_relaxed):
            os_relaxed += 0.05
            if os_relaxed > 10:
                break
    
    if not pairwise_disjoint(inner_relaxed):
        # If still overlapping, use initial configuration
        inner_final = inner_init
        os_final, oc_final, oa_final = os_init, oc_init, oa_init
        while not all_inside_outer(inner_final, oc_final, os_final, oa_final):
            os_final += 0.05
            if os_final > 10:
                break
    else:
        inner_final = inner_relaxed
        os_final, oc_final, oa_final = os_relaxed, oc_relaxed, oa_relaxed
    
    # Try local refinement with Nelder-Mead on a smooth objective
    def smooth_objective(p):
        """Smooth objective using penalty for overlaps and containment."""
        positions = []
        for i in range(N):
            positions.append((p[3*i], p[3*i+1], p[3*i+2]))
        
        # Compute outer hexagon
        os_val, oc_val, oa_val = compute_min_outer_side(positions)
        
        # Check containment with penalty
        penalty = 0.0
        for cx, cy, angle in positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, oc_val[0], oc_val[1], os_val, oa_val):
                    # Compute distance outside
                    dx = vx - oc_val[0]
                    dy = vy - oc_val[1]
                    angle_rad = np.radians(oa_val)
                    cos_a = np.cos(-angle_rad)
                    sin_a = np.sin(-angle_rad)
                    local_x = dx * cos_a - dy * sin_a
                    local_y = dx * sin_a + dy * cos_a
                    apothem = os_val * np.sqrt(3) / 2
                    normal_angles = np.radians(np.arange(30, 360, 60))
                    max_violation = 0
                    for na in normal_angles:
                        nx = np.cos(na)
                        ny = np.sin(na)
                        dist = local_x * nx + local_y * ny
                        if dist > apothem:
                            max_violation = max(max_violation, dist - apothem)
                    penalty += max_violation * 5.0
        
        # Check pairwise disjointness with penalty
        for i in range(N):
            for j in range(i+1, N):
                ci = (positions[i][0], positions[i][1])
                cj = (positions[j][0], positions[j][1])
                ai = positions[i][2]
                aj = positions[j][2]
                if not hexagons_disjoint(ci, 1.0, ai, cj, 1.0, aj):
                    # Compute center distance penalty
                    dx = ci[0] - cj[0]
                    dy = ci[1] - cj[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 2.0:
                        penalty += (2.0 - dist) * 5.0
        
        return os_val + penalty
    
    # Local refinement
    if time.time() < deadline - 5:
        try:
            res = minimize(smooth_objective, relaxed_params, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 0.01, 'fatol': 0.005, 'adaptive': True})
            if res.fun < os_final:
                # Validate the refined result
                inner_refined = []
                for i in range(N):
                    inner_refined.append((res.x[3*i], res.x[3*i+1], res.x[3*i+2]))
                
                os_ref, oc_ref, oa_ref = compute_min_outer_side(inner_refined)
                
                if all_inside_outer(inner_refined, oc_ref, os_ref, oa_ref) and pairwise_disjoint(inner_refined):
                    inner_final = inner_refined
                    os_final, oc_final, oa_final = os_ref, oc_ref, oa_ref
        except Exception:
            pass
    
    # Final result
    inner = [[p[0], p[1], p[2]] for p in inner_final]
    
    return inner, [oc_final[0], oc_final[1]], os_final, oa_final
# EVOLVE-BLOCK-END
