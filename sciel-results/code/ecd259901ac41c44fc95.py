import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    sqrt3 = math.sqrt(3)
    
    # Unit hexagon vertices (flat-top: 0, 60, 120, 180, 240, 300)
    unit_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, angle_deg=0.0):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts @ rot.T
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, outer_angle):
        """
        positions: (N, 2) array of centers.
        outer_angle: rotation of outer hexagon in degrees.
        Returns side length S.
        """
        theta_rad = math.radians(outer_angle)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3), 
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            # Inner hexagons are fixed at angle 0 for this optimization phase
            verts = get_hex_verts(cx, cy, 0.0)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / sqrt3

    def overlap_penalty(positions):
        """
        Calculate a penalty based on overlaps between inner hexagons.
        Returns 0 if no overlap, else a positive value.
        """
        penalty = 0.0
        # Pre-compute vertices for all hexagons to avoid redundant calculation
        # Note: This is expensive, but N=12 is small.
        verts_list = [get_hex_verts(positions[i][0], positions[i][1], 0.0) for i in range(N)]
        
        for i in range(N):
            for j in range(i+1, N):
                v1 = verts_list[i]
                v2 = verts_list[j]
                # Check separation using SAT
                # Axes from v1
                separated = False
                min_penetration = float('inf')
                
                # We need to check all 12 normals (6 from v1, 6 from v2)
                axes = []
                for k in range(6):
                    edge = v1[(k+1)%6] - v1[k]
                    norm = np.array([-edge[1], edge[0]])
                    norm = norm / np.linalg.norm(norm)
                    axes.append(norm)
                for k in range(6):
                    edge = v2[(k+1)%6] - v2[k]
                    norm = np.array([-edge[1], edge[0]])
                    norm = norm / np.linalg.norm(norm)
                    axes.append(norm)
                    
                for axis in axes:
                    proj1 = v1 @ axis
                    proj2 = v2 @ axis
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    if max1 < min2 or max2 < min1:
                        separated = True
                        break
                    else:
                        # Calculate penetration depth in this direction
                        # If they overlap, the gap is negative.
                        # Gap = min(min2 - max1, min1 - max2) ?
                        # If max1 >= min2 and max2 >= min1:
                        # Overlap extent = min(max1, max2) - max(min1, min2)
                        # This is the depth of overlap along this axis.
                        # We want the MINIMUM overlap depth over all axes?
                        # No, if they overlap, they overlap.
                        # The "penetration depth" is the minimum distance to separate them.
                        # This is the minimum of (max1 - min2, max2 - min1) ??
                        # Actually, if they overlap, max1 >= min2 and max2 >= min1.
                        # The amount we need to push them apart along this axis is max(min2 - max1, min1 - max2) ?
                        # If max1 > min2, they overlap. Push amount = max1 - min2 ? No.
                        # Let's just use a simple indicator: if not separated, add penalty.
                        # To make it smooth, we can use the minimum "gap" (which is negative).
                        # gap = min(min2 - max1, min1 - max2)
                        # If gap < 0, they overlap.
                        # Penalty += gap^2
                        gap = min(min2 - max1, min1 - max2)
                        if gap < 0:
                            # They overlap. The depth is -gap.
                            # We take the minimum depth over all axes?
                            # Actually, if they overlap, they are NOT separated by ANY axis.
                            # So we just need to know they overlap.
                            # For a smooth penalty, we can sum the squares of negative gaps.
                            penalty += (-gap)**2
                        # If gap >= 0, no overlap in this direction.
                
                if separated:
                    # If separated in any axis, no overlap.
                    # But we already added penalty for previous axes?
                    # No, we break out of the loop.
                    # So if separated, penalty contribution is 0.
                    pass
                else:
                    # If not separated, we accumulated penalty.
                    # But wait, if they overlap, they are not separated by ANY axis.
                    # So the loop will finish without breaking.
                    # The penalty will be the sum of squared negative gaps.
                    # This is a valid smooth penalty.
                    pass

        return penalty

    # Initial configuration: 3x4 lattice
    # v1 = (1.5, sqrt3/2), v2 = (0, sqrt3)
    # Let's create a 4x3 grid (4 columns, 3 rows)
    coords = []
    for r in range(3):
        for c in range(4):
            x = 1.5 * c
            y = sqrt3 * (r + 0.5 * c) # Offset every other column?
            # Standard hex grid:
            # x = 1.5 * c
            # y = sqrt3 * (r + 0.5 * (c % 2))
            # Let's use the standard axial coordinates to cartesian
            # q = c, r = r
            # x = 1.5 * q
            # y = sqrt3 * (r + 0.5 * q)
            coords.append((x, y))
            
    # Center the coords
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    cx = sum(xs) / N
    cy = sum(ys) / N
    init_pos = np.array([(x - cx, y - cy) for x, y in coords])
    
    # Variables: x1, y1, ..., x12, y12, alpha_outer
    # Total 25 variables
    init_vars = np.append(init_pos.flatten(), 0.0)
    
    def objective(vars):
        if time.time() > deadline:
            return 1e9
        pos = vars[:24].reshape((N, 2))
        alpha_outer = vars[24]
        
        s_outer = calc_min_outer_side(pos, alpha_outer)
        pen = overlap_penalty(pos)
        
        # Weight for penalty
        # If pen is large, s_outer is irrelevant
        return s_outer + 10.0 * pen

    best_result = None
    best_score = float('inf')
    
    # Try a few random perturbations of the initial lattice
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        noise = np.random.normal(0, 0.1, 24)
        x0 = np.append(init_pos.flatten() + noise, np.random.uniform(0, 60))
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res
        except Exception:
            pass
            
    if best_result is None:
        # Fallback to initial lattice
        pos = init_pos
        alpha_outer = 0.0
        s_outer = calc_min_outer_side(pos, alpha_outer)
        # Check validity
        if overlap_penalty(pos) > 1e-6:
             # Should not happen for lattice
             pass
        best_positions = pos
        best_angles = np.zeros(N)
        best_outer_angle = 0.0
        best_side = s_outer
    else:
        best_positions = best_result.x[:24].reshape((N, 2))
        best_angles = np.zeros(N)
        best_outer_angle = best_result.x[24]
        best_side = calc_min_outer_side(best_positions, best_outer_angle)
        
        # Final validity check
        if overlap_penalty(best_positions) > 1e-4:
            # If invalid, revert to lattice
            best_positions = init_pos
            best_angles = np.zeros(N)
            best_outer_angle = 0.0
            best_side = calc_min_outer_side(best_positions, best_outer_angle)

    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], best_angles[i]])
    
    return inner, (0.0, 0.0), best_side, best_outer_angle
