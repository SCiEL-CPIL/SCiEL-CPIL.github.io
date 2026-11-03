"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def get_gap(v1, v2):
        """
        Returns the maximum gap between two hexagons.
        If gap > 0, they are separated.
        If gap < 0, they overlap.
        """
        max_gap = -float('inf')
        for i in range(3):
            e1 = v1[(i+1)%6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                n1 = n1 / norm1
            
            e2 = v2[(i+1)%6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                n2 = n2 / norm2
            
            for n in [n1, n2]:
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    overlap_amount = min(max1, max2) - max(min1, min2)
                    gap = -overlap_amount
                    
                if gap > max_gap:
                    max_gap = gap
                    
        return max_gap

    def compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg):
        """
        Computes the minimum side length of the outer hexagon centered at (outer_cx, outer_cy)
        with rotation outer_angle_deg to contain all inner hexagons.
        """
        outer_center = np.array([outer_cx, outer_cy])
        outer_angle_rad = math.radians(outer_angle_deg)
        
        max_proj = 0.0
        for i in range(N):
            cx, cy = inner_centers[i]
            ang = inner_angles[i]
            verts = hex_vertices(cx, cy, ang)
            rel_verts = verts - outer_center
            for k in range(6):
                normal_angle = outer_angle_rad + k * math.pi / 3
                normal = np.array([math.cos(normal_angle), math.sin(normal_angle)])
                proj = np.dot(rel_verts, normal)
                max_proj = max(max_proj, np.max(proj))
        
        return 2.0 * max_proj / math.sqrt(3)

    def objective(params):
        # Unpack parameters
        # 0..23: inner centers (x, y) for 12 hexagons
        # 24..35: inner angles (degrees) for 12 hexagons
        # 36..38: outer cx, cy, angle (degrees)
        
        inner_centers = []
        inner_angles = []
        for i in range(N):
            cx = params[2*i]
            cy = params[2*i + 1]
            inner_centers.append((cx, cy))
            inner_angles.append(params[24 + i]) # Degrees

        outer_cx = params[36]
        outer_cy = params[37]
        outer_angle_deg = params[38]
        
        # Calculate side length
        side = compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg)
        
        # Calculate overlap penalty
        penalty = 0.0
        verts_list = [hex_vertices(inner_centers[i][0], inner_centers[i][1], inner_angles[i]) for i in range(N)]
        
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < 0:
                    # Penalize overlap. gap is negative.
                    penalty += (-gap) ** 2
        
        # Weight the penalty heavily
        return side + 100.0 * penalty

    # Initial configuration: 12 hexagons in a circle of radius 5
    x0 = np.zeros(39)
    for i in range(N):
        angle = i * 2 * math.pi / N
        x0[2*i] = 5.0 * math.cos(angle)
        x0[2*i+1] = 5.0 * math.sin(angle)
        x0[24+i] = 0.0 # Inner angles in degrees
    
    x0[36] = 0.0
    x0[37] = 0.0
    x0[38] = 0.0
    
    best_side = float('inf')
    best_params = x0
    best_valid = False

    # Run optimization
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-5}
        )
        
        # Evaluate result
        inner_centers = []
        inner_angles = []
        for i in range(N):
            cx = res.x[2*i]
            cy = res.x[2*i + 1]
            inner_centers.append((cx, cy))
            inner_angles.append(res.x[24 + i])
            
        outer_cx = res.x[36]
        outer_cy = res.x[37]
        outer_angle_deg = res.x[38]
        
        side = compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg)
        
        # Check validity
        verts_list = [hex_vertices(inner_centers[i][0], inner_centers[i][1], inner_angles[i]) for i in range(N)]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < -1e-5: # Allow tiny numerical error
                    valid = False
                    break
            if not valid: break
            
        if valid:
            best_side = side
            best_params = res.x
            best_valid = True
    except Exception as e:
        pass

    # If optimization didn't yield a valid result, try a simple valid grid
    if not best_valid:
        # Construct a valid 3x4 grid with correct hexagonal spacing
        centers = []
        for r in range(4):
            for c in range(3):
                x = c * math.sqrt(3)
                y = r * 1.5
                # Stagger
                if r % 2 == 1:
                    x += math.sqrt(3) / 2
                centers.append((x, y))
                
        # We have 12 centers.
        # Center the grid at 0,0
        cx_mean = sum(c[0] for c in centers) / N
        cy_mean = sum(c[1] for c in centers) / N
        adjusted_centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
        
        inner_angles = [0.0] * N
        outer_cx = 0.0
        outer_cy = 0.0
        outer_angle_deg = 0.0
        
        side = compute_outer_side(adjusted_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg)
        
        # Check validity of this fallback
        verts_list = [hex_vertices(c[0], c[1], 0.0) for c in adjusted_centers]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < -1e-5:
                    valid = False
                    break
            if not valid: break
            
        if valid:
            best_side = side
            # Map back to params format for consistency, though we just return directly
            inner_list = []
            for c, a in zip(adjusted_centers, inner_angles):
                inner_list.append([float(c[0]), float(c[1]), float(a)])
            return inner_list, (float(outer_cx), float(outer_cy)), float(best_side * 1.01), float(outer_angle_deg)
        else:
            # Absolute fallback: huge box
            inner_list = [[0.0, 0.0, 0.0] for _ in range(N)]
            return inner_list, (0.0, 0.0), 100.0, 0.0

    # Construct return value from best_params
    inner = []
    for i in range(N):
        cx = best_params[2*i]
        cy = best_params[2*i + 1]
        ang_deg = best_params[24 + i]
        inner.append([float(cx), float(cy), float(ang_deg)])
        
    outer_cx = best_params[36]
    outer_cy = best_params[37]
    outer_angle_deg = best_params[38]
    
    # Add safety margin
    final_side = best_side * 1.01 + 1e-4
    
    return inner, (float(outer_cx), float(outer_cy)), float(final_side), float(outer_angle_deg)
# EVOLVE-BLOCK-END
