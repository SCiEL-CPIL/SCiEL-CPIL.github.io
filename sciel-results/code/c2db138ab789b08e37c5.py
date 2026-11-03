"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

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

    def objective(params):
        # Unpack parameters
        inner_centers = []
        inner_angles = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i + 1]
            ang = params[3*i + 2]
            inner_centers.append((cx, cy))
            inner_angles.append(ang)
        
        outer_cx = params[36]
        outer_cy = params[37]
        outer_angle_deg = params[38]
        
        # Calculate side length
        side = compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg)
        
        # Approximate overlap penalty for speed
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = inner_centers[i][0] - inner_centers[j][0]
                dy = inner_centers[i][1] - inner_centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Minimum distance for non-overlapping unit hexagons is 2.0 (if aligned)
                # Actually, if rotated, they can be closer. But 2.0 is a good lower bound for center distance
                # if they are not interlocking. If they are interlocking, dist can be small.
                # A safe threshold for "potential overlap" is 2.0.
                if dist < 2.0:
                    penalty += (2.0 - dist) ** 2
        
        return side + 100.0 * penalty

    # Generate initial configurations
    def gen_honeycomb(rows, cols, offset_x=0, offset_y=0, angle_offset=0):
        centers = []
        s = 1.0
        dx = 2.0 * s
        dy = math.sqrt(3) * s
        offset = s
        count = 0
        for r in range(rows):
            for c in range(cols):
                if count >= N:
                    break
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += offset
                centers.append((x + offset_x, y + offset_y))
                count += 1
        # Pad if necessary
        while len(centers) < N:
            centers.append((0,0))
        # Center
        cx_mean = sum(c[0] for c in centers) / N
        cy_mean = sum(c[1] for c in centers) / N
        adjusted_centers = [(x - cx_mean, y - cy_mean) for x, y in centers]
        angles = [angle_offset] * N
        return adjusted_centers, angles

    initial_configs = []
    
    # Config 0: 4x3 Honeycomb
    c0, a0 = gen_honeycomb(4, 3)
    initial_configs.append((c0, a0))
    
    # Config 1: 3x4 Honeycomb
    c1, a1 = gen_honeycomb(3, 4)
    initial_configs.append((c1, a1))
    
    # Config 2: 4x3 Honeycomb rotated 30 deg
    c2, a2 = gen_honeycomb(4, 3, angle_offset=30)
    initial_configs.append((c2, a2))
    
    # Config 3: 2x6 Honeycomb
    c3, a3 = gen_honeycomb(2, 6)
    initial_configs.append((c3, a3))
    
    # Config 4: 6x2 Honeycomb
    c4, a4 = gen_honeycomb(6, 2)
    initial_configs.append((c4, a4))
    
    best_side = float('inf')
    best_params = None

    for idx, (centers, angles) in enumerate(initial_configs):
        if time.time() > deadline:
            break
            
        x0 = np.zeros(39)
        for i in range(N):
            x0[3*i] = centers[i][0]
            x0[3*i+1] = centers[i][1]
            x0[3*i+2] = angles[i]
        
        x0[36] = 0.0
        x0[37] = 0.0
        x0[38] = 0.0
        
        # Run optimization
        try:
            res = minimize(
                objective,
                x0,
                method='Powell',
                options={
                    'maxiter': 500,
                    'maxfev': 1000,
                    'xtol': 1e-4,
                    'ftol': 1e-4
                }
            )
            
            # Evaluate result
            inner_centers = []
            inner_angles = []
            for i in range(N):
                cx = res.x[3*i]
                cy = res.x[3*i + 1]
                inner_centers.append((cx, cy))
                inner_angles.append(res.x[3*i+2])
                
            outer_cx = res.x[36]
            outer_cy = res.x[37]
            outer_angle_deg = res.x[38]
            
            side = compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg)
            
            # Check validity with exact gap
            verts_list = [hex_vertices(inner_centers[i][0], inner_centers[i][1], inner_angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    gap = get_gap(verts_list[i], verts_list[j])
                    if gap < -1e-5:
                        valid = False
                        break
                if not valid: break
                
            if valid and side < best_side:
                best_side = side
                best_params = res.x.copy()
                
        except Exception as e:
            pass

    # If no valid result found, use the first config as fallback
    if best_params is None:
        centers, angles = initial_configs[0]
        inner_centers = centers
        inner_angles = angles
        outer_cx = 0.0
        outer_cy = 0.0
        outer_angle_deg = 0.0
        side = compute_outer_side(inner_centers, inner_angles, outer_cx, outer_cy, outer_angle_deg)
        
        # Check validity
        verts_list = [hex_vertices(c[0], c[1], a) for c, a in zip(inner_centers, inner_angles)]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                gap = get_gap(verts_list[i], verts_list[j])
                if gap < -1e-5:
                    valid = False
                    break
            if not valid: break
            
        if not valid:
             # Last resort: return a very large box
             inner = [[0.0, 0.0, 0.0] for _ in range(N)]
             return inner, (0.0, 0.0), 100.0, 0.0
             
        best_side = side
        best_params = np.zeros(39)
        for i in range(N):
            best_params[3*i] = inner_centers[i][0]
            best_params[3*i+1] = inner_centers[i][1]
            best_params[3*i+2] = inner_angles[i]
        best_params[36] = outer_cx
        best_params[37] = outer_cy
        best_params[38] = outer_angle_deg

    # Construct return value
    inner = []
    for i in range(N):
        cx = best_params[3*i]
        cy = best_params[3*i+1]
        ang_deg = best_params[3*i+2]
        inner.append([float(cx), float(cy), float(ang_deg)])
        
    outer_cx = best_params[36]
    outer_cy = best_params[37]
    outer_angle_deg = best_params[38]
    
    # Add safety margin
    final_side = best_side * 1.01 + 1e-4
    
    return inner, (float(outer_cx), float(outer_cy)), float(final_side), float(outer_angle_deg)
# EVOLVE-BLOCK-END
