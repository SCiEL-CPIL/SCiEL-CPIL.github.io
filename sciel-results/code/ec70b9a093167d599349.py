"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    SQRT3 = math.sqrt(3)

    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            vals = (x, y)
            verts.append(vals)
        return verts

    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2):
        all_axes = axes1 + axes2
        for axis in all_axes:
            nx, ny = axis
            min1 = min(vx * nx + vy * ny for vx, vy in hex1_verts)
            max1 = max(vx * nx + vy * ny for vx, vy in hex1_verts)
            min2 = min(vx * nx + vy * ny for vx, vy in hex2_verts)
            max2 = max(vx * nx + vy * ny for vx, vy in hex2_verts)
            if max1 < min2 or max2 < min1:
                return True
        return False

    def calc_outer_side(hexagons, outer_angle_deg):
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))

        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj

        return 2.0 * max_proj / SQRT3

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    def pbd_solve(initial_params, max_iters=3000, shrink_start=50, shrink_rate=0.995, initial_side=15.0):
        # params: [x1, y1, a1, x2, y2, a2, ...]
        params = initial_params.copy()
        current_side = initial_side
        
        # Precompute hexagon geometry relative to center
        # Vertices at angle 0: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        base_verts = np.array([
            [1.0, 0.0],
            [0.5, SQRT3/2],
            [-0.5, SQRT3/2],
            [-1.0, 0.0],
            [-0.5, -SQRT3/2],
            [0.5, -SQRT3/2]
        ])
        
        # Normals for SAT
        base_normals = np.array([
            [0.0, 1.0], # Normal to edge 0-1 (angle 30 deg? No, edge 0 is (1,0)->(0.5, s/2). Midpoint angle 30. Normal is 90+30=120? 
            # Let's use the standard 6 normals for a hexagon aligned with x-axis:
            # Edges are at angles 0, 60, 120, 180, 240, 300.
            # Normals are perpendicular: 90, 150, 210, 270, 330, 30.
            # Let's just compute them from base_verts to be safe.
        ])
        
        # Actually, let's stick to the python loop for SAT but optimize it slightly or just use it.
        # For N=11, 55 pairs. 6+6 axes = 12 checks. 660 checks per iter. 3000 iters = 2M checks. Python might be slow.
        # Let's try to vectorize SAT or use a simpler distance metric if PBD is too slow.
        # However, PBD requires projection. 
        # Alternative: Use a "soft" constraint solver via gradient descent on a smooth distance function?
        # The prompt asks for PBD. Let's implement a simplified PBD that uses a distance field approximation or just SAT separation.
        
        # To make it fast enough in Python, we will limit iterations or use a coarser initial shrink.
        # Let's try to implement a vectorized SAT check for all pairs? No, axes depend on angle.
        
        # We will run PBD for a limited number of iterations to stay within time.
        
        best_params = params.copy()
        best_side = current_side
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            # 1. Calculate current bounding side
            # We can approximate the side length by max projection.
            # For PBD, we need to know if they are inside a hexagon of side 'current_side'.
            # Hexagon containment: For each of the 6 normals n_k, |v . n_k| <= apothem = side * sqrt(3)/2.
            # So max_proj <= side * sqrt(3)/2.
            
            # Let's compute the required side for current configuration
            max_proj = 0.0
            hexagons = []
            axes_list = []
            for i in range(N):
                cx = params[i*3]
                cy = params[i*3+1]
                a_deg = params[i*3+2]
                verts = get_hex_vertices(cx, cy, a_deg)
                hexagons.append(verts)
                axes_list.append(get_hex_axes(a_deg))
                
                # Check containment
                for vx, vy in verts:
                    # Project onto 6 axes of the container (aligned with 0, 60, etc relative to center 0,0? No, container is fixed at 0,0)
                    # The container axes are fixed at 30, 90, 150, 210, 270, 330 degrees.
                    for k in range(6):
                        angle_rad = math.radians(30 + k*60)
                        nx = math.cos(angle_rad)
                        ny = math.sin(angle_rad)
                        proj = abs(vx * nx + vy * ny)
                        if proj > max_proj:
                            max_proj = proj
                        
            required_side = 2.0 * max_proj / SQRT3
            
            if required_side < best_side:
                best_side = required_side
                best_params = params.copy()
                
            # Shrink the target side
            if it > shrink_start:
                current_side *= shrink_rate
            else:
                current_side = required_side + 0.5 # Start loose
            
            # 2. Resolve constraints
            # a) Containment: If any vertex projects beyond apothem = current_side * sqrt(3)/2, push center back.
            apothem = current_side * SQRT3 / 2.0
            for i in range(N):
                cx = params[i*3]
                cy = params[i*3+1]
                a_deg = params[i*3+2]
                verts = get_hex_vertices(cx, cy, a_deg)
                
                max_violation = 0.0
                push_dir = (0.0, 0.0)
                
                for vx, vy in verts:
                    for k in range(6):
                        angle_rad = math.radians(30 + k*60)
                        nx = math.cos(angle_rad)
                        ny = math.sin(angle_rad)
                        proj = vx * nx + vy * ny
                        abs_proj = abs(proj)
                        if abs_proj > apothem:
                            violation = abs_proj - apothem
                            if violation > max_violation:
                                max_violation = violation
                                # Push direction is -sign(proj) * normal
                                if proj > 0:
                                    push_dir = (-nx, -ny)
                                else:
                                    push_dir = (nx, ny)
                
                if max_violation > 0:
                    # Move center by half the violation (PBD standard is 1.0 or 0.5, let's use 0.5 for stability)
                    params[i*3] += push_dir[0] * max_violation * 0.5
                    params[i*3+1] += push_dir[1] * max_violation * 0.5

            # b) Pairwise Non-Penetration
            # Re-fetch vertices after containment move? Or just use current params?
            # For efficiency, we'll just use current params for SAT.
            for i in range(N):
                for j in range(i+1, N):
                    cx1 = params[i*3]
                    cy1 = params[i*3+1]
                    a1 = params[i*3+2]
                    cx2 = params[j*3]
                    cy2 = params[j*3+1]
                    a2 = params[j*3+2]
                    
                    # Quick reject based on center distance
                    dx = cx1 - cx2
                    dy = cy1 - cy2
                    dist_sq = dx*dx + dy*dy
                    min_dist = SQRT3 # Distance between centers of touching hexagons
                    if dist_sq > (min_dist * 1.5)**2: # Loose bound
                        continue
                    
                    verts1 = get_hex_vertices(cx1, cy1, a1)
                    verts2 = get_hex_vertices(cx2, cy2, a2)
                    axes1 = get_hex_axes(a1)
                    axes2 = get_hex_axes(a2)
                    
                    # SAT
                    min_overlap = float('inf')
                    best_axis = None
                    
                    for axis in axes1 + axes2:
                        nx, ny = axis
                        min1 = min(vx * nx + vy * ny for vx, vy in verts1)
                        max1 = max(vx * nx + vy * ny for vx, vy in verts1)
                        min2 = min(vx * nx + vy * ny for vx, vy in verts2)
                        max2 = max(vx * nx + vy * ny for vx, vy in verts2)
                        
                        if max1 < min2 or max2 < min1:
                            break # Disjoint
                            
                        overlap = min(max1, max2) - max(min1, min2)
                        if overlap < min_overlap:
                            min_overlap = overlap
                            best_axis = axis
                            
                    if best_axis is not None:
                        # Resolve overlap
                        # Determine direction to push
                        # If center1 is to the "right" of center2 along axis, push 1 right, 2 left
                        nx, ny = best_axis
                        dot1 = cx1 * nx + cy1 * ny
                        dot2 = cx2 * nx + cy2 * ny
                        
                        push_mag = min_overlap * 0.5
                        if dot1 > dot2:
                            params[i*3] += nx * push_mag
                            params[i*3+1] += ny * push_mag
                            params[j*3] -= nx * push_mag
                            params[j*3+1] -= ny * push_mag
                        else:
                            params[i*3] -= nx * push_mag
                            params[i*3+1] -= ny * push_mag
                            params[j*3] += nx * push_mag
                            params[j*3+1] += ny * push_mag

        return best_params, best_side

    # Initial configurations
    coords1 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    
    def make_initial_params(coords, spread=1.0):
        params = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing=spread)
            # Add small random noise to break symmetry
            x += random.gauss(0, 0.1)
            y += random.gauss(0, 0.1)
            params.extend([x, y, random.uniform(-10, 10)])
        return np.array(params)

    best_params_global = None
    best_score_global = float('inf')

    # Run PBD from multiple starts
    for restart in range(5):
        if time.time() > deadline - 20.0:
            break
            
        # Vary the initial spread and noise
        spread = 1.0 + restart * 0.2
        params = make_initial_params(coords1, spread=spread)
        
        # Run PBD
        p_params, p_score = pbd_solve(params, max_iters=2000 + restart*500)
        
        if p_score < best_score_global:
            best_score_global = p_score
            best_params_global = p_params

    # Final Polish: Local optimization using Nelder-Mead on the exact side length
    if best_params_global is not None and time.time() < deadline - 5.0:
        final_params = best_params_global.copy()
        
        def objective(p):
            centers = []
            angles = []
            for i in range(N):
                centers.append((p[i*3], p[i*3+1]))
                angles.append(p[i*3+2])
                
            cx_avg = sum(c[0] for c in centers) / N
            cy_avg = sum(c[1] for c in centers) / N
            shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
            
            hexagons = []
            hex_axes = []
            for i in range(N):
                cx, cy = shifted_centers[i]
                a = angles[i]
                hex_verts = get_hex_vertices(cx, cy, a)
                hexagons.append(hex_verts)
                hex_axes.append(get_hex_axes(a))
                
            # Check validity
            for i in range(N):
                for j in range(i+1, N):
                    if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                        return 100.0 # Penalty for overlap
            
            # Minimize side length by optimizing outer rotation? 
            # Actually, we just return the min side length over a few outer rotations
            min_side = float('inf')
            for oa in range(0, 60, 10):
                side = calc_outer_side(hexagons, oa)
                if side < min_side:
                    min_side = side
            return min_side

        try:
            from scipy.optimize import minimize
            res = minimize(objective, final_params, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_score_global:
                best_score_global = res.fun
                best_params_global = res.x
        except Exception:
            pass

    # Extract final result
    if best_params_global is None:
        # Fallback
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg_fb = sum(p[0] for p in inner_list) / N
        cy_avg_fb = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg_fb, y - cy_avg_fb, a] for x, y, a in inner_list]
        hexagons_fb = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side_fb = calc_outer_side(hexagons_fb, 0.0)
        return (inner_list, (0.0, 0.0), best_side_fb, 0.0)

    final_params = best_params_global.copy()
    centers = []
    angles = []
    for i in range(N):
        centers.append((final_params[i*3], final_params[i*3+1]))
        angles.append(final_params[i*3+2])

    cx_avg = sum(c[0] for c in centers) / N
    cy_avg = sum(c[1] for c in centers) / N
    shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]

    hexagons = []
    hex_axes = []
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = angles[i]
        hex_verts = get_hex_vertices(cx, cy, a)
        hexagons.append(hex_verts)
        hex_axes.append(get_hex_axes(a))

    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                valid = False
                break
        if not valid:
            break

    if not valid:
        # Fallback to initial grid
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg_fb = sum(p[0] for p in inner_list) / N
        cy_avg_fb = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg_fb, y - cy_avg_fb, a] for x, y, a in inner_list]
        hexagons_fb = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side_fb = calc_outer_side(hexagons_fb, 0.0)
        return (inner_list, (0.0, 0.0), best_side_fb, 0.0)

    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 600):
        oa = oa_deg * 0.1
        side = calc_outer_side(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa = oa

    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])

    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
