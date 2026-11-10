"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    SQRT3 = math.sqrt(3)
    
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    def calc_outer_side(hexagons, outer_angle_deg):
        base_angles = [30, 90, 150, 210, 270, 330]
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for a in base_angles:
                    rad = math.radians(a + outer_angle_deg)
                    nx = math.cos(rad)
                    ny = math.sin(rad)
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / SQRT3

    def get_axes(poly):
        axes = []
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            norm = math.sqrt(dx*dx + dy*dy)
            if norm == 0: continue
            nx = -dy / norm
            ny = dx / norm
            axes.append((nx, ny))
        return axes

    def project(poly, axis):
        dots = [p[0]*axis[0] + p[1]*axis[1] for p in poly]
        return min(dots), max(dots)

    def penetration_depth(poly1, poly2):
        axes = get_axes(poly1) + get_axes(poly2)
        min_pen = float('inf')
        for axis in axes:
            min1, max1 = project(poly1, axis)
            min2, max2 = project(poly2, axis)
            if max1 < min2 or max2 < min1:
                return 0.0
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_pen:
                min_pen = overlap
        return min_pen

    def axial_to_cartesian(q, r, spacing=2.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    base_7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    extra = [(2,0), (2,-1), (2,1), (1,1)]
    coords = base_7 + extra

    def init_positions(spacing=2.2):
        positions = []
        angles = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing=spacing)
            positions.append([x, y])
            angles.append(0.0)
        cx_avg = sum(p[0] for p in positions) / N
        cy_avg = sum(p[1] for p in positions) / N
        for i in range(N):
            positions[i][0] -= cx_avg
            positions[i][1] -= cy_avg
        return positions, angles

    def compute_score(positions, angles):
        if time.time() > deadline - 2.0:
            return None
        hexagons = [get_hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if penetration_depth(hexagons[i], hexagons[j]) > 1e-5:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            return None
        best_side = float('inf')
        best_oa = 0.0
        for oa_deg in range(0, 60, 2):
            side = calc_outer_side(hexagons, oa_deg)
            if side < best_side:
                best_side = side
                best_oa = oa_deg
        return best_side, best_oa

    best_side = float('inf')
    best_state = None
    best_oa = 0.0

    # Phase 1: Multi-start PBD
    for start_idx in range(12):
        if time.time() > deadline - 40.0:
            break
        positions, angles = init_positions(spacing=2.2 + start_idx * 0.1)
        
        # Add some random noise to positions
        for i in range(N):
            positions[i][0] += random.gauss(0, 0.2)
            positions[i][1] += random.gauss(0, 0.2)
            angles[i] += random.gauss(0, 10.0)
        
        # Center again
        cx_avg = sum(p[0] for p in positions) / N
        cy_avg = sum(p[1] for p in positions) / N
        for i in range(N):
            positions[i][0] -= cx_avg
            positions[i][1] -= cy_avg

        num_frames = 800
        dt = 1.0
        stiffness = 0.5
        pressure_strength = 0.08 + start_idx * 0.01
        damping = 0.85
        
        velocities = [[0.0, 0.0] for _ in range(N)]
        angular_velocities = [0.0 for _ in range(N)]
        
        local_best_side = float('inf')
        local_best_state = None
        local_best_oa = 0.0

        for frame in range(num_frames):
            if time.time() > deadline - 40.0:
                break
            
            # Update velocities with pressure
            for i in range(N):
                fx = -positions[i][0] * pressure_strength
                fy = -positions[i][1] * pressure_strength
                velocities[i][0] += fx * dt
                velocities[i][1] += fy * dt
                angular_velocities[i] += random.gauss(0, 0.5)
                angular_velocities[i] *= 0.9

            # Integrate
            for i in range(N):
                positions[i][0] += velocities[i][0] * dt
                positions[i][1] += velocities[i][1] * dt
                angles[i] += angular_velocities[i] * dt
                velocities[i][0] *= damping
                velocities[i][1] *= damping

            # Collision Resolution
            iterations = 5
            for _ in range(iterations):
                for i in range(N):
                    for j in range(i+1, N):
                        verts_i = get_hex_vertices(positions[i][0], positions[i][1], angles[i])
                        verts_j = get_hex_vertices(positions[j][0], positions[j][1], angles[j])
                        pen = penetration_depth(verts_i, verts_j)
                        if pen > 1e-6:
                            dx = positions[j][0] - positions[i][0]
                            dy = positions[j][1] - positions[i][1]
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist < 1e-6:
                                dx = random.uniform(-1, 1)
                                dy = random.uniform(-1, 1)
                                dist = math.sqrt(dx*dx + dy*dy)
                            corr = pen * stiffness
                            nx = dx / dist
                            ny = dy / dist
                            positions[i][0] -= nx * corr * 0.5
                            positions[i][1] -= ny * corr * 0.5
                            positions[j][0] += nx * corr * 0.5
                            positions[j][1] += ny * corr * 0.5

            # Check validity every 10 frames
            if frame % 10 == 0:
                score = compute_score(positions, angles)
                if score:
                    side, oa = score
                    if side < local_best_side:
                        local_best_side = side
                        local_best_state = [p[:] for p in positions]
                        local_best_angles = angles[:]
                        local_best_oa = oa
                        if side < best_side:
                            best_side = side
                            best_state = [p[:] for p in positions]
                            best_angles = angles[:]
                            best_oa = oa

    # Phase 2: Nelder-Mead Polish on the best state found
    if best_state is not None:
        if time.time() < deadline - 15.0:
            # We optimize the positions and angles of all hexagons simultaneously using NM
            # Parameters: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10]
            x0 = [best_state[i][0] for i in range(N)] + [best_state[i][1] for i in range(N)] + [best_angles[i] for i in range(N)]
            
            def objective_nm(params):
                if time.time() > deadline - 5.0:
                    return 1e6
                positions = [[params[i], params[N+i]] for i in range(N)]
                angles = params[2*N:3*N]
                
                # Check overlaps
                hexagons = [get_hex_vertices(positions[i][0], positions[i][1], angles[i]) for i in range(N)]
                pen_sum = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        pen = penetration_depth(hexagons[i], hexagons[j])
                        if pen > 0:
                            pen_sum += pen ** 2
                if pen_sum > 1e-4:
                    return 1000 + pen_sum
                
                # Minimize outer side
                best_s = float('inf')
                for oa_deg in range(0, 60, 5):
                    s = calc_outer_side(hexagons, oa_deg)
                    if s < best_s:
                        best_s = s
                return best_s

            # Run NM
            result = minimize(objective_nm, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if result.success or result.fun < best_side:
                final_params = result.x
                final_positions = [[final_params[i], final_params[N+i]] for i in range(N)]
                final_angles = final_params[2*N:3*N]
                
                # Final validation and side calculation
                score = compute_score(final_positions, final_angles)
                if score:
                    side, oa = score
                    if side < best_side:
                        best_side = side
                        best_state = [p[:] for p in final_positions]
                        best_angles = final_angles
                        best_oa = oa

    # Final Polish of Outer Angle
    if best_state is not None:
        hexagons = [get_hex_vertices(best_state[i][0], best_state[i][1], best_angles[i]) for i in range(N)]
        final_side = float('inf')
        final_oa = 0.0
        for oa_deg in np.arange(0, 60, 0.5):
            side = calc_outer_side(hexagons, oa_deg)
            if side < final_side:
                final_side = side
                final_oa = oa_deg
        best_side = final_side
        best_oa = final_oa

    if best_state is None:
        # Fallback
        positions, angles = init_positions(2.2)
        best_state = positions
        best_angles = angles
        hexagons = [get_hex_vertices(best_state[i][0], best_state[i][1], best_angles[i]) for i in range(N)]
        for oa_deg in range(0, 60, 1):
            side = calc_outer_side(hexagons, oa_deg)
            if side < best_side:
                best_side = side
                best_oa = oa_deg

    inner_list = []
    for i in range(N):
        inner_list.append([best_state[i][0], best_state[i][1], best_angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
