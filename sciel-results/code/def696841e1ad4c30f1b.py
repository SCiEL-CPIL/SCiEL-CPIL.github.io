import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top)
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
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint_fast(poly1, poly2):
        """
        Fast check for disjointness using SAT.
        Returns True if disjoint (or touching), False if overlapping.
        """
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
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return True
                
        return False

    def is_valid(positions, angles):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    return False
        return True

    def compute_overlap_penalty(positions, angles):
        """
        Compute a smooth penalty based on minimum distance between hexagons.
        Uses the SAT framework but returns a continuous measure.
        """
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        total_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Compute minimum separation using SAT
                min_sep = float('inf')
                for k in range(6):
                    edge1 = polys[i][(k+1)%6] - polys[i][k]
                    normal1 = np.array([-edge1[1], edge1[0]])
                    edge2 = polys[j][(k+1)%6] - polys[j][k]
                    normal2 = np.array([-edge2[1], edge2[0]])
                    for axis in [normal1, normal2]:
                        norm = np.linalg.norm(axis)
                        if norm < 1e-9:
                            continue
                        axis = axis / norm
                        proj1 = polys[i] @ axis
                        proj2 = polys[j] @ axis
                        sep = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
                        if sep < min_sep:
                            min_sep = sep
                # If min_sep < 0, they overlap. Penalize quadratically.
                if min_sep < 0:
                    total_pen += min_sep * min_sep
        return total_pen

    # Phase 1: Symmetry-constrained optimization
    # We exploit the D6 symmetry of the outer hexagon.
    # Possible configurations:
    # 1. One central hexagon + 5 orbits of 6 (30 total - too many)
    # 2. One central + 1 orbit of 6 + 1 orbit of 5? No, orbits must be 1 or 6 for D6.
    #    Actually, with N=12:
    #    Option A: 1 central + 1 orbit of 6 + ... wait 1+6=7. Need 5 more.
    #    Option B: 2 orbits of 6.
    #    Option C: 1 central + ... 11 left? No.
    #    Let's stick to standard orbits under D6:
    #    - Center (1)
    #    - Orbit of 6 (6)
    #    - Orbit of 12 (12) -> too many.
    
    # So likely:
    # Case 1: 1 central + 1 orbit of 6 + ...? No, 1+6=7. 12-7=5. Can't make orbit of 5.
    # Case 2: 2 orbits of 6.
    # Case 3: 1 central + ...? Maybe the central is part of a larger structure?
    # Actually, for N=12, a common packing is 1 central + 6 surrounding + 6 more?
    # If we have 1 central, the 6 neighbors form an orbit of 6. The remaining 6 must form another orbit of 6.
    # So: 1 Central + Orbit A (6) + Orbit B (6) = 13? No. 1+6+6 = 13.
    # Wait, N=12.
    # If we have 1 central, we have 11 left. Not divisible by 6.
    # So either NO central hexagon, or the "central" one is not fixed at center but part of an orbit?
    # No, D6 symmetry forces the center to be a fixed point. If a hexagon is at the center, it's unique.
    # If N=12, we cannot have a single central hexagon and D6 symmetry unless the other 11 break symmetry or form orbits.
    # Orbits of size 6. 12 is divisible by 6.
    # So Case 1: 2 orbits of 6 hexagons each.
    # Case 2: 1 central + ...? If 1 central, remaining 11 must be packed. 11 is not multiple of 6.
    #    Unless the 1 central is actually 1 hexagon, and the other 11 are arranged such that some are in orbits of 6 and some in orbits of ...?
    #    No, D6 symmetry acts on the set of 12.
    #    Stabilizer of a point not at center is of order 1 or 2 (on axis).
    #    If a point is on an axis (60 deg), its orbit size is 6/2 = 3?
    #    Let's check: A point on the x-axis. Reflection across x-axis fixes it? No, rotation by 60 moves it.
    #    The stabilizer of a point on an axis (but not center) under D6 is a reflection group of order 2?
    #    Yes, the reflection across that axis fixes the point.
    #    So orbit size = 6/2 = 3.
    #    So we can have orbits of size 1 (center), 3 (on axis), or 6 (general).
    
    # Possible combinations for N=12:
    # A) 6 + 6
    # B) 6 + 3 + 3
    # C) 6 + 3 + 3 (same)
    # D) 3 + 3 + 3 + 3
    # E) 1 + ... 11 left. 11 = 6+3+2? No orbit of 2. 11 = 3+3+3+2? No.
    #    So if there is a central hexagon, the remaining 11 must be composed of orbits of 3 and 6.
    #    11 = 6 + 3 + 2 (no)
    #    11 = 3 + 3 + 3 + 2 (no)
    #    So it seems impossible to have a central hexagon with full D6 symmetry for N=12?
    #    Wait, is there an orbit of size 1 other than center? No.
    #    Is there an orbit of size 2? No, D6 has no subgroups of index 2 that stabilize a non-center point?
    #    Actually, reflection fixes a line. Points on the axis of reflection are fixed by that reflection.
    #    The stabilizer is {e, r}. Order 2. Orbit size 6/2=3.
    #    So orbits are size 1, 3, 6.
    #    12 = 6+6
    #    12 = 6+3+3
    #    12 = 3+3+3+3
    #    12 = 1 + ... 11 cannot be formed by 3s and 6s. (11 mod 3 = 2).
    #    So NO central hexagon in a D6 symmetric packing of 12 unit hexagons.

    # We will optimize over the parameter space of these symmetric configurations.
    # We'll use a generic optimization over the free parameters of the orbits.
    
    # Let's try to define the objective for a specific symmetric configuration.
    # We will try all valid combinations of orbit sizes.
    
    # Configuration 1: Two orbits of 6.
    # Orbit 1: Center (r1, theta1). The 6 points are at angles theta1 + k*60.
    # Orbit 2: Center (r2, theta2). The 6 points are at angles theta2 + k*60.
    # Parameters: r1, theta1, r2, theta2, angle1 (rotation of hex1), angle2 (rotation of hex2).
    # Note: Due to symmetry, the rotation angle of the hexagon at (r, theta) is determined by theta?
    # If we rotate the whole system by 60 deg, the hexagon at (r, theta) moves to (r, theta+60).
    # For the configuration to be invariant, the hexagon at the new position must have the same orientation relative to the radial vector?
    # Yes, typically the hexagon orientation is locked to the radial direction or a fixed offset.
    # Let's assume the angle of hex i is angle0 + theta_i.
    
    def build_positions_orbits(orbit_params):
        """
        orbit_params: list of tuples (r, theta_base, angle_offset)
        Returns positions (12, 2) and angles (12,)
        """
        positions = []
        angles = []
        for r, theta_base, angle_offset in orbit_params:
            for k in range(6):
                theta = theta_base + k * math.pi / 3
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                positions.append([x, y])
                ang = angle_offset + theta # Lock rotation to radial direction + offset
                angles.append(ang)
        return np.array(positions), np.array(angles)

    def objective_sym(params, orbit_sizes):
        """
        params: flattened parameters for all orbits.
        For each orbit, we have 3 params: r, theta_base, angle_offset.
        """
        positions = []
        angles = []
        idx = 0
        for size in orbit_sizes:
            if size == 6:
                r = params[idx]
                theta_base = params[idx+1]
                angle_offset = params[idx+2]
                idx += 3
                for k in range(6):
                    theta = theta_base + k * math.pi / 3
                    x = r * math.cos(theta)
                    y = r * math.sin(theta)
                    positions.append([x, y])
                    angles.append(angle_offset + theta)
            elif size == 3:
                # Orbit of 3: Points on axes?
                # If on x-axis, angles are 0, 120, 240?
                # Or 60, 180, 300?
                # Let's assume theta_base is the angle of the first point.
                # The other two are theta_base + 120, theta_base + 240.
                r = params[idx]
                theta_base = params[idx+1]
                angle_offset = params[idx+2]
                idx += 3
                for k in range(3):
                    theta = theta_base + k * 2 * math.pi / 3
                    x = r * math.cos(theta)
                    y = r * math.sin(theta)
                    positions.append([x, y])
                    angles.append(angle_offset + theta)
            else:
                raise ValueError("Invalid orbit size")
        
        positions = np.array(positions)
        angles = np.array(angles)
        
        # Penalty for overlap
        if not is_valid(positions, angles):
            return 1000.0 # Large penalty
        
        return calc_min_outer_side(positions, angles, 0.0)

    best_score = float('inf')
    best_result = None

    # Try different symmetric configurations
    configs = [
        [6, 6],
        [6, 3, 3],
        [3, 3, 3, 3]
    ]

    for sizes in configs:
        num_params = 3 * len(sizes)
        
        # Initial guess
        # Spread them out
        if sizes == [6, 6]:
            # Two rings
            x0 = [2.0, 0.0, 0.0, 5.0, 30.0, 0.0] # r1, th1, a1, r2, th2, a2
        elif sizes == [6, 3, 3]:
            x0 = [2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 60.0, 0.0]
        else: # [3,3,3,3]
            x0 = [2.0, 0.0, 0.0, 2.0, 60.0, 0.0, 2.0, 120.0, 0.0, 2.0, 180.0, 0.0]

        # Bounds
        # r > 0, theta any, angle any
        bounds = []
        for _ in range(len(sizes)):
            bounds.append((0.1, 10.0)) # r
            bounds.append((0.0, 360.0)) # theta
            bounds.append((0.0, 360.0)) # angle

        # Convert angles to radians for optimization?
        # SLSQP handles radians better usually, or we can just work in radians.
        # Let's work in radians.
        
        # Re-initialize x0 in radians
        x0_rad = []
        for i, val in enumerate(x0):
            if i % 3 == 1 or i % 3 == 2: # theta or angle
                x0_rad.append(math.radians(val))
            else:
                x0_rad.append(val)
        
        # Bounds in radians
        bounds_rad = []
        for _ in range(len(sizes)):
            bounds_rad.append((0.1, 10.0))
            bounds_rad.append((0.0, 2*math.pi))
            bounds_rad.append((0.0, 2*math.pi))

        def obj_func(p):
            if time.time() > deadline - 5:
                return 1000.0
            return objective_sym(p, sizes)

        try:
            res = minimize(
                obj_func, 
                x0_rad, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res.fun < best_score and res.fun < 100.0:
                best_score = res.fun
                best_result = (res.x, sizes)
        except Exception:
            pass

    # If no symmetric solution found or score is bad, fall back to general optimization
    if best_result is None or best_score > 5.0:
        # General optimization with penalty
        # Initial guess: Honeycomb subset
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
        
        def obj_general(params):
            if time.time() > deadline - 5:
                return 1000.0
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            
            if not is_valid(positions, angles):
                # Penalty proportional to overlap depth?
                # Simple penalty: count of overlapping pairs * 10
                polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
                pen = 0
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint_fast(polys[i], polys[j]):
                            pen += 10.0
                return calc_min_outer_side(positions, angles, 0.0) + pen
            
            return calc_min_outer_side(positions, angles, 0.0)

        try:
            res = minimize(
                obj_general, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if res.fun < best_score and res.fun < 100.0:
                best_score = res.fun
                best_result = (res.x, 'general')
        except Exception:
            pass

    # Finalize result
    if best_result is not None:
        params, type_str = best_result
        
        if type_str == 'general':
            final_positions = params[:2*N].reshape(N, 2)
            final_angles = params[2*N:]
        else:
            final_positions, final_angles = build_positions_orbits([(params[i*3], params[i*3+1], params[i*3+2]) for i in range(len(type_str))])
        
        # Verify validity one last time
        if not is_valid(final_positions, final_angles):
            # Try to fix with local optimization
            x0 = np.concatenate([final_positions.flatten(), final_angles])
            
            def obj_fix(params):
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                pen = compute_overlap_penalty(positions, angles)
                if pen > 0.01:
                    return calc_min_outer_side(positions, angles, 0.0) + 100.0 * pen
                return calc_min_outer_side(positions, angles, 0.0)
            
            try:
                res = minimize(
                    obj_fix, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9}
                )
                if res.fun < best_score:
                    final_positions = res.x[:2*N].reshape(N, 2)
                    final_angles = res.x[2*N:]
                    best_score = res.fun
            except Exception:
                pass
            
            # If still invalid, use fallback
            if not is_valid(final_positions, final_angles):
                final_positions = np.array([
                    [0, 0], [2.1, 0], [-2.1, 0],
                    [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
                    [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
                    [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
                ])
                final_angles = np.zeros(N)
            
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    else:
        # Emergency fallback
        final_positions = np.array([
            [0, 0], [2.1, 0], [-2.1, 0],
            [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
            [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
            [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
        ])
        final_angles = np.zeros(N)
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([float(final_positions[i][0]), float(final_positions[i][1]), float(final_angles[i])])
    
    return inner, (0.0, 0.0), float(final_side), 0.0
