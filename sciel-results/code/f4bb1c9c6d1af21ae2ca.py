import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
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
        The side length s is related to the max projection R on normals by R = s * sqrt(3)/2.
        So s = 2 * R / sqrt(3).
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

    def smooth_sat_penalty(p1, p2, beta=100.0):
        """
        Differentiable approximation of the minimum separation distance between two convex polygons.
        Uses the Separating Axis Theorem (SAT) on the 12 edge normals of both hexagons.
        Returns a smooth penalty that is ~0 when separated and large when overlapping.
        """
        axes = []
        for i in range(6):
            edge1 = p1[(i+1)%6] - p1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            edge2 = p2[(i+1)%6] - p2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
            
        gaps = []
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            proj1 = p1 @ axis
            proj2 = p2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            # Gap is the distance between the intervals
            # s > 0 if disjoint, s < 0 if overlapping
            # s = min( min2 - max1, min1 - max2 )
            s = min(min2 - max1, min1 - max2)
            gaps.append(s)
            
        if not gaps:
            return 0.0
            
        gaps = np.array(gaps)
        # We want to penalize when min(gaps) < 0.
        # Let d = min(gaps).
        # Penalty should be smooth max(0, -d).
        # Smooth max(0, x) can be approximated by log(1 + exp(beta*x))/beta.
        # Here x = -d = -min(gaps) = max(-gaps).
        # So we compute a smooth max of -gaps, then apply softplus.
        
        # Actually, it's easier to just compute the smooth max of the "overlap depths".
        # Overlap depth on axis i is max(0, -gap_i).
        # Total overlap is min_gap. If min_gap < 0, they overlap.
        # We want a penalty that grows when min_gap becomes negative.
        
        # Let's use the log-sum-exp trick on the negative gaps to find a smooth "minimum gap".
        # smooth_min(gaps) = -1/beta * log(sum(exp(-beta * gap)))
        # If all gaps are positive (separated), smooth_min is close to min(gaps) > 0.
        # If any gap is negative (overlapping), smooth_min is close to min(gaps) < 0.
        
        # We want penalty = smooth_max(0, -smooth_min(gaps)).
        # Let m = smooth_min(gaps).
        # Penalty = log(1 + exp(beta * (-m))) / beta.
        
        # To avoid overflow in exp, we can use the standard log-sum-exp trick.
        # Let v = -gaps. We want smooth_max(v).
        # smooth_max(v) = 1/beta * log(sum(exp(beta * v)))
        # Then penalty = smooth_max(0, -m) where m = -smooth_max(v) => penalty = smooth_max(0, smooth_max(v)).
        # This is getting complicated.
        
        # Simpler: Just use the raw min gap but smooth it slightly? No, raw min is non-diff.
        # Let's stick to: Penalty = log(1 + exp(beta * (-min_gap_raw))) / beta.
        # But min_gap_raw is non-diff.
        
        # Alternative: Compute the sum of squared overlaps on all axes?
        # This penalizes any overlap on any axis, which is a valid constraint satisfaction strategy.
        # Sum_{axes} max(0, -gap_axis)^2.
        # This is differentiable almost everywhere (kinks at 0, but gradient is 0 at 0 from positive side, linear from negative).
        # L-BFGS handles this well.
        
        total_pen = 0.0
        for g in gaps:
            if g < 0:
                total_pen += g * g
        return total_pen

    def objective_smooth(params):
        if time.time() > deadline - 5:
            return 1000.0
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Calculate outer side length
        side = calc_min_outer_side(positions, angles, 0.0)
        
        # Calculate overlap penalty
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                pen += smooth_sat_penalty(polys[i], polys[j])
        
        # We want to minimize side length, but ensure no overlap.
        # If there is overlap, the penalty should dominate.
        # Scale penalty to be significant relative to side length (~4-5).
        return side + 10.0 * pen

    def is_valid(positions, angles, tol=1e-4):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                # Check strict non-overlap
                # Use the same SAT logic but without smoothing
                axes = []
                for k in range(6):
                    edge1 = polys[i][(k+1)%6] - polys[i][k]
                    normal1 = np.array([-edge1[1], edge1[0]])
                    axes.append(normal1)
                    edge2 = polys[j][(k+1)%6] - polys[j][k]
                    normal2 = np.array([-edge2[1], edge2[0]])
                    axes.append(normal2)
                
                min_gap = float('inf')
                for axis in axes:
                    norm = np.linalg.norm(axis)
                    if norm < 1e-9: continue
                    axis = axis / norm
                    proj1 = polys[i] @ axis
                    proj2 = polys[j] @ axis
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    s = min(min2 - max1, min1 - max2)
                    if s < min_gap:
                        min_gap = s
                    if min_gap > tol:
                        break # Found a separating axis, they are disjoint
                
                if min_gap <= tol:
                    return False
        return True

    best_score = float('inf')
    best_params = None

    def generate_initial_guesses():
        guesses = []
        sqrt3 = math.sqrt(3)
        
        # 1. Honeycomb subset (compact)
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
        guesses.append(np.concatenate([init_positions.flatten(), init_angles]))
        
        # 2. 2x6 Grid (rectangular block)
        # Hexagons in a grid need to be rotated to fit? 
        # Standard hex grid has spacing 1.5 in x and sqrt(3) in y for adjacent rows.
        # Let's create a 2x6 block.
        pos = []
        for row in range(2):
            for col in range(6):
                x = col * 1.5
                y = row * sqrt3
                if row % 2 == 1:
                    x += 0.75
                pos.append([x, y])
        pos = np.array(pos)
        pos -= np.mean(pos, axis=0)
        ang = np.zeros(N)
        guesses.append(np.concatenate([pos.flatten(), ang]))
        
        # 3. Linear Chain
        pos = []
        for i in range(12):
            pos.append([i * 1.5, 0])
        pos = np.array(pos)
        pos -= np.mean(pos, axis=0)
        ang = np.zeros(N)
        guesses.append(np.concatenate([pos.flatten(), ang]))
        
        # 4. Random Scatter
        for _ in range(5):
            pos = np.random.uniform(-3, 3, (N, 2))
            ang = np.random.uniform(0, 360, N)
            guesses.append(np.concatenate([pos.flatten(), ang]))
            
        return guesses

    guesses = generate_initial_guesses()
    
    # Optimize each guess
    for i, x0 in enumerate(guesses):
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.fun < best_score and res.fun < 100.0:
                # Check validity of the result
                final_positions = res.x[:2*N].reshape(N, 2)
                final_angles = res.x[2*N:]
                
                # Re-calculate true score (without penalty) if valid
                if is_valid(final_positions, final_angles):
                    true_score = calc_min_outer_side(final_positions, final_angles, 0.0)
                    if true_score < best_score:
                        best_score = true_score
                        best_params = res.x
        except Exception:
            pass

    # If no good result found, use a safe fallback
    if best_params is None:
        # Fallback to a known valid configuration (even if not optimal)
        # A large hexagon containing 12 small ones
        final_positions = np.array([
            [0, 0], [2.1, 0], [-2.1, 0],
            [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
            [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
            [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
        ])
        final_angles = np.zeros(N)
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    else:
        final_positions = best_params[:2*N].reshape(N, 2)
        final_angles = best_params[2*N:]
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
