"""
Strategy: Deterministic Multi-Start Global Optimization using Differential Evolution.
Replaces Simulated Annealing with scipy's DE for robustness and speed.
Uses a smooth Log-Sum-Exp approximation for the maximum projection (container size)
and a smooth penalty for overlaps (based on center distance for speed).
"""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit

    SQRT3 = math.sqrt(3)
    
    # Pre-compute hexagon unit vectors and normals for speed
    # Vertices are at angles 0, 60, 120, 180, 240, 300
    # Normals (outward) are at 30, 90, 150, 210, 270, 330
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
    
    unit_verts = np.stack([np.cos(base_angles), np.sin(base_angles)], axis=1) # (6, 2)
    unit_normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1) # (6, 2)

    def get_hex_verts_batch(centers, angles):
        """
        centers: (N, 2)
        angles: (N,) in radians
        returns: (N, 6, 2)
        """
        N = len(centers)
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            c = centers[i]
            a = angles[i]
            # Rotation matrix
            cos_a = math.cos(a)
            sin_a = math.sin(a)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            # Rotate unit vertices and translate
            verts[i] = unit_verts @ R.T + c
        return verts

    def get_hex_axes_batch(angles):
        """
        angles: (N,) in radians
        returns: (N, 6, 2) - outward normals
        """
        N = len(angles)
        axes = np.zeros((N, 6, 2))
        for i in range(N):
            a = angles[i]
            cos_a = math.cos(a)
            sin_a = math.sin(a)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            axes[i] = unit_normals @ R.T
        return axes

    def calc_outer_side_smooth(hex_verts, outer_angle_deg, beta=20.0):
        """
        hex_verts: (N, 6, 2)
        outer_angle_deg: float
        returns: smooth approximation of outer side length
        """
        oa_rad = math.radians(outer_angle_deg)
        # Outer hexagon normals
        # The "radius" of the outer hexagon in direction n is R.
        # Side length S is related to R by R = S * SQRT3 / 2.
        # We want min S such that all points are inside.
        # Point p is inside if dot(p, n) <= R for all 6 normals n.
        # So R >= max_p dot(p, n).
        # S = 2 * R / SQRT3.
        
        # Normals for outer hexagon
        oa_normals = unit_normals @ np.array([[math.cos(oa_rad), -math.sin(oa_rad)], 
                                              [math.sin(oa_rad), math.cos(oa_rad)]]) # (6, 2)
        
        # Project all vertices onto all outer normals
        # hex_verts: (N, 6, 2) -> (N, 6, 1, 2)
        # oa_normals: (6, 2) -> (1, 1, 6, 2)
        # Projections: (N, 6, 6)
        projs = np.einsum('nij,kj->nik', hex_verts, oa_normals)
        
        # We need max over all vertices and all 6 directions?
        # Actually, the condition is: for each normal direction k, max over all vertices i,j of dot(v_ij, n_k) <= R.
        # So R = max_k ( max_{i,j} dot(v_ij, n_k) ).
        
        # max over vertices for each normal
        max_projs_per_normal = np.max(projs, axis=(0, 1)) # (6,)
        
        # Smooth max over the 6 normals? No, the max over normals is discrete and small (6 items).
        # But the max over vertices is the expensive part.
        # Let's smooth the max over vertices first.
        # log-sum-exp for max over vertices for each normal
        # projs shape (N, 6, 6)
        # max along axis 0 (N) and 1 (6 vertices)
        
        # To be safe and fast, let's just take the hard max over vertices, but smooth over normals?
        # Or smooth over vertices?
        # The bottleneck is the discontinuity of max.
        # Let's use log-sum-exp over all (N*6) projections for each normal direction.
        
        # Reshape projs to (6, N*6)
        projs_flat = projs.transpose(2, 0, 1).reshape(6, -1) # (6, N*6)
        
        # Log-sum-exp for each row
        max_vals = np.max(projs_flat, axis=1)
        # Handle case where max is -inf or 0
        # exp((x - max) * beta)
        # If beta is large, this approximates max.
        
        # To avoid overflow, we do:
        # sum(exp((x - max) * beta)) / beta + max
        # If beta -> inf, it becomes max.
        
        # Note: we need abs? No, the hexagon is centered at origin?
        # The centers are shifted to mean 0 in the evaluation, but the container is centered at 0.
        # The vertices can be negative.
        # The condition is dot(p, n) <= R.
        # Since the hexagon is symmetric, if we center the inner cluster at 0,
        # we need R >= max( max(dot(p, n)), max(dot(p, -n)) ).
        # But our normals cover all 6 directions.
        # So R = max_k max_{p} dot(p, n_k).
        
        # Wait, if the cluster is not centered at 0, we shift it.
        # The objective function shifts centers to mean 0.
        
        # So we compute R_smooth = max_k (LSE(projs_k))
        
        # LSE
        # To prevent overflow in exp:
        # We can use np.logaddexp.reduce, but it's slow.
        # Manual LSE:
        # m = max(x)
        # lse = m + log(sum(exp(x-m)))
        
        # Let's use a moderate beta for smoothness, or just hard max if it's fast enough.
        # N=11, 6 verts, 6 normals. 11*6*6 = 396 projections.
        # Hard max is very fast. The non-differentiability might be an issue for DE?
        # DE is gradient-free, so hard max is fine!
        # DE doesn't need gradients. So we can use the exact hard max for outer side.
        
        # Exact R
        R = np.max(max_projs_per_normal)
        
        # Ensure R is non-negative (should be, as max of projections of centered points)
        if R < 0:
            R = 0.0
            
        return 2.0 * R / SQRT3

    def evaluate_config(params, outer_angle_deg=0.0):
        """
        params: (N*3,) [x1, y1, a1, x2, y2, a2, ...]
        returns: scalar score (outer side length + overlap penalty)
        """
        # Extract centers and angles
        centers = params[0::3].reshape(-1, 2)
        angles_deg = params[1::3]
        angles_rad = np.deg2rad(angles_deg)
        
        # Center the configuration
        centers = centers - np.mean(centers, axis=0)
        
        # Get vertices and axes
        hex_verts = get_hex_verts_batch(centers, angles_rad)
        hex_axes = get_hex_axes_batch(angles_rad)
        
        # Calculate outer side length
        side = calc_outer_side_smooth(hex_verts, outer_angle_deg)
        
        # Calculate overlap penalty
        # Use Separating Axis Theorem (SAT) for exact disjoint check?
        # SAT is fast for 11 hexagons (55 pairs).
        # But for optimization, a smooth penalty is better.
        # Let's use a smooth penalty based on center distance.
        # Min distance between centers for non-overlapping hexagons is SQRT3 (if aligned).
        # If rotated, it can be smaller? No, inradius is SQRT3/2.
        # Distance between centers must be >= 2 * inradius = SQRT3 for aligned.
        # For arbitrary rotation, the minimum distance to avoid overlap is roughly SQRT3.
        # Actually, the "width" of a unit hexagon is SQRT3.
        # So if centers are closer than SQRT3, they might overlap.
        # A simple penalty: sum over pairs max(0, SQRT3 - dist)^2
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Threshold: SQRT3 is the distance between parallel sides.
                # If dist < SQRT3, they definitely overlap?
                # If aligned, yes. If rotated, they might touch at vertices with dist < SQRT3?
                # Inradius r = SQRT3/2.
                # If two hexagons are tangent, distance between centers is at least 2r = SQRT3.
                # So dist < SQRT3 implies overlap.
                # dist > SQRT3 implies no overlap?
                # Not necessarily, if they are rotated, they can interlock?
                # No, regular convex polygons. If distance between centers > sum of circumradii, they are disjoint.
                # Circumradius = 1.
                # So dist > 2 implies disjoint.
                # dist < SQRT3 implies overlap.
                # Between SQRT3 and 2, it depends on rotation.
                
                # For a robust smooth penalty, let's use a threshold of 1.8 (approx).
                # Or better, use SAT for validity check in the final step,
                # and a soft penalty for the optimizer.
                
                # Soft penalty: 
                # If dist < SQRT3, penalty is high.
                # If dist is close to SQRT3, penalty is lower.
                
                # Let's use a simple quadratic penalty for dist < SQRT3
                if dist < SQRT3:
                    penalty += (SQRT3 - dist) ** 2 * 10.0
                # Also penalize if dist is small (close to 0)
                # The above covers it.
                
        return side + penalty

    # Define bounds for DE
    # Centers: roughly within [-3, 3]
    # Angles: [0, 360]
    bounds = [(-3, 3)] * N + [(-3, 3)] * N + [(0, 360)] * N
    
    # Initial population
    # Use a honeycomb-like initial guess
    # Axial coords to cartesian
    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Generate a few initial configs
    initial_configs = []
    
    # Config 1: Honeycomb
    coords1 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    p1 = []
    for q, r in coords1:
        x, y = axial_to_cartesian(q, r)
        p1.extend([x, y, 0.0])
    initial_configs.append(np.array(p1))
    
    # Config 2: Random
    random.seed(42)
    p2 = []
    for i in range(N):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        a = random.uniform(0, 360)
        p2.extend([x, y, a])
    initial_configs.append(np.array(p2))
    
    # Run Differential Evolution
    best_params = None
    best_score = float('inf')
    
    # DE settings
    # popsize=15, maxiter=100, tol=1e-5
    # time budget: 120s
    
    # We will run DE multiple times with different initial populations or seeds if time permits.
    # But DE is stochastic.
    
    # Use a wrapper to check time
    class TimeExceededException(Exception):
        pass

    def objective_with_timeout(p):
        if time.time() > deadline - 5.0:
            raise TimeExceededException()
        return evaluate_config(p)

    # Since DE doesn't support timeout directly, we run it and catch exceptions.
    # However, DE is a loop. We can't easily interrupt it mid-loop in scipy.
    # We can reduce maxiter and check time between restarts.
    
    # Let's run DE with a conservative maxiter.
    # Estimate: 100 iterations * 150 (popsize) * eval_cost.
    # eval_cost is ~50us? 100*150*50us = 0.75s. Very fast.
    # So we can afford many iterations.
    
    # Let's try maxiter=200, popsize=30.
    
    for restart in range(5): # 5 restarts
        if time.time() > deadline - 20.0:
            break
            
        # Randomize initial population for each restart
        # DE generates its own population, but we can seed it.
        # scipy's DE uses `seed` or `popsize`.
        # To get different starts, we can vary the seed.
        
        try:
            res = differential_evolution(
                objective_with_timeout,
                bounds,
                maxiter=200,
                popsize=20,
                tol=1e-5,
                seed=restart,
                init='sobol' # Sobol sequence for better coverage
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                
        except TimeExceededException:
            break
        except Exception as e:
            # print(e)
            continue

    # Refine with Nelder-Mead if time permits
    if best_params is not None and time.time() < deadline - 10.0:
        def objective_nm(p):
            return evaluate_config(p)
            
        try:
            from scipy.optimize import minimize
            res_nm = minimize(objective_nm, best_params, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    # Final verification and extraction
    if best_params is None:
        # Fallback to initial config 1
        best_params = initial_configs[0]
        
    # Extract centers and angles
    centers = best_params[0::3].reshape(-1, 2)
    angles_deg = best_params[1::3]
    
    # Center the configuration
    centers = centers - np.mean(centers, axis=0)
    
    # Check validity with SAT
    def is_disjoint_sat(hex1_verts, hex2_verts, axes1, axes2):
        all_axes = list(axes1) + list(axes2)
        for axis in all_axes:
            nx, ny = axis
            min1 = min(vx * nx + vy * ny for vx, vy in hex1_verts)
            max1 = max(vx * nx + vy * ny for vx, vy in hex1_verts)
            min2 = min(vx * nx + vy * ny for vx, vy in hex2_verts)
            max2 = max(vx * nx + vy * ny for vx, vy in hex2_verts)
            if max1 < min2 or max2 < min1:
                return True
        return False

    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    valid = True
    hexagons = []
    hex_axes = []
    for i in range(N):
        cx, cy = centers[i]
        a = angles_deg[i]
        h_verts = get_hex_vertices(cx, cy, a)
        h_axes = get_hex_axes(a)
        hexagons.append(h_verts)
        hex_axes.append(h_axes)
        
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint_sat(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                valid = False
                break
        if not valid:
            break

    if not valid:
        # If invalid, try to fix by pushing apart?
        # Or just return the best valid config from the search history?
        # We didn't store history.
        # Let's try the initial config which is valid.
        # But the initial config might be worse.
        # Let's try to find a valid config among the DE population? No access.
        # Fallback: Return the initial honeycomb config.
        # It is valid and has a known score.
        p_fb = initial_configs[0]
        centers_fb = p_fb[0::3].reshape(-1, 2)
        angles_fb = p_fb[1::3]
        centers_fb = centers_fb - np.mean(centers_fb, axis=0)
        
        inner_list = []
        for i in range(N):
            inner_list.append([centers_fb[i, 0], centers_fb[i, 1], angles_fb[i]])
            
        # Calculate side for fallback
        side_fb = evaluate_config(p_fb) # This includes penalty, but penalty is 0 for valid
        # Recalculate side accurately
        hex_verts_fb = get_hex_verts_batch(centers_fb, np.deg2rad(angles_fb))
        side_fb = calc_outer_side_smooth(hex_verts_fb, 0.0)
        
        return (inner_list, (0.0, 0.0), side_fb, 0.0)

    # Calculate final side length
    # We should optimize the outer angle too?
    # The current code tries to find best outer angle.
    # With DE, we can include outer angle in the optimization?
    # Or just sweep it.
    
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 600, 5): # Sweep 0-60 deg with 5 deg step
        oa = float(oa_deg)
        # Recalculate side with this outer angle
        hex_verts = get_hex_verts_batch(centers, np.deg2rad(angles_deg))
        side = calc_outer_side_smooth(hex_verts, oa)
        if side < best_side:
            best_side = side
            best_oa = oa

    inner_list = []
    for i in range(N):
        inner_list.append([centers[i, 0], centers[i, 1], angles_deg[i]])

    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
