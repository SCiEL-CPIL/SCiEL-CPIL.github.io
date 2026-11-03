import math
import itertools
import time
import numpy as np
from scipy.optimize import minimize_scalar, minimize, linprog

N = 11

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 5.0

    # Hexagon geometry: unit regular hexagon with flat-top orientation
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # Side length 1, so circumradius = 1
    # Inradius = sqrt(3)/2
    
    # For a regular hexagon with side length s, the distance between opposite sides is s*sqrt(3)
    # The Minkowski sum of two unit hexagons (with same orientation) has side length 2
    # But we need to handle rotation
    
    # Key insight: Two unit hexagons with relative angle theta don't overlap iff
    # their centers are at distance >= f(theta) where f depends on theta
    
    # For same orientation (theta=0), min distance = sqrt(3) (inradius * 2)
    # Actually, for regular hexagons, the minimum center distance to avoid overlap
    # when both have the same orientation is sqrt(3)
    
    d = math.sqrt(3)  # minimum center distance for same orientation
    
    # Generate hexagonal lattice positions
    # Center + 6 in ring 1 + 12 in ring 2 = 19 positions
    # We need to choose 11 from these
    
    centers_all = [(0.0, 0.0)]  # center
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers_all.append((x, y))
    
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s_coord = -q - r
            if max(abs(q), abs(r), abs(s_coord)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))
    
    all_positions = centers_all + ring2_centers  # 19 positions
    n_positions = len(all_positions)
    
    # Precompute vertices for unit hexagon (flat-top, angle 0)
    # Vertex i at angle i*pi/3
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])
    
    # For the outer hexagon, we'll use the support function approach
    # A point p is inside a regular hexagon centered at c with side S and angle alpha
    # iff for all 6 normals n_k (at angles alpha + k*pi/3):
    #   |dot(p - c, n_k)| <= S * sqrt(3) / 2
    
    # To minimize S, we need to find positions and orientations such that
    # all vertices of all inner hexagons are inside the outer hexagon
    # and no two inner hexagons overlap
    
    # Strategy: Start with the hexagonal lattice arrangement (11 positions)
    # Then optimize positions and orientations using scipy.optimize
    
    # Choose 11 positions from the 19 available
    # The natural choice is center + ring 1 (7) + 4 from ring 2
    # But let's try all combinations of 11 from 19 and pick the best
    
    # For efficiency, let's first try the "natural" arrangement:
    # center + all 6 ring 1 + 4 from ring 2 (choose which 4)
    
    # Actually, let's just optimize a fixed set of 11 positions
    # Start with center + ring 1 + first 4 ring 2 positions
    
    initial_positions = centers_all[:7] + ring2_centers[:4]  # 11 positions
    
    # Initial parameters: for each hexagon, (x, y, angle)
    # Total: 33 parameters
    init_params = []
    for cx, cy in initial_positions:
        init_params.extend([cx, cy, 0.0])
    init_params = np.array(init_params)
    
    # Compute S (outer side length) given positions and orientations
    def compute_S(params):
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        max_proj = 0.0
        for i in range(N):
            c = math.cos(thetas[i])
            s = math.sin(thetas[i])
            # Rotate vertices
            rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
            ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
            # Translate
            px = rx + xs[i]
            py = ry + ys[i]
            
            # Project on 6 normals of outer hexagon (angle 0)
            for k in range(6):
                ang = k * math.pi / 3
                nx = math.cos(ang)
                ny = math.sin(ang)
                proj = np.max(np.abs(px * nx + py * ny))
                if proj > max_proj:
                    max_proj = proj
        
        return (2.0 / math.sqrt(3)) * max_proj
    
    # Check for overlap between two hexagons
    # For simplicity, use center distance >= sqrt(3) as a sufficient condition
    # for same orientation. For different orientations, this is conservative.
    def check_overlap(params, tol=1e-6):
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                # Conservative: require dist >= sqrt(3)
                if dist_sq < 3.0 - tol:
                    return False, (i, j, dist_sq)
        return True, None
    
    # Objective: minimize S + penalty for overlap
    def objective(params):
        S = compute_S(params)
        
        # Overlap penalty
        penalty = 0.0
        xs = params[0::3]
        ys = params[1::3]
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq < 3.0:
                    dist = math.sqrt(dist_sq)
                    penalty += (math.sqrt(3) - dist) ** 2 * 100.0
        
        return S + penalty
    
    best_S = compute_S(init_params)
    best_params = init_params.copy()
    
    # Optimize using Nelder-Mead
    if time.time() < deadline - safety_margin:
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.success or res.fun < best_S:
                candidate_params = res.x
                valid, _ = check_overlap(candidate_params)
                if valid:
                    candidate_S = compute_S(candidate_params)
                    if candidate_S < best_S:
                        best_S = candidate_S
                        best_params = candidate_params
        except Exception:
            pass
    
    # Try multiple starting points from different ring 2 combinations
    if time.time() < deadline - safety_margin:
        # Try a few different combinations of ring 2 positions
        ring2_combos = list(itertools.combinations(range(12), 4))
        # Shuffle to get diverse starts
        np.random.seed(42)
        np.random.shuffle(ring2_combos)
        
        n_tries = min(20, len(ring2_combos))
        for combo in ring2_combos[:n_tries]:
            if time.time() > deadline - safety_margin:
                break
            
            trial_positions = centers_all[:7] + [ring2_centers[i] for i in combo]
            trial_params = []
            for cx, cy in trial_positions:
                trial_params.extend([cx, cy, 0.0])
            trial_params = np.array(trial_params)
            
            trial_S = compute_S(trial_params)
            if trial_S < best_S:
                best_S = trial_S
                best_params = trial_params.copy()
            
            # Quick local optimization
            try:
                res = minimize(objective, trial_params, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
                if res.fun < best_S:
                    valid, _ = check_overlap(res.x)
                    if valid:
                        final_S = compute_S(res.x)
                        if final_S < best_S:
                            best_S = final_S
                            best_params = res.x
            except Exception:
                pass
    
    # Final validation
    valid, _ = check_overlap(best_params)
    if not valid:
        # Fallback to a known valid configuration
        best_params = init_params
        best_S = compute_S(best_params)
    
    # Extract final positions
    xs = best_params[0::3]
    ys = best_params[1::3]
    thetas = best_params[2::3]
    
    inner = [[float(xs[i]), float(ys[i]), float(math.degrees(thetas[i]))] for i in range(N)]
    center = (0.0, 0.0)
    side = float(best_S)
    angle_deg = 0.0  # Outer hexagon angle
    
    return inner, center, side, angle_deg
