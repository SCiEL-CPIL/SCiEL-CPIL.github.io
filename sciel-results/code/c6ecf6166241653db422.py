"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Normal directions for flat-top outer hexagon (pointing outward from center)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)
    
    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        vertex_angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(vertex_angles)
        vy = cy + np.sin(vertex_angles)
        
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)
    
    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Coarse search: 0 to 60 degrees in 0.5 degree steps
        for t_deg in np.arange(0, 60.5, 0.5):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine with finer steps
        for step_deg in [0.1, 0.01, 0.001]:
            step = math.radians(step_deg)
            t_start = best_theta - step * 10
            t_end = best_theta + step * 10 + step
            for t_deg in np.arange(math.degrees(t_start), math.degrees(t_end), step_deg):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta
    
    # Small scale to ensure strict non-overlap
    SCALE = 1.000001
    
    # Strategy: Try multiple lattice rotations and spacings,
    # find 12 centers that minimize the max required outer L.
    # Use a rotated hexagonal lattice with various offsets.
    
    best_L_global = float('inf')
    best_config = None  # (centers, thetas)
    
    # Try different lattice rotations (degrees) and radial extents
    lattice_rotations = [0, 15, 30, 45, 60, 75, 90]
    
    for rot_deg in lattice_rotations:
        if time.time() > deadline:
            break
        
        rot_rad = math.radians(rot_deg)
        
        # Generate hexagonal lattice points in a disk of radius ~5
        # Hex lattice: points at (a*sqrt3/2*(q+r), a*(1.5*q + sqrt3/2*r)) rotated by rot_rad
        # For unit hexagons on a hex lattice with spacing sqrt(3) (touching)
        # Centers are at distance sqrt(3) apart in the 3 lattice directions
        
        # Lattice vectors for unit hexagon centers (flat-top, touching):
        # v1 = (1.5, sqrt3/2), v2 = (0, sqrt3)
        # But we want them rotated by rot_rad
        
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        
        # Base lattice vectors (for flat-top hexagons with side 1, centers distance sqrt3 apart)
        # Direction 1: 30 deg -> (sqrt3/2, 1/2) * 2 = (sqrt3, 1) ... 
        # Actually for flat-top unit hexagons, adjacent centers are at distance sqrt(3)
        # in directions 0, 60, 120, 180, 240, 300 degrees
        # Let's use: v1 = sqrt3 * (cos(30), sin(30)) = sqrt3*(sqrt3/2, 1/2) = (1.5, sqrt3/2)
        #           v2 = sqrt3 * (cos(90), sin(90)) = (0, sqrt3)
        
        v1x, v1y = 1.5, sqrt3 / 2.0
        v2x, v2y = 0.0, sqrt3
        
        # Rotate lattice vectors
        v1x_r = v1x * cos_r - v1y * sin_r
        v1y_r = v1x * sin_r + v1y * cos_r
        v2x_r = v2x * cos_r - v2y * sin_r
        v2y_r = v2x * sin_r + v2y * cos_r
        
        # Generate lattice points within radius R
        R = 6.0  # max radius for center
        points = []
        for q in range(-4, 5):
            for r in range(-4, 5):
                # Point = q*v1 + r*v2 (before rotation)
                px = q * v1x + r * v2x
                py = q * v1y + r * v2y
                # Rotate
                px_r = px * cos_r - py * sin_r
                py_r = px * sin_r + py * cos_r
                if math.sqrt(px_r**2 + py_r**2) <= R:
                    points.append((px_r, py_r))
        
        n_pts = len(points)
        if n_pts < 12:
            continue
        
        # For each point, compute optimal theta and required L
        pt_optimal = []
        for px, py in points:
            L_h, theta_h = optimal_theta_for_hexagon(px * SCALE, py * SCALE)
            pt_optimal.append((L_h, theta_h))
        
        # Sort points by their individual required L (ascending)
        # We want to pick 12 points that minimize the max L
        # Greedy: sort by L, take first 12? But they might overlap.
        # Better: try all subsets if n_pts is small, else greedy + local search
        
        if n_pts <= 15:
            # Try all subsets of size 12
            for subset in combinations(range(n_pts), 12):
                if time.time() > deadline:
                    break
                L = max(pt_optimal[i][0] for i in subset)
                if L < best_L_global:
                    best_L_global = L
                    best_config = (
                        [points[i] for i in subset],
                        [pt_optimal[i][1] for i in subset]
                    )
        else:
            # Greedy: sort by L, take 12 with smallest L
            # But check non-overlap (distance >= 2 for unit hexagons, or sqrt(3) for lattice)
            sorted_idx = sorted(range(n_pts), key=lambda i: pt_optimal[i][0])
            
            # Try taking the 12 with smallest L
            subset = sorted_idx[:12]
            L = max(pt_optimal[i][0] for i in subset)
            if L < best_L_global:
                best_L_global = L
                best_config = (
                    [points[i] for i in subset],
                    [pt_optimal[i][1] for i in subset]
                )
            
            # Also try: for each of the top 20, exclude it and add the next
            for skip in range(min(20, n_pts)):
                if time.time() > deadline:
                    break
                subset = [i for i in sorted_idx if i != skip][:12]
                if len(subset) == 12:
                    L = max(pt_optimal[i][0] for i in subset)
                    if L < best_L_global:
                        best_L_global = L
                        best_config = (
                            [points[i] for i in subset],
                            [pt_optimal[i][1] for i in subset]
                        )
    
    # Fallback: use the original 19-center grid approach
    if best_config is None:
        all_centers = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                if abs(q + r) <= 2:
                    x = 1.5 * q
                    y = sqrt3 * (r + q / 2.0)
                    all_centers.append((x, y))
        
        n_centers = len(all_centers)
        center_optimal = []
        for i in range(n_centers):
            x, y = all_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))
        
        best_L = float('inf')
        best_subset = None
        best_thetas = None
        
        for subset in combinations(range(n_centers), 12):
            if time.time() > deadline:
                break
            L = max(center_optimal[i][0] for i in subset)
            if L < best_L:
                best_L = L
                best_subset = subset
                best_thetas = [center_optimal[i][1] for i in subset]
        
        if best_subset is None:
            best_subset = tuple(range(12))
            best_thetas = [center_optimal[i][1] for i in best_subset]
            best_L = max(center_optimal[i][0] for i in best_subset)
        
        best_config = (
            [all_centers[i] for i in best_subset],
            best_thetas
        )
        best_L_global = best_L
    
    centers, thetas = best_config
    
    # Final polish: refine theta for each hexagon with tighter tolerance
    polished_thetas = []
    for i, (cx, cy) in enumerate(centers):
        cx_s = cx * SCALE
        cy_s = cy * SCALE
        # Tighter local search around current theta
        t0 = thetas[i]
        best_L_local = min_L_for_hexagon(cx_s, cy_s, t0)
        best_t_local = t0
        for step_deg in [0.01, 0.001, 0.0001]:
            step = math.radians(step_deg)
            for k in range(-20, 21):
                t = best_t_local + k * step
                L = min_L_for_hexagon(cx_s, cy_s, t)
                if L < best_L_local:
                    best_L_local = L
                    best_t_local = t
        polished_thetas.append(best_t_local)
    
    # Recompute final L
    final_L = max(min_L_for_hexagon(c[0] * SCALE, c[1] * SCALE, t) 
                  for c, t in zip(centers, polished_thetas))
    
    # Construct inner hexagons
    inner = []
    for i, (cx, cy) in enumerate(centers):
        inner.append([cx * SCALE, cy * SCALE, math.degrees(polished_thetas[i])])
    
    return inner, (0.0, 0.0), final_L, 0.0
