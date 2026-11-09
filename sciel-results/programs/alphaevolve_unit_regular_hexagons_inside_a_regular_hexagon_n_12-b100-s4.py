"""Combined approach: combinatorial grid search + continuous rotation refinement."""
import math
import time
import numpy as np
from itertools import combinations

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian (flat-top hexagons)
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))

    # Normal directions for flat-top outer hexagon
    normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
    normal_cos = np.array([math.cos(a) for a in normal_angles])
    normal_sin = np.array([math.sin(a) for a in normal_angles])

    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        vertex_angles = [math.radians(60 * k) + theta for k in range(6)]
        vx = cx + np.array([math.cos(a) for a in vertex_angles])
        vy = cy + np.array([math.sin(a) for a in vertex_angles])

        # Projection of each vertex onto each normal direction
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)

    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Sample theta from 0 to 60 degrees (period is 60 degrees due to hexagonal symmetry)
        for t_deg in range(0, 61, 1):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine around best with finer steps
        for _ in range(3):
            step = 0.5 if _ == 0 else (0.1 if _ == 1 else 0.01)
            lo = best_theta - step * 10
            hi = best_theta + step * 10
            for t_deg in np.arange(lo, hi + step, step):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta

    # Small scale to ensure strict non-overlap
    SCALE = 1.0000001

    best_L = float('inf')
    best_subset = None
    best_thetas = None
    best_cluster_angle = 0.0

    # Search over cluster rotation angles with finer resolution
    cluster_angles_to_try = []
    # Coarse search every 2 degrees
    for a_deg in range(0, 61, 2):
        cluster_angles_to_try.append(math.radians(a_deg))
    # Add specific promising angles
    for extra_deg in [30, 15, 45, 10, 20, 25, 35, 40, 50, 55]:
        cluster_angles_to_try.append(math.radians(extra_deg))

    for cluster_angle in cluster_angles_to_try:
        if time.time() > deadline:
            break

        # Rotate all centers by cluster_angle
        ca = math.cos(cluster_angle)
        sa = math.sin(cluster_angle)
        rotated_centers = []
        for x, y, q, r in all_centers:
            rx = x * ca - y * sa
            ry = x * sa + y * ca
            rotated_centers.append((rx, ry, q, r))

        # Precompute optimal theta for each center (independent of subset)
        center_optimal = []
        for i in range(19):
            x, y, _, _ = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))

        # Try all subsets of size 12 from 19 centers
        # Optimize: pre-sort by L to quickly prune
        sorted_indices = sorted(range(19), key=lambda i: center_optimal[i][0])
        
        for subset in combinations(range(19), 12):
            if time.time() > deadline:
                break
            # L is the max over all hexagons in subset
            L = max(center_optimal[i][0] for i in subset)
            if L < best_L:
                best_L = L
                best_subset = subset
                best_thetas = [center_optimal[i][1] for i in subset]
                best_cluster_angle = cluster_angle

    if best_subset is None:
        # Fallback: use first 12 with no cluster rotation
        best_cluster_angle = 0.0
        rotated_centers = all_centers
        center_optimal = []
        for i in range(19):
            x, y, _, _ = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))
        best_subset = tuple(range(12))
        best_thetas = [center_optimal[i][1] for i in best_subset]
        best_L = max(center_optimal[i][0] for i in best_subset)

    # Joint refinement: simultaneously optimize cluster angle and individual rotations
    if time.time() < deadline - 10.0:
        # Start from current best and do coordinate descent
        cur_cluster_angle = best_cluster_angle
        cur_thetas = list(best_thetas)
        
        for iteration in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Optimize cluster angle with fixed thetas
            best_ca = cur_cluster_angle
            best_ca_L = float('inf')
            for delta_deg in [-5, -4, -3, -2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2, 3, 4, 5]:
                test_ca = cur_cluster_angle + math.radians(delta_deg)
                ca = math.cos(test_ca)
                sa = math.sin(test_ca)
                L_max = 0.0
                for idx, i in enumerate(best_subset):
                    x, y, _, _ = all_centers[i]
                    rx = (x * ca - y * sa) * SCALE
                    ry = (x * sa + y * ca) * SCALE
                    L_h = min_L_for_hexagon(rx, ry, cur_thetas[idx])
                    L_max = max(L_max, L_h)
                if L_max < best_ca_L:
                    best_ca_L = L_max
                    best_ca = test_ca
            
            if best_ca != cur_cluster_angle:
                cur_cluster_angle = best_ca
            
            # Optimize each theta individually with fixed cluster angle
            ca = math.cos(cur_cluster_angle)
            sa = math.sin(cur_cluster_angle)
            for idx, i in enumerate(best_subset):
                x, y, _, _ = all_centers[i]
                rx = (x * ca - y * sa) * SCALE
                ry = (x * sa + y * ca) * SCALE
                L_h, theta_h = optimal_theta_for_hexagon(rx, ry)
                cur_thetas[idx] = theta_h
            
            # Recompute L
            ca = math.cos(cur_cluster_angle)
            sa = math.sin(cur_cluster_angle)
            L_new = 0.0
            for idx, i in enumerate(best_subset):
                x, y, _, _ = all_centers[i]
                rx = (x * ca - y * sa) * SCALE
                ry = (x * sa + y * ca) * SCALE
                L_h = min_L_for_hexagon(rx, ry, cur_thetas[idx])
                L_new = max(L_new, L_h)
            
            if L_new < best_L:
                best_L = L_new
                best_cluster_angle = cur_cluster_angle
                best_thetas = list(cur_thetas)
            else:
                break

    # Construct inner hexagons
    inner = []
    ca = math.cos(best_cluster_angle)
    sa = math.sin(best_cluster_angle)
    for idx, i in enumerate(best_subset):
        x, y, _, _ = all_centers[i]
        # Rotate by cluster angle
        rx = x * ca - y * sa
        ry = x * sa + y * ca
        inner.append([rx * SCALE, ry * SCALE, math.degrees(best_thetas[idx])])

    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
