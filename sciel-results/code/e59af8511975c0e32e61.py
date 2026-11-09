"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    # Using Flat-Top hexagon grid coordinates
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian (flat-top hexagons)
                # x = 1.5 * q, y = sqrt(3) * (r + q/2)
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))

    # Normal directions for flat-top outer hexagon (pointing outward from center)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
    normal_cos = np.array([math.cos(a) for a in normal_angles])
    normal_sin = np.array([math.sin(a) for a in normal_angles])

    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        # Vertex angles relative to hexagon center
        vertex_angles = [math.radians(60 * k) + theta for k in range(6)]
        vx = cx + np.array([math.cos(a) for a in vertex_angles])
        vy = cy + np.array([math.sin(a) for a in vertex_angles])

        # Projection of each vertex onto each normal direction
        # proj[i,j] = vx[i]*cos(normal_j) + vy[i]*sin(normal_j)
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)

    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Sample theta from 0 to 60 degrees (period is 60 degrees due to hexagonal symmetry)
        # Coarse search
        for t_deg in range(0, 61, 2):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Fine search around best with multiple passes
        for step in [1.0, 0.1, 0.01, 0.001]:
            for t_deg in np.arange(max(0, best_theta - step*5), min(math.radians(60), best_theta + step*5) + step, step):
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

    # Search over cluster rotation angles
    # The cluster has 6-fold symmetry, so we only need to search 0 to 60 degrees
    # But the optimal might be at 30 degrees (rotating the "corners" to align with "sides")
    cluster_angles_to_try = []
    # Coarse search
    for a_deg in range(0, 61, 5):
        cluster_angles_to_try.append(math.radians(a_deg))
    # Add key angles explicitly
    for a_deg in [15, 30, 45, 52.5]:
        cluster_angles_to_try.append(math.radians(a_deg))
    # Deduplicate
    cluster_angles_to_try = sorted(set(cluster_angles_to_try))

    for cluster_angle in cluster_angles_to_try:
        if time.time() > deadline:
            break

        # Rotate all centers by cluster_angle
        rotated_centers = []
        for x, y, q, r in all_centers:
            rx = x * math.cos(cluster_angle) - y * math.sin(cluster_angle)
            ry = x * math.sin(cluster_angle) + y * math.cos(cluster_angle)
            rotated_centers.append((rx, ry, q, r))

        # Precompute optimal theta for each center (independent of subset)
        center_optimal = []
        for i in range(19):
            x, y, _, _ = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x * SCALE, y * SCALE)
            center_optimal.append((L_h, theta_h))
        
        # Sort centers by their individual L requirement to help subset selection
        sorted_indices = sorted(range(19), key=lambda i: center_optimal[i][0])

        # Try all subsets of size 12 from 19 centers
        # Use sorted order to prune: if the 12th smallest L is already >= best_L, skip
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
                # Early termination if we find a very good solution
                if best_L < 9.9:
                    break

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

    # Build initial positions from best discrete solution
    init_positions = np.zeros((N, 2))
    init_thetas = np.zeros(N)
    for idx, i in enumerate(best_subset):
        x, y, _, _ = all_centers[i]
        rx = x * math.cos(best_cluster_angle) - y * math.sin(best_cluster_angle)
        ry = x * math.sin(best_cluster_angle) + y * math.cos(best_cluster_angle)
        init_positions[idx, 0] = rx * SCALE
        init_positions[idx, 1] = ry * SCALE
        init_thetas[idx] = best_thetas[idx]

    # Smooth optimization using scipy if available
    if HAS_SCIPY and time.time() < deadline - 10.0:
        def min_separation(dtheta):
            """
            Compute the minimum center-to-center distance for two unit hexagons
            with relative rotation dtheta to be non-overlapping.
            """
            dtheta = abs(dtheta) % (2 * math.pi)
            dtheta = min(dtheta, 2*math.pi - dtheta)
            
            # For two regular hexagons of side 1:
            # The support function in direction phi is max over vertices of dot(v, u_phi)
            # where u_phi = (cos(phi), sin(phi))
            # For hexagon 1 with rotation theta1, vertices at angles theta1 + k*pi/3
            # For hexagon 2 with rotation theta2, vertices at angles theta2 + k*pi/3
            # 
            # The minimum separation is the minimum over all directions phi of:
            #   h1(phi) + h2(phi + pi)
            # where h1 is the support function of hexagon 1 in direction phi
            # and h2 is the support function of hexagon 2 in direction phi+pi
            #
            # For a unit regular hexagon with rotation theta:
            # h(phi) = max_{k=0..5} cos(phi - theta - k*pi/3)
            #        = max over 6 values
            #
            # Due to 6-fold symmetry, we can restrict phi to [0, pi/6]
            # and use the fact that h(phi) = cos(phi - theta_mod) where theta_mod
            # is theta mod pi/6, adjusted for which sector we're in.
            
            # Simpler approach: sample the minimum over a fine grid of directions
            # Since dtheta is the relative rotation, set theta1 = 0, theta2 = dtheta
            min_sep = float('inf')
            # Sample 72 directions (every 5 degrees)
            for phi_deg in range(0, 180, 5):
                phi = math.radians(phi_deg)
                # Support function of hexagon 1 (rotation 0) in direction phi
                # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
                h1 = max(math.cos(phi - math.radians(60*k)) for k in range(6))
                # Support function of hexagon 2 (rotation dtheta) in direction phi+pi
                # Vertices at angles dtheta, dtheta+60, ..., dtheta+300
                h2 = max(math.cos(phi + math.pi - dtheta - math.radians(60*k)) for k in range(6))
                sep = h1 + h2
                if sep < min_sep:
                    min_sep = sep
            
            return min_sep
        
        def smooth_objective(params):
            """
            params: [x0, y0, theta0, x1, y1, theta1, ..., x11, y11, theta11, L]
            Minimize L + penalty for overlaps
            """
            L = params[-1]
            positions = params[:2*N].reshape(N, 2)
            thetas = params[2*N:3*N]
            
            # Compute required L for each hexagon
            L_required = np.zeros(N)
            for i in range(N):
                L_required[i] = min_L_for_hexagon(positions[i, 0], positions[i, 1], thetas[i])
            
            # Soft-max approximation of max(L_required)
            # Use log-sum-exp for smoothness
            beta = 30.0  # Higher = sharper max approximation
            max_lr = np.max(L_required)
            if max_lr > 0:
                soft_max = max_lr + math.log(np.sum(np.exp(beta * (L_required - max_lr)))) / beta
            else:
                soft_max = 0.0
            
            # Overlap penalty: for each pair, check if hexagons overlap
            overlap_penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    d = math.sqrt(dx*dx + dy*dy)
                    
                    # Relative rotation
                    dtheta = abs(thetas[i] - thetas[j]) % (2 * math.pi)
                    dtheta = min(dtheta, 2*math.pi - dtheta)
                    
                    # Use exact min separation
                    min_dist = min_separation(dtheta)
                    
                    if d < min_dist:
                        # Smooth penalty
                        deficit = min_dist - d
                        overlap_penalty += deficit * deficit * 50.0
            
            # Also penalize if L is too small to contain all hexagons
            containment_penalty = 0.0
            for i in range(N):
                if L_required[i] > L:
                    deficit = L_required[i] - L
                    containment_penalty += deficit * deficit * 50.0
            
            return soft_max + overlap_penalty + containment_penalty
        
        # Initial guess for L
        L_init = best_L * 1.005
        init_params = np.concatenate([init_positions.flatten(), init_thetas, [L_init]])
        
        # Try multiple optimization methods
        best_opt_L = float('inf')
        best_opt_params = None
        
        methods_to_try = ['Nelder-Mead', 'Powell']
        
        for method in methods_to_try:
            if time.time() > deadline - 5.0:
                break
            
            try:
                if method == 'Nelder-Mead':
                    result = minimize(
                        smooth_objective,
                        init_params,
                        method='Nelder-Mead',
                        options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-9}
                    )
                elif method == 'Powell':
                    result = minimize(
                        smooth_objective,
                        init_params,
                        method='Powell',
                        options={'maxiter': 400, 'xtol': 1e-7, 'ftol': 1e-9}
                    )
                
                if result.success or result.fun < smooth_objective(init_params):
                    opt_positions = result.x[:2*N].reshape(N, 2)
                    opt_thetas = result.x[2*N:3*N]
                    
                    # Check actual max L required
                    actual_max_L = 0.0
                    for i in range(N):
                        actual_max_L = max(actual_max_L, min_L_for_hexagon(opt_positions[i,0], opt_positions[i,1], opt_thetas[i]))
                    
                    # Check overlaps with exact min separation
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = opt_positions[i, 0] - opt_positions[j, 0]
                            dy = opt_positions[i, 1] - opt_positions[j, 1]
                            d = math.sqrt(dx*dx + dy*dy)
                            dtheta = abs(opt_thetas[i] - opt_thetas[j]) % (2 * math.pi)
                            dtheta = min(dtheta, 2*math.pi - dtheta)
                            min_dist = min_separation(dtheta)
                            if d < min_dist - 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid and actual_max_L < best_opt_L:
                        best_opt_L = actual_max_L
                        best_opt_params = result.x.copy()
                        
            except Exception:
                pass
        
        # If optimization found a better solution, use it
        if best_opt_params is not None and best_opt_L < best_L:
            best_L = best_opt_L
            for i in range(N):
                init_positions[i, 0] = best_opt_params[2*i]
                init_positions[i, 1] = best_opt_params[2*i+1]
                init_thetas[i] = best_opt_params[2*N+i]
        
        # Final high-precision polish pass if time remains
        if HAS_SCIPY and time.time() < deadline - 3.0 and best_opt_params is not None:
            try:
                result = minimize(
                    smooth_objective,
                    best_opt_params,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                
                if result.fun < smooth_objective(best_opt_params):
                    opt_positions = result.x[:2*N].reshape(N, 2)
                    opt_thetas = result.x[2*N:3*N]
                    
                    actual_max_L = 0.0
                    for i in range(N):
                        actual_max_L = max(actual_max_L, min_L_for_hexagon(opt_positions[i,0], opt_positions[i,1], opt_thetas[i]))
                    
                    valid = True
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = opt_positions[i, 0] - opt_positions[j, 0]
                            dy = opt_positions[i, 1] - opt_positions[j, 1]
                            d = math.sqrt(dx*dx + dy*dy)
                            dtheta = abs(opt_thetas[i] - opt_thetas[j]) % (2 * math.pi)
                            dtheta = min(dtheta, 2*math.pi - dtheta)
                            min_dist = min_separation(dtheta)
                            if d < min_dist - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid and actual_max_L < best_L:
                        best_L = actual_max_L
                        for i in range(N):
                            init_positions[i, 0] = opt_positions[i, 0]
                            init_positions[i, 1] = opt_positions[i, 1]
                            init_thetas[i] = opt_thetas[i]
            except Exception:
                pass

    # Construct inner hexagons
    inner = []
    for idx in range(N):
        inner.append([init_positions[idx, 0], init_positions[idx, 1], math.degrees(init_thetas[idx])])

    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
