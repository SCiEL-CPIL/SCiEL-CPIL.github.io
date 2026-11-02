"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 140  # seconds, safety margin from 150s limit

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])
VERTICES = np.array([V0, V1, V2])

def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the unit equilateral triangle area."""
    # points: (N, 2) array of the 11 interior points
    # We consider all triangles formed by any 3 of the 14 points (11 interior + 3 vertices)
    all_pts = np.vstack([VERTICES, points])
    n = len(all_pts)
    x = all_pts[:, 0]
    y = all_pts[:, 1]
    
    # Compute all pairwise differences
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    
    # Cross product for all triples (i, j, k)
    # area(i,j,k) = 0.5 * |dx(i,j)*dy(i,k) - dy(i,j)*dx(i,k)|
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    # Extract i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
    
    min_area = np.min(valid_areas)
    return min_area / TRI_AREA

def _is_inside_triangle(p):
    """Check if point p is inside the equilateral triangle with small margin."""
    x, y = p
    eps = 1e-9
    if y < -eps or y > S3H + eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    if y > S3H * (1 - x) + eps:
        return False
    if y > S3H * x + eps:
        return False
    return True

def _penalized_min_area(x):
    """Objective for minimization: minimize negative min area plus penalty for out-of-bounds."""
    points = x.reshape(-1, 2)
    
    # Penalty for points outside triangle
    penalty = 0.0
    for i in range(N):
        px, py = points[i]
        # Distances to boundaries (positive if outside)
        d_bottom = -py
        d_top = py - S3H
        d_left = -px
        d_right = px - 1
        d_slant_right = py - S3H * (1 - px)
        d_slant_left = py - S3H * px
        
        dist = max(0.0, d_bottom, d_top, d_left, d_right, d_slant_right, d_slant_left)
        penalty += dist ** 2
    
    min_a = _min_area_normalized(points)
    # Large penalty to ensure points stay inside
    return -min_a + 100.0 * penalty

def _repulsive_energy(x, k_rep=100.0, k_con=50.0):
    """
    Objective: Minimize total potential energy.
    Energy = sum of repulsive forces between all pairs (11 interior + 3 vertices)
             + constraint forces pushing points towards center if near boundary.
    """
    points = x.reshape(-1, 2)
    all_pts = np.vstack([VERTICES, points])
    n = len(all_pts)
    
    # Repulsive energy: sum over pairs 1/d^p
    # We use p=2 for standard Coulomb-like, but higher p can help with local minima?
    # Let's stick to p=2 for smoothness, or maybe p=4 to be steeper.
    # The directive suggests "inversely proportional to distance raised to a high power".
    # Let's try p=4.
    p = 4
    
    energy = 0.0
    # Vectorized calculation of pairwise distances
    # dx[i, j] = x[j] - x[i]
    dx = all_pts[:, 0][:, None] - all_pts[:, 0][None, :]
    dy = all_pts[:, 1][:, None] - all_pts[:, 1][None, :]
    dist_sq = dx**2 + dy**2
    
    # Avoid division by zero (should not happen if points are distinct)
    # Add small epsilon to diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Calculate energy for all pairs i < j
    # dist = sqrt(dist_sq)
    # term = 1 / dist^p = 1 / dist_sq^(p/2)
    # Since p=4, term = 1 / dist_sq^2
    
    # We only want upper triangle to avoid double counting
    i_idx, j_idx = np.triu_indices(n, k=1)
    dist_sq_pairs = dist_sq[i_idx, j_idx]
    
    # Clip to prevent overflow if points are very close
    dist_sq_pairs = np.clip(dist_sq_pairs, 1e-6, None)
    
    repulsive_terms = 1.0 / (dist_sq_pairs ** 2) # p=4
    energy += np.sum(repulsive_terms)
    
    # Constraint energy: Push points towards center if they are near boundary
    # Center of triangle: (0.5, S3H/3)
    center = np.array([0.5, S3H / 3.0])
    
    # Distance to boundary for each interior point
    # We can reuse the logic from _penalized_min_area
    for i in range(N):
        px, py = points[i]
        d_bottom = -py
        d_top = py - S3H
        d_left = -px
        d_right = px - 1
        d_slant_right = py - S3H * (1 - px)
        d_slant_left = py - S3H * px
        
        # If any distance is negative (outside), add large penalty
        # If distance is small (near boundary), add penalty to push in
        # Let's define "near" as < 0.1
        min_dist = min(0, d_bottom, d_top, d_left, d_right, d_slant_right, d_slant_left)
        if min_dist < 0:
            # Outside
            energy += k_con * min_dist**2
        else:
            # Inside, but check if close to boundary
            # We want to keep them away from boundary to maximize area
            # So we penalize being close to boundary
            # distance to boundary = min of the positive distances
            dist_to_bdry = min(d_bottom, d_top, d_left, d_right, d_slant_right, d_slant_left)
            if dist_to_bdry < 0.1:
                energy += k_con * (0.1 - dist_to_bdry)**2

    return energy

def _repulsive_grad(x, k_rep=100.0, k_con=50.0):
    """
    Gradient of _repulsive_energy.
    """
    points = x.reshape(-1, 2)
    all_pts = np.vstack([VERTICES, points])
    n = len(all_pts)
    
    p = 4
    
    grad = np.zeros_like(x)
    
    # Repulsive gradient
    # E_rep = sum_{i<j} 1 / dist_ij^p
    # dE/dx_i = -p * (x_i - x_j) / dist_ij^(p+2)
    
    dx = all_pts[:, 0][:, None] - all_pts[:, 0][None, :]
    dy = all_pts[:, 1][:, None] - all_pts[:, 1][None, :]
    dist_sq = dx**2 + dy**2
    np.fill_diagonal(dist_sq, 1.0)
    
    # dist = sqrt(dist_sq)
    # term = dist_sq^(-(p+2)/2) = dist_sq^(-3) for p=4
    # coeff = -p * term
    
    # We need to sum gradients for each point i
    # For point i, grad_i = sum_{j != i} -p * (x_i - x_j) / dist_ij^(p+2)
    
    # Vectorized approach:
    # Create matrix of coefficients C[i, j] = -p * dist_ij^(-(p+2))
    # Then grad_i = sum_j C[i, j] * (x_i - x_j)
    
    # dist_sq is (n, n)
    # dist_pow = dist_sq^(-(p+2)/2)
    # For p=4, exponent is -3
    # We need to handle diagonal (should be 0 contribution)
    
    dist_pow = np.power(dist_sq, -3.0)
    np.fill_diagonal(dist_pow, 0.0)
    
    coeff_matrix = -p * dist_pow
    
    # dx[i, j] = x_j - x_i
    # We want (x_i - x_j) = -dx[i, j]
    # So grad_i = sum_j coeff_matrix[i, j] * (-dx[i, j])
    #             = - sum_j coeff_matrix[i, j] * dx[i, j]
    #             = - sum_j coeff_matrix[i, j] * (x_j - x_i)
    
    # Let's compute sum_j coeff_matrix[i, j] * dx[i, j]
    # This is matrix multiplication: coeff_matrix @ dx
    # Result shape (n, 1)
    
    grad_x = -(coeff_matrix @ dx)[:, 0]
    grad_y = -(coeff_matrix @ dy)[:, 0]
    
    # The gradient for the first 3 points (vertices) is not needed as they are fixed
    # The gradient for the next N points (interior) is what we return
    grad[:N*2:2] = grad_x[3:3+N]
    grad[1:N*2:2] = grad_y[3:3+N]
    
    # Constraint gradient
    # E_con = sum_i k_con * (0.1 - dist_to_bdry_i)^2 for dist_to_bdry_i < 0.1
    # dE/dx = 2 * k_con * (0.1 - dist) * (-d(dist)/dx)
    # dist is min of several linear functions. The gradient is the negative gradient of the active constraint.
    # If multiple constraints are active, we take the sum? Or just the min?
    # Usually, for min of smooth functions, the gradient is undefined or we pick one.
    # Let's pick the one with the smallest distance.
    
    for i in range(N):
        px, py = points[i]
        d_bottom = -py
        d_top = py - S3H
        d_left = -px
        d_right = px - 1
        d_slant_right = py - S3H * (1 - px)
        d_slant_left = py - S3H * px
        
        dists = [d_bottom, d_top, d_left, d_right, d_slant_right, d_slant_left]
        min_dist = min(dists)
        
        if min_dist < 0.1:
            # Find which constraint is active
            idx = dists.index(min_dist)
            
            # Gradient of distance to boundary
            # d_bottom = -y -> grad (0, -1)
            # d_top = y - S3H -> grad (0, 1)
            # d_left = -x -> grad (-1, 0)
            # d_right = x - 1 -> grad (1, 0)
            # d_slant_right = y - S3H*(1-x) = y - S3H + S3H*x -> grad (S3H, 1)
            # d_slant_left = y - S3H*x -> grad (-S3H, 1)
            
            grads = [
                np.array([0.0, -1.0]),
                np.array([0.0, 1.0]),
                np.array([-1.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([S3H, 1.0]),
                np.array([-S3H, 1.0])
            ]
            
            # dE/dx = 2 * k_con * (0.1 - min_dist) * (-grad_dist)
            # Note: dist is a function of x. If dist = ax + by + c, grad_dist = (a, b).
            # We want to push x INWARD.
            # If dist is small, we want to increase dist.
            # So we want to move in direction of grad_dist.
            # dE/dx should point in direction of DECREASE in E.
            # E = k * (0.1 - dist)^2. dE/ddist = -2k(0.1 - dist).
            # dE/dx = dE/ddist * ddist/dx = -2k(0.1 - dist) * grad_dist.
            # So grad_E = -2 * k_con * (0.1 - min_dist) * grads[idx]
            
            factor = -2.0 * k_con * (0.1 - min_dist)
            grad[2*i] += factor * grads[idx][0]
            grad[2*i+1] += factor * grads[idx][1]
            
    return grad

def construct():
    start_time = time.time()
    
    # Initial guess: Centroid + small perturbations
    center = np.array([0.5, S3H / 3.0])
    rng = np.random.RandomState(42)
    
    # Create a grid or random points
    # Let's try a few random initializations and pick the best one for L-BFGS-B
    
    best_points = None
    best_score = -float('inf')
    
    # Try 5 different random starts
    for attempt in range(5):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Random points inside triangle
        pts = []
        for _ in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = S3H * v
                    # Check if inside with margin
                    if 0.05 < x < 0.95 and 0.05 < y < S3H * 0.95:
                        if y < S3H * (1 - x) * 0.95 and y < S3H * x * 0.95:
                            pts.append([x, y])
                            break
        x0 = np.array(pts).flatten()
        
        try:
            res = minimize(
                _repulsive_energy,
                x0,
                method='L-BFGS-B',
                jac=_repulsive_grad,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            polished = res.x.reshape(-1, 2)
            
            # Check validity
            valid = all(_is_inside_triangle(p) for p in polished)
            if valid:
                score = _min_area_normalized(polished)
                if score > best_score:
                    best_score = score
                    best_points = polished.copy()
                    
        except Exception as e:
            #print(e)
            pass

    # If L-BFGS-B failed or didn't produce a good result, fallback to Nelder-Mead on the penalized objective
    if best_points is None or best_score < 0.01:
        if time.time() - start_time < DEADLINE - 5:
            # Use the best points from L-BFGS-B if available, else random
            if best_points is None:
                pts = []
                for _ in range(N):
                    while True:
                        u, v = rng.random(), rng.random()
                        if u + v < 1:
                            x = u + 0.5 * v
                            y = S3H * v
                            if 0.05 < x < 0.95 and 0.05 < y < S3H * 0.95:
                                if y < S3H * (1 - x) * 0.95 and y < S3H * x * 0.95:
                                    pts.append([x, y])
                                    break
                best_points = np.array(pts)
            
            x0 = best_points.flatten()
            try:
                res_nm = minimize(
                    _penalized_min_area,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                polished = res_nm.x.reshape(-1, 2)
                if all(_is_inside_triangle(p) for p in polished):
                    score = _min_area_normalized(polished)
                    if score > best_score:
                        best_points = polished
                        best_score = score
            except Exception:
                pass

    # Final Polish with Nelder-Mead on the penalized objective to ensure we are at a local max of min_area
    if time.time() - start_time < DEADLINE - 2:
        try:
            x0 = best_points.flatten()
            res_nm = minimize(
                _penalized_min_area,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            polished = res_nm.x.reshape(-1, 2)
            if all(_is_inside_triangle(p) for p in polished):
                score = _min_area_normalized(polished)
                if score > best_score:
                    best_points = polished
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
