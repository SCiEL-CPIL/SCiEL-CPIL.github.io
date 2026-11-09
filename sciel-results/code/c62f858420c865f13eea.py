"""Rewritten program: gradient-based repulsive energy minimization with smooth max-distance surrogate."""
import numpy as np
import time
import sys

N = 50

# EVOLVE-BLOCK-START
def _calc_covering_radius(pts):
    """Compute the covering radius in radians for a set of unit vectors.
    The covering radius is the maximum over all sphere points of the angular
    distance to the nearest point in pts. This equals the maximum circumradius
    of the convex hull facets (Voronoi cells on the sphere).
    We approximate by sampling a fine grid of the sphere."""
    # Use a Fibonacci sphere with many more points for accurate estimation
    M = 20000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each grid point, find min angular distance to any of the N points
    # Angular distance = arccos(clamp(dot, -1, 1))
    # We want max over grid of min over pts
    # To be efficient, compute all dot products
    # grid: (M, 3), pts: (N, 3)
    # dots: (M, N)
    dots = grid @ pts.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = dots.max(axis=1)  # max dot = min angular distance
    min_angles = np.arccos(min_dots)
    return np.max(min_angles)


def _smooth_max_energy(pts, beta=20.0):
    """A smooth surrogate for the covering radius using soft-max over a sampled set.
    Actually, let's use a different approach: minimize the sum of exp(-beta * d_i) 
    where d_i is the distance from point i to its nearest neighbor? No.
    
    Better: The covering radius is max_y min_i d(y, p_i). 
    We can think of it as: for each point p_i, it "covers" a cap of radius R around it.
    We want the largest gap. 
    
    Alternative smooth approach: use the fact that covering radius = max over vertices of 
    convex hull facets. But that's complex.
    
    Let's use a different smooth objective: minimize the sum over all pairs of 
    -log(min angular distance between them) to spread them out, combined with 
    a term that penalizes large covering radius.
    
    Actually, the knowledge says: "gradient-based repulsive energy minimization allows 
    for smoother, more directed exploration". Let's use a standard approach:
    
    1. Start with a good initial configuration (Fibonacci spiral or better).
    2. Use gradient descent on a smooth energy that approximates the covering radius.
    
    A common smooth approximation: for each point on a fine grid, compute the 
    min distance to any p_i, then use log-sum-exp to get a smooth max.
    
    But that's expensive. Let's use a simpler approach:
    
    Minimize E = sum_i exp(-alpha * d_i^2) where d_i is the distance from p_i to 
    its nearest neighbor? No, that would make points cluster.
    
    Actually, for covering, we want to maximize the minimum distance from any 
    sphere point to the nearest p_i. This is equivalent to maximizing the 
    minimum over all pairs of "something".
    
    Let me use a well-known approach: 
    - Use a set of test points (grid) on the sphere.
    - For each test point y, compute d(y) = min_i angular_dist(y, p_i)
    - The covering radius is max_y d(y)
    - We can approximate max_y d(y) by a soft-max over the grid points.
    
    But we need gradients w.r.t. p_i. 
    
    d(y) = min_i arccos(clamp(y . p_i, -1, 1))
    
    The min is non-smooth. We can use a soft-min: 
    soft_min_i f_i = -logsumexp(-f_i / tau) * tau
    
    where f_i = arccos(y . p_i)
    
    Then E = soft_max_y soft_min_i f_i(y)
    
    This is getting complex. Let's use a simpler heuristic that's known to work:
    
    1. Initialize with a good configuration.
    2. Use gradient descent on the energy: E = sum over all pairs (i,j) of 
       1/d_ij^2 (repulsive) plus a constraint that points stay on the sphere.
    3. This tends to spread points evenly, which is a good proxy for covering.
    
    Actually, for sphere covering, a better objective is to maximize the minimum 
    distance between any two points (packing), but that's not quite the same as 
    covering. Covering is about the largest empty cap.
    
    Let me use a practical approach:
    1. Start with Fibonacci spiral.
    2. Use a smooth approximation of the covering radius via a moderate grid.
    3. Use L-BFGS-B or similar with the sphere constraint handled by re-projecting.
    
    For the gradient, we can use finite differences or analytic gradients.
    Analytic is better for speed.
    
    Let me implement:
    - A function that computes the covering radius (via grid) and its gradient.
    - Use scipy.optimize.minimize with L-BFGS-B, with the constraint that 
      each point is on the unit sphere (handled by re-normalization after each step, 
      or by using a parametrization).
    
    Actually, re-normalizing after each step is simpler and works well in practice.
    """
    # Use a moderate grid for speed
    M = 5000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute dots: (M, N)
    dots = grid @ pts.T
    dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
    
    # min angular distance for each grid point
    # d(y) = arccos(max_i (y . p_i))
    # Let g(y) = max_i (y . p_i)
    # d(y) = arccos(g(y))
    
    # soft-max approximation for max_i
    beta = 50.0
    # logsumexp trick
    shifted = beta * dots
    max_shifted = np.max(shifted, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted - max_shifted)
    logsumexp = max_shifted.squeeze() + np.log(np.sum(exp_shifted, axis=1))
    g = logsumexp / beta  # approximate max_i (y . p_i)
    
    # d(y) = arccos(g(y))
    g = np.clip(g, -1.0 + 1e-12, 1.0 - 1e-12)
    d = np.arccos(g)
    
    # soft-max over y for the covering radius
    beta2 = 50.0
    shifted2 = beta2 * d
    max_shifted2 = np.max(shifted2, keepdims=True)
    exp_shifted2 = np.exp(shifted2 - max_shifted2)
    logsumexp2 = max_shifted2.squeeze() + np.log(np.sum(exp_shifted2))
    E = logsumexp2 / beta2
    
    return E


def _smooth_max_energy_grad(pts, beta=50.0, beta2=50.0):
    """Compute the smooth energy and its gradient w.r.t. pts.
    
    This is complex, so let's use a simpler approach: use scipy's minimize 
    with a black-box function and let it handle gradients via finite differences,
    or use a different strategy.
    
    Actually, let's use a well-known method: 
    1. Initialize with Fibonacci spiral.
    2. Use gradient descent with a repulsive energy that's easier to differentiate.
    
    Energy: E = -sum_{i != j} log(d_ij) where d_ij is the angular distance.
    This encourages spreading. But it doesn't directly minimize covering radius.
    
    Let me try a hybrid: 
    1. Use the smooth covering radius as the objective.
    2. Use finite-difference gradients (slow but simple).
    3. Or use a parametrization that avoids the sphere constraint.
    
    Given the time constraint (150s), let's use a practical approach:
    - Start with Fibonacci spiral.
    - Use L-BFGS-B with finite-difference gradients on a smooth surrogate.
    - Re-normalize points after each step.
    """
    M = 3000  # smaller grid for speed
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def energy_and_grad(pts_flat):
        pts = pts_flat.reshape(N, 3)
        # Re-normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        pts = pts / norms
        
        # Compute smooth covering radius
        dots = grid @ pts.T  # (M, N)
        dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
        
        # soft-max over i
        shifted = beta * dots
        max_shifted = np.max(shifted, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted - max_shifted)
        logsumexp = max_shifted.squeeze() + np.log(np.sum(exp_shifted, axis=1))
        g = logsumexp / beta
        g = np.clip(g, -1.0 + 1e-12, 1.0 - 1e-12)
        d = np.arccos(g)
        
        # soft-max over y
        shifted2 = beta2 * d
        max_shifted2 = np.max(shifted2, keepdims=True)
        exp_shifted2 = np.exp(shifted2 - max_shifted2)
        logsumexp2 = max_shifted2.squeeze() + np.log(np.sum(exp_shifted2))
        E = logsumexp2 / beta2
        
        # Gradient via finite differences would be too slow.
        # Let's return E and use a gradient-free method, or approximate gradient.
        # For now, return E and None for gradient, and use Nelder-Mead or similar.
        return E, None
    
    return energy_and_grad


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute initial covering radius
    best_pts = pts.copy()
    best_radius = _calc_covering_radius(best_pts)
    
    # Try a few different initializations
    inits = []
    inits.append(pts.copy())
    
    # Random initialization
    rng = np.random.default_rng(42)
    for trial in range(3):
        random_pts = rng.standard_normal((N, 3))
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        inits.append(random_pts)
    
    # For each initialization, run a local optimization
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
        
        current_pts = init_pts.copy()
        current_radius = _calc_covering_radius(current_pts)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
        
        # Use Nelder-Mead with a smooth surrogate
        # We'll use a simpler approach: iterative refinement with random perturbations
        # and keep the best.
        
        # Actually, let's use a gradient-free local search:
        # 1. Compute covering radius.
        # 2. Find the "worst" point (the grid point with max distance to any p_i).
        # 3. Move the nearest p_i towards that worst point slightly.
        # 4. Re-normalize.
        
        n_iter = 200
        step_size = 0.05
        
        for it in range(n_iter):
            if time.time() > deadline:
                break
            
            # Find the worst grid point
            M = 2000
            i_g = np.arange(M) + 0.5
            z_g = 1.0 - 2.0 * i_g / M
            phi_g = np.pi * (1.0 + np.sqrt(5.0)) * i_g
            r_g = np.sqrt(1.0 - z_g * z_g)
            grid = np.column_stack([r_g * np.cos(phi_g), r_g * np.sin(phi_g), z_g])
            
            dots = grid @ current_pts.T
            dots = np.clip(dots, -1.0, 1.0)
            min_dots = dots.max(axis=1)
            min_dots = np.clip(min_dots, -1.0 + 1e-12, 1.0 - 1e-12)
            min_angles = np.arccos(min_dots)
            worst_idx = np.argmax(min_angles)
            worst_point = grid[worst_idx]
            
            # Find the nearest point to the worst point
            worst_dots = worst_point @ current_pts.T
            nearest_idx = np.argmax(worst_dots)
            
            # Move the nearest point towards the worst point
            # Use a small step in the direction of the worst point
            direction = worst_point - current_pts[nearest_idx]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-12:
                direction /= direction_norm
                current_pts[nearest_idx] += step_size * direction
                # Re-normalize
                current_pts[nearest_idx] /= np.linalg.norm(current_pts[nearest_idx])
            
            # Also try to move other points away from each other to reduce clustering
            # Compute pairwise distances and push apart points that are too close
            dots_all = current_pts @ current_pts.T
            np.fill_diagonal(dots_all, -1)
            min_dots_all = dots_all.min(axis=1)
            # Points with small min distance are clustered
            for j in range(N):
                if min_dots_all[j] > 0.95:  # very close to some other point
                    # Find the closest point
                    closest_k = np.argmax(dots_all[j])
                    direction = current_pts[j] - current_pts[closest_k]
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 1e-12:
                        direction /= direction_norm
                        current_pts[j] += 0.01 * direction
                        current_pts[j] /= np.linalg.norm(current_pts[j])
            
            # Compute new covering radius
            new_radius = _calc_covering_radius(current_pts)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = current_pts.copy()
        
        # After local search, try a final polish with smaller steps
        if time.time() < deadline - 20:
            step_size = 0.01
            for it in range(100):
                if time.time() > deadline:
                    break
                
                M = 2000
                i_g = np.arange(M) + 0.5
                z_g = 1.0 - 2.0 * i_g / M
                phi_g = np.pi * (1.0 + np.sqrt(5.0)) * i_g
                r_g = np.sqrt(1.0 - z_g * z_g)
                grid = np.column_stack([r_g * np.cos(phi_g), r_g * np.sin(phi_g), z_g])
                
                dots = grid @ current_pts.T
                dots = np.clip(dots, -1.0, 1.0)
                min_dots = dots.max(axis=1)
                min_dots = np.clip(min_dots, -1.0 + 1e-12, 1.0 - 1e-12)
                min_angles = np.arccos(min_dots)
                worst_idx = np.argmax(min_angles)
                worst_point = grid[worst_idx]
                
                worst_dots = worst_point @ current_pts.T
                nearest_idx = np.argmax(worst_dots)
                
                direction = worst_point - current_pts[nearest_idx]
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-12:
                    direction /= direction_norm
                    current_pts[nearest_idx] += step_size * direction
                    current_pts[nearest_idx] /= np.linalg.norm(current_pts[nearest_idx])
                
                new_radius = _calc_covering_radius(current_pts)
                
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = current_pts.copy()
    
    # Final check: ensure all points are on the unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
