# EVOLVE-BLOCK-START
import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 620  # Leave 10s margin from 630s limit
    
    # Precompute indices for all 286 triangles
    tri_indices = np.array(list(itertools.combinations(range(n), 3)))
    i_idx, j_idx, k_idx = tri_indices[:, 0], tri_indices[:, 1], tri_indices[:, 2]

    def calculate_score(points):
        # points: (13, 2)
        try:
            hull = scipy.spatial.ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 1e-10:
                return 0.0
        except Exception:
            return 0.0
        
        # Vectorized triangle area calculation
        pi = points[i_idx]
        pj = points[j_idx]
        pk = points[k_idx]
        
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 1e-10:
            return 0.0
            
        return A_min / A_hull

    def neg_score(points):
        s = calculate_score(points)
        return -s

    best_score = 0.0
    best_points = None

    # Generate initial guesses
    initial_guesses = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_guesses.append(reg_poly)
    
    rng = np.random.default_rng(42)
    
    # 2. Jittered Regular Polygon (30 variations with varying jitter)
    for _ in range(30):
        jitter_scale = 0.1 + 0.3 * rng.random()
        jitter = rng.normal(0, jitter_scale, (n, 2))
        guess = reg_poly + jitter
        initial_guesses.append(guess)
        
    # 3. Random points (30 variations)
    for _ in range(30):
        guess = rng.random((n, 2))
        initial_guesses.append(guess)
        
    # 4. "Square" based (4x4 grid subset)
    grid_x = np.linspace(0, 1, 4)
    grid_y = np.linspace(0, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    for _ in range(15):
        idx = rng.choice(16, 13, replace=False)
        guess = grid_points[idx]
        initial_guesses.append(guess)
    
    # 5. Near-optimal structured pattern: 3 concentric rings
    # Inner: 1 center point
    # Middle: 4 points (square)
    # Outer: 8 points (octagon)
    center = np.array([[0.0, 0.0]])
    sq_angles = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
    sq_r = 0.5
    sq_pts = np.column_stack([sq_r * np.cos(sq_angles), sq_r * np.sin(sq_angles)])
    oct_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    oct_r = 1.0
    oct_pts = np.column_stack([oct_r * np.cos(oct_angles), oct_r * np.sin(oct_angles)])
    ring_guess = np.vstack([center, sq_pts, oct_pts])
    initial_guesses.append(ring_guess)
    
    # Jittered ring patterns
    for _ in range(10):
        jitter = rng.normal(0, 0.1, (n, 2))
        initial_guesses.append(ring_guess + jitter)

    # Optimization Loop
    for guess in initial_guesses:
        if time.time() - start_time > time_limit - 20:
            break
            
        # Normalize
        guess_centered = guess - np.mean(guess, axis=0)
        max_dist = np.max(np.linalg.norm(guess_centered, axis=1))
        if max_dist > 1e-10:
            guess_normalized = guess_centered / max_dist
        else:
            guess_normalized = guess_centered
        
        # Run Nelder-Mead with adaptive maxiter based on remaining time
        remaining = time_limit - (time.time() - start_time)
        maxiter = min(5000, max(500, int(remaining * 50)))
        
        try:
            result = scipy.optimize.minimize(
                neg_score, 
                guess_normalized, 
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            current_score = -result.fun
            if current_score > best_score:
                best_score = current_score
                best_points = result.x.copy()
        except Exception:
            pass

    # If time remains, refine the best point with Powell
    if best_points is not None and time.time() - start_time < time_limit - 40:
        try:
            result_powell = scipy.optimize.minimize(
                neg_score, 
                best_points, 
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9}
            )
            if -result_powell.fun > best_score:
                best_score = -result_powell.fun
                best_points = result_powell.x.copy()
        except Exception:
            pass
    
    # Final polish: coordinate descent on the most constrained triangles
    if best_points is not None and time.time() - start_time < time_limit - 5:
        try:
            current_points = best_points.copy()
            for _ in range(50):
                if time.time() - start_time > time_limit - 2:
                    break
                # Find the 5 smallest triangles
                pi = current_points[i_idx]
                pj = current_points[j_idx]
                pk = current_points[k_idx]
                cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
                areas = 0.5 * np.abs(cross)
                
                # Get indices of points in the 5 smallest triangles
                small_tri_idx = np.argsort(areas)[:5]
                involved_points = set()
                for ti in small_tri_idx:
                    involved_points.add(int(i_idx[ti]))
                    involved_points.add(int(j_idx[ti]))
                    involved_points.add(int(k_idx[ti]))
                
                # Try small perturbations of involved points
                improved = False
                for pt_idx in list(involved_points)[:5]:
                    for dx in [-0.01, 0.01]:
                        for dy in [-0.01, 0.01]:
                            test_points = current_points.copy()
                            test_points[pt_idx, 0] += dx
                            test_points[pt_idx, 1] += dy
                            test_score = calculate_score(test_points)
                            if test_score > best_score:
                                best_score = test_score
                                current_points = test_points
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    break
            best_points = current_points
        except Exception:
            pass

    if best_points is None:
        best_points = initial_guesses[0]
        
    return best_points

# EVOLVE-BLOCK-END
