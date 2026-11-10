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
    time_limit = 625  # Leave 5s margin from 630s limit

    # Precompute indices for all 286 triangles
    tri_indices = np.array(list(itertools.combinations(range(n), 3)))
    i_idx, j_idx, k_idx = tri_indices[:, 0], tri_indices[:, 1], tri_indices[:, 2]

    def calculate_score(points):
        """Calculate the exact Heilbronn score: A_min / A_hull"""
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
        
        # Cross product magnitude / 2
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min <= 1e-10:
            return 0.0
            
        return A_min / A_hull

    def neg_score(points):
        s = calculate_score(points)
        return -s

    def physics_relax(points, iterations=2000, dt=0.001):
        """
        Physics-based relaxation using repulsive forces.
        Points repel each other (inverse square law) and are weakly attracted to the centroid
        to keep the configuration compact. This helps spread points evenly.
        """
        pts = points.copy().astype(float)
        # Normalize initial size
        centroid = np.mean(pts, axis=0)
        pts -= centroid
        max_dist = np.max(np.linalg.norm(pts, axis=1))
        if max_dist > 0:
            pts /= max_dist
        
        # Adaptive time step
        current_dt = 0.01
        min_dist_thresh = 0.01
        
        for i in range(iterations):
            if time.time() - start_time > time_limit - 10:
                break
                
            # Calculate forces
            forces = np.zeros_like(pts)
            
            # Repulsion between all pairs
            for a in range(n):
                for b in range(a + 1, n):
                    delta = pts[a] - pts[b]
                    dist_sq = np.dot(delta, delta)
                    if dist_sq < 1e-12:
                        # Jitter if too close
                        delta += np.random.normal(0, 0.001, 2)
                        dist_sq = np.dot(delta, delta)
                    
                    # Coulomb-like repulsion: F = k / r^2
                    # We use a softening factor to avoid singularity
                    dist = np.sqrt(dist_sq)
                    force_mag = 1.0 / (dist_sq + 1e-6)
                    force_vec = force_mag * delta / dist
                    forces[a] += force_vec
                    forces[b] -= force_vec
            
            # Weak attraction to centroid to prevent expansion
            centroid = np.mean(pts, axis=0)
            for a in range(n):
                forces[a] -= 0.1 * (pts[a] - centroid)
            
            # Update positions
            pts += current_dt * forces
            
            # Adaptive step size: reduce if points get too close
            min_pair_dist = np.inf
            for a in range(n):
                for b in range(a + 1, n):
                    d = np.linalg.norm(pts[a] - pts[b])
                    if d < min_pair_dist:
                        min_pair_dist = d
            
            if min_pair_dist < min_dist_thresh:
                current_dt *= 0.5
            else:
                current_dt = min(current_dt * 1.05, 0.01)
            
            # Re-center
            pts -= np.mean(pts, axis=0)
            
        # Normalize final result
        centroid = np.mean(pts, axis=0)
        pts -= centroid
        max_dist = np.max(np.linalg.norm(pts, axis=1))
        if max_dist > 0:
            pts /= max_dist
            
        return pts

    best_score = 0.0
    best_points = None

    def update_best(points, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Generate diverse initial guesses
    initial_guesses = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_guesses.append(reg_poly)
    
    rng = np.random.default_rng(42)
    
    # 2. Jittered Regular Polygon
    for _ in range(10):
        jitter = rng.normal(0, 0.1, (n, 2))
        guess = reg_poly + jitter
        initial_guesses.append(guess)
        
    # 3. Random points inside unit circle
    for _ in range(10):
        angles_rand = rng.uniform(0, 2 * np.pi, n)
        radii_rand = rng.uniform(0.5, 1.0, n) # Keep away from center to avoid clustering
        guess = np.column_stack([radii_rand * np.cos(angles_rand), radii_rand * np.sin(angles_rand)])
        initial_guesses.append(guess)

    # Phase 1: Physics-based relaxation for each initial guess
    for guess in initial_guesses:
        if time.time() - start_time > time_limit - 60:
            break
        
        # Relax the configuration
        relaxed_pts = physics_relax(guess, iterations=1000)
        
        # Evaluate
        score = calculate_score(relaxed_pts)
        update_best(relaxed_pts, score)
        
        # Local refinement from the relaxed state
        if time.time() - start_time < time_limit - 20:
            try:
                result = scipy.optimize.minimize(
                    neg_score, 
                    relaxed_pts, 
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                current_score = -result.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = result.x.copy()
            except Exception:
                pass

    # Phase 2: If we have time, try a few more random starts with shorter relaxation
    if best_points is not None and time.time() - start_time < time_limit - 40:
        for _ in range(5):
            if time.time() - start_time > time_limit - 20:
                break
            guess = rng.random((n, 2)) * 2 - 1
            relaxed_pts = physics_relax(guess, iterations=500)
            score = calculate_score(relaxed_pts)
            update_best(relaxed_pts, score)
            
            if time.time() - start_time < time_limit - 10:
                try:
                    result = scipy.optimize.minimize(
                        neg_score, 
                        relaxed_pts, 
                        method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                    current_score = -result.fun
                    if current_score > best_score:
                        best_score = current_score
                        best_points = result.x.copy()
                except Exception:
                    pass

    # Phase 3: Final high-precision polish
    if best_points is not None and time.time() - start_time < time_limit - 5:
        try:
            result_final = scipy.optimize.minimize(
                neg_score, 
                best_points, 
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if result_final.fun < -best_score:
                best_score = -result_final.fun
                best_points = result_final.x.copy()
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon
        best_points = initial_guesses[0]
        
    return best_points

# EVOLVE-BLOCK-END
