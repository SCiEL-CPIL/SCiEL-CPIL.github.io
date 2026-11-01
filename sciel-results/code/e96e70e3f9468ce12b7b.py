"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def get_hull_and_radius(pts):
        """
        Computes the covering radius exactly via convex hull facets.
        The covering radius is the maximum circumradius of the spherical triangles (facets) 
        formed by the points.
        """
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            for simplex in hull.simplices:
                p1, p2, p3 = pts[simplex]
                # Calculate circumcenter of the triangle on the sphere
                # The circumcenter c is proportional to (p1 + p2 + p3) only if it's an equilateral triangle? No.
                # For a spherical triangle, the circumcenter is the intersection of the great circles.
                # Alternatively, the circumradius R is related to the side lengths a, b, c.
                # Spherical law of cosines: cos(c) = cos(a)cos(b) + sin(a)sin(b)cos(C)
                # There is a formula for circumradius R:
                # cot(R) = (cot(A/2) + cot(B/2) + cot(C/2)) / 3 ? No.
                # Simpler: The circumcenter of a spherical triangle is the normalized vector 
                # n = (p1 - p2) x (p2 - p3) ... wait, that's the normal.
                # The circumcenter is the point on the sphere equidistant from p1, p2, p3.
                # It lies in the plane defined by p1, p2, p3.
                # Let c = alpha*p1 + beta*p2 + gamma*p3.
                # |c|^2 = 1. c . p1 = c . p2 = c . p3 = cos(R).
                # This is a linear system for alpha, beta, gamma.
                
                # Construct matrix A where rows are p1, p2, p3.
                # We want c such that A @ c = [cos(R), cos(R), cos(R)]^T ? No, c is a vector in R3.
                # Let c = x*p1 + y*p2 + z*p3.
                # c . p1 = x + y*p1.p2 + z*p1.p3 = cosR
                # c . p2 = x*p2.p1 + y + z*p2.p3 = cosR
                # c . p3 = x*p3.p1 + y*p3.p2 + z = cosR
                # And |c|^2 = 1.
                
                # Let's solve for the coefficients.
                G = np.array([
                    [1.0, np.dot(p1, p2), np.dot(p1, p3)],
                    [np.dot(p2, p1), 1.0, np.dot(p2, p3)],
                    [np.dot(p3, p1), np.dot(p3, p2), 1.0]
                ])
                
                # The vector of unknowns is [x, y, z].
                # The RHS is [cosR, cosR, cosR].
                # But we also have the constraint |c|^2=1.
                # Actually, the circumcenter of the planar triangle formed by p1,p2,p3 (in 3D) is NOT the spherical circumcenter.
                # The spherical circumcenter is the intersection of the sphere with the line passing through the origin and the circumcenter of the planar triangle? No.
                # The spherical circumcenter is the normalized vector c such that c is in the span of p1, p2, p3 and c.p1 = c.p2 = c.p3.
                
                # Let's solve G @ v = [1, 1, 1] is not right.
                # We want c . p_i = k.
                # c = sum w_i p_i.
                # c . p_j = sum w_i (p_i . p_j) = k delta_ij? No, k for all j.
                # So G @ w = [k, k, k].
                # w = G^{-1} [k, k, k].
                # Then c = sum w_i p_i = P @ w where P is the matrix with columns p1, p2, p3.
                # c = P @ G^{-1} @ [k, k, k].
                # Let v = G^{-1} @ [1, 1, 1]. Then c = k * P @ v.
                # We need |c| = 1.
                # |c|^2 = k^2 * (P @ v) . (P @ v) = 1.
                # k = 1 / || P @ v ||.
                # Then cos(R) = c . p1 = k * (P @ v) . p1.
                
                try:
                    v = np.linalg.solve(G, np.array([1.0, 1.0, 1.0]))
                    P_mat = np.column_stack([p1, p2, p3])
                    c_vec = P_mat @ v
                    norm_c = np.linalg.norm(c_vec)
                    if norm_c < 1e-10:
                        # Degenerate triangle, likely near antipodal or coincident
                        # Fallback to max distance to vertices
                        d1 = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
                        d2 = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
                        d3 = np.arccos(np.clip(np.dot(p3, p1), -1, 1))
                        max_d = np.max([d1, d2, d3]) / 2.0 # Rough estimate
                        if max_d > max_rad:
                            max_rad = max_d
                        continue
                    
                    c_norm = c_vec / norm_c
                    cos_R = np.dot(c_norm, p1)
                    cos_R = np.clip(cos_R, -1.0, 1.0)
                    R = np.arccos(cos_R)
                    if R > max_rad:
                        max_rad = R
                except np.linalg.LinAlgError:
                    # Degenerate
                    pass
            
            return max_rad
        except Exception:
            # If hull fails, fallback to grid (slow but safe)
            # This should be rare
            tree = cKDTree(pts)
            # Sparse grid
            lat = np.linspace(-np.pi/2, np.pi/2, 30)
            lon = np.linspace(0, 2*np.pi, 60)
            lat, lon = np.meshgrid(lat, lon)
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dists, _ = tree.query(grid_points)
            angles = 2.0 * np.arcsin(np.clip(dists / 2.0, 0.0, 1.0))
            return np.max(angles)

    # Objective wrapper
    def objective(pts):
        return get_hull_and_radius(pts)
    
    # Simulated Annealing
    current_points = points.copy()
    current_score = objective(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    alpha = 0.999  # Cooling rate
    step_size = 0.1
    
    while time.time() < deadline:
        # Generate a perturbation
        # Move a random point in a random direction
        idx = np.random.randint(N)
        rand_dir = np.random.randn(3)
        rand_dir /= np.linalg.norm(rand_dir)
        
        new_points = current_points.copy()
        new_points[idx] = new_points[idx] + step_size * rand_dir
        new_points[idx] /= np.linalg.norm(new_points[idx])
        
        new_score = objective(new_points)
        
        # Acceptance criterion
        if new_score < current_score or np.random.rand() < np.exp((current_score - new_score) / T):
            current_points = new_points
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_points = current_points.copy()
                
        T *= alpha
        if T < T_min:
            break
            
    # Final polish: Local search on the best found
    # Use a simple gradient-free local search (Nelder-Mead is too slow for 150 vars, 
    # so we do a random hill climbing with decreasing step size)
    polish_points = best_points.copy()
    polish_score = best_score
    step = 0.05
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            for _ in range(5):
                rand_dir = np.random.randn(3)
                rand_dir /= np.linalg.norm(rand_dir)
                old_p = polish_points[i]
                new_p = polish_points[i] + step * rand_dir
                new_p /= np.linalg.norm(new_p)
                polish_points[i] = new_p
                new_score = objective(polish_points)
                if new_score < polish_score:
                    polish_score = new_score
                    improved = True
                else:
                    polish_points[i] = old_p
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break
                
    if polish_score < best_score:
        best_points = polish_points
        best_score = polish_score

    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
