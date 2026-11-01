"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return pts / norms

    def compute_covering_radius(pts):
        """
        Computes the covering radius of the point configuration on the sphere.
        The covering radius is the maximum angular distance from any point on the sphere
        to its nearest neighbor in the configuration.
        This is equivalent to the maximum circumradius of the spherical triangles
        (facets) of the convex hull of the points.
        
        For a spherical triangle with vertices p1, p2, p3:
        1. Compute the circumcenter c of the planar triangle.
        2. If c is inside the spherical triangle (i.e., c is a positive linear combination
           of p1, p2, p3), the circumradius is the angular distance from c to any vertex.
        3. If c is outside, the maximum distance is at the midpoint of the longest edge.
        """
        pts = normalize(pts)
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.pi  # Invalid or degenerate, return large radius
            
        max_radius = 0.0
        
        for simplex in hull.simplices:
            i, j, k = simplex
            p1, p2, p3 = pts[i], pts[j], pts[k]
            
            # Compute circumcenter of the planar triangle p1, p2, p3
            u = p2 - p1
            v = p3 - p1
            uu = np.dot(u, u)
            vv = np.dot(v, v)
            uv = np.dot(u, v)
            
            det = uu * vv - uv * uv
            if abs(det) < 1e-10:
                continue
                
            # Coefficients for circumcenter in basis {p1, u, v}
            a = (0.5 * uu * vv - 0.5 * uv * vv) / det
            b = (0.5 * vv * uu - 0.5 * uu * uv) / det
            
            c_planar = p1 + a * u + b * v
            c_norm = np.linalg.norm(c_planar)
            if c_norm < 1e-10:
                continue
                
            c_sph = c_planar / c_norm
            
            # Check if c_sph is inside the spherical triangle.
            # This is true if c_sph can be written as alpha*p1 + beta*p2 + gamma*p3
            # with alpha, beta, gamma > 0.
            P = np.column_stack([p1, p2, p3])
            try:
                coeffs = np.linalg.solve(P, c_sph)
                if np.all(coeffs > -1e-6):
                    # Circumcenter is inside
                    d_c = np.arccos(np.clip(np.dot(c_sph, p1), -1.0, 1.0))
                    if d_c > max_radius:
                        max_radius = d_c
                else:
                    # Circumcenter is outside
                    # The maximum distance is at the midpoint of the longest edge
                    m12 = (p1 + p2) / np.linalg.norm(p1 + p2)
                    m23 = (p2 + p3) / np.linalg.norm(p2 + p3)
                    m31 = (p3 + p1) / np.linalg.norm(p3 + p1)
                    
                    d_m12 = np.arccos(np.clip(np.dot(m12, p1), -1.0, 1.0))
                    d_m23 = np.arccos(np.clip(np.dot(m23, p2), -1.0, 1.0))
                    d_m31 = np.arccos(np.clip(np.dot(m31, p3), -1.0, 1.0))
                    
                    d_facet = max(d_m12, d_m23, d_m31)
                    if d_facet > max_radius:
                        max_radius = d_facet
            except np.linalg.LinAlgError:
                continue
                
        return max_radius

    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + 5.0**0.5) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_pts = x0.copy()
    current_radius = compute_covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius
    
    # Simulated Annealing Parameters
    # Initial temperature should be larger than typical step improvements
    T = 0.05
    T_min = 1e-6
    cooling_rate = 0.9995
    
    # Perturbation magnitude
    # We perturb a random subset of points by a random amount
    num_perturb = 5
    perturb_scale = 0.1
    
    steps = 0
    last_improve_step = 0
    
    while time.time() < deadline:
        steps += 1
        
        # Create a perturbed copy
        new_pts = current_pts.copy()
        if np.random.random() < 0.1:
            # Occasionally perturb more points with larger scale to jump out of basins
            n_p = np.random.randint(10, 20)
            scale = perturb_scale * 2
        else:
            n_p = num_perturb
            scale = perturb_scale
            
        idxs = np.random.choice(N, n_p, replace=False)
        # Generate random directions on the sphere
        random_dirs = np.random.randn(n_p, 3)
        random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True)
        
        # Apply perturbation: move points in random directions
        # We use a simple addition and renormalization to stay on sphere
        new_pts[idxs] = current_pts[idxs] + scale * random_dirs
        new_pts = normalize(new_pts)
        
        # Evaluate new configuration
        new_radius = compute_covering_radius(new_pts)
        
        # Acceptance criterion
        if new_radius < current_radius:
            # Accept improvement
            current_pts = new_pts
            current_radius = new_radius
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = current_pts.copy()
                last_improve_step = steps
        else:
            # Metropolis criterion
            delta = new_radius - current_radius
            if T > 0:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current_pts = new_pts
                    current_radius = new_radius
        
        # Cooling
        T *= cooling_rate
        
        # Reheating if stuck
        if steps - last_improve_step > 500:
            T = max(T, 0.02)
            # Also try a bigger jump
            current_pts = best_pts + np.random.randn(N, 3) * 0.5
            current_pts = normalize(current_pts)
            current_radius = compute_covering_radius(current_pts)
            last_improve_step = steps
            
        # Adaptive perturb scale: decrease slightly over time
        if steps % 100 == 0:
            perturb_scale *= 0.95
            if perturb_scale < 0.01:
                perturb_scale = 0.01

    # Final polish: simple gradient descent on the best solution found
    # This helps refine the local minimum
    pts = best_pts.copy()
    step_size = 0.01
    for it in range(100):
        if time.time() > deadline:
            break
            
        # Compute gradient approximation using finite differences on active facets?
        # Or just use the previous gradient logic? 
        # To keep it simple and robust, we do a small random search around the best point
        # and accept if better. This is a local hill-climbing polish.
        
        improved = False
        for _ in range(10):
            trial_pts = pts + np.random.randn(N, 3) * 0.01
            trial_pts = normalize(trial_pts)
            trial_radius = compute_covering_radius(trial_pts)
            if trial_radius < current_radius:
                pts = trial_pts
                current_radius = trial_radius
                improved = True
        
        if not improved:
            break
    
    if current_radius < best_radius:
        best_pts = pts
        best_radius = current_radius

    return normalize(best_pts)
# EVOLVE-BLOCK-END
