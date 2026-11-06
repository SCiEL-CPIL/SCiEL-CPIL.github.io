"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s eval / 150s kill

    def compute_covering_radius(points):
        """Compute covering radius using convex hull facets."""
        try:
            hull = ConvexHull(points)
            max_radius = 0.0
            for simplex in hull.simplices:
                face_points = points[simplex]
                a, b, c = face_points
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    continue
                normal = normal / normal_norm
                
                d = 2.0 * (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1]))
                if abs(d) < 1e-12:
                    continue
                
                a2 = np.dot(a, a)
                b2 = np.dot(b, b)
                c2 = np.dot(c, c)
                
                ux = ((a2 * (b[1] * c[2] - c[1] * b[2]) - b2 * (a[1] * c[2] - c[1] * a[2]) + c2 * (a[1] * b[2] - b[1] * a[2])) / d)
                uy = ((a2 * (b[0] * c[2] - c[0] * b[2]) - b2 * (a[0] * c[2] - c[0] * a[2]) + c2 * (a[0] * b[2] - b[0] * a[2])) / d)
                uz = ((a2 * (b[0] * c[1] - c[0] * b[1]) - b2 * (a[0] * c[1] - c[0] * a[1]) + c2 * (a[0] * b[1] - b[0] * a[1])) / d)
                
                circumcenter = np.array([ux, uy, uz])
                cc_norm = np.linalg.norm(circumcenter)
                if cc_norm < 1e-12:
                    continue
                circumcenter = circumcenter / cc_norm
                
                dot_vals = np.clip(np.dot(circumcenter, face_points), -1.0, 1.0)
                angles = np.arccos(dot_vals)
                facet_radius = np.max(angles)
                max_radius = max(max_radius, facet_radius)
            return max_radius
        except Exception:
            return np.pi

    def project_to_sphere(x):
        """Project points onto unit sphere."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return x / norms

    def smooth_covering_surrogate(x, beta=50.0):
        """
        Smooth surrogate for covering radius using soft-min of pairwise distances.
        For each point, compute the minimum distance to all other points.
        Then take the max over points of that minimum distance.
        We use a soft-min and soft-max to make it differentiable.
        """
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        pts = pts / norms
        
        # Pairwise angular distances
        dots = np.clip(pts @ pts.T, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # For each point i, find the soft-min of distances to all other points
        # soft_min_j(d_ij) = -1/beta * log(sum_j exp(-beta * d_ij))
        # We exclude self (d_ii = 0) by setting it to a large value
        dists_filled = dists.copy()
        np.fill_diagonal(dists_filled, 10.0)  # Large value so it doesn't affect min
        
        # Soft min for each row
        # logsumexp trick: -1/beta * log(sum exp(-beta * d))
        # = -1/beta * (max_d + log(sum exp(-beta * (d - max_d))))
        # But we want min, so:
        # min_j d_ij ≈ -1/beta * log(sum_j exp(-beta * d_ij))
        
        neg_beta_dists = -beta * dists_filled
        max_neg = np.max(neg_beta_dists, axis=1, keepdims=True)
        log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_beta_dists - max_neg), axis=1, keepdims=True))
        soft_min_dists = -log_sum_exp / beta  # This is the soft-min of distances
        
        # Now we want the max over i of soft_min_dists[i]
        # soft_max_i(s_i) = 1/beta * log(sum_i exp(beta * s_i))
        beta_s = beta * soft_min_dists
        max_bs = np.max(beta_s, axis=1, keepdims=True)
        log_sum_exp_s = max_bs + np.log(np.sum(np.exp(beta_s - max_bs), axis=1, keepdims=True))
        soft_max_val = log_sum_exp_s / beta
        
        return soft_max_val[0, 0]

    def smooth_covering_gradient(x, beta=50.0):
        """Numerical gradient of the smooth surrogate."""
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = smooth_covering_surrogate(x, beta)
        for i in range(0, len(x), 3):
            for j in range(3):
                x_plus = x.copy()
                x_plus[i + j] += eps
                f_plus = smooth_covering_surrogate(x_plus, beta)
                grad[i + j] = (f_plus - f0) / eps
        return grad

    # Initialize with Fibonacci spiral points
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_pts = pts.copy()
    best_radius = compute_covering_radius(best_pts)

    # Phase 1: Optimize smooth surrogate with L-BFGS
    # This is a structural change: using a smooth differentiable surrogate 
    # for the covering radius, rather than non-smooth potential energy.
    
    def objective_surrogate(x):
        return smooth_covering_surrogate(x, beta=50.0)

    def gradient_surrogate(x):
        return smooth_covering_gradient(x, beta=50.0)

    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective_surrogate,
            x0,
            jac=gradient_surrogate,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50}
        )
        if result.success or result.fun < objective_surrogate(x0):
            new_pts = result.x.reshape(N, 3)
            new_pts = project_to_sphere(new_pts)
            new_radius = compute_covering_radius(new_pts)
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = new_pts
    except Exception:
        pass

    # Phase 2: Fine polish with facet-based forces
    def compute_facet_forces(points):
        """
        Compute forces on each vertex based on the facet circumcenters.
        """
        try:
            hull = ConvexHull(points)
            forces = np.zeros_like(points)
            
            for simplex in hull.simplices:
                face_points = points[simplex]
                a, b, c = face_points
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    continue
                normal = normal / normal_norm
                
                d = 2.0 * (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1]))
                if abs(d) < 1e-12:
                    continue
                
                a2 = np.dot(a, a)
                b2 = np.dot(b, b)
                c2 = np.dot(c, c)
                
                ux = ((a2 * (b[1] * c[2] - c[1] * b[2]) - b2 * (a[1] * c[2] - c[1] * a[2]) + c2 * (a[1] * b[2] - b[1] * a[2])) / d)
                uy = ((a2 * (b[0] * c[2] - c[0] * b[2]) - b2 * (a[0] * c[2] - c[0] * a[2]) + c2 * (a[0] * b[2] - b[0] * a[2])) / d)
                uz = ((a2 * (b[0] * c[1] - c[0] * b[1]) - b2 * (a[0] * c[1] - c[0] * a[1]) + c2 * (a[0] * b[1] - b[0] * a[1])) / d)
                
                circumcenter = np.array([ux, uy, uz])
                cc_norm = np.linalg.norm(circumcenter)
                if cc_norm < 1e-12:
                    continue
                circumcenter = circumcenter / cc_norm
                
                dot_vals = np.clip(np.dot(circumcenter, face_points), -1.0, 1.0)
                angles = np.arccos(dot_vals)
                facet_radius = np.max(angles)
                
                weight = facet_radius / (np.pi / 6.0)
                
                for idx in simplex:
                    vertex = points[idx]
                    diff = circumcenter - vertex
                    diff_norm = np.linalg.norm(diff)
                    if diff_norm > 1e-10:
                        forces[idx] += weight * diff / diff_norm
            
            return forces
        except Exception:
            return np.zeros_like(points)

    pts = best_pts.copy()
    lr = 0.01
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
        if best_radius < 0.1971:
            break

        forces = compute_facet_forces(pts)
        
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-10, 1.0, force_norms)
        forces = forces / force_norms
        
        new_pts = pts + lr * forces
        new_pts = project_to_sphere(new_pts)
        
        new_radius = compute_covering_radius(new_pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts
            pts = new_pts
            lr *= 0.995
        else:
            lr *= 0.9

    return best_pts
# EVOLVE-BLOCK-END
