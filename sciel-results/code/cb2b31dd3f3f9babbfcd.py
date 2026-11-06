"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill

    # 1. Generate a high-resolution grid on the sphere for the covering energy
    # We use a Fibonacci spiral for the grid points to ensure uniform distribution.
    n_grid = 20000
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    grid_norms = np.linalg.norm(grid_pts, axis=1, keepdims=True)
    grid_pts /= grid_norms

    def compute_covering_radius(pts):
        """
        Compute the exact covering radius in degrees using ConvexHull.
        """
        try:
            hull = ConvexHull(pts)
            max_r_rad = 0.0
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                v1 = p0 - p1
                v2 = p0 - p2
                c_vec = np.cross(v1, v2)
                if np.linalg.norm(c_vec) < 1e-10:
                    continue
                c_vec /= np.linalg.norm(c_vec)
                centroid = (p0 + p1 + p2) / 3.0
                centroid /= np.linalg.norm(centroid)
                if np.dot(c_vec, centroid) < 0:
                    c_vec = -c_vec
                n1 = np.cross(p1, p2)
                n2 = np.cross(p2, p0)
                n3 = np.cross(p0, p1)
                if np.dot(centroid, n1) < 0:
                    n1 = -n1
                if np.dot(centroid, n2) < 0:
                    n2 = -n2
                if np.dot(centroid, n3) < 0:
                    n3 = -n3
                if np.dot(c_vec, n1) >= -1e-8 and np.dot(c_vec, n2) >= -1e-8 and np.dot(c_vec, n3) >= -1e-8:
                    dist = np.arccos(np.clip(np.dot(c_vec, p0), -1.0, 1.0))
                    if dist > max_r_rad:
                        max_r_rad = dist
            return np.degrees(max_r_rad)
        except Exception:
            return 180.0

    def soft_max_energy_and_grad(x, beta=30.0):
        """
        Compute smooth soft-max covering energy and gradient.
        x: flattened array of N*3 points.
        """
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms

        dots = grid_pts @ pts.T  # (n_grid, N)
        # Soft max over points for each grid point
        max_dots_raw = np.max(dots, axis=1)
        exp_dots = np.exp(beta * dots)
        sum_exp = np.sum(exp_dots, axis=1, keepdims=True)
        probs = exp_dots / sum_exp  # (n_grid, N)
        soft_max_dots = np.sum(probs * dots, axis=1)  # (n_grid,)

        # Energy: sum of exp(beta * (1 - soft_max_dot))
        vals = np.exp(beta * (1.0 - soft_max_dots))
        energy = np.sum(vals)

        # Gradient
        # dE/dp_j = sum_k vals[k] * (-beta) * d(soft_max_dot_k)/dp_j
        # d(soft_max_dot_k)/dp_j = sum_i probs[k,i] * d(grid_k . p_i)/dp_j
        # If i == j, d(grid_k . p_j)/dp_j = grid_k
        # If i != j, 0
        # So d(soft_max_dot_k)/dp_j = probs[k,j] * grid_k
        # Thus dE/dp_j = -beta * sum_k vals[k] * probs[k,j] * grid_k

        # grad_j = -beta * (grid_pts.T @ (vals * probs[:, j]))
        # We need to compute this for all j efficiently.
        # weighted_probs = probs * vals[:, None]  # (n_grid, N)
        # grad = -beta * (grid_pts.T @ weighted_probs)  # (3, N)
        weighted_probs = probs * vals[:, None]
        grad_flat = -beta * (grid_pts.T @ weighted_probs).flatten()

        # Project gradient onto tangent space of the sphere
        # grad_tangent = grad - (grad . p) * p
        grad_pts = grad_flat.reshape(N, 3)
        grad_tangent = grad_pts - np.sum(grad_pts * pts, axis=1, keepdims=True) * pts
        grad_tangent_flat = grad_tangent.flatten()

        return energy, grad_tangent_flat

    # Initial best: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_pts = fib_pts
    best_cr = compute_covering_radius(fib_pts)

    # L-BFGS-B optimization from Fibonacci spiral
    x0 = fib_pts.flatten()
    try:
        res = minimize(
            soft_max_energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.success or res.fun < np.inf:
            pts = res.x.reshape(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            cr = compute_covering_radius(pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts
    except Exception:
        pass

    # Final refinement: Facet-driven local search
    # Explicitly move points to reduce the largest circumradius of the convex hull facets
    if time.time() < deadline - 5:
        current_pts = best_pts.copy()
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            try:
                hull = ConvexHull(current_pts)
                # Find the facet with the largest circumradius
                max_r_rad = 0.0
                max_face_idx = -1
                max_circumcenter = None
                for idx, face in enumerate(hull.simplices):
                    p0, p1, p2 = current_pts[face[0]], current_pts[face[1]], current_pts[face[2]]
                    v1 = p0 - p1
                    v2 = p0 - p2
                    c_vec = np.cross(v1, v2)
                    if np.linalg.norm(c_vec) < 1e-10:
                        continue
                    c_vec /= np.linalg.norm(c_vec)
                    centroid = (p0 + p1 + p2) / 3.0
                    centroid /= np.linalg.norm(centroid)
                    if np.dot(c_vec, centroid) < 0:
                        c_vec = -c_vec
                    n1 = np.cross(p1, p2)
                    n2 = np.cross(p2, p0)
                    n3 = np.cross(p0, p1)
                    if np.dot(centroid, n1) < 0:
                        n1 = -n1
                    if np.dot(centroid, n2) < 0:
                        n2 = -n2
                    if np.dot(centroid, n3) < 0:
                        n3 = -n3
                    if np.dot(c_vec, n1) >= -1e-8 and np.dot(c_vec, n2) >= -1e-8 and np.dot(c_vec, n3) >= -1e-8:
                        dist = np.arccos(np.clip(np.dot(c_vec, p0), -1.0, 1.0))
                        if dist > max_r_rad:
                            max_r_rad = dist
                            max_face_idx = idx
                            max_circumcenter = c_vec
                if max_face_idx == -1:
                    break
                # Move the vertices of the worst facet slightly towards the circumcenter
                face = hull.simplices[max_face_idx]
                for idx in face:
                    p = current_pts[idx]
                    # Move p slightly towards the circumcenter
                    step = 0.01
                    new_p = p + step * (max_circumcenter - p)
                    new_p /= np.linalg.norm(new_p)
                    current_pts[idx] = new_p
                cr_new = compute_covering_radius(current_pts)
                if cr_new < best_cr:
                    best_cr = cr_new
                    best_pts = current_pts
                else:
                    # Revert if not improved
                    current_pts = best_pts.copy()
            except Exception:
                break
        best_pts = current_pts

    return best_pts
# EVOLVE-BLOCK-END
