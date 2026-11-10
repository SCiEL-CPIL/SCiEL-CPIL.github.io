import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 610  # 10s margin
    
    best_score = -1.0
    best_points = np.zeros((n, d))
    
    # Helper to calculate score based on squared distances
    def get_score(pts):
        # Calculate squared pairwise distances
        dists = pdist(pts, metric='sqeuclidean')
        if len(dists) == 0:
            return 0.0
        d_min_sq = np.min(dists)
        d_max_sq = np.max(dists)
        
        # If points coincide or are identical
        if d_max_sq < 1e-12:
            return 0.0
        
        # Metric is (d_min / d_max)^2 = d_min_sq / d_max_sq
        return d_min_sq / d_max_sq

    # Smooth objective for L-BFGS-B: Minimize repulsive potential on the sphere
    # This maximizes the minimum distance while keeping points on the sphere.
    def repulsive_obj(x, p=2.0, lambda_sphere=100.0):
        pts = x.reshape(n, d)
        
        # Sphere constraint penalty
        norms = np.linalg.norm(pts, axis=1)
        sphere_pen = lambda_sphere * np.sum((norms - 1.0)**2)
        
        # Repulsive potential
        dists = pdist(pts)
        # Avoid division by zero
        dists = np.maximum(dists, 1e-6)
        
        # Sum of inverse powers
        V = np.sum(1.0 / np.power(dists, p))
        
        return V + sphere_pen

    def grad_repulsive(x, p=2.0, lambda_sphere=100.0):
        pts = x.reshape(n, d)
        g = np.zeros_like(pts)
        
        # Sphere gradient
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms_safe = np.maximum(norms, 1e-6)
        sphere_grad = 2.0 * lambda_sphere * (norms - 1.0) * pts / norms_safe
        g += sphere_grad
        
        # Repulsive gradient
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_mat = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(n, dtype=bool)
        
        dists_mat_safe = np.where(mask, dists_mat, 1.0)
        factor = np.where(mask, 1.0 / np.power(dists_mat_safe, p+2), 0.0)
        
        repulsive_grad = -p * np.sum(diff * factor[:, :, np.newaxis], axis=1)
        
        g += repulsive_grad
        
        return g.flatten()

    def run_l_bfgs(start_pts, p=2.0, lambda_sphere=100.0):
        x0 = start_pts.flatten()
        
        def obj(x):
            return repulsive_obj(x, p=p, lambda_sphere=lambda_sphere)
            
        def grad(x):
            return grad_repulsive(x, p=p, lambda_sphere=lambda_sphere)

        res = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        best_pts = res.x.reshape(n, d)
        return best_pts

    # Initial population
    def get_initial_points():
        # Icosahedron + 2 random on sphere
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        
        # 2 random points on sphere
        v1 = np.random.randn(3)
        v1 /= np.linalg.norm(v1)
        v2 = np.random.randn(3)
        v2 /= np.linalg.norm(v2)
        
        pts = np.vstack([ico, [v1, v2]])
        return pts

    def get_random_points():
        pts = np.random.randn(n, d)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    # Simulated Annealing to escape local minima
    def simulated_annealing(pts, temp_start=0.1, temp_end=0.01, iterations=500):
        current_pts = pts.copy()
        current_score = get_score(current_pts)
        
        for i in range(iterations):
            if time.time() > deadline - 5:
                break
            
            temp = temp_start * (temp_end / temp_start) ** (i / iterations)
            
            # Perturb points
            noise_scale = temp * 0.5
            new_pts = current_pts + np.random.randn(n, d) * noise_scale
            
            # Project back to sphere to maintain scale constraint roughly
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            new_score = get_score(new_pts)
            
            # Accept if better or with probability exp((new - current) / temp)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
            else:
                delta = new_score - current_score
                prob = np.exp(delta / temp)
                if np.random.rand() < prob:
                    current_pts = new_pts
                    current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
                
        return current_pts, current_score

    # Run multiple restarts with different strategies
    num_restarts = 0
    while time.time() < deadline - 10:
        # Choose initialization strategy
        if num_restarts % 3 == 0:
            start_pts = get_initial_points()
        else:
            start_pts = get_random_points()
        
        # Run L-BFGS-B to find a local optimum
        try:
            local_best = run_l_bfgs(start_pts)
            local_score = get_score(local_best)
            
            if local_score > best_score:
                best_score = local_score
                best_points = local_best
        except Exception:
            pass
        
        num_restarts += 1
        
        # If we have time, try Simulated Annealing from the best point found so far
        # This helps escape local minima
        if time.time() < deadline - 60:
            try:
                sa_pts, sa_score = simulated_annealing(best_points, temp_start=0.2, temp_end=0.001, iterations=200)
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_pts
            except Exception:
                pass

        # If we are very good, break?
        if best_score > 0.95:
            break

    return best_points
