"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 150

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Generate multiple diverse initial configurations
    def fibonacci_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_points():
        pts = np.random.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        return pts / norms

    def grid_based():
        # Create a roughly uniform grid on sphere
        pts = []
        # Use golden angle spiral with different parameters
        for offset in [0.0, 0.5, 0.1, 0.9]:
            i = np.arange(N) + offset
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset * 10
            r = np.sqrt(1.0 - z * z)
            pts.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
        return np.array(pts)  # shape (4, N, 3)

    initial_configs = [fibonacci_spiral()]
    np.random.seed(42)
    for _ in range(3):
        initial_configs.append(random_points())
    
    # 2. Compute energy and gradient
    def compute_energy(pts):
        # pts: (N, 3)
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)  # Avoid log(0)
        
        # Energy = sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
        # Only upper triangle
        iu = np.triu_indices(N, 1)
        log_dists = np.log(dist[iu])
        energy = -math.fsum(log_dists.tolist())
        return energy

    def compute_grad(pts):
        # pts: (N, 3)
        # Gradient of energy w.r.t. pts[i]
        # E = sum_{j != i} log(1/|x_i - x_j|) = -sum_{j != i} log(|x_i - x_j|)
        # dE/dx_i = -sum_{j != i} d/dx_i [log(|x_i - x_j|)]
        #          = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        #          = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
        
        grad = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = pts[j] - pts[i]
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    continue
                grad[i] += diff / (d * d)
        
        # Project gradient to tangent space: grad_t = grad - (grad . x_i) * x_i
        for i in range(N):
            dot = np.dot(grad[i], pts[i])
            grad[i] -= dot * pts[i]
        
        return grad

    def project_to_sphere(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        return pts / norms

    # 3. Riemannian gradient descent with adaptive step size
    def optimize(pts, max_iters=5000, tol=1e-12):
        best_pts = pts.copy()
        best_energy = compute_energy(pts)
        step_size = 0.01
        min_step = 1e-6
        max_step = 0.1
        patience = 0
        max_patience = 200

        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            grad = compute_grad(pts)
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < tol:
                break
            
            # Adaptive step size with line search
            improved = False
            for _ in range(20):
                new_pts = pts - step_size * grad
                new_pts = project_to_sphere(new_pts)
                
                new_energy = compute_energy(new_pts)
                
                if new_energy < best_energy:
                    best_pts = new_pts.copy()
                    best_energy = new_energy
                    pts = new_pts
                    improved = True
                    patience = 0
                    # Increase step size slightly
                    step_size = min(step_size * 1.05, max_step)
                    break
                else:
                    step_size *= 0.8
                    if step_size < min_step:
                        break
            
            if not improved:
                patience += 1
                # Try smaller step or restart
                if patience > max_patience:
                    # Perturb and restart
                    if time.time() > deadline - 5:
                        break
                    perturbation = np.random.randn(*pts.shape) * 0.01
                    new_pts = project_to_sphere(pts + perturbation)
                    new_energy = compute_energy(new_pts)
                    if new_energy < best_energy:
                        pts = new_pts
                        best_pts = new_pts.copy()
                        best_energy = new_energy
                        patience = 0
                        step_size = 0.01
                    else:
                        break
            else:
                if it % 100 == 0 and it > 0:
                    # Check convergence
                    pass
        
        return best_pts, best_energy

    # 4. Optimize each initial configuration
    best_overall = None
    best_energy_overall = float('inf')

    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
        
        try:
            optimized_pts, energy = optimize(init_pts)
            if energy < best_energy_overall:
                best_energy_overall = energy
                best_overall = optimized_pts
        except Exception:
            continue

    # 5. Final high-precision polish
    if best_overall is not None and time.time() < deadline - 5:
        # Run a final refinement with smaller steps
        best_overall, final_energy = optimize(best_overall, max_iters=2000, tol=1e-14)
        if final_energy < best_energy_overall:
            best_energy_overall = final_energy

    if best_overall is None:
        best_overall = fibonacci_spiral()

    # Ensure all points are on unit sphere and finite
    best_overall = project_to_sphere(best_overall)
    if not np.all(np.isfinite(best_overall)):
        best_overall = fibonacci_spiral()

    return best_overall
# EVOLVE-BLOCK-END
