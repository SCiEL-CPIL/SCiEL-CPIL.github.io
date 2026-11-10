# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin of 20 seconds

    n = 16

    def compute_score(points):
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_and_grad(x, beta):
        """
        Smooth surrogate: maximize soft_min / soft_max using log-sum-exp.
        Returns (objective_value, gradient) where we minimize negative of the ratio.
        """
        points = x.reshape(n, 2)
        d = pdist(points)
        
        # Avoid zero distances
        d_safe = np.maximum(d, 1e-12)
        
        # Soft min via log-sum-exp
        d_min_shift = d_safe - np.min(d_safe)
        log_sum_exp_min = np.log(np.sum(np.exp(-beta * d_min_shift))) + np.min(d_safe)
        soft_min = log_sum_exp_min / beta
        
        # Soft max via log-sum-exp
        d_max_shift = d_safe - np.max(d_safe)
        log_sum_exp_max = np.log(np.sum(np.exp(beta * d_max_shift))) + np.max(d_safe)
        soft_max = log_sum_exp_max / beta
        
        if soft_max < 1e-12:
            return 0.0, np.zeros_like(x)
        
        ratio = soft_min / soft_max
        score = ratio * ratio
        
        # Weights for soft min
        exp_min = np.exp(-beta * d_min_shift)
        w_min = exp_min / np.sum(exp_min)
        
        # Weights for soft max
        exp_max = np.exp(beta * d_max_shift)
        w_max = exp_max / np.sum(exp_max)
        
        # Build gradients using vectorized approach
        idx_i, idx_j = np.triu_indices(n, k=1)
        diffs = points[idx_i] - points[idx_j]
        dists = d_safe
        
        # Unit vectors
        unit_vecs = diffs / dists[:, np.newaxis]
        
        # Accumulate gradients
        grad_soft_min = np.zeros((n, 2))
        grad_soft_max = np.zeros((n, 2))
        
        w_min_scaled = w_min / beta
        w_max_scaled = w_max / beta
        
        np.add.at(grad_soft_min, idx_i, w_min_scaled[:, np.newaxis] * unit_vecs)
        np.add.at(grad_soft_min, idx_j, -w_min_scaled[:, np.newaxis] * unit_vecs)
        
        np.add.at(grad_soft_max, idx_i, w_max_scaled[:, np.newaxis] * unit_vecs)
        np.add.at(grad_soft_max, idx_j, -w_max_scaled[:, np.newaxis] * unit_vecs)
        
        grad_soft_min = grad_soft_min.flatten()
        grad_soft_max = grad_soft_max.flatten()
        
        grad_score = 2 * ratio / soft_max * (grad_soft_min - ratio * grad_soft_max)
        
        return -score, -grad_score

    def optimize_from(points_init, beta_schedule=None, max_iters=200):
        x0 = points_init.flatten()
        
        if beta_schedule is None:
            beta_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0, 5000.0, 10000.0]
        
        best_x = x0.copy()
        best_score = compute_score(points_init)
        
        for beta in beta_schedule:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    lambda x, b=beta: objective_and_grad(x, b)[0],
                    x0,
                    jac=lambda x, b=beta: objective_and_grad(x, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': max_iters, 'gtol': 1e-10, 'ftol': 1e-12}
                )
                
                x0 = res.x
                points = x0.reshape(n, 2)
                score = compute_score(points)
                
                if score > best_score:
                    best_score = score
                    best_x = x0.copy()
            except Exception:
                continue
        
        # Final local search with small random perturbations
        if time.time() < deadline - 10:
            np.random.seed(42)
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                perturbation = np.random.randn(n, 2) * 0.005
                x_perturbed = best_x.reshape(n, 2) + perturbation
                x0_pert = x_perturbed.flatten()
                try:
                    res = minimize(
                        lambda x, b=10000.0: objective_and_grad(x, b)[0],
                        x0_pert,
                        jac=lambda x, b=10000.0: objective_and_grad(x, b)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 80, 'gtol': 1e-10, 'ftol': 1e-12}
                    )
                    points = res.x.reshape(n, 2)
                    score = compute_score(points)
                    if score > best_score:
                        best_score = score
                        best_x = res.x.copy()
                except Exception:
                    continue
        
        return best_x.reshape(n, 2), best_score

    def regular_polygon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def concentric_rings(r_inn, n_outer, n_inner, phase_inner=0):
        """Create concentric ring configuration."""
        angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
        angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + phase_inner
        pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        pts_inner = r_inn * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        return np.vstack([pts_outer, pts_inner])

    def two_circles(r1, r2, n1, n2, phase=0):
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + phase
        pts1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
        pts2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        return np.vstack([pts1, pts2])

    def random_points(seed):
        np.random.seed(seed)
        return np.random.randn(n, 2)

    def radial_warp(points, alpha):
        """Apply radial distortion: r' = r^(1+alpha) for r < 1, preserving direction."""
        pts = points.copy()
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        r_safe = np.maximum(r, 1e-12)
        r_new = np.where(r < 1.0, np.power(r_safe, 1.0 + alpha), r)
        mask = r < 1.0
        if np.any(mask):
            pts[mask, 0] = pts[mask, 0] * (r_new[mask] / r_safe[mask])
            pts[mask, 1] = pts[mask, 1] * (r_new[mask] / r_safe[mask])
        return pts

    def lattice_with_warp(alpha):
        """Create 4x4 lattice and apply radial warp."""
        coords = []
        for i in range(4):
            for j in range(4):
                coords.append([i - 1.5, j - 1.5])
        pts = np.array(coords, dtype=float)
        max_r = np.max(np.sqrt(pts[:, 0]**2 + pts[:, 1]**2))
        pts = pts / max_r
        return radial_warp(pts, alpha)

    def fibonacci_spiral(n_pts):
        golden_angle = np.pi * (3 - np.sqrt(5))
        angles = golden_angle * np.arange(n_pts)
        radii = np.sqrt(np.arange(n_pts) / n_pts)
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    def three_rings(r1, r2, r3, n1, n2, n3, phase1=0, phase2=0, phase3=0):
        a1 = np.linspace(0, 2*np.pi, n1, endpoint=False) + phase1
        a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + phase2
        a3 = np.linspace(0, 2*np.pi, n3, endpoint=False) + phase3
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            r2 * np.column_stack([np.cos(a2), np.sin(a2)]),
            r3 * np.column_stack([np.cos(a3), np.sin(a3)])
        ])
        return pts

    def hexagonal(n_pts):
        pts = []
        pts.append([0, 0])
        for k in range(6):
            angle = k * np.pi / 3
            pts.append([np.cos(angle), np.sin(angle)])
        for k in range(12):
            angle = k * np.pi / 6
            r = np.sqrt(3)
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        pts = np.array(pts[:n_pts])
        max_r = np.max(np.sqrt(pts[:,0]**2 + pts[:,1]**2))
        return pts / max_r

    def polar_symmetric(n_shells, points_per_shell, phases):
        """
        Generate points with rotational symmetry.
        n_shells: list of radii
        points_per_shell: list of how many points on each shell
        phases: list of phase offsets for each shell
        """
        pts = []
        for r, npoints, phase in zip(n_shells, points_per_shell, phases):
            angles = np.linspace(0, 2*np.pi, npoints, endpoint=False) + phase
            shell_pts = r * np.column_stack([np.cos(angles), np.sin(angles)])
            pts.append(shell_pts)
        return np.vstack(pts)

    # Generate diverse initializations
    initializations = []
    
    # Regular polygon (16-gon)
    initializations.append(regular_polygon())
    
    # Concentric rings with various inner radii and point distributions
    for r_inn in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
        for n_outer, n_inner in [(11, 5), (12, 4), (10, 6), (8, 8), (14, 2)]:
            for ph in [0, np.pi/n_inner]:
                initializations.append(concentric_rings(r_inn, n_outer, n_inner, ph))
    
    # Two circles with various radii and phases
    for r1, r2, n1, n2, phase in [
        (1.0, 0.5, 8, 8, 0),
        (1.0, 0.5, 8, 8, np.pi/8),
        (1.0, 0.5, 8, 8, np.pi/16),
        (1.0, 0.6, 10, 6, 0),
        (1.0, 0.6, 10, 6, np.pi/10),
        (1.0, 0.4, 12, 4, 0),
        (1.0, 0.4, 12, 4, np.pi/12),
        (1.0, 0.7, 6, 10, np.pi/6),
        (1.0, 0.3, 14, 2, 0),
        (1.0, 0.55, 9, 7, np.pi/9),
        (1.0, 0.65, 11, 5, np.pi/11),
    ]:
        initializations.append(two_circles(r1, r2, n1, n2, phase))
    
    # Lattice with radial warp for various alpha values
    for alpha in [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.6]:
        initializations.append(lattice_with_warp(alpha))
    
    # Random initializations
    for seed in range(25):
        initializations.append(random_points(seed))
    
    # Grid-based with perturbations
    for seed in range(10):
        np.random.seed(seed)
        side = int(np.ceil(np.sqrt(n)))
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) < n:
                    pts.append([i, j])
        pts = np.array(pts[:n], dtype=float)
        pts += np.random.randn(n, 2) * 0.3
        initializations.append(pts)
    
    # Fibonacci spiral
    initializations.append(fibonacci_spiral(n))
    
    # Three concentric rings
    for r1, r2, r3, n1, n2, n3 in [
        (1.0, 0.6, 0.2, 8, 5, 3),
        (1.0, 0.5, 0.15, 9, 5, 2),
        (1.0, 0.65, 0.3, 7, 6, 3),
        (1.0, 0.55, 0.25, 10, 4, 2),
        (1.0, 0.7, 0.35, 6, 6, 4),
        (1.0, 0.45, 0.1, 12, 3, 1),
    ]:
        for ph in [0, np.pi/n1, np.pi/n2]:
            initializations.append(three_rings(r1, r2, r3, n1, n2, n3, ph, ph*2, ph*3))
    
    # Hexagonal packing inspired
    initializations.append(hexagonal(n))
    
    # Polar symmetric configurations
    for shells, per_shell, phases in [
        ([1.0, 0.6, 0.2], [8, 6, 2], [0, 0, 0]),
        ([1.0, 0.6, 0.2], [8, 6, 2], [0, np.pi/6, 0]),
        ([1.0, 0.5, 0.3], [10, 4, 2], [0, 0, 0]),
        ([1.0, 0.5, 0.3], [10, 4, 2], [0, np.pi/4, 0]),
        ([1.0, 0.7, 0.35], [6, 6, 4], [0, 0, 0]),
        ([1.0, 0.7, 0.35], [6, 6, 4], [0, np.pi/6, 0]),
        ([1.0, 0.4, 0.2], [12, 3, 1], [0, 0, 0]),
        ([1.0, 0.4, 0.2], [12, 3, 1], [0, np.pi/3, 0]),
        ([1.0, 0.55, 0.25], [9, 5, 2], [0, 0, 0]),
        ([1.0, 0.55, 0.25], [9, 5, 2], [0, np.pi/5, 0]),
    ]:
        initializations.append(polar_symmetric(shells, per_shell, phases))
    
    best_points = None
    best_score = -1.0
    n_inits = len(initializations)
    
    # Track time per initialization to allocate budget
    time_per_init = (deadline - time.time()) / n_inits
    
    for i, init in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        # Limit time per initialization
        local_deadline = time.time() + max(5, min(time_per_init, 40))
        old_deadline = deadline
        deadline = local_deadline
        
        points, score = optimize_from(init)
        
        deadline = old_deadline
        
        if score > best_score:
            best_score = score
            best_points = points
    
    if best_points is None:
        return regular_polygon()
    
    # Final polish with higher precision
    if time.time() < deadline - 30:
        points, score = optimize_from(best_points, beta_schedule=[5000.0, 10000.0, 20000.0], max_iters=400)
        if score > best_score:
            best_points = points
    
    return best_points


# EVOLVE-BLOCK-END
