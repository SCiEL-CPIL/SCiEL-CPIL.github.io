import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def fibonacci_sphere(n, seed=42):
    """Generate n points on a sphere using Fibonacci lattice."""
    rng = np.random.default_rng(seed)
    golden_angle = np.pi * (3 - np.sqrt(5))
    points = np.zeros((n, D))
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        r = np.sqrt(max(0, 1 - z**2))
        theta = golden_angle * i + rng.uniform(0, 2*np.pi) * 0.01
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)
        points[i, 2] = z
    return points

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d)

def normalize_points(points):
    """
    Center and normalize points to unit mean squared norm.
    This removes translational and scale invariance.
    """
    points = points - np.mean(points, axis=0)
    mean_norm_sq = np.mean(np.sum(points**2, axis=1))
    if mean_norm_sq > 1e-12:
        points = points / np.sqrt(mean_norm_sq)
    return points

def objective_and_grad(x, tau=100.0):
    """
    Compute objective and gradient for smooth log-ratio.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    sum_exp_max = np.sum(exp_max)
    log_sum_exp_max = m_max + np.log(sum_exp_max) / tau
    
    # Soft min
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    sum_exp_min = np.sum(exp_min)
    log_sum_exp_min = m_min - np.log(sum_exp_min) / tau
    
    obj_val = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    # Gradient w.r.t points (normalized)
    p = exp_max / sum_exp_max
    q = exp_min / sum_exp_min
    
    iu_i, iu_k = np.triu_indices(N, k=1)
    coeffs = p - q
    
    grad_points = np.zeros_like(points)
    
    for j in range(len(coeffs)):
        i = iu_i[j]
        k = iu_k[j]
        diff = points[i] - points[k]
        c = 2.0 * coeffs[j]
        grad_points[i] += c * diff
        grad_points[k] -= c * diff
        
    # Transform to grad_x
    mean_x = np.mean(x.reshape(N, D), axis=0)
    P_raw = x.reshape(N, D) - mean_x
    sigma = np.sqrt(np.mean(np.sum(P_raw**2, axis=1)))
    
    if sigma < 1e-12:
        return obj_val, np.zeros_like(x)
        
    grad_x = np.zeros_like(x.reshape(N, D))
    
    # Term 1
    term1 = grad_points / sigma
    
    # Term 3
    w = np.zeros((N, D))
    for j in range(N):
        s = np.zeros(D)
        for i in range(N):
            dot_val = np.dot(P_raw[j], grad_points[i])
            s += P_raw[i] * dot_val
        w[j] = s
        
    term3 = -w / (N * sigma**3)
    
    grad_x = term1 + term3
    grad_x_flat = grad_x.flatten()
    
    return obj_val, grad_x_flat

def construct():
    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        pts = normalize_points(pts)
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Fibonacci sphere (well-distributed)
    fib_pts = fibonacci_sphere(N, seed=42)
    candidates.append(fib_pts.flatten())
    
    # Add perturbed Fibonacci variants
    for _ in range(5):
        perturbed = fib_pts + rng.normal(size=(N, D)) * 0.1
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        candidates.append(perturbed.flatten())
    
    # 2. Random points on sphere
    for _ in range(15):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # 3. Icosahedron + 2 random points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Symmetric candidate: Ico + Poles
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    candidates.append(pts_sym.flatten())
    
    for _ in range(8):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())

    # 4. Cuboctahedron vertices (12 points) + 2 random
    perms = np.array([
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
    ])
    cubo = perms / np.linalg.norm(perms, axis=1, keepdims=True)
    for _ in range(8):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([cubo, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())
    
    # 5. Concentric shells: 8 on inner shell, 6 on outer shell
    inner = fibonacci_sphere(8, seed=100) * 0.7
    outer = fibonacci_sphere(6, seed=200) * 1.0
    shell_pts = np.vstack([inner, outer])
    shell_pts = shell_pts / np.linalg.norm(shell_pts, axis=1, keepdims=True)
    candidates.append(shell_pts.flatten())
    
    for _ in range(5):
        inner_pert = inner + rng.normal(size=(8, D)) * 0.1
        outer_pert = outer + rng.normal(size=(6, D)) * 0.1
        shell_pert = np.vstack([inner_pert, outer_pert])
        shell_pert = shell_pert / np.linalg.norm(shell_pert, axis=1, keepdims=True)
        candidates.append(shell_pert.flatten())

    # Optimization Loop
    n_restarts = len(candidates)
    
    # Phase 1: Smooth Log-Ratio Annealing with extended tau range
    taus = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 20:
            break
            
        x0 = candidates[i].copy()
        x0 = x0.reshape(N, D)
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        try:
            current_x = x0
            
            # Phase 1: Standard smooth optimization with annealing
            for tau in taus:
                if time.time() - start_time > timeout - 20:
                    break
                
                res = minimize(
                    objective_and_grad, 
                    current_x, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if res.x is not None and res.fun < np.inf:
                    current_x = res.x
                    pts_current = current_x.reshape(N, D)
                    update_best(pts_current)
            
            # Phase 2: Local perturbation escape from local minima
            if time.time() - start_time < timeout - 15:
                best_local = current_x.copy()
                for _ in range(3):
                    if time.time() - start_time > timeout - 15:
                        break
                    
                    # Add small random perturbation
                    perturbation = rng.normal(size=(N, D)) * 0.05
                    x_pert = best_local.reshape(N, D) + perturbation
                    x_pert = normalize_points(x_pert).flatten()
                    
                    # Optimize from perturbed point with high tau
                    for tau in [1000.0, 5000.0]:
                        if time.time() - start_time > timeout - 15:
                            break
                        
                        res = minimize(
                            objective_and_grad, 
                            x_pert, 
                            method='L-BFGS-B', 
                            args=(tau,),
                            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                        )
                        
                        if res.x is not None and res.fun < np.inf:
                            x_pert = res.x
                            pts_current = x_pert.reshape(N, D)
                            update_best(pts_current)
                            best_local = x_pert.copy()

        except Exception:
            pass
    
    # Final high-precision polish pass on best point
    if best_points is not None and time.time() - start_time < timeout - 5:
        try:
            x_polish = best_points.flatten()
            
            # Multi-stage polish with increasing precision
            for tau in [5000.0, 20000.0, 50000.0]:
                if time.time() - start_time > timeout - 5:
                    break
                    
                res = minimize(
                    objective_and_grad, 
                    x_polish, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
                )
                if res.x is not None and res.fun < np.inf:
                    x_polish = res.x
                    pts_polish = x_polish.reshape(N, D)
                    update_best(pts_polish)
        except Exception:
            pass

    if best_points is None:
        best_points = normalize_points(candidates[0].reshape(N, D))
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
