import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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

def _icosahedron():
    """Return 12 icosahedron vertices on unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    return ico

def _cuboctahedron():
    """Return 12 cuboctahedron vertices on unit sphere."""
    perms = np.array([
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
    ])
    return perms / np.linalg.norm(perms, axis=1, keepdims=True)

def _tetrahedron():
    """Return 4 regular tetrahedron vertices on unit sphere."""
    pts = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _octahedron():
    """Return 6 regular octahedron vertices on unit sphere."""
    pts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    return pts

def _square_antiprism():
    """Return 8 square antiprism vertices on unit sphere."""
    angles_bottom = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    angles_top = angles_bottom + np.pi/4
    z = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]) * 0.577
    r = np.sqrt(1 - z**2)
    angles = np.concatenate([angles_top, angles_bottom])
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def get_pairwise_sq_dists(points):
    """Compute squared pairwise distances for N points in D dimensions."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Calculate the actual score: (max_dist / min_dist)^2."""
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d)

def normalize_points(points):
    """Center and normalize points to unit mean squared norm."""
    points = points - np.mean(points, axis=0)
    mean_norm_sq = np.mean(np.sum(points**2, axis=1))
    if mean_norm_sq > 1e-12:
        points = points / np.sqrt(mean_norm_sq)
    return points

def objective_smooth(x, tau=100.0):
    """Smooth approximation of log ratio of max to min squared distance."""
    points = x.reshape(N, D)
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    obj_val = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    return obj_val

def objective_exact(x):
    """Exact objective: (max_dist / min_dist)^2."""
    points = x.reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return 1e6
    return np.max(sq_dists) / min_d

def objective_combined(x, tau=100.0, w_ratio=1.0, w_spread=0.05):
    """Combined objective: log-ratio + spread penalty."""
    points = x.reshape(N, D)
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    obj_ratio = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    log_dists = np.log(sq_dists)
    spread = np.var(log_dists)
    
    return w_ratio * obj_ratio + w_spread * spread

def coordinate_descent(points, max_iter=200, tol=1e-10):
    """Simple coordinate descent to refine points."""
    points = normalize_points(points).copy()
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            best_p = points[i].copy()
            best_score = raw_score(points)
            # Try small perturbations in each dimension
            for dim in range(D):
                for delta in [0.01, 0.005, 0.001]:
                    for sign in [1, -1]:
                        trial = points.copy()
                        trial[i, dim] += sign * delta
                        trial = normalize_points(trial)
                        s = raw_score(trial)
                        if s < best_score - tol:
                            best_score = s
                            best_p = trial[i].copy()
            if np.any(best_p != points[i]):
                points[i] = best_p
                improved = True
        if not improved:
            break
    return normalize_points(points)

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        pts = normalize_points(pts)
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Fibonacci sphere
    fib_pts = fibonacci_sphere(N, seed=42)
    candidates.append(fib_pts.flatten())
    
    for _ in range(3):
        perturbed = fib_pts + rng.normal(size=(N, D)) * 0.1
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        candidates.append(perturbed.flatten())
    
    # 2. Random points on sphere
    for _ in range(6):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # 3. Icosahedron + 2 points
    ico = _icosahedron()
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    candidates.append(pts_sym.flatten())
    
    for _ in range(4):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())

    # 4. Cuboctahedron + 2 points
    cubo = _cuboctahedron()
    for _ in range(4):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([cubo, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())
    
    # 5. Square antiprism + tetrahedron + 2 octahedron points
    antiprism = _square_antiprism()
    tetra = _tetrahedron()
    octa = _octahedron()
    combo1 = np.vstack([antiprism, tetra, octa[:2]])
    candidates.append(combo1.flatten())
    
    # 6. Concentric shells
    inner = fibonacci_sphere(6, seed=100) * 0.6
    outer = fibonacci_sphere(8, seed=200) * 1.0
    shell_pts = np.vstack([inner, outer])
    shell_pts = shell_pts / np.linalg.norm(shell_pts, axis=1, keepdims=True)
    candidates.append(shell_pts.flatten())
    
    for _ in range(2):
        inner_pert = inner + rng.normal(size=(6, D)) * 0.1
        outer_pert = outer + rng.normal(size=(8, D)) * 0.1
        shell_pert = np.vstack([inner_pert, outer_pert])
        shell_pert = shell_pert / np.linalg.norm(shell_pert, axis=1, keepdims=True)
        candidates.append(shell_pert.flatten())
    
    # 7. Two concentric arrangements (7+7)
    inner7 = fibonacci_sphere(7, seed=300) * 0.5
    outer7 = fibonacci_sphere(7, seed=400) * 1.0
    shell2 = np.vstack([inner7, outer7])
    shell2 = shell2 / np.linalg.norm(shell2, axis=1, keepdims=True)
    candidates.append(shell2.flatten())

    # Evaluate all candidates
    for cand in candidates:
        pts = cand.reshape(N, D)
        update_best(pts)
    
    # Phase 1: Differential evolution for global search (limited)
    if time.time() - start_time < timeout - 60:
        try:
            bounds = [(-2, 2)] * (N * D)
            de_result = differential_evolution(
                objective_exact,
                bounds=bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                update_interval=1,
                polish=False,
                workers=1
            )
            if de_result.x is not None:
                pts_de = de_result.x.reshape(N, D)
                update_best(pts_de)
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B with tau annealing from best candidates
    if time.time() - start_time < timeout - 30:
        # Use top 5 candidates for local optimization
        scored_candidates = []
        seen = set()
        for cand in candidates:
            pts = cand.reshape(N, D)
            pts_norm = normalize_points(pts)
            key = tuple(np.round(pts_norm, 4).flatten())
            if key not in seen:
                seen.add(key)
                s = raw_score(pts_norm)
                scored_candidates.append((s, cand))
        
        scored_candidates.sort(key=lambda x: x[0])
        top_candidates = [c[1] for c in scored_candidates[:5]]
        
        if best_points is not None:
            top_candidates.append(best_points.flatten())
        
        taus = [5.0, 20.0, 50.0, 100.0, 300.0, 1000.0]
        
        for i, x0 in enumerate(top_candidates):
            if time.time() - start_time > timeout - 30:
                break
            
            x0 = x0.reshape(N, D)
            x0 = normalize_points(x0)
            x0 = x0.flatten()
            
            try:
                current_x = x0.copy()
                
                for tau in taus:
                    if time.time() - start_time > timeout - 30:
                        break
                    
                    res = minimize(
                        objective_smooth, 
                        current_x, 
                        method='L-BFGS-B', 
                        args=(tau,),
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                    )
                    
                    if res.x is not None and res.fun < np.inf:
                        current_x = res.x
                        pts_current = current_x.reshape(N, D)
                        update_best(pts_current)
                
                # Local perturbation escape
                if time.time() - start_time < timeout - 25:
                    for _ in range(2):
                        if time.time() - start_time > timeout - 25:
                            break
                        
                        perturbation = rng.normal(size=(N, D)) * 0.02
                        x_pert = current_x.reshape(N, D) + perturbation
                        x_pert = normalize_points(x_pert).flatten()
                        
                        for tau in [500.0, 2000.0]:
                            if time.time() - start_time > timeout - 25:
                                break
                            
                            res = minimize(
                                objective_smooth, 
                                x_pert, 
                                method='L-BFGS-B', 
                                args=(tau,),
                                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                            )
                            
                            if res.x is not None and res.fun < np.inf:
                                x_pert = res.x
                                pts_current = x_pert.reshape(N, D)
                                update_best(pts_current)
                                current_x = x_pert.copy()

            except Exception:
                pass
    
    # Phase 3: Coordinate descent refinement
    if best_points is not None and time.time() - start_time < timeout - 15:
        try:
            pts_cd = coordinate_descent(best_points, max_iter=100)
            update_best(pts_cd)
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead polish on exact objective
    if best_points is not None and time.time() - start_time < timeout - 10:
        try:
            x_polish = best_points.flatten()
            res_nm = minimize(
                objective_exact,
                x_polish,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.x is not None:
                pts_nm = res.x.reshape(N, D)
                update_best(pts_nm)
        except Exception:
            pass
    
    # Phase 5: Very high-precision L-BFGS-B polish
    if best_points is not None and time.time() - start_time < timeout - 3:
        try:
            x_polish = best_points.flatten()
            
            for tau in [5000.0, 20000.0]:
                if time.time() - start_time > timeout - 3:
                    break
                    
                res = minimize(
                    objective_smooth, 
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
