import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 14
D = 3

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def objective_smooth(x, tau=20.0):
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    # Soft max
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Soft min
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-10:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def construct():
    start_time = time.time()
    timeout = 110.0
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts

    # Initial candidates
    candidates = []
    
    # 1. Icosahedron vertices (12) + 2 points
    # Icosahedron vertices can be generated using golden ratio
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True) * 2.0 # Scale to radius 2
    # Add 2 points at center? No, must be distinct. Add 2 points slightly off center or random.
    # Let's add 2 random points inside the sphere
    rng = np.random.default_rng(42)
    for _ in range(5):
        p1 = rng.normal(size=D) * 0.5
        p2 = rng.normal(size=D) * 0.5
        pts = np.vstack([ico, p1, p2])
        candidates.append(pts.flatten())
        
    # 2. Random points in sphere
    for _ in range(10):
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * 2.0
        candidates.append(pts.flatten())
        
    # 3. FCC Lattice subset
    # Generate FCC points in a box and select 14
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) scaled by 2
    # Just use a simple grid for now
    grid_pts = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                grid_pts.append([x, y, z])
    grid_pts = np.array(grid_pts[:14], dtype=float)
    grid_pts = grid_pts - np.mean(grid_pts, axis=0)
    candidates.append(grid_pts.flatten())

    # Optimization Loop
    # Use Differential Evolution to find good basins, then Polish with L-BFGS-B
    # DE bounds: [-5, 5] for each coordinate
    
    bounds = [(-5, 5)] * (N * D)
    
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout:
            break
            
        # Skip DE for the first few to save time? No, DE is fast.
        # But DE doesn't use the init_x. It starts from random population.
        # So we can just run DE once or twice.
        
        if i == 0:
            # Run DE
            try:
                res_de = differential_evolution(
                    objective_smooth, 
                    bounds, 
                    maxiter=100, 
                    popsize=15, 
                    seed=42,
                    tol=1e-8,
                    polish=False # We will polish manually
                )
                if res_de.x is not None:
                    # Polish with L-BFGS-B
                    res_polish = minimize(
                        objective_smooth, 
                        res_de.x, 
                        method='L-BFGS-B', 
                        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if res_polish.x is not None:
                        update_best(res_polish.x.reshape(N, D))
            except Exception:
                pass
            
            # Also polish the DE result without L-BFGS if it failed? No.
            
            # Run DE again with different seed
            if time.time() - start_time < timeout - 30:
                try:
                    res_de2 = differential_evolution(
                        objective_smooth, 
                        bounds, 
                        maxiter=100, 
                        popsize=15, 
                        seed=123,
                        tol=1e-8,
                        polish=False
                    )
                    if res_de2.x is not None:
                        res_polish2 = minimize(
                            objective_smooth, 
                            res_de2.x, 
                            method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                        )
                        if res_polish2.x is not None:
                            update_best(res_polish2.x.reshape(N, D))
                except Exception:
                    pass

        # For other candidates, just polish directly (faster)
        else:
            try:
                res_polish = minimize(
                    objective_smooth, 
                    init_x, 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res_polish.x is not None:
                    update_best(res_polish.x.reshape(N, D))
            except Exception:
                pass

    # Fallback
    if best_points is None:
        # Use the first candidate
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
