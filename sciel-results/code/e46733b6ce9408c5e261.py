import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 14
D = 3

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
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def objective_log_ratio(x, tau=100.0):
    """
    Objective: Minimize log(soft_max(d^2)) - log(soft_min(d^2)).
    This is a smooth approximation of log(max(d^2)/min(d^2)).
    """
    points = x.reshape(N, D)
    # Center points to remove translation invariance
    points = points - np.mean(points, axis=0, keepdims=True)
    
    sq_dists = get_pairwise_sq_dists(points)
    
    # Ensure positivity for log
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max: log(sum(exp(tau * x))) / tau
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    # Soft min: -log(sum(exp(-tau * x))) / tau
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    # We want to minimize max/min, so minimize log(max) - log(min)
    return log_sum_exp_max - log_sum_exp_min

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin
    
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
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate initial guesses
    # 1. Icosahedron + 2 random points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    initial_guesses = []
    
    # Icosahedron based
    for _ in range(5):
        p1 = rng.normal(size=D); p1 /= np.linalg.norm(p1)
        p2 = rng.normal(size=D); p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1, p2])
        # Add small noise to break symmetry
        pts += rng.normal(scale=0.1, size=pts.shape)
        initial_guesses.append(pts)
        
    # Cuboctahedron based (12 vertices) + 2
    # Cuboctahedron vertices are permutations of (0, 1, 1)
    co = np.array([
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
    ], dtype=float)
    co = co / np.linalg.norm(co, axis=1, keepdims=True)
    for _ in range(5):
        p1 = rng.normal(size=D); p1 /= np.linalg.norm(p1)
        p2 = rng.normal(size=D); p2 /= np.linalg.norm(p2)
        pts = np.vstack([co, p1, p2])
        pts += rng.normal(scale=0.1, size=pts.shape)
        initial_guesses.append(pts)
        
    # Random points on sphere
    for _ in range(10):
        pts = rng.normal(size=(N, D))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        initial_guesses.append(pts)

    # Optimization loop
    for i, guess in enumerate(initial_guesses):
        if time.time() - start_time > timeout - 10:
            break
            
        x0 = guess.flatten()
        
        # Try different tau values for the soft min/max approximation
        # Higher tau is sharper, lower is smoother
        taus = [50.0, 100.0, 200.0]
        
        for tau in taus:
            if time.time() - start_time > timeout - 5:
                break
                
            try:
                res = minimize(
                    objective_log_ratio,
                    x0,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                pts = res.x.reshape(N, D)
                update_best(pts)
            except Exception:
                pass

    # Final Polish
    if best_points is not None:
        # Use a very high tau for the final polish to get closer to the true min/max
        # and tighter tolerances
        if time.time() - start_time < timeout - 2:
            try:
                res = minimize(
                    objective_log_ratio,
                    best_points.flatten(),
                    args=(1000.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                pts = res.x.reshape(N, D)
                update_best(pts)
            except Exception:
                pass

    if best_points is None:
        # Fallback
        best_points = initial_guesses[0]
        best_score = raw_score(best_points)
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
