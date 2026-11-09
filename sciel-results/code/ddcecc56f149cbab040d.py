import numpy as np
import time
from scipy.spatial.distance import pdist

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def _soft_objective(points, temperature):
    """
    Objective for simulated annealing.
    We want to maximize (d_min / d_max)^2.
    Equivalently, minimize -log(d_min/d_max) = log(d_max) - log(d_min).
    To make the landscape smoother and help escape local minima where d_min is stuck,
    we can use a potential energy function that penalizes close approaches more strongly
    when temperature is low, but allows exploration when high.
    
    A common approach for packing problems is to minimize the potential energy:
    E = sum_{i<j} 1/d_ij^k  for k > 0, but this blows up.
    
    Alternatively, we can minimize:
    f = - (d_min / d_max)^2
    But this is non-smooth at the minimum.
    
    Let's use a differentiable proxy:
    Minimize: log(d_max) - log(d_min)
    This is equivalent to minimizing log(d_max/d_min) = -log(d_min/d_max).
    Minimizing this maximizes the ratio.
    
    To help with the non-smoothness of d_min (which depends on the single closest pair),
    we can use a 'soft' minimum:
    d_min_soft = -beta * log(sum(exp(-beta * d_ij)))
    This approximates min(d_ij) as beta -> inf.
    
    So objective = log(d_max) - (-beta * log(sum(exp(-beta * d_ij))))
    = log(d_max) + beta * log(sum(exp(-beta * d_ij)))
    
    We will optimize this.
    """
    dists = pdist(points, metric='euclidean')
    if np.any(dists == 0) or np.any(np.isnan(dists)):
        return 1e10
    
    d_max = np.max(dists)
    if d_max == 0:
        return 1e10
        
    # Soft min calculation
    # To avoid overflow in exp, subtract max (which is min of -dists, so max of dists? No, -beta * d)
    # Let u = -beta * dists
    # log(sum(exp(u))) = max(u) + log(sum(exp(u - max(u))))
    u = -temperature * dists
    u_max = np.max(u)
    soft_min = -u_max / temperature - np.log(np.sum(np.exp(u - u_max))) / temperature
    # Wait, d_min_soft = -1/beta * log(sum(exp(-beta d)))
    # So log(d_min_soft) is not simply related.
    
    # Let's stick to the ratio but smooth the min.
    # We want to maximize R = d_min / d_max.
    # Let's minimize -log(R) = log(d_max) - log(d_min).
    # Use soft_min for d_min.
    
    # Re-calculate soft_min properly
    # d_min_soft = -1/temperature * log(sum(exp(-temperature * d_ij)))
    # Note: as temperature increases, soft_min approaches the arithmetic mean? No.
    # As temperature -> inf, exp(-t*d) is dominated by the largest d? No, smallest d.
    # exp(-t*d) is largest when d is smallest.
    # So sum is dominated by smallest d.
    # log(sum) ~ log(exp(-t*d_min)) = -t*d_min.
    # -1/t * (-t*d_min) = d_min. Correct.
    
    # However, if temperature is too high, it becomes less sensitive to the min.
    # We want to sharpen it as we cool.
    
    # Let's just use the raw score but add a small noise penalty?
    # Actually, SA usually minimizes an energy function.
    # Let's define Energy = 1 / (d_min / d_max)^2 = (d_max / d_min)^2.
    # Minimizing this is equivalent.
    # Energy is non-smooth.
    
    # Let's try minimizing: sum_{i<j} (1/d_ij^2) * w_ij ?
    # If we minimize the sum of inverse squared distances, we push points apart.
    # But we also need to control d_max.
    # If we constrain points to a unit sphere (or normalize), d_max is bounded by 2.
    # Then maximizing d_min is the main goal.
    
    # Strategy:
    # 1. Normalize points to unit sphere after each step? Or keep them in 3D and normalize at the end?
    # Normalizing at every step keeps them on the sphere, reducing dimensionality effectively.
    # Let's work on the sphere S^2.
    
    dists = pdist(points, metric='euclidean')
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_min == 0:
        return 1e10
    ratio = d_min / d_max
    # We want to maximize ratio.
    # Objective to minimize: -log(ratio) = log(d_max) - log(d_min)
    return np.log(d_max) - np.log(d_min)

def _generate_seed_points(seed_type, n=14):
    """Generate diverse starting configurations."""
    if seed_type == 0:
        # Random on sphere
        pts = np.random.randn(n, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts
    elif seed_type == 1:
        # Icosahedron + 2 poles (distorted)
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico /= np.linalg.norm(ico, axis=1, keepdims=True)
        extra = np.array([[0, 0, 1], [0, 0, -1]])
        pts = np.vstack([ico, extra])
        # Add small random noise
        pts += np.random.randn(n, 3) * 0.1
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts
    elif seed_type == 2:
        # Cube vertices + face centers (14 points? Cube has 8 vertices + 6 centers = 14)
        # Vertices of cube: (+-1, +-1, +-1)
        # Centers: (+-1, 0, 0), (0, +-1, 0), (0, 0, +-1)
        verts = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ])
        centers = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        pts = np.vstack([verts, centers])
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        pts += np.random.randn(n, 3) * 0.1
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts
    elif seed_type == 3:
        # Two rings of 7? Or 6+6+2?
        # Let's try 2 poles + 12 points in 3 rings of 4 (square rings)
        pts = []
        z_pole = 1.0
        pts.append([0, 0, z_pole])
        pts.append([0, 0, -z_pole])
        r = 1.0
        h = 0.5
        for k in range(3):
            z = -h + k * h # -0.5, 0, 0.5
            for i in range(4):
                theta = i * np.pi / 2 + (k * np.pi / 4) # Twist
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                pts.append([x, y, z])
        pts = np.array(pts)
        pts += np.random.randn(n, 3) * 0.1
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts
    else:
        return _generate_seed_points(0, n)

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    np.random.seed(12345)
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 0.01
    cooling_rate = 0.995
    
    # Number of restarts
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Choose a seed type
        seed_type = restart % 4
        points = _generate_seed_points(seed_type, 14)
        
        current_score = _score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        T = T_start
        n_iter = 5000  # Iterations per restart
        
        for i in range(n_iter):
            if time.time() > deadline - 5:
                break
                
            # Generate a random perturbation
            # Perturb one point slightly
            idx = np.random.randint(0, 14)
            delta = np.random.randn(3) * 0.1
            new_points = points.copy()
            new_points[idx] += delta
            # Normalize to sphere
            new_points[idx] /= np.linalg.norm(new_points[idx])
            
            new_score = _score(new_points)
            
            # Acceptance probability
            if new_score > current_score:
                accept = True
            else:
                # Metropolis criterion for maximization:
                # P = exp((new_score - current_score) / T)
                # We want to accept worse solutions with prob exp(-DeltaE/T)
                # Here DeltaScore = current - new > 0
                # P = exp(-(current - new) / T)
                diff = current_score - new_score
                if diff < 0:
                    accept = True
                else:
                    if diff < 1e-10:
                        accept = True
                    else:
                        p = np.exp(-diff / T)
                        accept = np.random.rand() < p
            
            if accept:
                points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    
            T *= cooling_rate
            
    # Final Polish: Use L-BFGS-B on the best point found to refine locally
    if best_points is not None and time.time() < deadline - 10:
        try:
            flat_params = best_points.flatten()
            
            def _objective_flat(params):
                pts = params.reshape(14, 3)
                # Normalize
                norms = np.linalg.norm(pts, axis=1)
                norms[norms == 0] = 1.0
                pts = pts / norms[:, np.newaxis]
                s = _score(pts)
                return -s
                
            from scipy.optimize import minimize
            res = minimize(
                _objective_flat, 
                flat_params, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            
            score_polish = -res.fun
            if score_polish > best_score:
                best_score = score_polish
                pts_polish = res.x.reshape(14, 3)
                norms = np.linalg.norm(pts_polish, axis=1)
                norms[norms == 0] = 1.0
                best_points = pts_polish / norms[:, np.newaxis]
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = _generate_seed_points(1, 14)

    # Ensure finite and correct shape
    best_points = np.nan_to_num(best_points, nan=0.0, posinf=0.0, neginf=0.0)
    if best_points.shape != (14, 3):
        best_points = _generate_seed_points(1, 14)
        
    return best_points
