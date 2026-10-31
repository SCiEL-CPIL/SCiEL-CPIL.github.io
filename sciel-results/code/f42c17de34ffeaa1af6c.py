import numpy as np
import itertools
from scipy.optimize import minimize
import time

N = 16
D = 2

def compute_score(pts):
    """
    Computes the squared ratio of max to min pairwise distance.
    """
    # pts: (N, D)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    mask = ~np.eye(N, dtype=bool)
    dist_sq = dist_sq[mask]
    
    if len(dist_sq) == 0:
        return 1e10
        
    d_min = np.min(dist_sq)
    d_max = np.max(dist_sq)
    
    if d_min < 1e-12:
        return 1e10
        
    return d_max / d_min

def smooth_objective(flat_pts, tau1, tau2):
    """
    Smooth surrogate for log(max_dist_sq) - log(min_dist_sq).
    """
    pts = flat_pts.reshape(N, D)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    mask = ~np.eye(N, dtype=bool)
    dist_sq = dist_sq[mask]
    
    # Soft max: log(sum(exp(x/tau))) * tau
    # To avoid overflow, we can use logsumexp if available, or manual
    # scipy.special.logsumexp is available in SciPy
    from scipy.special import logsumexp
    
    max_term = logsumexp(dist_sq / tau1) * tau1
    min_term = logsumexp(-dist_sq / tau2) * tau2
    
    return max_term - min_term

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = np.inf
    best_pts = None
    
    # 1. Generate Initial Candidates
    
    # Candidate 1: Grid
    grid_pts = np.array(list(itertools.product(range(4), repeat=2)), dtype=float)
    
    # Candidate 2: Best subset of Triangular Lattice Hexagon (19 points)
    # Generate 19-point hexagon in axial coords (q, r)
    axial_pts = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if abs(q) <= 2 and abs(r) <= 2 and abs(s) <= 2:
                # Convert to cartesian
                x = 1.5 * q
                y = np.sqrt(3) / 2 * (q + 2 * r)
                axial_pts.append((x, y))
    lattice_hex = np.array(axial_pts)
    
    best_lattice_score = np.inf
    best_lattice_pts = None
    
    # Try all combinations of 16 points from 19
    # 19 choose 3 = 969
    for combo in itertools.combinations(range(19), 3):
        # Remove these 3 points
        mask = np.ones(19, dtype=bool)
        mask[list(combo)] = False
        pts = lattice_hex[mask]
        s = compute_score(pts)
        if s < best_lattice_score:
            best_lattice_score = s
            best_lattice_pts = pts.copy()
    
    # Candidate 3: Random points
    random_pts = np.random.randn(N, D) * 2
    
    # Evaluate initial candidates
    candidates = [grid_pts, best_lattice_pts, random_pts]
    for pts in candidates:
        if pts is None:
            continue
        s = compute_score(pts)
        if s < best_score:
            best_score = s
            best_pts = pts.copy()
    
    # 2. Optimization Phase
    # Use smooth surrogate to polish
    
    if time.time() < deadline:
        # Try different tau values (annealing)
        taus = [(1.0, 1.0), (0.5, 0.5), (0.2, 0.2), (0.1, 0.1), (0.05, 0.05)]
        
        # We will run optimization from the best initial guess
        # and also from a few random restarts if time permits
        
        starts = [best_pts.copy()]
        
        # Add a few random restarts
        for _ in range(5):
            if time.time() > deadline:
                break
            rp = np.random.randn(N, D) * 2
            starts.append(rp)
        
        for start_pts in starts:
            if time.time() > deadline:
                break
            
            current_pts = start_pts.copy()
            
            for tau1, tau2 in taus:
                if time.time() > deadline:
                    break
                
                x0 = current_pts.flatten()
                
                try:
                    res = minimize(
                        smooth_objective,
                        x0,
                        args=(tau1, tau2),
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-9}
                    )
                    
                    if res.success:
                        new_pts = res.x.reshape(N, D)
                        s = compute_score(new_pts)
                        if s < best_score:
                            best_score = s
                            best_pts = new_pts
                            current_pts = new_pts
                except Exception:
                    continue
        
        # 3. Local Polish
        # Simple random perturbations on the best point found
        
        if time.time() < deadline:
            T = 0.05
            step_size = 0.1
            
            for _ in range(20000):
                if time.time() > deadline:
                    break
                
                idx = np.random.randint(N)
                perturbation = np.random.randn(D) * step_size
                new_pts = best_pts.copy()
                new_pts[idx] += perturbation
                
                s = compute_score(new_pts)
                if s < best_score:
                    best_score = s
                    best_pts = new_pts
                else:
                    # Accept with probability exp(-(s - best_score) / T)
                    if np.random.rand() < np.exp(-(s - best_score) / T):
                        best_pts = new_pts
                
                # Slowly cool down
                if _ % 1000 == 0:
                    T *= 0.9
                    step_size *= 0.95

    return best_pts

# EVOLVE-BLOCK-END
