"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def fibonacci_sphere(n):
    """Generate n points on a sphere using the Fibonacci lattice method."""
    points = np.zeros((n, 3))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        z = 1.0 - (i / (n - 1)) * 2.0
        radius_at_z = np.sqrt(1.0 - z * z)
        theta = golden_angle * i
        points[i, 0] = np.cos(theta) * radius_at_z
        points[i, 1] = np.sin(theta) * radius_at_z
        points[i, 2] = z
    return points

def random_initialization(n, d, rng):
    """Generate random initial points."""
    return rng.normal(0, 1, size=(n, d))

def get_score(pts):
    """Calculate the squared ratio of max to min pairwise distance."""
    dists = pdist(pts)
    if np.any(dists < 1e-10):
        return 1e10
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d == 0:
        return 1e10
    return (max_d / min_d) ** 2

def simulate_annealing(initial_pts, max_iter=30000, rng=None):
    """Run simulated annealing to optimize point configuration."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    current_pts = initial_pts.copy()
    current_score = get_score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    # SA Parameters
    T = 1.0
    T_min = 1e-8
    alpha = 0.9997  # Cooling rate
    step_size = 0.2
    
    # Track acceptance rate for adaptive step size
    accept_count = 0
    total_count = 0
    window_size = 500
    
    for i in range(max_iter):
        # Generate new point
        # Perturb one or more points
        num_perturbed = rng.integers(1, 3)  # 1 or 2 points
        idx = rng.choice(N, size=num_perturbed, replace=False)
        new_pts = current_pts.copy()
        new_pts[idx] += rng.normal(0, step_size, size=(num_perturbed, D))
        
        new_score = get_score(new_pts)
        
        # Metropolis criterion
        accepted = False
        if new_score < current_score:
            accepted = True
        else:
            # Accept with probability exp(-delta / T)
            delta = new_score - current_score
            # Avoid overflow in exp
            if delta < 700 * T:
                prob = np.exp(-delta / T)
            else:
                prob = 0.0
            
            if rng.random() < prob:
                accepted = True
        
        if accepted:
            current_pts = new_pts
            current_score = new_score
            accept_count += 1
            if current_score < best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        
        total_count += 1
        
        # Adaptive step size based on acceptance rate
        if total_count % window_size == 0:
            accept_rate = accept_count / window_size
            # Target acceptance rate around 0.2-0.3
            if accept_rate < 0.1:
                step_size *= 0.8
            elif accept_rate > 0.5:
                step_size *= 1.2
            else:
                step_size *= 0.98
            
            # Keep step size in reasonable bounds
            step_size = np.clip(step_size, 0.001, 1.0)
            
            accept_count = 0
            total_count = 0
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
    
    return best_pts, best_score

def local_refinement(pts, maxiter=5000):
    """Perform local refinement using Nelder-Mead."""
    def objective(x):
        pts_local = x.reshape(N, D)
        return get_score(pts_local)

    x0 = pts.flatten()
    
    # Run NM for a limited number of iterations to polish
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10,
                           'adaptive': True})
    
    opt_points = res.x.reshape(N, D)
    return opt_points

def construct():
    rng = np.random.default_rng(42)
    
    # Try multiple initializations and keep the best result
    best_pts = None
    best_score = float('inf')
    
    # Different initialization strategies
    initializations = [
        fibonacci_sphere(N),
        random_initialization(N, D, rng),
        random_initialization(N, D, rng) * 0.5,
        random_initialization(N, D, rng) * 2.0,
    ]
    
    for init_pts in initializations:
        # Run SA from this initialization
        sa_pts, sa_score = simulate_annealing(init_pts, max_iter=25000, rng=rng)
        
        # Local refinement
        refined_pts = local_refinement(sa_pts, maxiter=3000)
        refined_score = get_score(refined_pts)
        
        if refined_score < best_score:
            best_score = refined_score
            best_pts = refined_pts
        elif sa_score < best_score:
            best_score = sa_score
            best_pts = sa_pts
    
    return best_pts

# EVOLVE-BLOCK-END
