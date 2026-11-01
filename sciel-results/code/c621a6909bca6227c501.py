"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def _pairwise_indices(n):
    """Return list of (i, j) pairs with i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
    # Compute all pairwise distances
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d > 1e-10:
                dists.append(d)
    if not dists:
        return 1e10
    min_d = min(dists)
    max_d = max(dists)
    if min_d < 1e-10:
        return 1e10
    return (max_d / min_d) ** 2

def construct():
    n = N
    d = D
    indices = _pairwise_indices(n)
    num_pairs = len(indices)
    
    # We will optimize the Gram matrix G (n x n)
    # G is symmetric PSD, rank <= 3
    # G_ij = x_i . x_j
    # ||x_i - x_j||^2 = G_ii + G_jj - 2 G_ij
    # We want to minimize (max_{i<j} ||x_i - x_j|| / min_{i<j} ||x_i - x_j||)^2
    # Let D_ij = ||x_i - x_j||^2 = G_ii + G_jj - 2 G_ij
    # We want to minimize (max D_ij / min D_ij)
    
    # To enforce rank <= 3, we can minimize the sum of eigenvalues beyond the top 3
    # Or we can simply optimize the coordinates directly, which naturally enforces rank <= 3.
    # The directive suggests optimizing the Gram matrix. Let's try that.
    # Variables: G (n x n), symmetric. We can parameterize the upper triangle.
    # But G must be PSD.
    
    # Alternative: Optimize coordinates directly but with a better objective and initialization.
    # The previous approach used Nelder-Mead on 42 variables.
    # Let's try a smooth objective function.
    
    def objective(x):
        pts = x.reshape(n, d)
        # Compute squared distances
        d_sq = np.zeros(num_pairs)
        for k, (i, j) in enumerate(indices):
            diff = pts[i] - pts[j]
            d_sq[k] = np.dot(diff, diff)
        
        # Avoid division by zero
        min_d_sq = np.min(d_sq)
        max_d_sq = np.max(d_sq)
        
        if min_d_sq < 1e-12:
            return 1e10
        
        # Smooth approximation of the ratio
        # f = log(sum exp(d_sq/T)) - log(sum exp(-d_sq/T))
        # This approximates 2*(max - min) roughly, but we want the ratio.
        # Let's just use the ratio directly, it's not that non-smooth if we avoid 0.
        return (max_d_sq / min_d_sq)
    
    # Initial configurations
    # 1. Grid
    side = int(np.ceil(n ** (1.0 / d)))
    grid = list(itertools.product(range(side), repeat=d))[:n]
    x0_grid = np.array(grid, dtype=float).flatten()
    
    # 2. Random
    np.random.seed(42)
    x0_rand = np.random.randn(n * d)
    
    # 3. Spherical code initialization (simple)
    x0_sph = np.zeros(n * d)
    for i in range(n):
        # Random direction
        vec = np.random.randn(d)
        vec /= np.linalg.norm(vec)
        x0_sph[i*d:(i+1)*d] = vec * 1.0
        
    # 4. Fibonacci sphere (better spherical distribution)
    x0_fib = np.zeros(n * d)
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        # Fibonacci sphere point
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        x0_fib[i*d:(i+1)*d] = np.array([x, y, z])
        
    best_x = None
    best_val = 1e10
    
    # Try optimizing from each initial point
    for x0 in [x0_grid, x0_rand, x0_sph, x0_fib]:
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    if best_x is None:
        # Fallback to grid
        return np.array(grid, dtype=float)
    
    points = best_x.reshape(n, d)
    return points

# EVOLVE-BLOCK-END
