"""Optimized logarithmic energy minimizer for 150 points on the unit sphere using Simulated Annealing."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10
            total += math.log(1.0 / d)
    return total

def _initial_fibonacci():
    """Generate initial Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _initial_random(seed=42):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _sa_optimize(points, iterations=50000, T_start=1.0, T_end=1e-6, step_size=0.5):
    """
    Simulated Annealing optimization.
    points: (N, 3) initial points
    iterations: number of SA steps
    T_start: initial temperature
    T_end: final temperature
    step_size: magnitude of perturbation
    """
    pts = points.copy()
    n = len(pts)
    
    # Calculate initial energy
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return pts, 1e10
            E += math.log(1.0 / d)
    
    best_E = E
    best_pts = pts.copy()
    
    rng = np.random.default_rng(0)
    
    for it in range(iterations):
        # Cooling schedule: exponential
        # T = T_start * (T_end / T_start) ** (it / iterations)
        T = T_start * (T_end / T_start) ** (it / iterations)
        
        # Pick a random point to move
        i = rng.integers(0, n)
        
        # Generate a random tangent vector
        # Random vector in R3
        rand_vec = rng.normal(size=3)
        # Project to tangent space: v - (v . p) p
        p_i = pts[i]
        v = rand_vec - np.dot(rand_vec, p_i) * p_i
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v /= v_norm
        
        # New position
        new_p = p_i + step_size * v
        new_p /= np.linalg.norm(new_p)
        
        # Calculate Delta E
        # E_i_old = sum_{j != i} log(1 / |p_i - p_j|)
        # E_i_new = sum_{j != i} log(1 / |new_p - p_j|)
        
        # Vectorized distance calculation for point i
        # dists_old: distance from p_i to all other points
        # dists_new: distance from new_p to all other points
        
        # To avoid O(N^2) total, we only calculate O(N) here.
        # We need the sum of logs.
        
        # Create a mask to exclude i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        # Calculate distances from p_i to all others
        # pts is (N, 3). p_i is (3,).
        # diff = pts - p_i -> (N, 3)
        # dists = norm(diff, axis=1) -> (N,)
        
        diff_old = pts - p_i
        dists_old = np.linalg.norm(diff_old, axis=1)
        dists_old[i] = 1.0 # Ignore self (set to 1 so log(1/1)=0)
        
        diff_new = pts - new_p
        dists_new = np.linalg.norm(diff_new, axis=1)
        dists_new[i] = 1.0
        
        # Clip distances to avoid log(0)
        # If any distance is 0, energy is inf.
        if np.any(dists_old < 1e-15) or np.any(dists_new < 1e-15):
            # If collision, reject move
            continue
            
        # Sum of logs
        # log(1/d) = -log(d)
        # We want sum_{j != i} -log(d_j)
        # Using math.fsum for precision? np.sum is usually fine for intermediate steps,
        # but let's be careful.
        
        # Since we only care about Delta E, we can compute:
        # delta_E = sum(-log(dists_new)) - sum(-log(dists_old))
        #          = -sum(log(dists_new)) + sum(log(dists_old))
        
        # We can sum only the non-i elements, but setting to 1.0 (log=0) makes it easy to sum all.
        
        sum_log_new = np.sum(np.log(dists_new))
        sum_log_old = np.sum(np.log(dists_old))
        
        delta_E = -(sum_log_new - sum_log_old)
        # Wait: E = sum log(1/d) = - sum log(d).
        # E_new_part = - sum(log(dists_new))
        # E_old_part = - sum(log(dists_old))
        # Delta E = E_new_part - E_old_part = -sum(log(dists_new)) - (-sum(log(dists_old)))
        #         = sum(log(dists_old)) - sum(log(dists_new))
        
        # Check acceptance
        if delta_E < 0 or (T > 0 and rng.random() < math.exp(-delta_E / T)):
            # Accept
            pts[i] = new_p
            E += delta_E
            if E < best_E:
                best_E = E
                best_pts = pts.copy()
                
    return best_pts, best_E

def construct():
    # Use multiple initializations and SA runs
    best_pts = None
    best_energy = float('inf')
    
    inits = [
        _initial_fibonacci(),
        _initial_random(42),
        _initial_random(123),
        _initial_random(999)
    ]
    
    # Run SA for each initialization
    # We can run for a fixed number of iterations
    # 50000 iterations * 4 inits = 200,000 steps.
    # Each step is O(N) = 150 ops.
    # Total 30M ops. Should be fast enough.
    
    for init in inits:
        try:
            pts, energy = _sa_optimize(init, iterations=30000, T_start=0.5, T_end=1e-5, step_size=0.3)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        except Exception:
            pass
            
    # If SA didn't find anything (shouldn't happen), fallback to Fibonacci
    if best_pts is None:
        best_pts = _initial_fibonacci()
        
    # Final projection and validation
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    best_pts = best_pts / norms
    
    # Check for close points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                best_pts[j] += 1e-6 * np.random.randn(3)
                best_pts[j] /= np.linalg.norm(best_pts[j])
                
    return best_pts
# EVOLVE-BLOCK-END
