"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # 1. Initial heuristic: Fibonacci sphere
    # This provides a good starting point for the optimizer if we were to use it,
    # but differential_evolution uses its own population.
    # However, we can use the Fibonacci points as a seed for a local refinement
    # if the global optimizer is too slow, or just rely on DE.
    
    # The objective for Differential Evolution is to MINIMIZE a function.
    # We want to MAXIMIZE the minimum pairwise distance.
    # So we minimize -min_pairwise_distance.
    
    # We represent points as (theta, phi) to ensure they are on the sphere.
    # theta in [0, pi], phi in [0, 2*pi]
    
    def objective(theta_phi_flat):
        # Unpack to (N, 2)
        theta_phi = theta_phi_flat.reshape(-1, 2)
        theta = theta_phi[:, 0]
        phi = theta_phi[:, 1]
        
        # Convert to Cartesian
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        points = np.column_stack([x, y, z])
        
        # Calculate pairwise distances (angular)
        # Using dot product for efficiency
        # d_ij = arccos(p_i . p_j)
        # We want min over i != j of d_ij
        # To avoid arccos cost in the inner loop, we can maximize min dot product? 
        # No, min angular distance corresponds to min dot product (since cos is decreasing in [0, pi]).
        # So we want to MAXIMIZE the minimum dot product between distinct points.
        # Wait, if points are close, dot product is ~1. If far, dot product is low.
        # Max angular distance -> Min dot product.
        # We want to spread points out, so we want the *closest* pair to be as far apart as possible.
        # So we want to maximize the minimum angular distance.
        # Equivalent to minimizing the maximum dot product of distinct pairs?
        # Let's check: 
        # If all points are evenly spread, the max dot product among distinct pairs is minimized (closest to -1 or small positive).
        # Actually, for covering radius, we care about the largest empty circle.
        # The covering radius is determined by the circumcircles of the Delaunay triangulation facets.
        # However, maximizing the minimum pairwise distance (packing) is a strong proxy and often yields good covering properties for N=50.
        
        # Let's compute all pairwise dot products
        dot_prods = points @ points.T
        # Set diagonal to -inf so we don't pick self-distance
        np.fill_diagonal(dot_prods, -np.inf)
        
        # The minimum angular distance corresponds to the maximum dot product among distinct pairs.
        # Because theta = arccos(dot). Min theta -> Max dot.
        # We want to MAXIMIZE min_theta.
        # So we want to MINIMIZE max_dot.
        
        max_dot = np.max(dot_prods)
        
        # We minimize this value in DE.
        return max_dot

    # Bounds for theta and phi
    bounds = [(0, np.pi), (0, 2 * np.pi)] * N
    
    # Run Differential Evolution
    # Strategy: 'best1bin' is standard.
    # popsize: 15 is default, maybe increase for better exploration? 20-30.
    # maxiter: We need to limit this based on time.
    # DE is expensive. Each objective eval is O(N^2). N=50 -> 2500 ops.
    # If popsize=20, maxiter=100, that's 2000 evals. 2000 * 2500 = 5M ops. Very fast.
    # We can afford many more iterations.
    
    # Let's try a robust configuration
    # We will run DE with a time-checking wrapper? 
    # SciPy DE doesn't have a built-in time callback that stops early easily without modifying the objective.
    # We can modify the objective to check time and return a bad value if time is up?
    # Or just run a fixed number of iterations that fits in the budget.
    
    # Let's estimate: 
    # N=50. 
    # DE popsize=15, maxiter=200.
    # Total evals ~ 15 * 200 = 3000.
    # Each eval: 50x50 matrix mult. ~2500 flops.
    # 3000 * 2500 = 7.5M flops. This is trivial for NumPy, likely < 1 second.
    # So we can go much higher.
    # Let's try popsize=30, maxiter=500.
    # 15000 evals. 15000 * 2500 = 37.5M flops. Still likely < 1-2 seconds.
    
    # However, DE can be slow in Python overhead if not careful.
    # Let's use a moderate setting to be safe, then refine.
    
    result = differential_evolution(
        objective,
        bounds,
        maxiter=300,
        popsize=25,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=42
    )
    
    # Check time
    if time.time() > deadline:
        # Return initial heuristic if we timed out during DE (unlikely with above settings, but safe)
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # Convert result back to Cartesian
    theta_phi = result.x.reshape(-1, 2)
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    points = np.column_stack([x, y, z])
    
    # Optional: Local refinement using Nelder-Mead or similar on the Cartesian points?
    # DE finds a good global candidate. A local polish can improve it.
    # But we must be careful with time.
    # Let's do a few steps of gradient descent or just return the DE result.
    # Given the "structural change" directive, DE alone is a strong candidate.
    # However, the "covering radius" is not exactly the same as "packing radius".
    # Maximizing min-pair-distance minimizes the covering radius in many symmetric cases.
    
    # To ensure we are valid (nonzero, normalized), they are on the sphere by construction.
    
    return points
# EVOLVE-BLOCK-END
