# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Strategy: Symmetry-Constrained Discrete Search + Vectorized Coordinate Descent.
    
    1. The problem is scale-invariant, so we can fix d_max=1 or work on the unit sphere initially.
    2. We hypothesize the optimal configuration for 14 points in 3D has high symmetry.
       Common high-symmetry structures for N=14 include:
       - Icosahedron (12) + 2 Poles (already tried, score ~17)
       - Two concentric rings of 7 points each? (Unlikely to be optimal due to clustering)
       - One ring of 14? (Unlikely)
       - Icosahedron + 2 points not at poles?
       
    3. New Approach:
       - Start with a diverse set of initial configurations, including ones with lower symmetry 
         to avoid local minima of the "Icosahedron+Poles" basin.
       - Use a fast, custom coordinate descent optimizer.
         Instead of Nelder-Mead which evaluates the full objective (O(N^2) distance calc) 
         for every simplex vertex (many evaluations), we update one coordinate at a time.
         For a given point i, we can analytically or via 1D search find the best position 
         that maximizes the minimum distance to all other points, while keeping others fixed.
         Actually, maximizing min_dist is non-smooth. 
         
       - Better: Use a "Repulsion" based local search with vectorized updates.
         Or, simply use a very fast multi-start Nelder-Mead but with a much faster objective function 
         by pre-computing indices or using C-optimized distance calculations.
         
       - Let's try a different angle: **Simulated Annealing** or **Basin Hopping** with a very fast 
         local optimizer.
         
    4. Implementation Plan:
       - Define a fast objective function.
       - Use multiple diverse starting points:
         a) Icosahedron + Poles
         b) Icosahedron + 2 points optimized in a small neighborhood
         c) Random points on sphere (multiple seeds)
         d) "Ring" structures: 7 points on equator, 7 points on a tilted ring?
       - Run a fast local optimization (Nelder-Mead with tight tolerances) for each.
       - Keep the best.
       
    5. Optimization:
       - The current program uses Nelder-Mead which is slow in high dimensions (42D).
       - We can reduce dimensionality by enforcing symmetry where possible, but for N=14, 
         full symmetry is hard.
       - However, we can use a "stiffness" trick: penalize points that are too close.
       - Let's stick to Cartesian Nelder-Mead but make it faster by:
         - Using a smaller initial simplex.
         - Using a time-budgeted loop that stops if no improvement is seen for a while.
         - More importantly, start from *better* initial guesses.
         
    6. Key Insight from Inspirations:
       - Inspiration 1 (score 34.6) mentions "materially different strategy" and "numpy, scipy, time".
       - It likely uses a more efficient global search or a better local optimizer.
       - Let's try **Differential Evolution** (DE) but with a very small population and fast convergence, 
         or **Basin Hopping** with a fast local minimizer.
         
    7. Let's implement a robust multi-start local search with a time guard.
       - We will generate 10-20 diverse initial configurations.
       - For each, run Nelder-Mead with a strict time limit per start.
       - We will also try a "shaking" step: if stuck, perturb and restart.
       
    8. Even better: Use **scipy.optimize.minimize** with method='Powell' or 'BFGS' if we can compute gradients?
       - Gradients of min distance are hard (non-differentiable at min).
       - So gradient-free is necessary.
       
    9. Let's focus on **Diversity of Initializations** and **Fast Evaluation**.
       - We will use a set of pre-computed "good" topologies:
         - Icosahedron + Poles
         - Icosahedron + 2 points at (0, 1, 0) and (0, -1, 0) rotated?
         - 14 points on a sphere with random distribution (Fibonacci sphere)
         - 14 points on a sphere with random distribution (Uniform)
       - Then refine each.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # 20s margin

    def get_score(pts):
        """Calculate (d_min / d_max)^2"""
        if pts.shape != (n, d):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def objective(params):
        """
        Objective for Cartesian coordinates.
        params: shape (3n, )
        Returns negative score for minimization.
        """
        if time.time() > deadline:
            return 1e10
        pts = params.reshape(n, 3)
        # Center of mass constraint to break translation invariance
        pts = pts - np.mean(pts, axis=0)
        score = get_score(pts)
        return -score

    best_score_global = -np.inf
    best_params_cart = None

    # Generate diverse initial configurations
    inits = []

    # 1. Icosahedron + Poles
    phi_golden = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi_golden], [0, -1, phi_golden],
        [0, 1, -phi_golden], [0, -1, -phi_golden],
        [1, phi_golden, 0], [-1, phi_golden, 0],
        [1, -phi_golden, 0], [-1, -phi_golden, 0],
        [phi_golden, 0, 1], [-phi_golden, 0, 1],
        [phi_golden, 0, -1], [-phi_golden, 0, -1]
    ])
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    inits.append(np.vstack([ico_verts, poles]))

    # 2. Icosahedron + 2 points at different positions (e.g., on equator, 90 deg apart from ico vertices)
    # Ico vertices on equator are at phi=0, 72, 144, 216, 288 deg.
    # Let's put extra points at 36, 108, 180, 252, 324? No, only 2 points.
    # Try 36 deg and 216 deg (opposite).
    extra1 = np.array([np.cos(np.deg2rad(36)), np.sin(np.deg2rad(36)), 0])
    extra2 = np.array([np.cos(np.deg2rad(216)), np.sin(np.deg2rad(216)), 0])
    inits.append(np.vstack([ico_verts, extra1, extra2]))

    # 3. Fibonacci Sphere (14 points)
    def fibonacci_sphere(n):
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    inits.append(fibonacci_sphere(14))

    # 4. Random Uniform on Sphere (Multiple seeds)
    np.random.seed(123)
    for _ in range(5):
        pts = np.random.randn(n, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)

    # 5. Ring Structure: 7 on equator, 7 on a tilted ring?
    # Let's try 1 North Pole, 1 South Pole, 6 on equator, 6 on a ring at 45 deg?
    # Total 14.
    ring1 = np.array([[np.cos(2*np.pi*i/6), np.sin(2*np.pi*i/6), 0] for i in range(6)])
    ring2 = np.array([[np.cos(2*np.pi*i/6 + np.pi/6), np.sin(2*np.pi*i/6 + np.pi/6), 1/np.sqrt(2)] for i in range(6)])
    # Scale ring2 to unit sphere? No, z is fixed, so x,y must be scaled by sqrt(1-z^2).
    # Let's just use spherical coords for this.
    # 1 at (0,0,1), 1 at (0,0,-1), 6 at theta=pi/2, 6 at theta=pi/3?
    # Let's stick to the previous inits which are more standard.
    
    # Run local optimization for each init
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Center and normalize
        init_pts = init_pts - np.mean(init_pts, axis=0)
        dists = pdist(init_pts)
        d_max = np.max(dists)
        if d_max > 0:
            init_pts = init_pts / d_max
        
        try:
            res = minimize(
                objective, 
                init_pts.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res.fun < -best_score_global:
                best_score_global = -res.fun
                best_params_cart = res.x.copy()
                
        except Exception as e:
            pass

    # If we have time, do a second pass with more iterations on the best found so far
    if best_params_cart is not None and time.time() < deadline - 30:
        try:
            res_refine = minimize(
                objective, 
                best_params_cart, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res_refine.fun < -best_score_global:
                best_score_global = -res_refine.fun
                best_params_cart = res_refine.x.copy()
                
        except Exception as e:
            pass

    # Fallback
    if best_params_cart is None:
        best_params_cart = inits[0].flatten()

    points = best_params_cart.reshape(n, 3)
    
    # Final cleanup
    points -= np.mean(points, axis=0)
    dists = pdist(points)
    d_max = np.max(dists)
    if d_max > 1e-12:
        points = points / d_max

    if not np.all(np.isfinite(points)):
        # Ultimate fallback
        points = inits[0]
        points -= np.mean(points, axis=0)
        dists = pdist(points)
        d_max = np.max(dists)
        if d_max > 0:
            points = points / d_max

    return points

from scipy.optimize import minimize

# EVOLVE-BLOCK-END
