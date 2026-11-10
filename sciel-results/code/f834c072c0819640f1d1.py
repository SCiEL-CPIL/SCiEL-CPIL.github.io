import numpy as np
import time
import scipy.optimize
import scipy.spatial.distance

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600 - 10  # 590 seconds to be safe
    
    def get_score(points):
        """Calculate the actual score R = (d_min/d_max)^2"""
        # Ensure points is (n, 2)
        pts = points.reshape(n, d)
        dists = scipy.spatial.distance.pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        """
        Objective for minimization (DE minimizes).
        We want to maximize R, so we minimize -R.
        """
        score = get_score(x)
        return -score

    # Generate diverse initial candidates to seed the best-so-far
    candidates = []
    
    # Candidate 1: 4x4 Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    candidates.append(grid_pts)
    
    # Candidate 2: Regular 16-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(circle_pts)
    
    # Candidate 3: Hexagonal-like packing (3 rings? 1+6+9? No. 1+6+12=19. 1+6+9=16? No.
    # Let's try a distorted grid with shifted rows
    hex_pts = grid_pts.copy()
    for i in range(1, 4, 2):
        hex_pts[i*4:(i+1)*4, 0] += 0.5
    candidates.append(hex_pts)
    
    # Candidate 4: Random
    np.random.seed(42)
    rand_pts = np.random.uniform(-2, 2, (n, d))
    candidates.append(rand_pts)
    
    best_points = None
    best_score = -np.inf
    
    for c in candidates:
        s = get_score(c)
        if s > best_score:
            best_score = s
            best_points = c.copy()
            
    # Strategy: Differential Evolution for Global Search
    # Bounds: The scale doesn't matter, but we need to constrain the domain to prevent divergence.
    # Normalizing the points so that d_max = 1 is helpful, but DE works on raw coords.
    # We can fix one point at the origin and another at (1,0) to remove translation and scale invariance?
    # That reduces dimensions to 2*(16-2) = 28.
    # However, standard DE on 32 dims is fine.
    # To help DE, we can bound the search space. A 4x4 grid with spacing 1 has extent ~3.
    # Let's use bounds [-5, 5] for all coordinates.
    
    bounds = [(-5, 5)] * (n * d)
    
    # DE Parameters:
    # popsize: 15 (15 * 32 = 480 individuals)
    # maxiter: 100
    # tol: 1e-8
    # mutation: (0.5, 1.0)
    # recombination: 0.7
    
    # We will run DE in chunks to respect the deadline and update best_points.
    
    # Since scipy's differential_evolution doesn't easily allow early stopping with callback updates to global state 
    # in a way that preserves the best found *during* the run for the final return if it crashes, 
    # we can implement a simple custom DE or use the callback.
    
    # Actually, scipy.optimize.differential_evolution has a callback function.
    # We can use it to update best_points.
    
    def de_callback(xk, convergence):
        nonlocal best_points, best_score
        # xk is the current best individual found by DE so far? 
        # Documentation: "callback(xk, convergence) is called after each generation.
        # xk is the current best individual."
        # Note: This is not the global best, but the best in the current population?
        # Actually, for DE, xk is the best individual in the current population.
        # We should evaluate it to be sure.
        s = get_score(xk)
        if s > best_score:
            best_score = s
            best_points = xk.reshape(n, d).copy()

    try:
        # Run DE
        # maxiter is a bit tricky to set dynamically based on time.
        # We estimate time per iteration.
        # Let's run for a fixed number of iterations that fits in time.
        # 480 individuals * 120 distances * ~1us per dist calc (vectorized) = ~0.06s per generation?
        # Actually, pdist is very fast. 120 distances.
        # Let's say 0.1s per generation. 100 generations = 10s.
        # We have 590s. We can afford many more.
        # Let's try popsize=20, maxiter=500.
        
        # To be safe and allow for multiple restarts or a final polish,
        # we will run DE with a moderate budget.
        
        res = scipy.optimize.differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=20, 
            tol=1e-8, 
            mutation=(0.5, 1.0), 
            recombination=0.7, 
            seed=42,
            callback=de_callback,
            polish=True, # Use Nelder-Mead on the best result to polish
            init='sobol' # Sobol sequence for better initial spread
        )
        
        # Update best with the final result of DE
        if res.success:
            final_pts = res.x.reshape(n, d)
            final_score = get_score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()
                
    except Exception:
        pass

    # If we still have time, perform a local refinement on the best point found
    if best_points is not None and time.time() < deadline - 5:
        try:
            # Use L-BFGS-B on the smoothed objective for final polish
            def smoothed_objective(x, beta=50.0):
                points = x.reshape(n, d)
                dists = scipy.spatial.distance.pdist(points)
                max_d = np.max(dists)
                term_min = np.exp(-beta * (dists - max_d))
                d_min_soft = max_d - (1.0 / beta) * np.log(np.sum(term_min))
                term_max = np.exp(beta * (dists - max_d))
                d_max_soft = max_d + (1.0 / beta) * np.log(np.sum(term_max))
                if d_max_soft == 0:
                    return 0.0
                return -(d_min_soft / d_max_soft)
            
            res2 = scipy.optimize.minimize(
                smoothed_objective, 
                best_points.flatten(), 
                args=(50.0,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res2.success:
                polished_pts = res2.x.reshape(n, d)
                polished_score = get_score(polished_pts)
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished_pts.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = candidates[0]
        
    return best_points
