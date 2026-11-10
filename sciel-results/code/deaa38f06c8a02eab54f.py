import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    n = 8
    
    def objective_radial(params):
        """
        params: 4 floats representing radial distances for angles 0, pi/4, pi/2, 3pi/4
        The points are generated as:
        P_i = r_i * [cos(theta_i), sin(theta_i)] for theta_i in [0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4]
        where r_i repeats for opposite angles: r_0, r_1, r_2, r_3, r_0, r_1, r_2, r_3.
        
        This enforces central symmetry (P_i = -P_{i+4}) and 90-degree rotational symmetry? 
        Wait, if we use r_0, r_1, r_2, r_3 for angles 0, 45, 90, 135, and repeat, 
        we enforce central symmetry. 
        Does it enforce 90-degree rotational symmetry? 
        Angle 0: r_0
        Angle 90: r_2
        If r_0 != r_2, it is not 90-degree rotationally symmetric. 
        So this allows general centrally symmetric octagons with vertices on the 8 standard rays.
        This is a 4D subspace of the 8D angular+radial space (or 16D Cartesian space).
        """
        if time.time() > deadline:
            return 1e10
            
        r = params
        
        # Ensure radii are positive
        if np.any(r < 1e-6):
            return 1e10
            
        angles = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
        radii = np.array([r[0], r[1], r[2], r[3], r[0], r[1], r[2], r[3]])
        
        points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        # Calculate pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(points[i] - points[j])
                dists[i, j] = d
                dists[j, i] = d
                if d < 1e-8:
                    return 1e10
        
        D = np.max(dists)
        if D < 1e-8:
            return 1e10
            
        mask = ~np.eye(n, dtype=bool)
        valid_dists = dists[mask]
        
        if np.any(valid_dists < 1e-8):
            return 1e10
            
        log_delta = np.sum(np.log(valid_dists))
        
        # Score = Delta_scaled / n^n
        # log_score = log_delta + 56*log(2) - 56*log(D) - n*log(n)
        
        log_score = log_delta + 56 * np.log(2) - 56 * np.log(D) - n * np.log(n)
        
        return -log_score

    def objective_general(params):
        """
        params: 16 floats (8 x, 8 y)
        """
        if time.time() > deadline:
            return 1e10
            
        points = params.reshape((n, 2))
        
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(points[i] - points[j])
                dists[i, j] = d
                dists[j, i] = d
                if d < 1e-8:
                    return 1e10
        
        D = np.max(dists)
        if D < 1e-8:
            return 1e10
            
        mask = ~np.eye(n, dtype=bool)
        valid_dists = dists[mask]
        
        if np.any(valid_dists < 1e-8):
            return 1e10
            
        log_delta = np.sum(np.log(valid_dists))
        
        log_score = log_delta + 56 * np.log(2) - 56 * np.log(D) - n * np.log(n)
        
        return -log_score

    best_result = None
    best_val = np.inf
    
    # Strategy 1: Optimize the 4 radial parameters
    # This is a low-dimensional problem, so DE can be very effective.
    # Bounds: [1e-3, 10]
    bounds_radial = [(1e-3, 10)] * 4
    
    try:
        result_radial = differential_evolution(
            objective_radial, 
            bounds_radial, 
            maxiter=500, 
            popsize=20, 
            tol=1e-10, 
            seed=42,
            polish=True,
            maxfev=20000
        )
        
        if result_radial.fun < best_val:
            best_val = result_radial.fun
            # Convert radial result to Cartesian for potential further refinement or return
            r = result_radial.x
            angles = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
            radii = np.array([r[0], r[1], r[2], r[3], r[0], r[1], r[2], r[3]])
            best_result = np.column_stack((radii * np.cos(angles), radii * np.sin(angles))).flatten()
            
    except Exception as e:
        pass
        
    # Strategy 2: If time remains, refine the best result using full 16D local search
    if time.time() < deadline - 10 and best_result is not None:
        try:
            # Use the current best as initial guess for local refinement
            # We can use L-BFGS-B or Nelder-Mead
            res = minimize(
                objective_general, 
                best_result, 
                method='L-BFGS-B', 
                options={'maxiter': 1000}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
                
        except Exception:
            pass
            
    # Fallback: Regular Octagon
    if best_result is None or best_val > 1e9:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_result = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
    
    # Final check: ensure distinct points
    points = best_result.reshape((n, 2))
    
    # Verify distinctness
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-6:
                # Perturb slightly
                points[j] += np.random.normal(0, 1e-5, 2)
                
    # Return as list of lists
    return [[float(p[0]), float(p[1])] for p in points]
