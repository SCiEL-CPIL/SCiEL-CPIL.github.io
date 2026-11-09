"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as so

N = 300

# EVOLVE-BLOCK-START
def _compute_radius(points):
    """
    Computes the maximum radius for a set of points.
    r = min(min_{i<j} dist(i,j)/2, min_i (1 - |p_i|))
    """
    if len(points) == 0:
        return 0.0
    
    # Distance to boundary
    dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_boundary)
    
    # Pairwise distances
    # For N=300, 300x300 matrix is 90k elements, very fast
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _hexagonal_lattice(n_points):
    """
    Generates a hexagonal lattice of points inside the unit circle.
    Returns an array of shape (n_points, 2).
    """
    # Estimate spacing to fit n_points
    # Area of hexagon approx 2.6 * r^2 per point? 
    # Let's just try a range of spacings.
    # Optimal density is pi / (2 * sqrt(3)) ~ 0.9069
    # Area of circle 1 is pi. Area of n points is n * pi * r^2.
    # n * pi * r^2 <= pi * 0.9069 => r^2 <= 0.9069 / n => r <= sqrt(0.9069/n)
    # For n=300, r <= sqrt(0.9069/300) ~ 0.055
    # Spacing d = 2r ~ 0.11
    
    # We generate a grid and pick the closest n_points to origin
    # Or generate rings.
    
    # Simple approach: generate a large hex grid and filter.
    # Spacing s.
    # Let's try s such that we get slightly more than N points.
    # Number of points in hex circle of radius R (in units of s) is approx 1 + 3*k*(k+1) where R ~ k*s.
    # Area ~ pi * (k*s)^2.
    # Let's just iterate on s.
    
    best_points = None
    best_r = 0.0
    
    # Range of s to try
    # s_min ~ 0.1, s_max ~ 0.2
    for s in np.linspace(0.10, 0.15, 20):
        # Generate hex lattice points
        # x = (i + 0.5*(j%2)) * s
        # y = j * s * sqrt(3)/2
        k_max = int(1.0 / s) + 2
        points = []
        for j in range(-k_max, k_max + 1):
            for i in range(-k_max, k_max + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                points.append([x, y])
        
        points = np.array(points)
        
        # Filter points inside unit circle (with some margin for radius)
        # We don't know r yet, but r ~ s/2.
        # Let's just take points with norm < 1.
        norms = np.linalg.norm(points, axis=1)
        valid_points = points[norms < 1.0]
        
        if len(valid_points) < N:
            continue
            
        # Sort by distance to origin to pick the "densest" core
        # This ensures we pick the most central points
        idx = np.argsort(norms[norms < 1.0])
        selected = valid_points[idx[:N]]
        
        # Compute radius for this configuration
        r = _compute_radius(selected)
        
        if r > best_r:
            best_r = r
            best_points = selected.copy()
            
    return best_points

def _objective(x):
    """
    Objective function to minimize.
    x = [r, x_1, y_1, ..., x_N, y_N]
    We want to maximize r, so we minimize -r.
    """
    return -x[0]

def _make_constraints(N):
    """
    Create constraint functions for scipy.optimize.minimize.
    Constraints are of the form g(x) >= 0.
    """
    constraints = []
    
    # Boundary constraint: ||c_i|| + r <= 1  =>  1 - r - ||c_i|| >= 0
    def boundary_constraints(x):
        r = x[0]
        points = x[1:].reshape(N, 2)
        norms = np.linalg.norm(points, axis=1)
        return 1.0 - r - norms
    
    constraints.append({'type': 'ineq', 'fun': boundary_constraints})
    
    # Non-overlap constraint: ||c_i - c_j|| >= 2r  =>  ||c_i - c_j|| - 2r >= 0
    # To save memory and computation, we can define this for all pairs i < j.
    # However, defining 45k separate constraint functions is slow to set up and call.
    # It is better to define a single function that returns an array of all pair constraints.
    
    def pair_constraints(x):
        r = x[0]
        points = x[1:].reshape(N, 2)
        
        # Compute all pairwise distances
        # diff[i, j] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # We only need i < j
        # Create a mask for i < j
        # Or simply compute for all and take upper triangle
        # dists is symmetric.
        
        # Constraint: dists[i, j] - 2r >= 0
        vals = dists - 2.0 * r
        
        # Extract upper triangle (excluding diagonal)
        # np.triu_indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        return vals[i_idx, j_idx]
        
    constraints.append({'type': 'ineq', 'fun': pair_constraints})
    
    return constraints

def _bounds(N):
    """
    Define bounds for the variables.
    r is in (0, 0.1]
    x, y are in [-1, 1]
    """
    bounds = [(0, 0.1)]
    for _ in range(N * 2):
        bounds.append((-1, 1))
    return bounds

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate initial hexagonal lattice
    init_points = _hexagonal_lattice(N)
    
    if init_points is None:
        # Fallback to random if lattice fails
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        init_points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
    init_r = _compute_radius(init_points)
    
    # Flatten initial points
    # x = [r, x1, y1, ..., xN, yN]
    x0 = np.zeros(1 + 2 * N)
    x0[0] = init_r
    x0[1:] = init_points.flatten()
    
    # 2. Setup optimization
    constraints = _make_constraints(N)
    bounds = _bounds(N)
    
    best_points = init_points.copy()
    best_r = init_r
    
    # Use SLSQP
    # maxiter might need to be high, but SLSQP is slow for N=300.
    # Each iteration involves solving a QP subproblem.
    # N=300 variables, 45k+300 constraints. This is heavy.
    # SLSQP might be too slow.
    # Let's try a smaller number of iterations or a different method if SLSQP fails.
    # However, the directive asks for SLSQP or Trust-Constr.
    # Trust-Constr is also slow.
    
    # Let's try SLSQP with a reasonable maxiter.
    # If it times out, we return the best found so far.
    
    try:
        result = so.minimize(
            _objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints, 
            options={'maxiter': 50, 'ftol': 1e-8}
        )
        
        if result.success:
            new_r = result.x[0]
            new_points = result.x[1:].reshape(N, 2)
            
            # Verify validity
            actual_r = _compute_radius(new_points)
            if actual_r > best_r:
                best_r = actual_r
                best_points = new_points.copy()
                
    except Exception:
        pass
        
    # Check time
    if time.time() < deadline - 5.0:
        # If we have time, maybe try a second run with a different initial guess or method?
        # Or just refine the current best with a local search?
        # The prompt asks for a structural change to constrained optimization.
        # Let's stick to the result of the optimization.
        
        # One potential issue: SLSQP might get stuck in a local minimum or fail to converge.
        # We can try a few random restarts if time permits.
        
        restarts = 0
        while time.time() < deadline - 5.0 and restarts < 3:
            restarts += 1
            # Perturb the initial points slightly
            noise = np.random.normal(0, 0.01, size=(N, 2))
            perturbed_points = init_points + noise
            # Project back inside
            norms = np.linalg.norm(perturbed_points, axis=1)
            mask = norms > 0.9
            perturbed_points[mask] *= 0.9 / norms[mask]
            
            perturbed_r = _compute_radius(perturbed_points)
            
            x0_pert = np.zeros(1 + 2 * N)
            x0_pert[0] = perturbed_r
            x0_pert[1:] = perturbed_points.flatten()
            
            try:
                result_pert = so.minimize(
                    _objective, 
                    x0_pert, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints, 
                    options={'maxiter': 20, 'ftol': 1e-8}
                )
                
                if result_pert.success:
                    new_r_pert = result_pert.x[0]
                    new_points_pert = result_pert.x[1:].reshape(N, 2)
                    
                    actual_r_pert = _compute_radius(new_points_pert)
                    if actual_r_pert > best_r:
                        best_r = actual_r_pert
                        best_points = new_points_pert.copy()
                        
            except Exception:
                pass

    return best_points
# EVOLVE-BLOCK-END
