"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import basinhopping
import itertools

def construct():
    # Equilateral triangle vertices
    h = np.sqrt(3) / 2
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    
    # Pre-compute indices for all triples
    triples = list(itertools.combinations(range(N), 3))
    tri_indices = np.array(triples)
    
    def params_to_points(params):
        """
        Convert parameters (u, v) pairs to points inside the triangle.
        params shape: (N, 2)
        u in [0, 1], v in [0, 1]
        Barycentric coords: a = u, b = v*(1-u), c = 1 - a - b
        Point = a*V[0] + b*V[1] + c*V[2]
        """
        u = params[:, 0]
        v = params[:, 1]
        a = u
        b = v * (1 - u)
        c = 1 - a - b
        points = np.zeros((N, 2))
        points[:, 0] = a * V[0, 0] + b * V[1, 0] + c * V[2, 0]
        points[:, 1] = a * V[0, 1] + b * V[1, 1] + c * V[2, 1]
        return points

    def min_area(points):
        """Calculate the minimum triangle area."""
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        
        dx1 = p2[:, 0] - p1[:, 0]
        dy1 = p2[:, 1] - p1[:, 1]
        dx2 = p3[:, 0] - p1[:, 0]
        dy2 = p3[:, 1] - p1[:, 1]
        
        cross = dx1 * dy2 - dy1 * dx2
        areas = 0.5 * np.abs(cross)
        return np.min(areas)

    def objective(x):
        """
        x: shape (2*N,) flattened (u1, v1, u2, v2, ...)
        Returns negative of min triangle area.
        """
        params = x.reshape(-1, 2)
        points = params_to_points(params)
        area = min_area(points)
        return -area

    # Initial guess: Random points inside the triangle
    rng = np.random.default_rng(42)
    x0 = rng.random((N, 2)).flatten()
    
    # Define the local minimizer
    # L-BFGS-B is faster than Nelder-Mead for smooth-ish functions.
    # We can provide a gradient? No, min() makes it non-differentiable.
    # But L-BFGS-B often works well enough.
    # Let's try Nelder-Mead for robustness, but it might be slow.
    # Let's try to use L-BFGS-B with a finite difference gradient.
    local_min = {'method': 'L-BFGS-B', 'options': {'maxiter': 100, 'ftol': 1e-8}}
    
    # Run basin hopping
    # stepsize: The step size for the random perturbation.
    # Since the domain is [0,1]^22, a stepsize of 0.1-0.2 is reasonable.
    # niter: 100-200 iterations.
    
    try:
        res = basinhopping(objective, x0, niter=100, T=1.0, stepsize=0.2, 
                           minimizer_kwargs=local_min, seed=42)
        x_opt = res.x
    except Exception:
        # Fallback
        x_opt = x0

    # Convert optimal parameters to points
    params_opt = x_opt.reshape(-1, 2)
    final_points = params_to_points(params_opt)
    
    return final_points
# EVOLVE-BLOCK-END
