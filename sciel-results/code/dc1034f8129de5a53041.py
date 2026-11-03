import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])


def _bary_to_cart(u, v):
    """Convert barycentric coordinates (u, v) to Cartesian coordinates.
    Point P = (1-u-v)*A + u*B + v*C
    Since A is (0,0), P = u*B + v*C
    """
    x = u * B[0] + v * C[0]
    y = u * B[1] + v * C[1]
    return np.array([x, y])


def _cart_to_bary(x, y):
    """Convert Cartesian coordinates to barycentric coordinates (u, v)."""
    v = 2.0 * y / np.sqrt(3)
    u = x - 0.5 * v
    return np.array([u, v])


def _clamp_bary(u, v):
    """Clamp barycentric coordinates to valid range [0,1] with u+v<=1."""
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] = u[mask] / s[mask]
        v[mask] = v[mask] / s[mask]
    return u, v


def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by the points.
    points: array of shape (N, 2)
    """
    min_area = np.inf
    n = len(points)
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            vec1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                vec2 = pk - pi
                area = 0.5 * abs(vec1[0] * vec2[1] - vec1[1] * vec2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _objective(params):
    """
    params: array of shape (2*N,) representing u_i, v_i for each point.
    Returns negative of the minimum area.
    """
    u = params[0:N]
    v = params[N:2 * N]
    u_clipped, v_clipped = _clamp_bary(u.copy(), v.copy())
    points = np.array([_bary_to_cart(ui, vi) for ui, vi in zip(u_clipped, v_clipped)])
    min_area = _min_triangle_area(points)
    return -min_area


def _structured_init(rng):
    """Generate a structured initial guess with good spread."""
    u = np.zeros(N)
    v = np.zeros(N)
    
    # Vertices
    u[0], v[0] = 0.0, 0.0
    u[1], v[1] = 1.0, 0.0
    u[2], v[2] = 0.5, 1.0 / np.sqrt(3) * 2 / 1.0  # Actually C is at (0.5, sqrt(3)/2), bary = (0,1)
    u[2], v[2] = 0.0, 1.0
    
    # Midpoints of edges
    u[3], v[3] = 0.5, 0.0  # Mid AB
    u[4], v[4] = 0.0, 0.5  # Mid AC
    u[5], v[5] = 0.5, 0.5  # Mid BC... wait, mid BC: B=(1,0), C=(0.5, sqrt(3)/2)
    # Mid BC in bary: average of B(1,0) and C(0,1) = (0.5, 0.5)
    
    # Centroid
    u[6], v[6] = 1/3, 1/3
    
    # Remaining 4 points: spread them
    # Use a smaller grid or random perturbations of the centroid
    # Place them at 1/6, 1/6, 1/3 positions or similar
    # Let's use random but constrained points for the rest
    for i in range(7, N):
        while True:
            ui = rng.random()
            vi = rng.random()
            if ui + vi <= 1.0:
                u[i], v[i] = ui, vi
                break
    
    # Add small random perturbations to avoid degeneracy
    u += rng.normal(0, 0.01, N)
    v += rng.normal(0, 0.01, N)
    u, v = _clamp_bary(u, v)
    
    return np.concatenate([u, v])


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_params = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initializations
    initializations = []
    
    # Structured initialization
    for _ in range(3):
        initializations.append(_structured_init(rng))
    
    # Random initializations
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N)
        s = u + v
        mask = s > 1.0
        u[mask] = u[mask] / s[mask]
        v[mask] = v[mask] / s[mask]
        initializations.append(np.concatenate([u, v]))
    
    # Phase 1: Differential Evolution for global optimization
    # Use DE to explore the space broadly
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Run DE with moderate population to find a good starting point
    try:
        # Use a moderate number of generations
        de_result = differential_evolution(
            _objective,
            bounds,
            maxiter=50,
            popsize=20,
            tol=1e-8,
            seed=42,
            polish=False,
            workers=-1
        )
        
        if de_result.fun < best_score:
            best_score = -de_result.fun
            best_params = de_result.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 30.0:
        # Not enough time for refinement, return best so far
        if best_params is not None:
            u = best_params[0:N]
            v = best_params[N:2*N]
            u, v = _clamp_bary(u, v)
            points = np.array([_bary_to_cart(ui, vi) for ui, vi in zip(u, v)])
            return points
        # Fallback
        u = rng.random(N)
        v = rng.random(N)
        s = u + v
        mask = s > 1.0
        u[mask] = u[mask] / s[mask]
        v[mask] = v[mask] / s[mask]
        points = np.array([_bary_to_cart(ui, vi) for ui, vi in zip(u, v)])
        return points
    
    # Phase 2: Local refinement with Powell/Nelder-Mead from best DE result and other good starts
    refinement_starts = []
    
    if best_params is not None:
        refinement_starts.append(best_params)
    
    # Add the best few initializations
    for init in initializations:
        score = -_objective(init)
        if score > best_score * 0.9:  # Within 10% of best
            refinement_starts.append(init)
    
    # Also add the DE result with small perturbations
    if best_params is not None:
        for _ in range(3):
            perturbed = best_params + rng.normal(0, 0.01, 2*N)
            refinement_starts.append(perturbed)
    
    # Run local optimization from each start
    for init in refinement_starts:
        if time.time() > deadline - 5.0:
            break
        
        try:
            res = minimize(_objective, init, method='Powell', 
                          options={'maxiter': 500, 'disp': False, 'xtol': 1e-10, 'ftol': 1e-10})
            
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x.copy()
                
        except Exception:
            pass
        
        # Try Nelder-Mead as well
        if time.time() > deadline - 5.0:
            break
        
        try:
            res_nm = minimize(_objective, init, method='Nelder-Mead',
                            options={'maxiter': 1000, 'disp': False, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x.copy()
                
        except Exception:
            pass
    
    # Final verification and return
    if best_params is None:
        # Fallback to random if optimization failed completely
        u = rng.random(N)
        v = rng.random(N)
        s = u + v
        mask = s > 1.0
        u[mask] = u[mask] / s[mask]
        v[mask] = v[mask] / s[mask]
        points = np.array([_bary_to_cart(ui, vi) for ui, vi in zip(u, v)])
        return points
    
    u = best_params[0:N]
    v = best_params[N:2*N]
    
    # Final clipping
    u, v = _clamp_bary(u, v)
    
    points = np.array([_bary_to_cart(ui, vi) for ui, vi in zip(u, v)])
    return points
