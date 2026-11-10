"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.optimize as opt
import time
import math

N = 150

# EVOLVE-BLOCK-START
def _energy_and_grad(x):
    """
    x: (N, 3) array of points on unit sphere (or near it).
    Returns (energy, grad) where energy = sum_{i<j} log(1/|x_i - x_j|).
    We project x to sphere first.
    """
    # Project to sphere
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    x = x / norms

    # Pairwise distances
    # Use (x_i - x_j) dot (x_i - x_j) = 2 - 2 * c_ij
    # d_ij = sqrt(2 - 2*c_ij)
    # log(1/d_ij) = -log(d_ij) = -0.5 * log(2 - 2*c_ij)

    # c_ij = x_i dot x_j
    c = x @ x.T
    # Set diagonal to 0 to avoid log(0)
    np.fill_diagonal(c, 0.0)

    # d_ij = sqrt(2 - 2*c_ij)
    # But we need to be careful: c_ij should be in [-1, 1]
    c = np.clip(c, -1.0, 1.0)

    # 2 - 2*c_ij
    two_minus_two_c = 2.0 - 2.0 * c
    # Avoid log(0)
    two_minus_two_c = np.maximum(two_minus_two_c, 1e-12)

    # energy = sum_{i<j} -0.5 * log(2 - 2*c_ij)
    # We compute the full matrix sum and divide by 2, subtract diagonal (which is 0 anyway)
    log_term = -0.5 * np.log(two_minus_two_c)
    np.fill_diagonal(log_term, 0.0)
    energy = np.sum(log_term) / 2.0

    # Gradient:
    # E = sum_{i<j} -0.5 log(2 - 2 c_ij)
    # dE/dx_i = sum_{j != i} d/dx_i [-0.5 log(2 - 2 c_ij)]
    # d/dx_i log(2 - 2 c_ij) = 1/(2-2 c_ij) * (-2 * d c_ij/dx_i) = -2/(2-2 c_ij) * (x_j) = -x_j/(1-c_ij)
    # So dE/dx_i = sum_{j != i} -0.5 * (-x_j/(1-c_ij)) = sum_{j != i} x_j / (2(1-c_ij))
    # Wait: dE/dx_i = sum_{j != i} -0.5 * (1/(2-2 c_ij)) * (-2 x_j) = sum_{j != i} x_j / (2(1-c_ij))
    
    # Let's recompute:
    # E_ij = -0.5 log(2 - 2 c_ij)
    # dE_ij/dx_i = -0.5 * 1/(2-2 c_ij) * (-2) * x_j = x_j / (2(1-c_ij))
    # So grad_i = sum_{j != i} x_j / (2(1-c_ij))
    
    # Note: 1 - c_ij can be small. 
    # We want: grad_i = sum_{j != i} x_j / (2 * (1 - c_ij))
    # Let's form a matrix W where W_ij = 1/(2(1-c_ij)) for i != j, 0 for i=j
    # Then grad = W @ x

    one_minus_c = 1.0 - c
    np.fill_diagonal(one_minus_c, 1.0)  # avoid div by zero, but we'll zero out diagonal later
    W = 1.0 / (2.0 * one_minus_c)
    np.fill_diagonal(W, 0.0)

    grad = W @ x
    return energy, grad

def _project(x):
    """Project points to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return x / norms

def _fibonacci_sphere(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # 10s safety margin
    
    best_x = None
    best_e = np.inf

    # Multiple initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_fibonacci_sphere(N, offset=0.0))
    inits.append(_fibonacci_sphere(N, offset=0.7))
    
    # Add a few random initializations
    rng = np.random.default_rng(42)
    for _ in range(3):
        x_rand = rng.normal(size=(N, 3))
        inits.append(_project(x_rand))

    for x0 in inits:
        if time.time() > deadline:
            break
        
        x0 = _project(x0)
        
        # L-BFGS-B optimization in R^3 (with projection in gradient step is tricky for L-BFGS-B)
        # Instead, we optimize on the sphere using a parameterization or just use L-BFGS-B in R^3 
        # and project at the end, but that's not strictly on the manifold.
        
        # Better: use scipy.optimize.minimize with method='L-BFGS-B' but we need to handle the constraint.
        # Since the energy is invariant to scaling (log(1/|xi-xj|) changes if we scale... wait)
        # If we scale x -> alpha x, then |xi - xj| -> alpha |xi - xj|, so log(1/(alpha d)) = -log(alpha) - log(d).
        # The sum changes by -N(N-1)/2 * log(alpha). So it's NOT scale invariant.
        # We MUST keep points on the sphere.
        
        # Option 1: Optimize in 2N variables (spherical coords).
        # Option 2: Use a retraction-based method.
        # Option 3: Just optimize in R^3 and project at each evaluation. This is a common heuristic.
        
        # Let's try optimizing in R^3 and projecting inside the function.
        def fun_and_grad(y):
            x = _project(y)
            e, g = _energy_and_grad(x)
            # Gradient of e w.r.t y:
            # y = alpha * x where x is on sphere. Actually y is projected to x.
            # This chain rule is messy.
            # Instead, let's just do L-BFGS-B on the sphere using a coordinate patch? No.
            
            # Let's stick to the plan: optimize in R^3, but the gradient needs to be correct for y.
            # If we use y as the variable, and x = y/|y|, then:
            # dE/dy_i = dE/dx_i * dx_i/dy_i
            # This is complicated.
            
            # Alternative: Use scipy's minimize with a constraint? No, L-BFGS-B doesn't support general constraints.
            
            # Let's switch strategy: Use spherical coordinates.
            # theta in [0, pi], phi in [0, 2*pi].
            # This avoids the singularity at poles if we're careful, or we can just use the R^3 projection heuristic which often works well enough for L-BFGS-B if we start on the sphere.
            
            # Actually, the most robust simple method for "on the sphere" with L-BFGS-B is to just optimize in R^3 and project. The gradient w.r.t R^3 coordinates is not the true gradient on the manifold, but L-BFGS-B often finds a good local minimum anyway because the projection is smooth and the energy landscape is similar.
            
            # However, a better approach for high precision is to use the "Gaussian" parametrization or just accept the R^3 heuristic and then do a polish step with a manifold-aware method.
            
            # Let's do this: 
            # 1. Optimize in R^3 with projection.
            # 2. Take the result, project to sphere.
            # 3. Do a few iterations of gradient descent on the sphere (retraction: x_new = x - step * grad_projected).
            
            return e, g

        # Let's implement a proper manifold optimizer for the polish phase.
        # For the main optimization, let's use the R^3 heuristic.
        
        res = opt.minimize(
            fun_and_grad, x0, method='L-BFGS-B', jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        x_opt = _project(res.x)
        
        # Polish phase: Gradient descent on the sphere
        # grad_sphere = grad - (grad . x) x  (project gradient to tangent space)
        for _ in range(50):
            if time.time() > deadline:
                break
            e, g = _energy_and_grad(x_opt)
            
            # Project gradient to tangent space
            grad_proj = g - np.sum(g * x_opt, axis=1, keepdims=True) * x_opt
            
            # Adaptive step size
            step = 0.01
            # Simple line search? Or fixed step with decay?
            # Let's try a simple backtracking line search
            alpha = 1.0
            for _ in range(10):
                x_try = x_opt - alpha * grad_proj
                x_try = _project(x_try)
                e_try, _ = _energy_and_grad(x_try)
                if e_try < e:
                    break
                alpha *= 0.5
            else:
                break
            
            x_opt = x_try
            if alpha < 1e-6:
                break

        e_final, _ = _energy_and_grad(x_opt)
        
        if e_final < best_e:
            best_e = e_final
            best_x = x_opt

    if best_x is None:
        return _fibonacci_sphere(N)
    
    return best_x
# EVOLVE-BLOCK-END

