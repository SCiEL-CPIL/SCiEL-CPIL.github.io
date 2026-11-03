"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _energy_and_grad(X):
    """Compute logarithmic energy and gradient.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    # X is (N, 3)
    # diff[i, j] = X[i] - X[j]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    d2 = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero and log(0)
    d2_safe = np.maximum(d2, 1e-15)
    
    # Energy: sum over i<j of -log(sqrt(d2)) = -0.5 * sum_{i<j} log(d2)
    # We only need upper triangle
    iu = np.triu_indices(N, k=1)
    d2_upper = d2[iu]
    energy = -0.5 * math.fsum(np.log(d2_upper).tolist())
    
    # Gradient
    # grad[i] = sum_{j!=i} diff[i,j] / d2[i,j]
    # Note: diagonal is 0/0, we set it to 0
    inv_d2 = np.zeros_like(d2_safe)
    np.fill_diagonal(inv_d2, 0.0)
    # Calculate for all off-diagonals
    # We can compute full gradient matrix then sum
    # grad[i] = sum_j (X[i]-X[j])/d2[i,j]
    # This is equivalent to:
    # G[i,j] = diff[i,j] * inv_d2[i,j]
    # grad = sum_j G[i,j]
    
    # Efficient computation:
    # We can't just divide directly because of the diagonal.
    # Create a mask for non-diagonal
    mask = ~np.eye(N, dtype=bool)
    inv_d2[mask] = 1.0 / d2_safe[mask]
    
    # Gradient = sum over j of (X[i] - X[j]) / d2[i,j]
    # grad[i] = sum_j X[i]/d2[i,j] - sum_j X[j]/d2[i,j]
    # sum_j X[i]/d2[i,j] = X[i] * sum_j 1/d2[i,j]
    # sum_j X[j]/d2[i,j] is a matrix-vector product: inv_d2 @ X
    
    sum_inv_d2 = np.sum(inv_d2, axis=1, keepdims=True) # (N, 1)
    term1 = X * sum_inv_d2
    term2 = inv_d2 @ X
    
    grad = term1 - term2
    return energy, grad

def _project(X):
    """Project points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _fibonacci_sphere(n):
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence."""
    # Use scipy's Sobol for quasi-random points in [0,1]^3
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    # Generate 2^k points, take first n
    m = 10
    while (1 << m) < n:
        m += 1
    points = sampler.random(1 << m)[:n]
    # Map [0,1]^3 to sphere
    # One method: use inverse CDF or just normalize random vectors
    # Better: map to uniform on sphere using spherical coordinates
    # u1, u2, u3 in [0,1]
    # z = 2*u1 - 1
    # theta = 2*pi*u2
    # r = sqrt(1 - z^2)
    # x = r*cos(theta), y = r*sin(theta)
    u = points
    z = 2.0 * u[:, 0] - 1.0
    theta = 2.0 * math.pi * u[:, 1]
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])

def _random_sphere(n, seed):
    """Generate n random points on unit sphere."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _riemannian_gd(X, max_iter=100, lr_init=0.1, deadline=None):
    """Riemannian Gradient Descent with backtracking line search."""
    X = _project(X.copy())
    E, G = _energy_and_grad(X)
    
    # Project gradient to tangent space
    # G_tangent = G - (G . x) x
    G_tan = G - np.sum(X * G, axis=1, keepdims=True) * X
    
    v = np.zeros_like(X)
    lr = lr_init
    momentum = 0.9
    
    for i in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Momentum step in tangent space
        v = momentum * v - lr * G_tan
        X_new = X + v
        X_new = _project(X_new)
        
        E_new, G_new = _energy_and_grad(X_new)
        
        if E_new < E:
            X = X_new
            E = E_new
            G = G_new
            # Decay learning rate slowly
            if i % 20 == 0 and i > 0:
                lr *= 0.95
        else:
            # Backtracking
            lr *= 0.5
            v = 0.5 * v
            if lr < 1e-8:
                break
                
    return X, E

def _polish_lbfgs(X, deadline):
    """Polish using L-BFGS-B with penalty method to stay on sphere."""
    # Objective: E(X) + penalty * sum(||x_i||^2 - 1)^2
    # Gradient: G_E + 4 * penalty * x_i * (||x_i||^2 - 1)
    
    penalty = 1e8
    
    def obj_and_grad(flat_x):
        X = flat_x.reshape(N, 3)
        E, G_E = _energy_and_grad(X)
        
        # Penalty term
        norms2 = np.sum(X * X, axis=1)
        penalty_term = penalty * np.sum((norms2 - 1.0) ** 2)
        
        # Gradient of penalty term: 4 * penalty * x_i * (||x_i||^2 - 1)
        G_pen = 4.0 * penalty * X * (norms2 - 1.0)[:, np.newaxis]
        
        return E + penalty_term, (G_E + G_pen).ravel()
    
    x0 = X.ravel()
    
    def callback(xk):
        return time.time() > deadline
    
    try:
        res = minimize(
            lambda x: obj_and_grad(x)[0],
            x0,
            jac=lambda x: obj_and_grad(x)[1],
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'ftol': 1e-15,
                'gtol': 1e-10,
                'callback': callback
            }
        )
        if res.success or np.isfinite(res.fun):
            X_res = _project(res.x.reshape(N, 3))
            E_res, _ = _energy_and_grad(X_res)
            return X_res, E_res
    except Exception:
        pass
    return X, _energy_and_grad(X)[0]

def construct():
    start_time = time.time()
    
    # Initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_sobol_sphere(N, seed=42))
    inits.append(_random_sphere(N, seed=123))
    inits.append(_random_sphere(N, seed=456))
    
    best_X = None
    best_E = float('inf')
    
    # Allocate time for each restart
    # Total time ~ 145s. Let's use 30s per restart for 4 restarts + final polish.
    time_per_restart = 30.0
    
    for i, X0 in enumerate(inits):
        deadline = start_time + time_per_restart * (i + 1)
        if time.time() > deadline - 5:
            break
            
        # Phase 1: Riemannian GD
        X_gd, E_gd = _riemannian_gd(X0, max_iter=200, lr_init=0.05, deadline=deadline - 5)
        
        # Phase 2: L-BFGS-B Polish
        X_pol, E_pol = _polish_lbfgs(X_gd, deadline - 2)
        
        if E_pol < best_E:
            best_E = E_pol
            best_X = X_pol
            
    # Final high-precision polish on the best result found
    if best_X is not None and time.time() < DEADLINE - 10:
        # Do a few more iterations of Riemannian GD with very small lr
        # Or another L-BFGS-B pass with higher penalty?
        # Let's do a final Riemannian GD polish with tight tolerance
        X_final, E_final = _riemannian_gd(best_X, max_iter=100, lr_init=0.01, deadline=DEADLINE - 5)
        if E_final < best_E:
            best_E = E_final
            best_X = X_final
            
        # One last L-BFGS-B pass
        X_last, E_last = _polish_lbfgs(best_X, DEADLINE - 2)
        if E_last < best_E:
            best_E = E_last
            best_X = X_last

    # Ensure valid output
    best_X = _project(best_X)
    
    # Check for close points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < 1e-9:
                # Perturb
                rng = np.random.default_rng(i+j)
                best_X[j] += 1e-8 * rng.standard_normal(3)
                best_X = _project(best_X)
    
    return best_X

# EVOLVE-BLOCK-END
