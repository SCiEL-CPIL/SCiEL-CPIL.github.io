import numpy as np
import time
import math
from scipy.spatial.distance import cdist

N = 100

def _log_energy(X):
    """Compute the pure logarithmic energy sum_{i<j} log(1/d_ij) with math.fsum for accuracy."""
    dists = cdist(X, X)
    np.fill_diagonal(dists, 1.0)
    d_safe = np.maximum(dists, 1e-12)
    
    # Extract upper triangle to sum each pair once
    iu = np.triu_indices(N, k=1)
    vals = -np.log(d_safe[iu])
    
    # Use math.fsum for highest accuracy as per contract
    return math.fsum(vals.tolist())

def _energy_and_tangent_grad(X):
    """
    Compute energy and gradient projected onto tangent spaces.
    X: (N, 3) array of unit vectors.
    Returns:
    E: scalar energy
    G: (N, 3) array of gradients in the tangent space (G[i] dot X[i] == 0)
    """
    # Compute pairwise distances
    # Using cdist for efficiency
    dists = cdist(X, X)
    np.fill_diagonal(dists, 1.0)
    
    # Safe distances to avoid log(0) or div(0)
    d_safe = np.maximum(dists, 1e-12)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # We compute sum over all i!=j and divide by 2
    log_dists = np.log(d_safe)
    E = -0.5 * np.sum(log_dists)
    
    # Gradient of energy w.r.t. coordinates in R^3 (ambient gradient)
    # dE/dx_i = -0.5 * sum_{j!=i} (x_i - x_j) / d_ij^2
    # Note: The factor 0.5 comes from summing over all pairs i!=j in the full matrix
    # Let's derive carefully.
    # E = sum_{i<j} -log(d_ij)
    # dE/dx_i = sum_{j!=i, j>i} -1/d_ij * d(d_ij)/dx_i + sum_{j<i} -1/d_ij * d(d_ij)/dx_i
    # d(d_ij)/dx_i = (x_i - x_j)/d_ij
    # So term is - (x_i - x_j) / d_ij^2
    # Total ambient gradient g_amb_i = - sum_{j!=i} (x_i - x_j) / d_ij^2
    
    # Vectorized computation of ambient gradient
    # diff[i,j] = x_i - x_j
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :] # (N, N, 3)
    
    # inv_dists_sq[i,j] = 1 / d_ij^2
    inv_dists_sq = np.zeros_like(dists)
    mask = dists > 1e-12
    inv_dists_sq[mask] = 1.0 / (dists[mask] ** 2)
    
    # g_amb_i = - sum_j (x_i - x_j) / d_ij^2
    # g_amb = - sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    g_amb = -np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    
    # Project ambient gradient onto tangent space of the sphere at x_i
    # proj(v) = v - (v . x_i) x_i
    # g_tan_i = g_amb_i - (g_amb_i . x_i) x_i
    
    # Compute dot products g_amb_i . x_i
    dots = np.sum(g_amb * X, axis=1) # (N,)
    
    # Project
    G = g_amb - dots[:, np.newaxis] * X
    
    return E, G

def _fibonacci_sphere(n, seed=0):
    """Generate a Fibonacci lattice on the sphere."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Add slight noise to break symmetry
    pts += rng.randn(n, 3) * 0.01
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _random_sphere(n, seed=0):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _rsgd_optimize(X_init, deadline):
    """
    Riemannian Stochastic Gradient Descent with momentum.
    Uses a coarse-to-fine learning rate schedule.
    """
    X = X_init.copy()
    # Momentum buffer
    M = np.zeros_like(X)
    
    # Hyperparameters
    # We use a decaying learning rate
    # Start with a larger step to escape local minima, then reduce for precision
    lr_start = 0.05
    lr_end = 1e-5
    momentum = 0.9
    
    # Total steps estimate: we have roughly 100-120 seconds.
    # Each step involves cdist (N^2) and some vector ops.
    # Let's estimate 1000 steps per second is too high, maybe 100-500 steps/sec.
    # Let's just run until deadline.
    
    best_E = _log_energy(X)
    best_X = X.copy()
    no_improve_count = 0
    patience = 2000 # steps without improvement before reducing LR further
    
    # We will adaptively reduce LR if stuck
    current_lr = lr_start
    
    while time.time() < deadline - 5.0:
        # Compute gradient
        E, G = _energy_and_tangent_grad(X)
        
        # Update momentum
        M = momentum * M - current_lr * G
        
        # Retraction: X_new = (X + M) / ||X + M||
        X_new = X + M
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-12)
        X_new = X_new / norms
        
        # Check for improvement
        E_new = _log_energy(X_new)
        
        if E_new < best_E:
            best_E = E_new
            best_X = X_new.copy()
            no_improve_count = 0
            # Optionally increase momentum or keep LR
        else:
            no_improve_count += 1
            # If stuck, decay LR
            if no_improve_count > patience:
                current_lr *= 0.5
                no_improve_count = 0
                # Reset momentum to avoid being trapped in a direction
                M *= 0.5
                
        X = X_new
        
    return best_X, best_E

def construct():
    start_time = time.time()
    # Safety margin: 145s deadline (5s buffer)
    deadline = start_time + 145.0
    
    best_X = None
    best_energy = float('inf')
    
    # Multi-start strategy
    # 1. Fibonacci lattice (structured)
    # 2. Random starts (diverse)
    
    inits = []
    
    # Fibonacci with different noise seeds
    for s in range(3):
        inits.append(_fibonacci_sphere(N, seed=s))
        
    # Random starts
    for s in range(10):
        inits.append(_random_sphere(N, seed=100 + s))
        
    # Shuffle inits to ensure diversity in time budget
    rng = np.random.RandomState(42)
    rng.shuffle(inits)
    
    # Allocate time to each start
    # We have ~145s. Let's give each start a slice, but allow early termination if stuck?
    # RSGD is simpler than L-BFGS, so we can run it for a fixed number of steps or time.
    # Let's try running each for a specific time slice.
    time_per_start = 10.0 # 10 seconds per start. 13 starts = 130s.
    
    for i, X_init in enumerate(inits):
        if time.time() > deadline - time_per_start:
            break
            
        # Run RSGD for a specific duration
        local_deadline = time.time() + time_per_start
        X_opt, E_opt = _rsgd_optimize(X_init, local_deadline)
        
        if E_opt < best_energy:
            best_energy = E_opt
            best_X = X_opt.copy()
            
    if best_X is None:
        # Fallback
        best_X = _fibonacci_sphere(N, seed=0)
        best_energy = _log_energy(best_X)
        
    # Final refinement phase:
    # Take the best result and run a longer, lower-LR RSGD to polish it.
    if time.time() < deadline - 10.0:
        X_polish, E_polish = _rsgd_optimize(best_X, deadline)
        if E_polish < best_energy:
            best_energy = E_polish
            best_X = X_polish.copy()
            
    # Final normalization
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    
    return best_X
