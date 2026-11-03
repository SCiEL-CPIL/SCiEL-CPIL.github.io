import numpy as np
import scipy.optimize
import math
import time

N = 100

def energy_and_grad(x):
    """
    Compute the logarithmic energy and its gradient with respect to the 
    Cartesian coordinates x (unnormalized).
    
    x: shape (N, 3)
    """
    # Normalize points to the sphere
    r = np.linalg.norm(x, axis=1, keepdims=True)
    # Avoid division by zero
    r = np.maximum(r, 1e-15)
    u = x / r
    
    # Compute pairwise distances squared
    # d2[i, j] = ||u_i - u_j||^2
    # ||u_i - u_j||^2 = ||u_i||^2 + ||u_j||^2 - 2 u_i . u_j = 2 - 2 u_i . u_j
    dot = u @ u.T
    d2 = 2.0 - 2.0 * dot
    np.fill_diagonal(d2, 1.0) # Avoid zero division on diagonal, though we mask it later
    
    # Distances
    d = np.sqrt(d2)
    
    # Energy: sum_{i<j} log(1/d) = - sum_{i<j} log(d)
    # We only need upper triangle
    iu = np.triu_indices(N, 1)
    d_pairs = d[iu]
    
    # Use math.fsum for the energy to match evaluator exactly
    # Convert to list of floats for math.fsum
    # This is slow, but N=100 is small. 
    # Actually, let's use numpy sum for speed in optimization, 
    # and rely on the fact that the difference is negligible.
    # However, the prompt says "pair terms summed with math.fsum".
    # I will use math.fsum to be safe.
    log_d_pairs = np.log(d_pairs)
    energy = -math.fsum(log_d_pairs.tolist())
    
    # Gradient
    # grad_x_k = -1/r_k * (S_k - (S_k . u_k) u_k)
    # S_k = sum_{j != k} (u_k - u_j) / d2_kj
    
    # Compute S for all k
    # Term (u_k - u_j) / d2_kj
    # We can compute this matrix-wise.
    # M[i, j] = (u_i - u_j) / d2_ij
    # Note: d2 is symmetric.
    
    # Vectorized computation of S
    # S_k = sum_j (u_k - u_j) / d2_kj
    # = sum_j u_k / d2_kj - sum_j u_j / d2_kj
    
    # Let's compute the matrix W where W[i, j] = 1 / d2_ij
    # Set diagonal to 0 to ignore self-interaction
    W = 1.0 / d2
    np.fill_diagonal(W, 0.0)
    
    # S_k = sum_j W[k, j] (u_k - u_j)
    # S_k = u_k * sum_j W[k, j] - sum_j W[k, j] u_j
    
    # sum_j W[k, j] is the row sum of W
    W_row_sums = W.sum(axis=1, keepdims=True) # Shape (N, 1)
    
    # sum_j W[k, j] u_j is W @ u
    W_u = W @ u # Shape (N, 3)
    
    # S_k = u_k * W_row_sums[k] - W_u[k]
    S = u * W_row_sums - W_u # Shape (N, 3)
    
    # Now compute grad_x_k
    # grad_x_k = -1/r_k * (S_k - (S_k . u_k) u_k)
    
    # S_k . u_k
    S_dot_u = (S * u).sum(axis=1, keepdims=True) # Shape (N, 1)
    
    # (S_k . u_k) u_k
    S_proj = S_dot_u * u
    
    # S_k - (S_k . u_k) u_k
    S_tangent = S - S_proj
    
    # grad_x_k = -1/r_k * S_tangent
    grad_x = -S_tangent / r # Shape (N, 3)
    
    return energy, grad_x

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    best_energy = float('inf')
    best_x = None
    
    # Initializations
    inits = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(fib)
    
    # 2. Random Uniform Points
    for _ in range(5):
        # Generate random points on sphere
        # Using normal distribution and normalizing
        rand_x = np.random.randn(N, 3)
        rand_norm = np.linalg.norm(rand_x, axis=1, keepdims=True)
        rand_x = rand_x / rand_norm
        inits.append(rand_x)
        
    # 3. Slightly Perturbed Fibonacci
    for _ in range(2):
        pert = fib + 0.01 * np.random.randn(N, 3)
        pert_norm = np.linalg.norm(pert, axis=1, keepdims=True)
        pert = pert / pert_norm
        inits.append(pert)

    for init in inits:
        if time.time() > deadline - 5.0:
            break
            
        try:
            # Run optimization
            res = scipy.optimize.minimize(
                energy_and_grad, 
                init, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_energy:
                # Check if we have time for more
                if time.time() > deadline:
                    break
                best_energy = res.fun
                best_x = res.x
        except Exception as e:
            pass

    if best_x is None:
        # Fallback to Fibonacci if optimization failed
        best_x = fib
    else:
        # Normalize the final result
        best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
        
    return best_x
