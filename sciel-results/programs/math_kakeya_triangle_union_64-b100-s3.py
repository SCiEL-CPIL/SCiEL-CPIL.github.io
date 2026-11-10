# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area(x, n=64, M=20000):
    """Compute the union area of the n triangles given base positions x."""
    a = (np.arange(n) + 1) / n  # a_i for i=0..n-1
    # y_m = (m + 0.5)/20000 for m=0..19999
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    
    # L_i = x_i + a_i * y, shape (n, M)
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1 - y[None, :]) / n  # shape (1, M)
    
    # Sort L along axis 0 for each column (height)
    L_sorted = np.sort(L, axis=0)  # shape (n, M)
    
    # Union length at each height: w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    diffs = L_sorted[1:, :] - L_sorted[:-1, :]  # shape (n-1, M)
    union_lengths = w[0] + np.sum(np.minimum(diffs, w[0]), axis=0)  # shape (M,)
    
    area = np.mean(union_lengths)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimize the area of the union of 64 Kakeya needle triangles.
    Uses the Keich construction as an initial guess and refines it using
    L-BFGS-B optimization, exploiting the convexity of the area function.
    """
    n = 64
    k = int(np.log2(n))
    
    # Generate Keich initial guess
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        # Get binary representation of i with k bits
        bits = list(f"{i:0{k}b}")
        eps = np.array(bits, dtype=int)
        # Calculate x_i
        # Formula: sum_{j=1}^k ((1 - j) / k) * eps_j * 2^{-j}
        # Note: The prompt's formula uses 1-based indexing for j in the sum?
        # "sum_{j=1}^k ((1 - j) / k) * eps_j * 2^{-j}"
        # My j array is 1..k.
        # eps corresponds to bits. Which bit is eps_1?
        # Usually MSB is bit 1? Or LSB?
        # The prompt says "expand a_i = i/n in binary as a_i = sum_j eps_j 2^{-j}".
        # This implies eps_1 is the MSB (value 1/2) if i/n < 1?
        # Actually a_i = (i+1)/n.
        # Let's stick to the prompt's implementation logic in the seed:
        # eps = np.array(list(str(bin(i))[2:]), dtype=int)
        # This gives MSB first.
        # The seed code used:
        # eps = np.pad(eps, (k - len(eps), 0), "constant")
        # positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
        # I will replicate this exactly.
        
        eps_seed = np.array(list(str(bin(i))[2:]), dtype=int)
        eps_seed = np.pad(eps_seed, (k - len(eps_seed), 0), "constant")
        positions.append(np.sum((1 - j) * eps_seed * (2.0 ** -j) / k))
        
    x0 = np.array(positions, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 500  # Leave 100s buffer
    
    # Define objective function for scipy
    def objective(x):
        return _compute_area(x, n, 20000)
    
    # Initial evaluation
    current_area = objective(x0)
    best_x = x0.copy()
    best_area = current_area
    
    # Optimization
    try:
        # Use L-BFGS-B. It is efficient for large scale problems.
        # We don't provide gradients, so it uses numerical approximation.
        # Since the function is convex, this should work well.
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_area:
            best_x = res.x
            best_area = res.fun
            
        # Check time
        if time.time() > deadline:
            pass # Just return best found
            
    except Exception:
        # If optimization fails, return the initial guess
        pass

    # Verify validity
    if not np.all(np.isfinite(best_x)):
        return x0
        
    return best_x

# EVOLVE-BLOCK-END
