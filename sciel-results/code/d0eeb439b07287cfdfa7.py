"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 256
    # We will use a continuous relaxation and then round to +/-1.
    # However, gradient descent on a discrete constraint is tricky.
    # A more robust approach for this specific problem (Erdos Discrepancy-like but for APs)
    # is often combinatorial or heuristic. But the directive asks for gradient-based SGD
    # with a smooth surrogate loss.
    
    # Let x be a vector of real values. We want to minimize the max absolute sum over all APs.
    # Let S_ap be the sum of x[i] for i in an arithmetic progression.
    # Loss L(x) = sum_{all APs} (S_ap^2) + lambda * sum_{i} (x_i^2 - 1)^2 (to push towards +/-1)
    # Or better: minimize max(S_ap^2). A soft approximation is sum(exp(S_ap^2 / T)) or just sum(S_ap^4) etc.
    # Given the 60s limit, we need to be efficient.
    
    # Precompute all arithmetic progressions?
    # Number of APs in [1, 256]:
    # Length 1: 256
    # Length 2: 128 * 127 / 2? No, start a, step d, length k.
    # a + (k-1)d <= n
    # Total number of APs is roughly O(n^3) which is 256^3 ~ 16M. This might be too slow for Python loops in 60s if we do many iterations.
    # However, we can vectorize the calculation of sums if we structure it well, or just use a subset of APs for the gradient step and evaluate the true max at the end.
    
    # Strategy:
    # 1. Initialize x with random noise.
    # 2. Define a loss function that approximates the max AP sum.
    #    Using sum of squares of AP sums is differentiable.
    #    L(x) = sum_{ap} (sum_{i in ap} x_i)^2
    # 3. Perform SGD steps.
    # 4. Project x to the hypercube [-1, 1] or directly to {-1, 1} periodically.
    # 5. Final output is sign(x).
    
    # To speed up, we can precompute the incidence matrix or just iterate over APs in a vectorized way if possible.
    # Since n=256 is small, we can generate indices for all APs.
    
    ap_indices = []
    for start in range(n):
        for step in range(1, n):
            # Generate AP: start, start+step, start+2*step, ... < n
            seq = []
            cur = start
            while cur < n:
                seq.append(cur)
                cur += step
            if len(seq) > 1: # Singletons are trivial, sum is x_i, max abs is 1. We can ignore them for minimization as they don't change with other variables, but they contribute to the loss. Actually, singletons are fixed at +/-1 so their sum is always +/-1. The max will be at least 1. We want to minimize the *other* sums.
                ap_indices.append(np.array(seq))
    
    # This list might be large. Let's estimate size.
    # For each step d, number of APs is roughly n^2 / (2d).
    # Sum_{d=1}^{n} n^2 / (2d) ~ n^2 / 2 * ln(n).
    # 256^2 / 2 * ln(256) ~ 32768 * 5.5 ~ 180,000 APs.
    # This is manageable.
    
    # Convert to a sparse matrix or just a list of arrays for fast summing.
    # Actually, computing the gradient of sum(ap_sum^2) is:
    # grad_x_i = 2 * sum_{ap containing i} ap_sum * (indicator that i is in ap)
    
    # Let's build a sparse representation or just use a dense matrix if memory allows.
    # 180k APs * 256 floats = 46M floats = ~360MB. Might be tight but okay.
    # Alternatively, we can just loop over APs in the gradient calculation. 180k iterations * 256 ops = 46M ops per gradient step.
    # In Python, 46M ops is ~0.1s-0.5s. 100 steps would be 10-50s. Too slow.
    
    # We need a faster gradient.
    # We can use the structure of APs.
    # Sum_{ap} (sum_{i in ap} x_i)^2 = x^T M x, where M is a matrix.
    # M_{ij} = number of APs that contain both i and j.
    # Then grad = 2 M x.
    # Computing M:
    # For each pair (i, j), how many APs contain both?
    # An AP is defined by (start, step).
    # If i and j are in an AP with step d, then (j-i) must be a multiple of d, and they must fit in the range.
    # This is still complex to compute M efficiently without loops.
    
    # Alternative: Use a subset of APs for the gradient, or use a different optimization.
    # Maybe Simulated Annealing or Local Search is better for 60s?
    # The directive says "design a new program... built on this approach... gradient-based SGD".
    # So I must stick to SGD.
    
    # Optimization:
    # 1. Precompute M (the Gram matrix of AP incidence).
    #    M is 256x256.
    #    M[i, j] = count of APs containing both i and j.
    #    We can compute this by iterating over all APs and adding 1 to M[i, j] for all pairs i, j in the AP.
    #    Total work: sum_{ap} (len(ap)^2).
    #    Average len(ap) is small. Max len is 256 (step 1).
    #    Step 1: 1 AP of len 256 -> 256^2 = 65k ops.
    #    Step 2: 2 APs of len 128 -> 2 * 128^2 = 32k ops.
    #    ...
    #    Total ops ~ sum_{d=1}^{n} (n/d) * (n/d)^2 = n^3 sum 1/d^3 ~ n^3 * 1.2 = 16M * 1.2 ~ 20M ops.
    #    This is fast in Python if done with NumPy or even pure Python loops (20M simple adds is < 1s).
    
    M = np.zeros((n, n), dtype=np.float64)
    
    # Build M
    for start in range(n):
        for step in range(1, n):
            seq = []
            cur = start
            while cur < n:
                seq.append(cur)
                cur += step
            k = len(seq)
            if k > 1:
                # Add 1 to M[i, j] for all i, j in seq
                # This is a rank-1 update of ones matrix for the subset seq
                # We can do: M[np.ix_(seq, seq)] += 1
                # But np.ix_ creates a view/copy. For small k it's fine.
                # To be faster, we can just loop if k is small, or use vectorized add.
                # Let's use the vectorized way for simplicity, it should be fast enough for 20M total elements touched.
                idx = np.array(seq)
                M[np.ix_(idx, idx)] += 1.0
                
    # Now Loss L(x) = x^T M x
    # Grad = 2 M x
    
    # We also want to push x towards +/-1.
    # Add a term: lambda * sum((x_i^2 - 1)^2)
    # Grad of this term: 4 * lambda * x_i * (x_i^2 - 1)
    
    # Let's run SGD.
    x = np.random.uniform(-0.5, 0.5, n)
    lr = 0.01
    lam = 10.0
    steps = 200
    
    # Precompute M for speed
    # M is 256x256, very small.
    
    for t in range(steps):
        # Compute gradient
        grad_M = 2.0 * M.dot(x)
        grad_reg = 4.0 * lam * x * (x**2 - 1.0)
        grad = grad_M + grad_reg
        
        # Update
        x -= lr * grad
        
        # Project to [-1, 1] to keep values bounded?
        # Or let the regularization handle it.
        # Clipping might help stability.
        x = np.clip(x, -1.0, 1.0)
        
        # Optionally, periodically snap to +/-1 to escape local minima?
        # No, let's keep it continuous until the end.
        
    # Final discretization
    signs = np.sign(x)
    # Ensure no zeros (if x is 0, sign is 0, but we need +/-1)
    signs[signs == 0] = 1.0
    
    return [int(s) for s in signs]

# EVOLVE-BLOCK-END
