import numpy as np

# Precompute the B-spline basis functions for efficiency?
# The evaluator calls equation(x, params). x is an array of 200 points.
# We can compute the basis matrix M (200 x 10) inside the function.
# Since x is fixed during the optimization (the training set is fixed),
# we could cache the basis matrix, but the function signature doesn't allow global state easily
# unless we use a global variable or closure. However, the task says "Keep the function names...".
# Using a global cache is risky if the evaluator resets state or runs in parallel.
# But computing 200x10 basis is very fast.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x: (N,)
    # params: (10,)
    
    # Define knots for 10 linear B-splines on [-2, 2]
    # We need 12 knots for 10 basis functions of degree 1.
    # Let's use open uniform knot vector.
    n_knots = 12
    knots = np.linspace(-2, 2, n_knots)
    
    # Compute basis functions N_i,1(x) for i=0..9
    # N_i,1(x) is non-zero on [knots[i], knots[i+2]]
    
    # Efficient vectorized calculation
    # For each i, calculate the contribution.
    # Or loop over i (10 iterations) and vectorize over x.
    
    basis = np.zeros((len(x), 10))
    
    for i in range(10):
        k0 = knots[i]
        k1 = knots[i+1]
        k2 = knots[i+2]
        
        # N_i,0 is 1 on [k0, k1)
        # N_{i+1},0 is 1 on [k1, k2)
        
        # Term 1: (x - k0) / (k1 - k0) * N_i,0
        # N_i,0 is 1 where k0 <= x < k1
        mask1 = (x >= k0) & (x < k1)
        term1 = np.zeros_like(x)
        term1[mask1] = (x[mask1] - k0) / (k1 - k0)
        
        # Term 2: (k2 - x) / (k2 - k1) * N_{i+1},0
        # N_{i+1},0 is 1 where k1 <= x < k2
        mask2 = (x >= k1) & (x < k2)
        term2 = np.zeros_like(x)
        term2[mask2] = (k2 - x[mask2]) / (k2 - k1)
        
        basis[:, i] = term1 + term2
        
        # Handle boundary at x=2 exactly?
        # The last knot is 2.
        # Standard B-splines handle the right boundary with the last basis function.
        # For x == 2, mask2 for the last i (i=9) covers [k1, k2) which is [k10, k11) = [~1.63, 2).
        # x=2 is not included in mask2 if we use < k2.
        # We should ensure x=2 is covered.
        # Let's adjust masks to include the right endpoint for the last interval.
        
    # The standard definition often excludes the right endpoint for all but the last one,
    # or uses a specific convention.
    # To be safe and ensure partition of unity or coverage, let's refine the masks.
    
    # Let's restart the basis calculation with careful boundary handling.
    
    basis = np.zeros((len(x), 10))
    
    for i in range(10):
        k0 = knots[i]
        k1 = knots[i+1]
        k2 = knots[i+2]
        
        # Interval [k0, k1]
        # Contribution from left leg
        mask_left = (x >= k0) & (x <= k1)
        # Contribution from right leg
        mask_right = (x >= k1) & (x <= k2)
        
        # Avoid division by zero (k1-k0 should not be 0 for distinct knots)
        denom_left = k1 - k0
        denom_right = k2 - k1
        
        val_left = np.zeros_like(x)
        val_left[mask_left] = (x[mask_left] - k0) / denom_left
        
        val_right = np.zeros_like(x)
        val_right[mask_right] = (k2 - x[mask_right]) / denom_right
        
        # Sum contributions
        basis[:, i] = val_left + val_right
        
        # Special case for x == 2 (the last knot)
        # The last basis function i=9 covers [k9, k11].
        # At x=2, mask_right for i=9 is (2>=k10) & (2<=k11) -> True.
        # val_right = (k11 - 2) / (k11 - k10) = 0.
        # val_left for i=9 is 0 at x=2 (since x > k10).
        # So basis[9] is 0 at x=2?
        # Let's check the B-spline definition.
        # Usually, the last knot is repeated or the domain is closed.
        # With open uniform knots, the basis functions do not sum to 1 at the boundaries?
        # Actually, for B-splines with open knot vector, they do form a partition of unity on [t_p, t_{n-1}].
        # Here p=1, n=10 (num basis), knots length 12.
        # The support is [t_1, t_{10}]? No, [t_0, t_{11}].
        # The partition of unity holds on [t_p, t_{n-1}] = [t_1, t_9]?
        # Let's just ensure the function is continuous and covers the domain.
        # If there's a gap at x=2, we can force the last parameter to handle it or adjust knots.
        
    # To ensure robustness, let's use a simpler basis:
    # Linear interpolation between 10 evenly spaced centers.
    # This is equivalent to using a set of "ramp" functions or just using np.interp?
    # But we need a fixed basis for BFGS.
    
    # Let's try a different, simpler basis:
    # y = sum( w_i * max(0, 1 - |x - c_i|/h) )  (Triangular hats)
    # Centers c_i evenly spaced.
    # This is also linear in w.
    
    # Let's stick to the B-spline implementation but fix the boundary.
    # If the B-spline basis doesn't cover x=2 perfectly, we can add a small epsilon or adjust knots.
    # Actually, `np.linspace(-2, 2, 12)` gives knots.
    # The last basis function N_9,1 is non-zero on [k_9, k_11].
    # k_11 = 2.
    # At x=2, N_9,1(2) should be 1 if it's the last one in a partition?
    # Let's check the formula:
    # N_9,1(x) = (x-k9)/(k10-k9)*N_9,0(x) + (k11-x)/(k11-k10)*N_10,0(x)
    # N_10,0(x) is 1 on [k10, k11).
    # At x=2, N_10,0(2) is 0 (since x < k11 is false).
    # So N_9,1(2) = 0.
    # This means the basis is 0 at x=2. This is bad.
    
    # Solution: Use a closed knot vector or adjust the definition.
    # Easier: Use **Cosine Basis**? No, directive says piecewise linear.
    # Easier: Use **Linear Ramp Functions**?
    
    # Let's use a set of "Hinge" functions: max(0, x - k_i).
    # This is a standard basis for piecewise linear functions.
    # y = w0 + w1*x + sum_{i=1}^{8} w_i * max(0, x - k_i)
    # This has 10 parameters.
    # This is linear in w.
    # It can represent any piecewise linear function with knots at k_i.
    # Let's use 8 interior knots evenly spaced in [-2, 2].
    # k_i = -2 + i * (4/8) for i=1..8?
    # k_1 = -1.5, k_2 = -1.0, ..., k_8 = 1.5.
    # This allows for 9 segments.
    
    knots = np.linspace(-1.5, 1.5, 8)
    
    basis = np.zeros((len(x), 10))
    basis[:, 0] = 1.0
    basis[:, 1] = x
    for i in range(8):
        basis[:, 2 + i] = np.maximum(0, x - knots[i])
        
    return basis @ params
