"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    using Chebyshev approximation for numerical stability and better generalization.
    
    The strategy is:
    1. Map the slow state x to the Chebyshev domain [-1, 1].
    2. Fit the Chebyshev coefficients (c_0 to c_5) using least squares.
       Chebyshev polynomials are orthogonal on [-1, 1], leading to a well-conditioned
       system compared to monomials.
    3. Convert the Chebyshev coefficients to the standard monomial basis (a_0 to a_5)
       as required by the output interface.
    
    Args:
        training_x: Flattened array of slow-state values
        training_subgrid_tendency: Flattened array of unresolved tendencies
        
    Returns:
        Array of 6 finite real polynomial coefficients in ascending powers of x
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Remove any potential NaNs/Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.zeros(6)

    # 1. Determine the mapping from x to t in [-1, 1]
    # We use the min/max of the training data.
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        # Degenerate case: all x are the same. Return a constant fit.
        c0 = np.mean(y)
        return np.array([c0, 0, 0, 0, 0, 0])
    
    # Mapping: t = (2*x - (x_max + x_min)) / (x_max - x_min)
    # This maps x_min -> -1 and x_max -> 1.
    # Inverse: x = (t * x_range + x_max + x_min) / 2
    # Let's define the affine transformation x = A*t + B
    A = x_range / 2.0
    B = (x_max + x_min) / 2.0
    
    # Compute t for training data
    t = (x - B) / A
    # Clip to [-1, 1] to handle any numerical drift or if we want to strictly stay in domain
    # Note: For Chebyshev fit, t is the variable.
    t = np.clip(t, -1.0, 1.0)

    # 2. Construct the Chebyshev design matrix
    # T_k(t) = cos(k * arccos(t))
    # We need degree 5, so k=0..5.
    # Using recurrence is more stable than direct cos for higher degrees, but 5 is low.
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    N = len(t)
    T = np.zeros((N, 6))
    T[:, 0] = 1.0
    if N > 0:
        T[:, 1] = t
        for k in range(2, 6):
            T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
    
    # 3. Solve for Chebyshev coefficients c_cheb
    # Minimize ||T @ c_cheb - y||^2
    # Using SVD or lstsq for robustness
    try:
        c_cheb, residuals, rank, sv = np.linalg.lstsq(T, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to simple mean if linear algebra fails
        c_cheb = np.zeros(6)
        c_cheb[0] = np.mean(y)

    # 4. Convert Chebyshev coefficients to Monomial coefficients
    # We have P(t) = sum_{k=0}^5 c_k T_k(t)
    # We want P(x) = sum_{j=0}^5 a_j x^j
    # Substitute t = (x - B) / A = (1/A)x - B/A
    # Let u = x. t = alpha * u + beta, where alpha = 1/A, beta = -B/A.
    # P(u) = sum_{k=0}^5 c_k T_k(alpha * u + beta)
    
    # We can compute the monomial coefficients by expanding each T_k(alpha*u + beta)
    # into powers of u and summing up the contributions.
    
    alpha = 1.0 / A
    beta = -B / A
    
    a_monomial = np.zeros(6)
    
    for k in range(6):
        ck = c_cheb[k]
        if abs(ck) < 1e-15:
            continue
            
        # We need the coefficients of T_k(alpha*u + beta) in powers of u.
        # Let z = alpha*u + beta.
        # T_k(z) is a polynomial in z of degree k.
        # T_k(z) = sum_{m=0}^{floor(k/2)} ... (explicit formula is complex)
        # Easier: Evaluate T_k at a set of points and fit? No, that's what we are avoiding.
        # Better: Use the fact that T_k(z) can be represented in monomial basis of z,
        # then substitute z = alpha*u + beta.
        
        # Step 4a: Get monomial coefficients of T_k(z) in powers of z.
        # Let T_k(z) = sum_{j=0}^k d_j z^j.
        # We can find d_j by evaluating T_k at k+1 points and inverting Vandermonde,
        # or using recurrence.
        # Since k is small (<=5), we can just evaluate T_k at 6 points and use polyfit?
        # Actually, there is a direct way using the Chebyshev recurrence on the basis.
        
        # Let's represent T_k(z) in the monomial basis of z: [1, z, z^2, z^3, z^4, z^5]
        # T_0(z) = 1
        # T_1(z) = z
        # T_2(z) = 2z^2 - 1
        # T_3(z) = 4z^3 - 3z
        # T_4(z) = 8z^4 - 8z^2 + 1
        # T_5(z) = 16z^5 - 20z^3 + 5z
        
        # Let's hardcode the monomial coefficients of T_k(z) for k=0..5 to avoid any ambiguity.
        # d[k][j] is coeff of z^j in T_k(z)
        
        d_coeffs = np.zeros((6, 6))
        d_coeffs[0, 0] = 1.0
        d_coeffs[1, 1] = 1.0
        d_coeffs[2, 2] = 2.0; d_coeffs[2, 0] = -1.0
        d_coeffs[3, 3] = 4.0; d_coeffs[3, 1] = -3.0
        d_coeffs[4, 4] = 8.0; d_coeffs[4, 2] = -8.0; d_coeffs[4, 0] = 1.0
        d_coeffs[5, 5] = 16.0; d_coeffs[5, 3] = -20.0; d_coeffs[5, 1] = 5.0
        
        # Now substitute z = alpha*u + beta.
        # z^j = (alpha*u + beta)^j = sum_{m=0}^j binom(j, m) alpha^m beta^(j-m) u^m
        # We want to add ck * sum_j d_coeffs[k, j] * (alpha*u + beta)^j to a_monomial.
        
        for j in range(k + 1):
            dj = d_coeffs[k, j]
            if abs(dj) < 1e-15:
                continue
                
            # Expand (alpha*u + beta)^j
            # Coeff of u^m is binom(j, m) * alpha^m * beta^(j-m)
            for m in range(j + 1):
                # Binomial coefficient C(j, m)
                binom_jm = 1.0
                for q in range(m):
                    binom_jm *= (j - q) / (q + 1)
                
                term = dj * binom_jm * (alpha ** m) * (beta ** (j - m))
                a_monomial[m] += ck * term

    # Check time
    if time.time() - start_time > DEADLINE:
        # Return best found so far
        pass

    # Final sanity check: ensure coefficients are finite
    if not np.all(np.isfinite(a_monomial)):
        # Fallback to simple linear fit in original domain if transformation failed
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            a_monomial = np.zeros(6)
            a_monomial[:2] = c_lin
        except:
            a_monomial = np.zeros(6)

    return a_monomial
# EVOLVE-BLOCK-END
