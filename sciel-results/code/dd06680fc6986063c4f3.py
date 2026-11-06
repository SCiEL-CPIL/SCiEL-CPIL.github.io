# EVOLVE-BLOCK-START
import numpy as np
from numpy.polynomial import polynomial as P
from math import comb


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation using Robust Iteratively
    Reweighted Least Squares (IRLS) with a Tukey's Biweight loss function.
    This approach is robust to outliers and high-variance regions, potentially
    generalizing better to unseen forcing regimes than standard OLS.
    
    Args:
        training_x: array-like, flattened slow-state values
        training_subgrid_tendency: array-like, flattened subgrid tendency values
    
    Returns:
        np.ndarray of shape (6,) with coefficients in ASCENDING powers of x
    """
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Filter out any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        return np.zeros(6)
    
    # Compute statistics for scaling
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    
    # Scale x to [-1, 1] range approximately for numerical stability
    x_scaled = (x - x_mean) / x_std
    
    # Scale y to help with weight calculation sensitivity
    y_scaled = (y - y_mean) / y_std
    
    # Build Vandermonde matrix for degree 5
    def make_vandermonde(x_vals, deg=5):
        powers = np.arange(deg + 1)
        return x_vals[:, None] ** powers
    
    V = make_vandermonde(x_scaled, 5)
    n = len(x_scaled)
    
    # Initialize weights
    w = np.ones(n)
    
    # IRLS Parameters
    # Tukey's biweight constant (1.5946 is standard for 95% efficiency)
    c = 1.5946
    # Scale estimate for residuals (MAD is robust)
    # We will update scale in each iteration or use initial robust scale
    # Initial scale: 1.4826 * MAD(y)
    med_y = np.median(y_scaled)
    mad_y = np.median(np.abs(y_scaled - med_y))
    if mad_y < 1e-12:
        scale = 1.0
    else:
        scale = 1.4826 * mad_y
    
    # Iterate IRLS
    max_iter = 50
    tol = 1e-8
    prev_coeff = None
    
    for _ in range(max_iter):
        # Weighted Least Squares: (V^T W V) c = V^T W y
        # W is diagonal with weights w
        W = np.diag(w)
        
        try:
            # Use solve for stability
            A = V.T @ W @ V
            b = V.T @ W @ y_scaled
            
            # Add small ridge for numerical stability if needed
            A += 1e-10 * np.eye(6)
            
            coeff_scaled = np.linalg.solve(A, b)
            
            # Check convergence
            if prev_coeff is not None:
                if np.linalg.norm(coeff_scaled - prev_coeff) < tol:
                    break
            
            prev_coeff = coeff_scaled
            
            # Compute residuals
            y_pred = V @ coeff_scaled
            residuals = y_scaled - y_pred
            
            # Update scale based on current residuals (optional, but robust)
            med_r = np.median(residuals)
            mad_r = np.median(np.abs(residuals - med_r))
            if mad_r > 1e-12:
                current_scale = 1.4826 * mad_r
            else:
                current_scale = scale
            
            # Update weights using Tukey's Biweight
            # u = residual / scale
            u = residuals / current_scale
            abs_u = np.abs(u)
            
            # Tukey's biweight weight function
            # w(u) = (1 - u^2)^2 if |u| <= c, else 0
            weights_new = np.zeros_like(u)
            valid = abs_u <= c
            weights_new[valid] = (1 - u[valid]**2)**2
            
            # Avoid zero weights causing singular matrix in next step
            # If all weights are zero, fallback to uniform
            if np.sum(weights_new) < 1e-10:
                weights_new = np.ones(n)
            else:
                # Normalize weights to sum to n for consistency
                weights_new = weights_new / np.sum(weights_new) * n
                
            w = weights_new
            
        except np.linalg.LinAlgError:
            # Fallback to simple OLS if weighted solve fails
            try:
                coeff_scaled, _, _, _ = np.linalg.lstsq(V, y_scaled, rcond=None)
            except:
                coeff_scaled = np.zeros(6)
            break

    # coeff_scaled is in terms of x_scaled and y_scaled
    # y_scaled = (y - y_mean) / y_std => y = y_scaled * y_std + y_mean
    # x_scaled = (x - x_mean) / x_std
    
    # Polynomial P(x_scaled) = sum c_i x_scaled^i approximates y_scaled
    # We want Q(x) = sum d_i x^i approximates y
    # y = y_std * P(x_scaled) + y_mean
    # P(x_scaled) = sum c_i ((x - x_mean) / x_std)^i
    # y = y_std * sum c_i / x_std^i * (x - x_mean)^i + y_mean
    
    # Expand (x - x_mean)^i into powers of x
    coeff_original = np.zeros(6)
    
    # The constant term from y_mean
    coeff_original[0] += y_mean
    
    for i in range(6):
        c_i = coeff_scaled[i]
        if abs(c_i) < 1e-15:
            continue
        
        # Term is y_std * c_i / x_std^i * (x - x_mean)^i
        scale_factor = y_std * c_i / (x_std ** i)
        
        # Expand (x - x_mean)^i = sum_{j=0}^i binom(i, j) x^j (-x_mean)^(i-j)
        for j in range(i + 1):
            coeff_original[j] += scale_factor * comb(i, j) * ((-x_mean) ** (i - j))
    
    # Clip to finite values
    coeff_original = np.where(np.isfinite(coeff_original), coeff_original, 0.0)
    
    return coeff_original
# EVOLVE-BLOCK-END
