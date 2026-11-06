"""Improved program for the suite task; fits an odd-degree polynomial closure exploiting antisymmetry."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Uses the physical insight that the unresolved forcing is an odd function of the local slow state x.
    This restricts the polynomial to odd powers (x, x^3, x^5), reducing the number of parameters
    and enforcing the correct symmetry, which should improve generalization.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Ensure inputs are float64 numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle edge case of empty or tiny training sets
    if len(x) < 3:
        return np.zeros(6)
        
    # Check time
    if time.time() - start_time > deadline:
        return np.zeros(6)
        
    try:
        # Build design matrix for odd powers only: x, x^3, x^5
        # We fit y = c1*x + c3*x^3 + c5*x^5
        n = len(x)
        V = np.column_stack([x, x**3, x**5]).astype(np.float64)
        
        # Compute scaling for regularization: normalize columns to unit variance
        # This makes regularization strength comparable across polynomial orders
        col_means = V.mean(axis=0)
        col_stds = V.std(axis=0)
        col_stds[col_stds < 1e-10] = 1.0  # avoid division by zero
        
        # Try multiple regularization strengths and pick the one with best validation
        # We'll use a simple cross-validation approach on the training data
        best_coeffs = None
        best_score = np.inf
        
        # Regularization strengths to try (in log space)
        # Alpha controls the strength of L2 regularization
        alphas = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
        
        for alpha in alphas:
            if time.time() - start_time > deadline - 10.0:
                break
                
            if alpha == 0.0:
                # Unregularized least squares
                try:
                    coeffs_raw = np.linalg.lstsq(V, y, rcond=None)[0]
                    if np.all(np.isfinite(coeffs_raw)):
                        # Evaluate on training data for comparison
                        pred = V @ coeffs_raw
                        mse = np.mean((pred - y) ** 2)
                        if mse < best_score:
                            best_score = mse
                            best_coeffs = coeffs_raw
                except Exception:
                    pass
            else:
                # Ridge regression: (V^T V + alpha * I) c = V^T y
                # But we need to scale appropriately
                # Use scaled regularization: alpha * diag(col_variances)
                col_variances = col_stds ** 2
                reg_matrix = np.diag(col_variances) * alpha
                
                try:
                    A = V.T @ V + reg_matrix
                    b = V.T @ y
                    coeffs_raw = np.linalg.solve(A, b)
                    if np.all(np.isfinite(coeffs_raw)):
                        # Evaluate on training data for comparison
                        pred = V @ coeffs_raw
                        mse = np.mean((pred - y) ** 2)
                        if mse < best_score:
                            best_score = mse
                            best_coeffs = coeffs_raw
                except Exception:
                    pass
        
        # If no good solution found, fallback to simple polyfit
        if best_coeffs is None:
            try:
                # Fallback to linear fit in x
                coeffs_lin = np.polynomial.polynomial.polyfit(x, y, 1)
                coeffs = np.zeros(6)
                coeffs[1] = coeffs_lin[1]
                return coeffs
            except Exception:
                return np.zeros(6)
        
        # Map the 3 coefficients (for x, x^3, x^5) to the 6-coefficient output
        coeffs = np.zeros(6)
        coeffs[1] = best_coeffs[0]
        coeffs[3] = best_coeffs[1]
        coeffs[5] = best_coeffs[2]
        
        return coeffs
        
    except Exception:
        # Fallback to linear fit in case of any errors
        try:
            coeffs_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeffs = np.zeros(6)
            coeffs[1] = coeffs_lin[1]
            return coeffs
        except Exception:
            return np.zeros(6)
# EVOLVE-BLOCK-END
