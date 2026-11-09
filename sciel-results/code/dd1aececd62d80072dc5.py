import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t_start = time.time()
    deadline = 145.0  # Safety margin from 150s
    
    # Convert to numpy arrays
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Handle length mismatch
    n = min(x.size, y.size)
    if n == 0:
        return np.zeros(6, dtype=np.float64)
        
    x = x[:n]
    y = y[:n]
    
    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    
    if n == 0:
        return np.zeros(6, dtype=np.float64)
        
    # If very few points, fall back to lower degree or zero
    if n < 6:
        # Fit with degree n-1 if possible, else 0
        deg = max(0, n - 1)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, deg)
            # Pad to length 6
            full_coeff = np.zeros(6)
            full_coeff[:len(coeff)] = coeff
            return full_coeff
        except Exception:
            return np.zeros(6, dtype=np.float64)

    # Check time
    if time.time() - t_start > deadline:
        # Fallback to linear fit if we are running out of time (unlikely for polyfit)
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            full_coeff = np.zeros(6)
            full_coeff[:2] = coeff
            return full_coeff
        except Exception:
            return np.zeros(6, dtype=np.float64)

    # Build Vandermonde matrix for degree 5 (ascending powers)
    V = np.vander(x, N=6, increasing=True)
    
    # Use ridge regression with tiny regularization for numerical stability
    # This helps with ill-conditioning of Vandermonde matrices
    best_coeff = None
    best_residual = np.inf
    
    # Try a range of very small regularization values
    lambdas = [0.0, 1e-14, 1e-12, 1e-10, 1e-8]
    
    for lam in lambdas:
        try:
            if lam == 0.0:
                coeff, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
            else:
                A = V.T @ V + lam * np.eye(6)
                b = V.T @ y
                coeff = np.linalg.solve(A, b)
            
            if not np.all(np.isfinite(coeff)):
                continue
            
            # Compute residual norm as a proxy for fit quality
            pred = V @ coeff
            residual = np.sum((pred - y) ** 2)
            
            if residual < best_residual:
                best_residual = residual
                best_coeff = coeff
                
        except Exception:
            continue
    
    if best_coeff is not None:
        return best_coeff.astype(np.float64)
    
    # Fallback to linear
    try:
        coeff = np.polynomial.polynomial.polyfit(x, y, 1)
        full_coeff = np.zeros(6)
        full_coeff[:2] = coeff
        return full_coeff
    except Exception:
        return np.zeros(6, dtype=np.float64)
