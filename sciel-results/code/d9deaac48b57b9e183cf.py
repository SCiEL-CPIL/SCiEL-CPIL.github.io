# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
import warnings


def fit(X_train, y_train):
    """
    Fit a regression model using adaptive polynomial detection and piecewise cubic Hermite interpolation.
    Strategy:
    1. Sort by input feature.
    2. Handle duplicate x values by averaging y.
    3. Try to fit low-degree polynomials (1-4) exactly. If a low-degree polynomial fits the noiseless data 
       within numerical tolerance, use it. This is the most stable and accurate for simple functions.
    4. If no low-degree polynomial fits, use PchipInterpolator (Piecewise Cubic Hermite Interpolating Polynomial).
       PCHIP is preferred over standard cubic splines because it preserves monotonicity and avoids 
       oscillatory behavior (overshoots) between data points, which is crucial for unknown functions.
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    # Ensure 1D input handling
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    # If multiple features, fallback to linear regression (though task says 1 input)
    if n_features > 1:
        X_1d = X[:, 0]
        # Simple linear regression fallback
        A = np.column_stack([X_1d, np.ones_like(X_1d)])
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        def predict_linear(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test[:, 0]
            A_test = np.column_stack([x_test, np.ones_like(x_test)])
            return A_test @ w
        return predict_linear

    # 1D case
    x = X[:, 0]
    
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Check for duplicate x values
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    if len(unique_x) < len(x_sorted):
        # Average y values for duplicate x
        y_unique = np.zeros(len(unique_x))
        for i, idx in enumerate(unique_indices):
            if i < len(unique_indices) - 1:
                next_idx = unique_indices[i+1]
                y_unique[i] = y_sorted[idx:next_idx].mean()
            else:
                y_unique[i] = y_sorted[idx:].mean()
        x_sorted = unique_x
        y_sorted = y_unique
        
    n_unique = len(x_sorted)
    
    # If there are very few points, use linear interpolation
    if n_unique < 4:
        def predict_linear_interp(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test[:, 0]
            return np.interp(x_test, x_sorted, y_sorted)
        return predict_linear_interp

    # Try to fit low-degree polynomials exactly
    # We check degrees 1, 2, 3, 4
    # Tolerance for "exact" fit
    tol = 1e-9 * max(1.0, np.max(np.abs(y_sorted)))
    
    for deg in range(1, 5):
        # Construct Vandermonde matrix
        # Using np.vander is efficient
        V = np.vander(x_sorted, N=deg+1, increasing=True)
        
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(V, y_sorted, rcond=None)
            
            # Check if the fit is exact
            y_pred = V @ coeffs
            residual = np.max(np.abs(y_pred - y_sorted))
            
            if residual < tol:
                # Found an exact polynomial fit
                def predict_poly(X_test, coeffs=coeffs):
                    X_test = np.asarray(X_test, dtype=np.float64)
                    if X_test.ndim == 1:
                        X_test = X_test.reshape(-1, 1)
                    x_test = X_test[:, 0]
                    V_test = np.vander(x_test, N=len(coeffs), increasing=True)
                    return V_test @ coeffs
                return predict_poly
        except Exception:
            continue
            
    # If no low-degree polynomial fits exactly, use PCHIP
    # PCHIP is robust against oscillations
    try:
        interp = PchipInterpolator(x_sorted, y_sorted)
    except Exception:
        # Fallback to cubic spline if PCHIP fails (unlikely)
        interp = CubicSpline(x_sorted, y_sorted)
        
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        x_test = X_test[:, 0]
        return interp(x_test)
        
    return predict
# EVOLVE-BLOCK-END
