import numpy as np
from scipy.optimize import curve_fit, least_squares, minimize
from scipy.interpolate import CubicSpline, PchipInterpolator
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    X = np.asarray(X_train).flatten().astype(float)
    y = np.asarray(y_train).flatten().astype(float)
    n = len(X)
    
    # Handle constant case
    if np.var(y) < 1e-12:
        mean_y = np.mean(y)
        def predict_const(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_const

    best_model = None
    best_mse = np.inf
    
    def update_best(mse, predict_func):
        nonlocal best_model, best_mse
        if mse < best_mse:
            best_mse = mse
            best_model = predict_func

    # 1. Try Polynomials (Degrees 1 to 15)
    for deg in range(1, 16):
        if time.time() > deadline - 5:
            break
        try:
            if deg > n - 1:
                break
            coef = np.polyfit(X, y, deg)
            y_pred = np.polyval(coef, X)
            mse = np.mean((y_pred - y)**2)
            
            if mse < best_mse:
                best_mse = mse
                c = coef.copy()
                def predict_poly(X, c=c):
                    x = np.asarray(X).flatten()
                    return np.polyval(c, x).reshape(np.asarray(X).shape)
                best_model = predict_poly
                
            if mse < 1e-10:
                return best_model
        except Exception:
            continue

    # 2. Try Fourier Series (Stable, good for periodic/smooth functions)
    # y = a0 + sum_{k=1}^{K} (a_k cos(kx) + b_k sin(kx))
    try:
        if time.time() < deadline - 10:
            # Normalize X to [-1, 1] to help with frequency selection? 
            # Or just use raw X. Since ranges are known-ish, let's try raw X.
            # If X is large, high frequencies oscillate fast.
            # Let's try a range of max frequencies.
            for K in [1, 2, 3, 4, 5, 8, 12, 16]:
                if time.time() > deadline - 5:
                    break
                try:
                    # Build design matrix
                    # Column 0: 1
                    # Columns 1..K: cos(kx)
                    # Columns K+1..2K: sin(kx)
                    X_fourier = np.ones((n, 1))
                    for k in range(1, K + 1):
                        X_fourier = np.hstack([X_fourier, np.cos(k * X).reshape(-1, 1)])
                        X_fourier = np.hstack([X_fourier, np.sin(k * X).reshape(-1, 1)])
                    
                    # Solve linear least squares
                    coeffs, residuals, rank, sv = np.linalg.lstsq(X_fourier, y, rcond=None)
                    y_pred = X_fourier @ coeffs
                    
                    if not np.all(np.isfinite(y_pred)):
                        continue
                        
                    mse = np.mean((y_pred - y)**2)
                    
                    if mse < best_mse:
                        best_mse = mse
                        c = coeffs.copy()
                        def predict_fourier(X, c=c, K=K):
                            x = np.asarray(X).flatten()
                            Xf = np.ones((len(x), 1))
                            for k in range(1, K + 1):
                                Xf = np.hstack([Xf, np.cos(k * x).reshape(-1, 1)])
                                Xf = np.hstack([Xf, np.sin(k * x).reshape(-1, 1)])
                            return (Xf @ c).reshape(np.asarray(X).shape)
                        best_model = predict_fourier
                        
                    if mse < 1e-10:
                        return best_model
                except Exception:
                    continue
    except Exception:
        pass

    # 3. Try RBF Network (Radial Basis Function)
    # y = sum_{i=1}^{M} w_i * exp(-alpha * (x - c_i)^2)
    # Use fixed centers (e.g., quantiles of X) and solve for weights and alpha.
    try:
        if time.time() < deadline - 10:
            # Use 20 fixed centers evenly spaced in the range of X
            x_min, x_max = np.min(X), np.max(X)
            if x_max - x_min < 1e-10:
                centers = np.array([x_min])
            else:
                M = 20
                centers = np.linspace(x_min, x_max, M)
            
            # Try a few values for alpha (inverse width)
            # Alpha controls the "bumpiness". 
            # If alpha is small, it's like a linear/quadratic fit.
            # If alpha is large, it interpolates.
            alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
            
            for alpha in alphas:
                if time.time() > deadline - 5:
                    break
                try:
                    # Build design matrix
                    # phi_ij = exp(-alpha * (X_i - centers_j)^2)
                    # X shape (n, 1), centers shape (M,)
                    diff = X.reshape(-1, 1) - centers.reshape(1, -1)
                    phi = np.exp(-alpha * diff**2)
                    
                    # Solve for weights
                    weights, residuals, rank, sv = np.linalg.lstsq(phi, y, rcond=None)
                    y_pred = phi @ weights
                    
                    if not np.all(np.isfinite(y_pred)):
                        continue
                        
                    mse = np.mean((y_pred - y)**2)
                    
                    if mse < best_mse:
                        best_mse = mse
                        w = weights.copy()
                        c = centers.copy()
                        a = alpha
                        def predict_rbf(X, w=w, c=c, a=a):
                            x = np.asarray(X).flatten().reshape(-1, 1)
                            diff = x - c.reshape(1, -1)
                            phi = np.exp(-a * diff**2)
                            return (phi @ w).reshape(np.asarray(X).shape)
                        best_model = predict_rbf
                        
                    if mse < 1e-10:
                        return best_model
                except Exception:
                    continue
    except Exception:
        pass

    # 4. Interpolation Fallbacks
    try:
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        
        # Handle duplicates
        unique_X, inverse_indices = np.unique(X_sorted, return_inverse=True)
        if len(unique_X) < n:
            unique_y = np.zeros(len(unique_X))
            counts = np.zeros(len(unique_X))
            np.add.at(unique_y, inverse_indices, y_sorted)
            np.add.at(counts, inverse_indices, 1)
            unique_y /= counts
            X_sorted = unique_X
            y_sorted = unique_y
            
        # Try Cubic Spline
        cs = CubicSpline(X_sorted, y_sorted, bc_type='natural')
        y_pred = cs(X)
        mse = np.mean((y_pred - y)**2)
        
        if mse < best_mse:
            best_mse = mse
            cs_copy = cs
            def predict_spline(X):
                x = np.asarray(X).flatten()
                return cs_copy(x).reshape(np.asarray(X).shape)
            best_model = predict_spline
            
        # Try PCHIP (Monotone Cubic)
        if time.time() < deadline - 5:
            pchip = PchipInterpolator(X_sorted, y_sorted)
            y_pred = pchip(X)
            mse = np.mean((y_pred - y)**2)
            
            if mse < best_mse:
                best_mse = mse
                pchip_copy = pchip
                def predict_pchip(X):
                    x = np.asarray(X).flatten()
                    return pchip_copy(x).reshape(np.asarray(X).shape)
                best_model = predict_pchip
            
    except Exception:
        pass

    # 5. Fallback: Mean
    if best_model is None:
        mean_y = np.mean(y)
        def predict_mean(X):
            return np.full(np.asarray(X).shape, mean_y)
        best_model = predict_mean

    return best_model
