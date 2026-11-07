import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import least_squares
import time
import warnings

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 295.0  # Safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    X_flat = X.flatten()
    y_flat = y.flatten()
    
    # Handle constant function
    if np.allclose(y_flat, y_flat[0]):
        c = y_flat[0]
        def predict_const(X):
            return np.full(X.shape, c)
        return predict_const
    
    # Sort by X
    sort_idx = np.argsort(X_flat)
    X_sorted = X_flat[sort_idx]
    y_sorted = y_flat[sort_idx]
    
    # Handle duplicate X values
    unique_X, inverse_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
    if len(unique_X) != len(X_sorted):
        y_sum = np.zeros_like(unique_X, dtype=float)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_sorted = y_sum / counts
        X_sorted = unique_X
    
    n = len(X_sorted)
    
    # Strategy 1: PCHIP interpolation - primary method for smooth noiseless functions
    # PCHIP is monotone-preserving and works well for noiseless data with enough points
    pchip = None
    try:
        if n >= 3:
            pchip = PchipInterpolator(X_sorted, y_sorted)
    except Exception:
        pass
    
    # Strategy 2: Try to identify if the function matches a simple parametric form
    # This gives exact answers for common mathematical functions
    best_parametric_mse = np.inf
    best_parametric_predict = None
    
    # Try linear: y = a*x + b
    try:
        A = np.vstack([X_sorted, np.ones(n)]).T
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, y_sorted, rcond=None)
        y_pred = A @ coeffs
        mse = np.mean((y_pred - y_sorted) ** 2)
        if mse < 1e-10:
            def predict_linear(X, coeffs=coeffs):
                x = np.asarray(X, dtype=float).flatten()
                return (coeffs[0] * x + coeffs[1]).reshape(np.asarray(X).shape)
            best_parametric_mse = mse
            best_parametric_predict = predict_linear
    except Exception:
        pass
    
    # Try quadratic: y = a*x^2 + b*x + c
    if best_parametric_mse > 1e-12:
        try:
            A = np.vstack([X_sorted**2, X_sorted, np.ones(n)]).T
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, y_sorted, rcond=None)
            y_pred = A @ coeffs
            mse = np.mean((y_pred - y_sorted) ** 2)
            if mse < 1e-10:
                def predict_quad(X, coeffs=coeffs):
                    x = np.asarray(X, dtype=float).flatten()
                    return (coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]).reshape(np.asarray(X).shape)
                best_parametric_mse = mse
                best_parametric_predict = predict_quad
        except Exception:
            pass
    
    # Try cubic: y = a*x^3 + b*x^2 + c*x + d
    if best_parametric_mse > 1e-12:
        try:
            A = np.vstack([X_sorted**3, X_sorted**2, X_sorted, np.ones(n)]).T
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, y_sorted, rcond=None)
            y_pred = A @ coeffs
            mse = np.mean((y_pred - y_sorted) ** 2)
            if mse < 1e-10:
                def predict_cubic(X, coeffs=coeffs):
                    x = np.asarray(X, dtype=float).flatten()
                    return (coeffs[0]*x**3 + coeffs[1]*x**2 + coeffs[2]*x + coeffs[3]).reshape(np.asarray(X).shape)
                best_parametric_mse = mse
                best_parametric_predict = predict_cubic
        except Exception:
            pass
    
    # Try exponential: y = a*exp(b*x) + c
    if best_parametric_mse > 1e-12 and n >= 3:
        try:
            def exp_model(params, x):
                a, b, c = params
                return a * np.exp(b * x) + c
            
            y_min, y_max = y_sorted.min(), y_sorted.max()
            a0 = max(abs(y_max - y_min), 1.0)
            c0 = y_min
            b0 = 1.0
            
            # Multiple initial guesses
            best_exp_mse = np.inf
            best_exp_params = None
            for b_init in [0.1, 0.5, 1.0, 2.0, -0.1, -0.5, -1.0]:
                for a_init in [a0, -a0]:
                    try:
                        res = least_squares(exp_model, [a_init, b_init, c0], args=(X_sorted,),
                                           method='lm', max_nfev=200)
                        y_pred = exp_model(res.x, X_sorted)
                        mse = np.mean((y_pred - y_sorted) ** 2)
                        if np.isfinite(mse) and mse < best_exp_mse:
                            best_exp_mse = mse
                            best_exp_params = res.x
                    except Exception:
                        continue
            
            if best_exp_params is not None and best_exp_mse < 1e-10:
                def predict_exp(X, params=best_exp_params):
                    x = np.asarray(X, dtype=float).flatten()
                    return exp_model(params, x).reshape(np.asarray(X).shape)
                best_parametric_mse = best_exp_mse
                best_parametric_predict = predict_exp
        except Exception:
            pass
    
    # Try sine: y = a*sin(b*x + c) + d
    if best_parametric_mse > 1e-12 and n >= 4:
        try:
            def sine_model(params, x):
                a, b, c, d = params
                return a * np.sin(b * x + c) + d
            
            y_min, y_max = y_sorted.min(), y_sorted.max()
            a0 = (y_max - y_min) / 2
            d0 = (y_max + y_min) / 2
            
            best_sin_mse = np.inf
            best_sin_params = None
            for b_init in [0.5, 1.0, 2.0, np.pi, 2*np.pi]:
                for c_init in [0, np.pi/2, np.pi]:
                    try:
                        res = least_squares(sine_model, [a0, b_init, c_init, d0], args=(X_sorted,),
                                           method='lm', max_nfev=200)
                        y_pred = sine_model(res.x, X_sorted)
                        mse = np.mean((y_pred - y_sorted) ** 2)
                        if np.isfinite(mse) and mse < best_sin_mse:
                            best_sin_mse = mse
                            best_sin_params = res.x
                    except Exception:
                        continue
            
            if best_sin_params is not None and best_sin_mse < 1e-10:
                def predict_sin(X, params=best_sin_params):
                    x = np.asarray(X, dtype=float).flatten()
                    return sine_model(params, x).reshape(np.asarray(X).shape)
                best_parametric_mse = best_sin_mse
                best_parametric_predict = predict_sin
        except Exception:
            pass
    
    # Try power law: y = a*x^b + c (for positive x)
    if best_parametric_mse > 1e-12 and n >= 3 and np.all(X_sorted > 0):
        try:
            def power_model(params, x):
                a, b, c = params
                return a * np.power(x, b) + c
            
            y_min, y_max = y_sorted.min(), y_sorted.max()
            a0 = (y_max - y_min) / 2
            c0 = y_min
            
            best_pow_mse = np.inf
            best_pow_params = None
            for b_init in [0.5, 1.0, 2.0]:
                for a_init in [a0, -a0]:
                    try:
                        res = least_squares(power_model, [a_init, b_init, c0], args=(X_sorted,),
                                           method='lm', max_nfev=200)
                        y_pred = power_model(res.x, X_sorted)
                        mse = np.mean((y_pred - y_sorted) ** 2)
                        if np.isfinite(mse) and mse < best_pow_mse:
                            best_pow_mse = mse
                            best_pow_params = res.x
                    except Exception:
                        continue
            
            if best_pow_params is not None and best_pow_mse < 1e-10:
                def predict_pow(X, params=best_pow_params):
                    x = np.asarray(X, dtype=float).flatten()
                    x_safe = np.maximum(x, 1e-10)
                    return power_model(params, x_safe).reshape(np.asarray(X).shape)
                best_parametric_mse = best_pow_mse
                best_parametric_predict = predict_pow
        except Exception:
            pass
    
    # Try tanh: y = a*tanh(b*x + c) + d
    if best_parametric_mse > 1e-12 and n >= 4:
        try:
            def tanh_model(params, x):
                a, b, c, d = params
                return a * np.tanh(b * x + c) + d
            
            y_min, y_max = y_sorted.min(), y_sorted.max()
            a0 = (y_max - y_min) / 2
            d0 = (y_max + y_min) / 2
            
            best_tanh_mse = np.inf
            best_tanh_params = None
            for b_init in [0.5, 1.0, 2.0]:
                for c_init in [0, 1.0, -1.0]:
                    try:
                        res = least_squares(tanh_model, [a0, b_init, c_init, d0], args=(X_sorted,),
                                           method='lm', max_nfev=200)
                        y_pred = tanh_model(res.x, X_sorted)
                        mse = np.mean((y_pred - y_sorted) ** 2)
                        if np.isfinite(mse) and mse < best_tanh_mse:
                            best_tanh_mse = mse
                            best_tanh_params = res.x
                    except Exception:
                        continue
            
            if best_tanh_params is not None and best_tanh_mse < 1e-10:
                def predict_tanh(X, params=best_tanh_params):
                    x = np.asarray(X, dtype=float).flatten()
                    return tanh_model(params, x).reshape(np.asarray(X).shape)
                best_parametric_mse = best_tanh_mse
                best_parametric_predict = predict_tanh
        except Exception:
            pass
    
    # Try Gaussian: y = a*exp(-(x-b)^2/(2*c^2)) + d
    if best_parametric_mse > 1e-12 and n >= 4:
        try:
            def gauss_model(params, x):
                a, b, c, d = params
                c_safe = max(abs(c), 0.1)
                return a * np.exp(-(x - b)**2 / (2 * c_safe**2)) + d
            
            y_min, y_max = y_sorted.min(), y_sorted.max()
            a0 = y_max - y_min
            b0 = X_sorted[np.argmax(y_sorted)]
            c0 = (X_sorted.max() - X_sorted.min()) / 4
            d0 = y_min
            
            best_gauss_mse = np.inf
            best_gauss_params = None
            for a_init in [a0, -a0]:
                for c_init in [c0, c0/2, c0*2]:
                    try:
                        res = least_squares(gauss_model, [a_init, b0, c_init, d0], args=(X_sorted,),
                                           method='lm', max_nfev=200)
                        y_pred = gauss_model(res.x, X_sorted)
                        mse = np.mean((y_pred - y_sorted) ** 2)
                        if np.isfinite(mse) and mse < best_gauss_mse:
                            best_gauss_mse = mse
                            best_gauss_params = res.x
                    except Exception:
                        continue
            
            if best_gauss_params is not None and best_gauss_mse < 1e-10:
                def predict_gauss(X, params=best_gauss_params):
                    x = np.asarray(X, dtype=float).flatten()
                    return gauss_model(params, x).reshape(np.asarray(X).shape)
                best_parametric_mse = best_gauss_mse
                best_parametric_predict = predict_gauss
        except Exception:
            pass
    
    # If we found an exact parametric fit, use it
    if best_parametric_predict is not None and best_parametric_mse < 1e-10:
        return best_parametric_predict
    
    # Otherwise, use PCHIP interpolation as the best general-purpose method
    if pchip is not None:
        def predict_pchip(X):
            x = np.asarray(X, dtype=float).flatten()
            return pchip(x).reshape(np.asarray(X).shape)
        return predict_pchip
    
    # Fallback: linear interpolation
    def predict_interp(X):
        x = np.asarray(X, dtype=float).flatten()
        return np.interp(x, X_sorted, y_sorted).reshape(np.asarray(X).shape)
    return predict_interp
