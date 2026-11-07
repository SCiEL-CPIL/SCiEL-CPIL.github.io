import numpy as np
from scipy.optimize import curve_fit, least_squares
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    X_flat = np.asarray(X_train).flatten().astype(float)
    y_flat = np.asarray(y_train).flatten().astype(float)
    n = len(X_flat)
    
    # Handle constant case
    if np.var(y_flat) < 1e-12:
        mean_y = np.mean(y_flat)
        def predict_const(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_const

    best_model = None
    best_mse = np.inf
    
    # Helper to update best model
    def update_best(mse, predict_func):
        nonlocal best_model, best_mse
        if mse < best_mse:
            best_mse = mse
            best_model = predict_func

    # 1. Try Polynomials (Degrees 1 to 12)
    # Polynomials are a strong baseline for smooth functions
    for deg in range(1, 13):
        if time.time() - start_time > deadline - 10:
            break
        try:
            # Use polyfit for speed, check conditioning
            if deg > n - 1:
                break
            coef = np.polyfit(X_flat, y_flat, deg)
            y_pred = np.polyval(coef, X_flat)
            mse = np.mean((y_pred - y_flat)**2)
            
            # If it's a very good fit, save it
            if mse < best_mse:
                best_mse = mse
                c = coef.copy()
                def predict_poly(X, c=c):
                    x = np.asarray(X).flatten()
                    return np.polyval(c, x).reshape(np.asarray(X).shape)
                best_model = predict_poly
                
            # If MSE is effectively zero, we found the exact function
            if mse < 1e-10:
                return best_model
        except Exception:
            continue

    # 2. Try Parametric Functions
    # Define candidate functions and their initial parameter guesses
    # We use curve_fit which handles the optimization
    
    x_min, x_max = np.min(X_flat), np.max(X_flat)
    y_min, y_max = np.min(y_flat), np.max(y_flat)
    y_range = y_max - y_min
    x_range = x_max - x_min
    
    if x_range < 1e-6:
        # Constant x? Just return mean y
        mean_y = np.mean(y_flat)
        def predict_const_x(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_const_x

    candidates = []
    
    # Sine: A*sin(B*x + C) + D
    def func_sin(x, A, B, C, D):
        return A * np.sin(B * x + C) + D
    candidates.append((func_sin, [y_range, 2*np.pi/x_range, 0, (y_min+y_max)/2]))
    
    # Cosine: A*cos(B*x + C) + D
    def func_cos(x, A, B, C, D):
        return A * np.cos(B * x + C) + D
    candidates.append((func_cos, [y_range, 2*np.pi/x_range, 0, (y_min+y_max)/2]))
    
    # Exponential: A*exp(B*x) + C
    def func_exp(x, A, B, C):
        return A * np.exp(B * x) + C
    # Initial guess: A ~ y_range, B ~ 1/x_range, C ~ y_min
    candidates.append((func_exp, [y_range, 1.0/x_range, y_min]))
    
    # Power Law: A*x^B + C (Only valid if x > 0 for non-integer B, but let's try for x>0 domains)
    if x_min > 0.01:
        def func_pow(x, A, B, C):
            return A * np.power(x, B) + C
        candidates.append((func_pow, [1.0, 1.0, y_min]))
        
        # Log: A*log(x) + B
        def func_log(x, A, B):
            return A * np.log(x) + B
        candidates.append((func_log, [y_range, y_min]))
        
        # Rational: A/(x+B) + C
        def func_rat(x, A, B, C):
            return A / (x + B) + C
        candidates.append((func_rat, [y_range*x_range, -x_min*0.1, y_min]))
        
    # Gaussian: A*exp(-B*x^2) + C
    def func_gauss(x, A, B, C):
        return A * np.exp(-B * x**2) + C
    candidates.append((func_gauss, [y_range, 1.0/(x_range**2), y_min]))
    
    # Hyperbolic Tangent: A*tanh(B*x) + C
    def func_tanh(x, A, B, C):
        return A * np.tanh(B * x) + C
    candidates.append((func_tanh, [y_range, 1.0/x_range, (y_min+y_max)/2]))
    
    # Quadratic (already covered by poly, but explicit fit can be more robust with bounds)
    # Cubic (covered by poly)
    
    # Absolute value: A*|x| + B (Piecewise linear)
    def func_abs(x, A, B):
        return A * np.abs(x) + B
    candidates.append((func_abs, [y_range/x_range, y_min]))
    
    # Square root: A*sqrt(|x|) + B (if domain allows)
    # Let's stick to smooth functions first.
    
    for func, p0 in candidates:
        if time.time() - start_time > deadline - 10:
            break
        try:
            # Set reasonable bounds to help curve_fit
            # A can be negative, B usually positive for frequency/decay, C/D offset
            # We'll use least_squares for more control or curve_fit with bounds
            # curve_fit can be slow or fail if p0 is bad.
            
            # Determine bounds
            # For simplicity, we'll try curve_fit first with default bounds (unbounded)
            # If it fails or is slow, we might need to be more careful.
            # Given 200 points, curve_fit should be fast.
            
            popt, pcov = curve_fit(func, X_flat, y_flat, p0=p0, maxfev=2000)
            y_pred = func(X_flat, *popt)
            mse = np.mean((y_pred - y_flat)**2)
            
            if mse < best_mse:
                best_mse = mse
                f = func
                p = popt.copy()
                def predict_param(X, f=f, p=p):
                    x = np.asarray(X).flatten()
                    return f(x, *p).reshape(np.asarray(X).shape)
                best_model = predict_param
                
            if mse < 1e-10:
                return best_model
                
        except Exception:
            continue

    # 3. Try a combination of basis functions via Linear Least Squares
    # This is effectively finding a sparse linear combination of basis functions
    # Basis: 1, x, x^2, x^3, sin(x), cos(x), sin(2x), cos(2x), exp(x), exp(-x), 1/x (if safe)
    
    if time.time() - start_time < deadline - 20:
        try:
            # Build design matrix
            X = X_flat
            cols = [np.ones_like(X)]
            cols.append(X)
            cols.append(X**2)
            cols.append(X**3)
            cols.append(X**4)
            cols.append(np.sin(X))
            cols.append(np.cos(X))
            cols.append(np.sin(2*X))
            cols.append(np.cos(2*X))
            cols.append(np.exp(X))
            cols.append(np.exp(-X))
            if np.all(X > 0.01):
                cols.append(1.0/X)
                cols.append(np.log(X))
            
            A_matrix = np.column_stack(cols)
            
            # Solve linear least squares
            # Use lstsq for robustness
            coeffs, residuals, rank, s = np.linalg.lstsq(A_matrix, y_flat, rcond=None)
            
            y_pred = A_matrix @ coeffs
            mse = np.mean((y_pred - y_flat)**2)
            
            if mse < best_mse:
                best_mse = mse
                A_m = A_matrix
                c = coeffs.copy()
                
                # Need to reconstruct the same basis for prediction
                def predict_basis(X):
                    x = np.asarray(X).flatten()
                    cols_p = [np.ones_like(x)]
                    cols_p.append(x)
                    cols_p.append(x**2)
                    cols_p.append(x**3)
                    cols_p.append(x**4)
                    cols_p.append(np.sin(x))
                    cols_p.append(np.cos(x))
                    cols_p.append(np.sin(2*x))
                    cols_p.append(np.cos(2*x))
                    cols_p.append(np.exp(x))
                    cols_p.append(np.exp(-x))
                    if np.all(x > 0.01):
                        cols_p.append(1.0/x)
                        cols_p.append(np.log(x))
                    A_p = np.column_stack(cols_p)
                    return (A_p @ c).reshape(np.asarray(X).shape)
                
                best_model = predict_basis
                
        except Exception:
            pass

    # 4. Fallback: High degree polynomial interpolation
    # If nothing else worked, a polynomial of degree n-1 will interpolate exactly.
    # However, it might oscillate wildly.
    # A safer fallback is a spline or a lower degree poly.
    # Let's use a degree 10 poly if not already beaten.
    if best_model is None or best_mse > 1e-4 * np.var(y_flat):
        try:
            deg = min(10, n-1)
            coef = np.polyfit(X_flat, y_flat, deg)
            y_pred = np.polyval(coef, X_flat)
            mse = np.mean((y_pred - y_flat)**2)
            
            c = coef.copy()
            def predict_fallback(X, c=c):
                x = np.asarray(X).flatten()
                return np.polyval(c, x).reshape(np.asarray(X).shape)
            
            if mse < best_mse:
                best_model = predict_fallback
                best_mse = mse
        except Exception:
            # Ultimate fallback: Mean
            mean_y = np.mean(y_flat)
            def predict_mean(X):
                return np.full(np.asarray(X).shape, mean_y)
            best_model = predict_mean

    return best_model
