import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import least_squares
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 280.0
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    X_flat = X.flatten()
    y_flat = y.flatten()
    
    if np.allclose(y_flat, y_flat[0]):
        c = y_flat[0]
        def predict_const(X):
            return np.full(X.shape, c)
        return predict_const
    
    sort_idx = np.argsort(X_flat)
    X_sorted = X_flat[sort_idx]
    y_sorted = y_flat[sort_idx]
    
    unique_X, inverse_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
    if len(unique_X) != len(X_sorted):
        y_sum = np.zeros_like(unique_X, dtype=float)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_sorted = y_sum / counts
        X_sorted = unique_X
    
    models = []
    
    # Model 1: Polynomial regression (degrees 1-15)
    best_poly_mse = np.inf
    poly_coeffs = None
    for deg in range(1, 16):
        if time.time() - start_time > deadline:
            break
        try:
            coeffs = np.polyfit(X_sorted, y_sorted, deg)
            y_pred = np.polyval(coeffs, X_sorted)
            mse = np.mean((y_pred - y_sorted) ** 2)
            if mse < best_poly_mse:
                best_poly_mse = mse
                poly_coeffs = coeffs
        except Exception:
            continue
    
    if poly_coeffs is not None:
        def predict_poly(X):
            X_eval = np.asarray(X, dtype=float).flatten()
            return np.polyval(poly_coeffs, X_eval).reshape(np.asarray(X).shape)
        models.append((best_poly_mse, predict_poly))
    
    # Model 2: Cubic Spline
    try:
        if len(X_sorted) >= 4:
            spline = CubicSpline(X_sorted, y_sorted)
            y_pred = spline(X_sorted)
            mse_spline = np.mean((y_pred - y_sorted) ** 2)
            def predict_spline(X):
                X_eval = np.asarray(X, dtype=float).flatten()
                return spline(X_eval).reshape(np.asarray(X).shape)
            models.append((mse_spline, predict_spline))
    except Exception:
        pass
    
    # Model 3: PCHIP
    try:
        if len(X_sorted) >= 3:
            pchip = PchipInterpolator(X_sorted, y_sorted)
            y_pred = pchip(X_sorted)
            mse_pchip = np.mean((y_pred - y_sorted) ** 2)
            def predict_pchip(X):
                X_eval = np.asarray(X, dtype=float).flatten()
                return pchip(X_eval).reshape(np.asarray(X).shape)
            models.append((mse_pchip, predict_pchip))
    except Exception:
        pass
    
    # Model 4: Exponential fit y = a * exp(b*x) + c
    if time.time() - start_time < deadline - 20:
        try:
            def exp_model(params, x):
                a, b, c = params
                return a * np.exp(b * x) + c
            
            # Initial guess
            y_min, y_max = y_sorted.min(), y_sorted.max()
            a0 = (y_max - y_min) / 2
            c0 = (y_max + y_min) / 2
            b0 = 0.5
            
            res = least_squares(exp_model, [a0, b0, c0], args=(X_sorted,), 
                               method='lm', max_nfev=1000)
            y_pred = exp_model(res.x, X_sorted)
            mse_exp = np.mean((y_pred - y_sorted) ** 2)
            
            if np.isfinite(mse_exp):
                def predict_exp(X):
                    X_eval = np.asarray(X, dtype=float).flatten()
                    return exp_model(res.x, X_eval).reshape(np.asarray(X).shape)
                models.append((mse_exp, predict_exp))
        except Exception:
            pass
    
    # Model 5: Power law fit y = a * x^b + c
    if time.time() - start_time < deadline - 20:
        try:
            # Only try if x has both positive and negative or all positive
            if np.all(X_sorted > 0) or np.all(X_sorted < 0):
                def power_model(params, x):
                    a, b, c = params
                    return a * np.power(np.abs(x), b) * np.sign(x) + c
                
                y_min, y_max = y_sorted.min(), y_sorted.max()
                a0 = (y_max - y_min) / 2
                c0 = (y_max + y_min) / 2
                b0 = 1.0
                
                res = least_squares(power_model, [a0, b0, c0], args=(X_sorted,),
                                   method='lm', max_nfev=1000)
                y_pred = power_model(res.x, X_sorted)
                mse_pow = np.mean((y_pred - y_sorted) ** 2)
                
                if np.isfinite(mse_pow):
                    def predict_pow(X):
                        X_eval = np.asarray(X, dtype=float).flatten()
                        return power_model(res.x, X_eval).reshape(np.asarray(X).shape)
                    models.append((mse_pow, predict_pow))
        except Exception:
            pass
    
    # Model 6: Sine fit y = a * sin(b*x + c) + d
    if time.time() - start_time < deadline - 20:
        try:
            def sine_model(params, x):
                a, b, c, d = params
                return a * np.sin(b * x + c) + d
            
            y_min, y_max = y_sorted.min(), y_sorted.max()
            a0 = (y_max - y_min) / 2
            d0 = (y_max + y_min) / 2
            b0 = 1.0
            c0 = 0.0
            
            res = least_squares(sine_model, [a0, b0, c0, d0], args=(X_sorted,),
                               method='lm', max_nfev=2000)
            y_pred = sine_model(res.x, X_sorted)
            mse_sin = np.mean((y_pred - y_sorted) ** 2)
            
            if np.isfinite(mse_sin):
                def predict_sin(X):
                    X_eval = np.asarray(X, dtype=float).flatten()
                    return sine_model(res.x, X_eval).reshape(np.asarray(X).shape)
                models.append((mse_sin, predict_sin))
        except Exception:
            pass
    
    # Model 7: Gaussian fit y = a * exp(-((x-b)/c)^2) + d
    if time.time() - start_time < deadline - 20:
        try:
            def gauss_model(params, x):
                a, b, c, d = params
                c_safe = max(abs(c), 0.1)
                return a * np.exp(-((x - b) / c_safe) ** 2) + d
            
            y_min, y_max = y_sorted.min(), y_sorted.max()
            a0 = y_max - y_min
            b0 = X_sorted.mean()
            c0 = (X_sorted.max() - X_sorted.min()) / 4
            d0 = y_min
            
            res = least_squares(gauss_model, [a0, b0, c0, d0], args=(X_sorted,),
                               method='lm', max_nfev=2000)
            y_pred = gauss_model(res.x, X_sorted)
            mse_gauss = np.mean((y_pred - y_sorted) ** 2)
            
            if np.isfinite(mse_gauss):
                def predict_gauss(X):
                    X_eval = np.asarray(X, dtype=float).flatten()
                    return gauss_model(res.x, X_eval).reshape(np.asarray(X).shape)
                models.append((mse_gauss, predict_gauss))
        except Exception:
            pass
    
    # Model 8: Rational function fit y = (a*x + b) / (c*x + d)
    if time.time() - start_time < deadline - 20:
        try:
            def rational_model(params, x):
                a, b, c, d = params
                denom = c * x + d
                # Avoid division by zero
                denom_safe = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
                return (a * x + b) / denom_safe
            
            y_min, y_max = y_sorted.min(), y_sorted.max()
            a0 = 1.0
            b0 = 0.0
            c0 = 0.1
            d0 = 1.0
            
            res = least_squares(rational_model, [a0, b0, c0, d0], args=(X_sorted,),
                               method='lm', max_nfev=2000)
            y_pred = rational_model(res.x, X_sorted)
            mse_rat = np.mean((y_pred - y_sorted) ** 2)
            
            if np.isfinite(mse_rat):
                def predict_rat(X):
                    X_eval = np.asarray(X, dtype=float).flatten()
                    return rational_model(res.x, X_eval).reshape(np.asarray(X).shape)
                models.append((mse_rat, predict_rat))
        except Exception:
            pass
    
    # Model 9: Simple neural network with more capacity
    if time.time() - start_time < deadline - 30:
        try:
            x_min, x_max = X_sorted.min(), X_sorted.max()
            if x_max - x_min < 1e-10:
                x_norm = np.zeros_like(X_sorted)
            else:
                x_norm = (X_sorted - x_min) / (x_max - x_min)
            
            y_min, y_max = y_sorted.min(), y_sorted.max()
            y_range = y_max - y_min
            if y_range < 1e-10:
                y_norm = np.zeros_like(y_sorted)
            else:
                y_norm = (y_sorted - y_min) / y_range
            
            np.random.seed(42)
            n_samples = len(x_norm)
            hidden_size = 64
            
            W1 = np.random.randn(1, hidden_size) * np.sqrt(2.0 / 1)
            b1 = np.zeros((1, hidden_size))
            W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
            b2 = np.zeros((1, 1))
            
            X_nn = x_norm.reshape(-1, 1)
            y_nn = y_norm.reshape(-1, 1)
            
            lr = 0.01
            for epoch in range(3000):
                if time.time() - start_time > deadline - 10:
                    break
                
                z1 = X_nn @ W1 + b1
                a1 = np.tanh(z1)
                z2 = a1 @ W2 + b2
                a2 = z2
                
                dz2 = 2 * (a2 - y_nn) / n_samples
                dW2 = a1.T @ dz2
                db2 = np.sum(dz2, axis=0, keepdims=True)
                da1 = dz2 @ W2.T
                dz1 = da1 * (1 - a1**2)
                dW1 = X_nn.T @ dz1
                db1 = np.sum(dz1, axis=0, keepdims=True)
                
                W2 -= lr * dW2
                b2 -= lr * db2
                W1 -= lr * dW1
                b1 -= lr * db1
            
            X_nn_eval = X_nn
            z1 = X_nn_eval @ W1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2 + b2
            y_pred_train = z2.flatten()
            
            mse_nn = np.mean((y_pred_train - y_sorted) ** 2)
            
            def predict_nn(X):
                X_eval = np.asarray(X, dtype=float).flatten()
                if x_max - x_min < 1e-10:
                    X_norm_eval = np.zeros_like(X_eval)
                else:
                    X_norm_eval = (X_eval - x_min) / (x_max - x_min)
                X_nn_eval = X_norm_eval.reshape(-1, 1)
                z1 = X_nn_eval @ W1 + b1
                a1 = np.tanh(z1)
                z2 = a1 @ W2 + b2
                y_pred = z2.flatten()
                if y_range < 1e-10:
                    y_pred = y_min * np.ones_like(y_pred)
                else:
                    y_pred = y_pred * y_range + y_min
                return y_pred.reshape(np.asarray(X).shape)
            
            models.append((mse_nn, predict_nn))
        except Exception:
            pass
    
    if not models:
        def predict_interp(X):
            X_eval = np.asarray(X, dtype=float).flatten()
            return np.interp(X_eval, X_sorted, y_sorted).reshape(np.asarray(X).shape)
        return predict_interp
    
    models.sort(key=lambda x: x[0])
    best_mse, best_predict = models[0]
    
    if best_mse < 1e-8:
        return best_predict
    
    good_models = [m for m in models if m[0] < best_mse * 10 + 1e-10]
    if len(good_models) > 1:
        predict_funcs = [m[1] for m in good_models]
        def predict_ensemble(X):
            preds = []
            for pf in predict_funcs:
                try:
                    preds.append(pf(X).flatten())
                except Exception:
                    pass
            if not preds:
                return np.zeros(X.shape)
            return np.mean(preds, axis=0).reshape(np.asarray(X).shape)
        return predict_ensemble
    
    return best_predict
