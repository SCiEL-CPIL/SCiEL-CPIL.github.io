import numpy as np
import time
from scipy.interpolate import CubicSpline, RBFInterpolator

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    if d != 1:
        # Fallback for d > 1 (unlikely per task contract)
        X_aug = np.column_stack([np.ones(n), X])
        w, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        def predict(X_new):
            X_new = np.asarray(X_new, dtype=np.float64)
            X_new_aug = np.column_stack([np.ones(X_new.shape[0]), X_new])
            return X_new_aug @ w
        return predict

    x = X.flatten()
    
    # --- Feature Generation ---
    def build_features(x):
        n = len(x)
        cols = [np.ones(n)]
        
        # Polynomials up to degree 20
        xk = np.ones(n)
        for k in range(1, 21):
            xk = xk * x
            cols.append(xk.copy())
        
        # Trig functions with various frequencies
        # Include pi multiples which are common in hidden equations
        freqs = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        pi_freqs = [np.pi/4, np.pi/2, np.pi, 2*np.pi, 3*np.pi, 4*np.pi]
        for f in freqs + pi_freqs:
            cols.append(np.sin(f * x))
            cols.append(np.cos(f * x))
            
        # Exponential functions
        exp_coeffs = [-3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
        for c in exp_coeffs:
            cols.append(np.exp(c * x))
            
        # Other common functions
        cols.append(np.abs(x))
        cols.append(np.sqrt(np.abs(x) + 1e-9))
        cols.append(np.log(np.abs(x) + 1e-9))
        cols.append(np.log(np.abs(x) + 1.0))
        cols.append(1.0 / (x + 1e-9))
        cols.append(1.0 / (x**2 + 1e-9))
        cols.append(1.0 / (np.abs(x) + 1.0))
        cols.append(np.tanh(x))
        cols.append(np.tanh(2*x))
        cols.append(np.tanh(0.5*x))
        cols.append(np.arctan(x))
        cols.append(np.arctan(2*x))
        cols.append(np.arctan(0.5*x))
        
        return np.column_stack(cols)

    Xf = build_features(x)
    n_features = Xf.shape[1]

    # --- Model 1: Ridge Regression with Cross-Validation ---
    def fit_ridge(Xf, y, alpha):
        # Do not regularize the intercept (first column)
        I = np.eye(Xf.shape[1])
        I[0, 0] = 0.0
        A = Xf.T @ Xf + alpha * I
        b = Xf.T @ y
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return w

    best_ridge_alpha = 1.0
    best_ridge_cv = np.inf
    best_ridge_w = None
    
    # 5-Fold CV for Ridge Alpha
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = max(1, n // 5)
    
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    
    for alpha in alphas:
        if time.time() > deadline - 30:
            break
        cv_mses = []
        for fold in range(5):
            val_idx = indices[fold*fold_size:(fold+1)*fold_size]
            train_idx = np.concatenate([indices[:fold*fold_size], indices[(fold+1)*fold_size:]])
            
            w = fit_ridge(Xf[train_idx], y[train_idx], alpha)
            pred = Xf[val_idx] @ w
            mse = np.mean((pred - y[val_idx])**2)
            cv_mses.append(mse)
        
        mean_cv = np.mean(cv_mses)
        if mean_cv < best_ridge_cv:
            best_ridge_cv = mean_cv
            best_ridge_alpha = alpha
            best_ridge_w = fit_ridge(Xf, y, alpha)

    def predict_ridge(X_new):
        x_new = np.asarray(X_new, dtype=np.float64).flatten()
        Xf_new = build_features(x_new)
        return Xf_new @ best_ridge_w

    # --- Model 2: Spline Interpolation ---
    def create_spline_model():
        try:
            idx = np.argsort(x)
            x_sorted = x[idx]
            y_sorted = y[idx]
            # Check for duplicate x values, which break CubicSpline
            if np.any(np.diff(x_sorted) == 0):
                # Handle duplicates by averaging y or removing duplicates
                # For simplicity, if duplicates exist, spline might be unstable or fail
                # Let's try to remove exact duplicates
                unique_x, unique_indices = np.unique(x_sorted, return_index=True)
                if len(unique_x) < len(x_sorted):
                    # Aggregate y values for duplicates
                    y_unique = np.zeros_like(unique_x)
                    counts = np.zeros_like(unique_x)
                    for i, orig_idx in enumerate(unique_indices):
                        pass # This logic is complex, let's just fail spline if duplicates are many or strict
                    # Actually, CubicSpline requires strictly increasing x.
                    # If we have duplicates, we can't use it directly without preprocessing.
                    # Given 200 points from a continuous range, exact float duplicates are rare unless the function is constant or steps.
                    # Let's just return None if duplicates are found to be safe.
                    return None
            cs = CubicSpline(x_sorted, y_sorted)
            def predict_spline(X_new):
                x_new = np.asarray(X_new, dtype=np.float64).flatten()
                return cs(x_new)
            return predict_spline
        except Exception:
            return None

    spline_model = create_spline_model()

    # --- Model 3: RBF Interpolation (SciPy) ---
    def create_rbf_model(epsilon):
        try:
            rbf_interp = RBFInterpolator(X, y, kernel='gaussian', epsilon=epsilon)
            # Check training error
            train_pred = rbf_interp(X)
            train_mse = np.mean((train_pred - y)**2)
            if np.isfinite(train_mse) and train_mse < 1e-4:
                def predict_rbf(X_new):
                    return rbf_interp(np.asarray(X_new, dtype=np.float64))
                return predict_rbf
            return None
        except Exception:
            return None

    rbf_models = []
    for eps in [0.5, 1.0, 2.0, 5.0]:
        if time.time() > deadline - 20:
            break
        m = create_rbf_model(eps)
        if m:
            rbf_models.append(m)

    # --- Model Selection via Validation ---
    # Use a hold-out set for final selection among the candidates
    # We already used CV for Ridge. For Spline and RBF, we can use a simple hold-out
    # or just pick the one with lowest training error if CV is not applicable, 
    # but RBF/Spline are interpolants so training error is ~0.
    # Better: Use the same CV folds or a separate validation set.
    # Let's use the last 20% as validation for the final choice among all models.
    
    val_size = 40
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_val = x[val_idx]
    y_val = y[val_idx]
    
    candidates = []
    
    # 1. Ridge (refit on all data, but evaluated on val set using the CV-selected alpha)
    # Note: The ridge_w is fit on ALL data. To fairly evaluate, we should have kept a model fit on train only.
    # However, the CV score is a good proxy. Let's evaluate the "CV performance" vs "Val performance" of the specific model instance.
    # Actually, for a fair comparison, let's evaluate the specific models on the validation set.
    # For Ridge, we can quickly refit on train_idx with best_ridge_alpha.
    w_val_ridge = fit_ridge(Xf[train_idx], y[train_idx], best_ridge_alpha)
    pred_ridge_val = Xf[val_idx] @ w_val_ridge
    mse_ridge_val = np.mean((pred_ridge_val - y_val)**2)
    candidates.append(('ridge', mse_ridge_val, predict_ridge))
    
    # 2. Spline
    if spline_model:
        # Spline is an interpolant. It will have ~0 error on training data.
        # On validation data, it might oscillate.
        try:
            pred_spline_val = spline_model(X_val.reshape(-1, 1))
            mse_spline_val = np.mean((pred_spline_val - y_val)**2)
            if np.isfinite(mse_spline_val):
                candidates.append(('spline', mse_spline_val, spline_model))
        except Exception:
            pass

    # 3. RBF Models
    for i, rbf_m in enumerate(rbf_models):
        try:
            pred_rbf_val = rbf_m(X_val.reshape(-1, 1))
            mse_rbf_val = np.mean((pred_rbf_val - y_val)**2)
            if np.isfinite(mse_rbf_val):
                candidates.append((f'rbf_{i}', mse_rbf_val, rbf_m))
        except Exception:
            pass

    # Select best model
    if not candidates:
        # Fallback: Linear regression
        X_aug = np.column_stack([np.ones(n), X])
        w, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        def predict(X_new):
            X_new = np.asarray(X_new, dtype=np.float64)
            X_new_aug = np.column_stack([np.ones(X_new.shape[0]), X_new])
            return X_new_aug @ w
        return predict

    candidates.sort(key=lambda item: item[1])
    best_name, best_mse, best_predict_func = candidates[0]

    # Optional: Ensemble? 
    # Given the "hidden equations" nature, a single strong model is often better than an ensemble that might average out sharp features.
    # However, if the best model is RBF/Spline, it might be overfitting to noise (though noiseless).
    # If the function is complex, Ridge might be smoother and generalize better.
    # Let's stick to the best single model based on validation MSE.
    
    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        try:
            res = best_predict_func(X_new)
            if not np.all(np.isfinite(res)):
                # Fallback if prediction fails
                return np.full(res.shape, np.mean(y))
            return res
        except Exception:
            return np.full(len(X_new), np.mean(y))
            
    return predict
