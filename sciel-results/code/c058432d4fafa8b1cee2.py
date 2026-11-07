import numpy as np
import time
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import differential_evolution

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Flatten X if needed, though task says 1D inputs
    if X_train.ndim == 2:
        x = X_train[:, 0].astype(float)
    else:
        x = np.asarray(X_train, dtype=float).flatten()
    y = np.asarray(y_train, dtype=float).flatten()
    
    n = len(x)
    
    # Handle constant y
    if np.std(y) < 1e-12:
        c = np.mean(y)
        return lambda X: np.full(np.asarray(X).shape, c)
    
    # Sort by x for interpolation methods
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Handle duplicate x values by averaging
    unique_x, inverse_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    if len(unique_x) != len(x_sorted):
        y_sum = np.zeros_like(unique_x, dtype=float)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_sorted = y_sum / counts
        x_sorted = unique_x
    n_unique = len(x_sorted)
    
    # Define candidate models
    candidates = []
    
    # 1. Cubic Spline
    def make_spline_predict(x_tr, y_tr):
        try:
            cs = CubicSpline(x_tr, y_tr)
            return lambda X: cs(np.asarray(X).flatten())
        except Exception:
            return lambda X: np.interp(np.asarray(X).flatten(), x_tr, y_tr)
    candidates.append(("spline", make_spline_predict))
    
    # 2. PCHIP
    def make_pchip_predict(x_tr, y_tr):
        try:
            pchip = PchipInterpolator(x_tr, y_tr)
            return lambda X: pchip(np.asarray(X).flatten())
        except Exception:
            return lambda X: np.interp(np.asarray(X).flatten(), x_tr, y_tr)
    candidates.append(("pchip", make_pchip_predict))
    
    # 3. Gaussian Process with specific length scales
    def make_gp_predict(x_tr, y_tr, length_scale):
        y_mean = np.mean(y_tr)
        y_std = np.std(y_tr)
        if y_std < 1e-8:
            y_std = 1.0
        y_scaled = (y_tr - y_mean) / y_std
        
        X_k = x_tr[:, np.newaxis]
        diff = X_k - X_k.T
        K = np.exp(-0.5 * (diff**2) / (length_scale**2))
        K += 1e-8 * np.eye(len(x_tr))
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))
        except np.linalg.LinAlgError:
            try:
                alpha = np.linalg.lstsq(K, y_scaled, rcond=None)[0]
            except Exception:
                return lambda X: np.full(len(np.asarray(X).flatten()), y_mean)
        
        def predict(X):
            x_new = np.asarray(X).flatten()
            diff_new = x_new[:, np.newaxis] - X_k
            K_star = np.exp(-0.5 * (diff_new**2) / (length_scale**2))
            y_pred_scaled = K_star @ alpha
            return y_pred_scaled * y_std + y_mean
            
        return predict

    # Add a few GP candidates with different length scales
    x_range = np.max(x_sorted) - np.min(x_sorted)
    if x_range > 0:
        typical_scale = x_range / 10.0
        for ls_factor in [0.5, 1.0, 2.0, 5.0]:
            ls = typical_scale * ls_factor
            candidates.append((f"gp_{ls:.2f}", lambda x_tr, y_tr, ls=ls: make_gp_predict(x_tr, y_tr, ls)))
    else:
        candidates.append(("gp_1.0", lambda x_tr, y_tr: make_gp_predict(x_tr, y_tr, 1.0)))
    
    # 4. Polynomial with Ridge (Degrees 4, 6, 8)
    def make_poly_predict(x_tr, y_tr, degree, alpha):
        V = np.vander(x_tr, N=degree+1, increasing=True)
        try:
            A = V.T @ V + alpha * np.eye(degree+1)
            b = V.T @ y_tr
            w = np.linalg.solve(A, b)
        except Exception:
            w = np.zeros(degree+1)
        
        def predict(X):
            x_new = np.asarray(X).flatten()
            V_new = np.vander(x_new, N=degree+1, increasing=True)
            return V_new @ w
        return predict

    for deg in [4, 6, 8]:
        for alpha in [1e-2, 1e-1, 1.0]:
            candidates.append((f"poly_{deg}_{alpha}", lambda x_tr, y_tr, d=deg, a=alpha: make_poly_predict(x_tr, y_tr, d, a)))
    
    # 5. Neural Network (Simple MLP)
    def make_nn_predict(x_tr, y_tr, hidden_size=32, epochs=500, lr=0.01):
        # Normalize X and Y
        x_min, x_max = x_tr.min(), x_tr.max()
        if x_max - x_min < 1e-10:
            x_norm = np.zeros_like(x_tr)
        else:
            x_norm = (x_tr - x_min) / (x_max - x_min)
        
        y_min, y_max = y_tr.min(), y_tr.max()
        y_range = y_max - y_min
        if y_range < 1e-10:
            y_norm = np.zeros_like(y_tr)
        else:
            y_norm = (y_tr - y_min) / y_range
        
        np.random.seed(42)
        n_samples = len(x_norm)
        
        W1 = np.random.randn(1, hidden_size) * np.sqrt(2.0 / 1)
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        b2 = np.zeros((1, 1))
        
        X_nn = x_norm.reshape(-1, 1)
        y_nn = y_norm.reshape(-1, 1)
        
        for epoch in range(epochs):
            if time.time() > deadline - 10:
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
        
        def predict(X):
            x_new = np.asarray(X).flatten()
            if x_max - x_min < 1e-10:
                x_norm_new = np.zeros_like(x_new)
            else:
                x_norm_new = (x_new - x_min) / (x_max - x_min)
            X_nn_eval = x_norm_new.reshape(-1, 1)
            z1 = X_nn_eval @ W1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2 + b2
            y_pred = z2.flatten()
            if y_range < 1e-10:
                y_pred = y_min * np.ones_like(y_pred)
            else:
                y_pred = y_pred * y_range + y_min
            return y_pred
        
        return predict
    
    candidates.append(("nn_32", lambda x_tr, y_tr: make_nn_predict(x_tr, y_tr, hidden_size=32, epochs=500, lr=0.01)))
    candidates.append(("nn_64", lambda x_tr, y_tr: make_nn_predict(x_tr, y_tr, hidden_size=64, epochs=500, lr=0.01)))
    
    # 6. Symbolic Regression via Differential Evolution
    # Basis functions: 1, x, x^2, x^3, x^4, sin(x), cos(x), exp(x) (clamped), log(abs(x)+eps)
    # We will fit coefficients for a linear combination of these.
    
    def symbolic_predict(x_tr, y_tr, coeffs):
        x = x_tr
        # Clamp x for exp and log to avoid issues
        x_safe = np.clip(x, -10, 10)
        basis = np.column_stack([
            np.ones_like(x),
            x,
            x**2,
            x**3,
            x**4,
            np.sin(x),
            np.cos(x),
            np.exp(x_safe * 0.5), # scaled to prevent overflow/underflow dominance
            np.log(np.abs(x) + 1e-6)
        ])
        
        def predict(X):
            x_new = np.asarray(X).flatten()
            x_new_safe = np.clip(x_new, -10, 10)
            basis_new = np.column_stack([
                np.ones_like(x_new),
                x_new,
                x_new**2,
                x_new**3,
                x_new**4,
                np.sin(x_new),
                np.cos(x_new),
                np.exp(x_new_safe * 0.5),
                np.log(np.abs(x_new) + 1e-6)
            ])
            return basis_new @ coeffs
        
        return predict

    # Optimize coefficients using differential_evolution
    def make_symbolic_candidate(x_tr, y_tr):
        # Calculate basis
        x = x_tr
        x_safe = np.clip(x, -10, 10)
        basis = np.column_stack([
            np.ones_like(x),
            x,
            x**2,
            x**3,
            x**4,
            np.sin(x),
            np.cos(x),
            np.exp(x_safe * 0.5),
            np.log(np.abs(x) + 1e-6)
        ])
        
        # Normalize basis to help optimization
        basis_means = np.mean(basis, axis=0)
        basis_stds = np.std(basis, axis=0)
        basis_stds[basis_stds < 1e-8] = 1.0
        basis_norm = (basis - basis_means) / basis_stds
        
        def objective(coeffs):
            pred = basis_norm @ coeffs
            return np.mean((pred - y_tr)**2)
        
        # Bounds for coefficients
        bounds = [(-10, 10)] * len(basis[0])
        
        try:
            res = differential_evolution(objective, bounds, seed=42, maxiter=50, popsize=10, tol=1e-6)
            best_coeffs = res.x
            
            # Denormalize coefficients? No, we just use the normalized basis in prediction
            # Wait, if we train on normalized basis, we must predict on normalized basis.
            def predict(X):
                x_new = np.asarray(X).flatten()
                x_new_safe = np.clip(x_new, -10, 10)
                basis_new = np.column_stack([
                    np.ones_like(x_new),
                    x_new,
                    x_new**2,
                    x_new**3,
                    x_new**4,
                    np.sin(x_new),
                    np.cos(x_new),
                    np.exp(x_new_safe * 0.5),
                    np.log(np.abs(x_new) + 1e-6)
                ])
                basis_new_norm = (basis_new - basis_means) / basis_stds
                return basis_new_norm @ best_coeffs
            
            return predict
            
        except Exception:
            return lambda X: np.full(len(np.asarray(X).flatten()), np.mean(y_tr))

    candidates.append(("symbolic", make_symbolic_candidate))
    
    # Cross-Validation to select the best model
    k_folds = 5
    indices = np.arange(n_unique)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_indices = np.array_split(indices, k_folds)
    
    model_scores = {name: [] for name, _ in candidates}
    
    for i in range(k_folds):
        if time.time() > deadline:
            break
        val_idx = fold_indices[i]
        tr_idx = np.concatenate([fold_indices[j] for j in range(k_folds) if j != i])
        
        x_val = x_sorted[val_idx]
        y_val = y_sorted[val_idx]
        x_tr = x_sorted[tr_idx]
        y_tr = y_sorted[tr_idx]
        
        for name, model_func in candidates:
            if time.time() > deadline:
                break
            try:
                predict_func = model_func(x_tr, y_tr)
                y_pred = predict_func(x_val.reshape(-1, 1))
                y_pred = np.asarray(y_pred).flatten()
                mse = np.mean((y_pred - y_val)**2)
                model_scores[name].append(mse)
            except Exception:
                model_scores[name].append(np.inf)
    
    # Select model with lowest mean CV MSE
    best_name = None
    best_score = np.inf
    for name, scores in model_scores.items():
        mean_score = np.mean(scores)
        if mean_score < best_score:
            best_score = mean_score
            best_name = name
    
    if best_name is None:
        return lambda X: np.interp(np.asarray(X).flatten(), x_sorted, y_sorted)
        
    # Find the function for the best model
    best_model_func = None
    for name, func in candidates:
        if name == best_name:
            best_model_func = func
            break
            
    if best_model_func is None:
        return lambda X: np.interp(np.asarray(X).flatten(), x_sorted, y_sorted)

    # Train the best model on ALL data
    final_predict = best_model_func(x_sorted, y_sorted)
    
    # Wrap to ensure correct output shape and finite values
    def final_wrapper(X):
        try:
            pred = final_predict(X)
            pred = np.asarray(pred).flatten()
            pred = np.nan_to_num(pred, nan=0.0, posinf=np.finfo(float).max, neginf=-np.finfo(float).max)
            return pred
        except Exception:
            return np.zeros(len(np.asarray(X).flatten()))
            
    return final_wrapper
