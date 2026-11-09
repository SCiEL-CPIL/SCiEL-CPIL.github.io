import numpy as np
import time
from scipy.interpolate import RBFInterpolator

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Strategy: Ensemble of multiple modelling families
    # 1. RBF Interpolation (scipy)
    # 2. High-order spline interpolation
    # 3. Neural network (small, from scratch)
    # 4. Ridge regression with rich features (fallback)
    
    models = []
    model_weights = []
    
    # --- Model 1: RBF Interpolation with Gaussian kernel ---
    try:
        if n >= 10:
            rbf_interp = RBFInterpolator(X, y, kernel='gaussian', epsilon=1.0)
            # Validate on training data
            rbf_train_pred = rbf_interp(X)
            rbf_train_mse = np.mean((rbf_train_pred - y)**2)
            if np.isfinite(rbf_train_mse) and rbf_train_mse < 1e-6:
                models.append(('rbf_gauss1', rbf_interp))
    except Exception:
        pass
    
    try:
        if n >= 10:
            rbf_interp2 = RBFInterpolator(X, y, kernel='gaussian', epsilon=0.5)
            rbf_train_pred = rbf_interp2(X)
            rbf_train_mse = np.mean((rbf_train_pred - y)**2)
            if np.isfinite(rbf_train_mse) and rbf_train_mse < 1e-6:
                models.append(('rbf_gauss05', rbf_interp2))
    except Exception:
        pass
    
    try:
        if n >= 10:
            rbf_interp3 = RBFInterpolator(X, y, kernel='thin_plate_spline')
            rbf_train_pred = rbf_interp3(X)
            rbf_train_mse = np.mean((rbf_train_pred - y)**2)
            if np.isfinite(rbf_train_mse) and rbf_train_mse < 1e-6:
                models.append(('rbf_tps', rbf_interp3))
    except Exception:
        pass
    
    # --- Model 2: Neural Network (small, from scratch) ---
    def train_nn(X, y, hidden_size=32, lr=0.01, epochs=500, seed=42):
        rng = np.random.RandomState(seed)
        n_samples, n_features = X.shape
        # Normalize inputs
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0) + 1e-8
        X_norm = (X - x_mean) / x_std
        y_mean = y.mean()
        y_std = y.std() + 1e-8
        y_norm = (y - y_mean) / y_std
        
        # Initialize weights
        W1 = rng.randn(n_features, hidden_size) * np.sqrt(2.0 / n_features)
        b1 = np.zeros(hidden_size)
        W2 = rng.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        b2 = np.zeros(1)
        
        # Adam optimizer
        mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
        mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
        mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
        mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)
        t = 0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        for epoch in range(epochs):
            if time.time() > deadline - 30:
                break
            # Forward
            z1 = X_norm @ W1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2 + b2
            pred_norm = z2.flatten()
            
            # Loss
            loss = np.mean((pred_norm - y_norm)**2)
            
            # Backward
            dz2 = 2 * (pred_norm - y_norm) / n_samples
            dW2 = a1.T @ dz2.reshape(-1, 1)
            db2 = dz2.sum()
            da1 = dz2.reshape(-1, 1) @ W2.T
            dz1 = da1 * (1 - a1**2)
            dW1 = X_norm.T @ dz1
            db1 = dz1.sum(axis=0)
            
            t += 1
            mW1 = beta1 * mW1 + (1 - beta1) * dW1
            vW1 = beta2 * vW1 + (1 - beta2) * dW1**2
            mW1_hat = mW1 / (1 - beta1**t)
            vW1_hat = vW1 / (1 - beta2**t)
            W1 -= lr * mW1_hat / (np.sqrt(vW1_hat) + eps)
            
            mb1 = beta1 * mb1 + (1 - beta1) * db1
            vb1 = beta2 * vb1 + (1 - beta2) * db1**2
            mb1_hat = mb1 / (1 - beta1**t)
            vb1_hat = vb1 / (1 - beta2**t)
            b1 -= lr * mb1_hat / (np.sqrt(vb1_hat) + eps)
            
            mW2 = beta1 * mW2 + (1 - beta1) * dW2
            vW2 = beta2 * vW2 + (1 - beta2) * dW2**2
            mW2_hat = mW2 / (1 - beta1**t)
            vW2_hat = vW2 / (1 - beta2**t)
            W2 -= lr * mW2_hat / (np.sqrt(vW2_hat) + eps)
            
            mb2 = beta1 * mb2 + (1 - beta1) * db2
            vb2 = beta2 * vb2 + (1 - beta2) * db2**2
            mb2_hat = mb2 / (1 - beta1**t)
            vb2_hat = vb2 / (1 - beta2**t)
            b2 -= lr * mb2_hat / (np.sqrt(vb2_hat) + eps)
        
        def nn_predict(X_new):
            Xn = (np.asarray(X_new, dtype=np.float64) - x_mean) / x_std
            z1 = Xn @ W1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2 + b2
            return (z2.flatten() * y_std) + y_mean
        
        return nn_predict, loss
    
    try:
        if n >= 20:
            nn_pred_func, nn_loss = train_nn(X, y, hidden_size=64, lr=0.005, epochs=800, seed=42)
            nn_train_pred = nn_pred_func(X)
            nn_train_mse = np.mean((nn_train_pred - y)**2)
            if np.isfinite(nn_train_mse) and nn_train_mse < 1e-4:
                models.append(('nn64', nn_pred_func))
    except Exception:
        pass
    
    try:
        if n >= 20:
            nn_pred_func2, nn_loss2 = train_nn(X, y, hidden_size=128, lr=0.003, epochs=1000, seed=123)
            nn_train_pred2 = nn_pred_func2(X)
            nn_train_mse2 = np.mean((nn_train_pred2 - y)**2)
            if np.isfinite(nn_train_mse2) and nn_train_mse2 < 1e-4:
                models.append(('nn128', nn_pred_func2))
    except Exception:
        pass
    
    # --- Model 3: Ridge regression with rich features (fallback) ---
    def build_features(X):
        n = X.shape[0]
        cols = [np.ones(n)]
        for j in range(d):
            x = X[:, j]
            xk = np.ones(n)
            for k in range(1, 10):
                xk = xk * x
                cols.append(xk.copy())
            for freq in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, np.pi/2, np.pi]:
                cols.append(np.sin(freq * x))
                cols.append(np.cos(freq * x))
            for c in [0.3, 0.5, 1.0, -0.3, -0.5, -1.0]:
                cols.append(np.exp(c * x))
            cols.append(np.abs(x))
            cols.append(np.sqrt(np.abs(x)))
            cols.append(np.log(np.abs(x) + 1e-6))
            cols.append(np.log(np.abs(x) + 1.0))
            cols.append(1.0 / (x + 1e-6))
            cols.append(1.0 / (x**2 + 1e-6))
            cols.append(1.0 / (np.abs(x) + 1.0))
        return np.column_stack(cols)
    
    def ridge_fit(Xf, y, alpha):
        n, p = Xf.shape
        A = Xf.T @ Xf + alpha * np.eye(p)
        b = Xf.T @ y
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]
        return w
    
    try:
        Xf_train = build_features(X)
        # CV for alpha
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        best_alpha = 1.0
        best_cv = np.inf
        indices = np.arange(n)
        fold_size = max(1, n // 5)
        for alpha in alphas:
            cv_scores = []
            for fold in range(5):
                val_idx = indices[fold*fold_size:(fold+1)*fold_size]
                train_idx = np.concatenate([indices[:fold*fold_size], indices[(fold+1)*fold_size:]])
                w = ridge_fit(Xf_train[train_idx], y[train_idx], alpha)
                pred = Xf_train[val_idx] @ w
                mse = np.mean((pred - y[val_idx])**2)
                cv_scores.append(mse)
            mean_cv = np.mean(cv_scores)
            if mean_cv < best_cv:
                best_cv = mean_cv
                best_alpha = alpha
        w = ridge_fit(Xf_train, y, best_alpha)
        
        def ridge_predict(X_new):
            Xf = build_features(np.asarray(X_new, dtype=np.float64))
            return Xf @ w
        
        models.append(('ridge', ridge_predict))
    except Exception:
        pass
    
    # If no models, use simple linear regression as absolute fallback
    if not models:
        try:
            X_aug = np.column_stack([np.ones(n), X])
            w, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            def predict(X):
                X_aug = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=np.float64)])
                return X_aug @ w
            return predict
        except Exception:
            # Absolute last resort: mean prediction
            y_mean = np.mean(y)
            def predict(X):
                return np.full(len(X), y_mean)
            return predict
    
    # Evaluate each model on training data to weight them
    y_var = np.var(y) + 1e-12
    train_mses = []
    for name, model in models:
        try:
            pred = model(X)
            mse = np.mean((pred - y)**2)
            if not np.isfinite(mse):
                mse = 1e10
            train_mses.append(mse)
        except Exception:
            train_mses.append(1e10)
    
    # Weight inversely to training MSE (with floor)
    inv_mses = [1.0 / (m + 1e-10) for m in train_mses]
    total_inv = sum(inv_mses)
    weights = [w / total_inv for w in inv_mses]
    
    # Final predict: weighted ensemble
    def predict(X):
        X_arr = np.asarray(X, dtype=np.float64)
        preds = np.zeros(len(X_arr))
        for i, (name, model) in enumerate(models):
            try:
                p = model(X_arr)
                if np.all(np.isfinite(p)):
                    preds += weights[i] * p
            except Exception:
                pass
        # If all failed, return mean
        if not np.all(np.isfinite(preds)):
            preds = np.where(np.isfinite(preds), preds, np.mean(y))
        return preds
    
    return predict
