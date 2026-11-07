"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    
    # Determine input dimensionality
    d = X_train.shape[1]
    n = X_train.shape[0]
    
    # For 1D problems (which is the case here), use Chebyshev spectral collocation
    # For higher dimensions, fall back to polynomial regression
    
    if d == 1:
        # Normalize input to [-1, 1] for Chebyshev basis
        x = X_train[:, 0]
        x_min = x.min()
        x_max = x.max()
        x_range = x_max - x_min
        if x_range < 1e-12:
            x_range = 1.0
            x_center = x_min
        else:
            x_center = (x_max + x_min) / 2.0
        
        # Map to [-1, 1]
        def map_to_unit(x_val):
            return 2.0 * (x_val - x_center) / x_range - 1.0
        
        x_norm = map_to_unit(x)
        
        # Try different Chebyshev degrees and select via regularization
        # Chebyshev polynomials: T_k(x) = cos(k * arccos(x))
        def chebyshev_basis(x_vals, max_degree):
            """Return matrix of Chebyshev polynomial values T_0(x), T_1(x), ..., T_max_degree(x)"""
            x_vals = np.asarray(x_vals)
            n_pts = len(x_vals)
            # Clip to [-1, 1] for numerical stability of arccos
            x_clipped = np.clip(x_vals, -1.0, 1.0)
            theta = np.arccos(x_clipped)
            degrees = np.arange(max_degree + 1)
            # T_k(cos(theta)) = cos(k * theta)
            basis = np.cos(np.outer(degrees, theta))
            return basis
        
        # Fit with different degrees, using L2 regularization to avoid overfitting
        best_degree = 2
        best_w = None
        best_score = np.inf
        
        # Try degrees from 2 up to a reasonable maximum
        max_deg_to_try = 30
        
        for deg in range(2, max_deg_to_try + 1):
            try:
                basis_train = chebyshev_basis(x_norm, deg)
                
                # Use ridge regression for stability
                # Regularization strength: small to allow good fit but prevent oscillation
                alpha = 1e-8 * (deg ** 2)  # Scale regularization with degree
                
                # Solve (B^T B + alpha I) w = B^T y
                BtB = basis_train.T @ basis_train
                Bty = basis_train.T @ y_train
                reg = alpha * np.eye(deg + 1)
                # Don't regularize the constant term as heavily
                reg[0, 0] = alpha * 0.01
                
                A = BtB + reg
                try:
                    w = np.linalg.solve(A, Bty)
                except np.linalg.LinAlgError:
                    w, *_ = np.linalg.lstsq(A, Bty, rcond=None)
                
                # Evaluate on training data to get in-sample score
                pred_train = basis_train @ w
                mse_train = np.mean((pred_train - y_train) ** 2)
                var_y = np.var(y_train)
                if var_y < 1e-12:
                    var_y = 1.0
                nrmse_train = mse_train / var_y
                
                # Use a simple criterion: prefer lower degree if in-sample error is small
                # Since data is noiseless, we want the simplest model that fits well
                if nrmse_train < 1e-6:
                    best_degree = deg
                    best_w = w
                    break
                elif nrmse_train < best_score:
                    best_score = nrmse_train
                    best_degree = deg
                    best_w = w
                    
            except Exception:
                continue
        
        # If we didn't break early, use the best found
        if best_w is None:
            # Fallback: use degree 2
            deg = 2
            basis_train = chebyshev_basis(x_norm, deg)
            alpha = 1e-8
            BtB = basis_train.T @ basis_train
            Bty = basis_train.T @ y_train
            reg = alpha * np.eye(deg + 1)
            reg[0, 0] = alpha * 0.01
            A = BtB + reg
            w, *_ = np.linalg.lstsq(A, Bty, rcond=None)
            best_w = w
            best_degree = deg
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            x_pred = X[:, 0]
            x_pred_norm = map_to_unit(x_pred)
            # Clip for safety in arccos
            x_pred_clipped = np.clip(x_pred_norm, -1.0, 1.0)
            theta = np.arccos(x_pred_clipped)
            degrees = np.arange(best_degree + 1)
            basis = np.cos(np.outer(degrees, theta))
            return basis @ best_w
        
        return predict
    
    else:
        # For multi-dimensional inputs, use polynomial feature expansion
        # Try different degrees
        def features(X, degree):
            cols = [np.ones(len(X))]
            # Linear terms
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            # Quadratic and higher terms up to specified degree
            if degree >= 2:
                for i in range(X.shape[1]):
                    for j in range(i, X.shape[1]):
                        cols.append(X[:, i] * X[:, j])
            if degree >= 3:
                for i in range(X.shape[1]):
                    for j in range(i, X.shape[1]):
                        for k in range(j, X.shape[1]):
                            cols.append(X[:, i] * X[:, j] * X[:, k])
            return np.column_stack(cols)
        
        best_degree = 2
        best_w = None
        best_score = np.inf
        
        for deg in [2, 3]:
            try:
                F = features(X_train, deg)
                alpha = 1e-6
                BtB = F.T @ F
                Bty = F.T @ y_train
                reg = alpha * np.eye(F.shape[1])
                A = BtB + reg
                w, *_ = np.linalg.lstsq(A, Bty, rcond=None)
                
                pred = F @ w
                mse = np.mean((pred - y_train) ** 2)
                var_y = np.var(y_train)
                if var_y < 1e-12:
                    var_y = 1.0
                nrmse = mse / var_y
                
                if nrmse < best_score:
                    best_score = nrmse
                    best_degree = deg
                    best_w = w
                    
            except Exception:
                continue
        
        if best_w is None:
            F = features(X_train, 2)
            best_w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
            best_degree = 2
        
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            F = features(X, best_degree)
            return F @ best_w
        
        return predict
# EVOLVE-BLOCK-END
