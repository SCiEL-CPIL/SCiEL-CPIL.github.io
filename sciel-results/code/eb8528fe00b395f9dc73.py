"""Symbolic regression over 4 hidden equations using robust non-parametric interpolation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # 290s safety margin out of 330s
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    d = X.shape[1]
    
    # If d == 1, use high-resolution monotone cubic interpolation (PCHIP-like)
    # Otherwise, use a combination of polynomial + RBF or kernel regression
    
    if d == 1:
        # Sort by input
        order = np.argsort(X[:, 0])
        x_sorted = X[order, 0]
        y_sorted = y[order]
        
        # Remove duplicates (keep mean)
        unique_x, unique_indices, inverse_indices = np.unique(x_sorted, return_index=True, return_inverse=True)
        if len(unique_x) < len(x_sorted):
            # Average y values for duplicate x
            y_unique = np.zeros(len(unique_x))
            counts = np.zeros(len(unique_x))
            for i in range(len(x_sorted)):
                y_unique[inverse_indices[i]] += y_sorted[i]
                counts[inverse_indices[i]] += 1
            y_sorted = y_unique / counts
            x_sorted = unique_x
        else:
            x_sorted = x_sorted
            y_sorted = y_sorted
        
        # Ensure strictly increasing x for PCHIP
        # If there are still duplicates after unique, handle edge cases
        if len(x_sorted) < 2:
            # Degenerate case
            c = y_sorted[0] if len(y_sorted) > 0 else 0.0
            return lambda X_test: np.full(X_test.shape[0], c)
        
        # Use scipy's PCHIPInterpolator if available, else implement simple monotone interpolation
        try:
            from scipy.interpolate import PCHIPInterpolator
            pchip = PCHIPInterpolator(x_sorted, y_sorted)
            
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                return pchip(X_test[:, 0])
        except ImportError:
            # Fallback: linear interpolation with numpy
            def predict(X_test):
                X_test = np.asarray(X_test, dtype=np.float64)
                return np.interp(X_test[:, 0], x_sorted, y_sorted)
        
        return predict
    
    else:
        # For multi-dimensional inputs, use a more robust approach
        # Try a combination of polynomial features + regularization
        
        # First, try a polynomial fit with higher degree
        def make_features(X, degree):
            cols = [np.ones(len(X))]
            for d_order in range(1, degree + 1):
                for i in range(X.shape[1]):
                    cols.append(X[:, i] ** d_order)
                # Cross terms
                for i in range(X.shape[1]):
                    for j in range(i + 1, X.shape[1]):
                        cols.append(X[:, i] ** (d_order) * X[:, j] ** (d_order - 1))
                        if d_order >= 2:
                            cols.append(X[:, i] ** (d_order - 1) * X[:, j] ** (d_order))
            return np.column_stack(cols)
        
        # Try several degrees and pick the best via cross-validation
        best_w = None
        best_score = np.inf
        best_degree = 2
        
        for degree in [2, 3, 4]:
            if time.time() > deadline:
                break
            try:
                F = make_features(X, degree)
                # Add small ridge regularization for stability
                reg = 1e-6 * np.eye(F.shape[1])
                w, *_ = np.linalg.lstsq(F + reg, y, rcond=None)
                
                # Simple holdout validation
                n = len(X)
                n_val = max(10, n // 5)
                idx = np.random.permutation(n)
                val_idx = idx[:n_val]
                tr_idx = idx[n_val:]
                
                pred_val = F[val_idx] @ w
                mse_val = np.mean((pred_val - y[val_idx]) ** 2)
                
                if mse_val < best_score:
                    best_score = mse_val
                    best_w = w
                    best_degree = degree
            except Exception:
                continue
        
        if best_w is None:
            # Fallback to linear
            F = np.column_stack([np.ones(len(X)), X])
            best_w, *_ = np.linalg.lstsq(F, y, rcond=None)
            best_degree = 1
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            F = make_features(X_test, best_degree)
            return F @ best_w
        
        return predict

# EVOLVE-BLOCK-END
