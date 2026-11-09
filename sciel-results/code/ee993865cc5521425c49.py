"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 330.0 - 15.0  # safety margin
    
    X = X_train[:, 0] if X_train.ndim == 2 else X_train
    y = y_train
    n = len(X)
    
    # Generate candidate basis functions
    def make_basis(x):
        cols = [np.ones_like(x)]
        names = ['1']
        
        # Polynomials
        for k in range(1, 12):
            cols.append(x ** k)
            names.append(f'x^{k}')
        
        # Trig functions
        for freq in [1, 2, 3, 4, 5, 0.5, np.pi/2, np.pi/3]:
            cols.append(np.sin(freq * x))
            names.append(f'sin({freq}*x)')
            cols.append(np.cos(freq * x))
            names.append(f'cos({freq}*x)')
        
        # More trig
        cols.append(np.sin(x))
        names.append('sin(x)')
        cols.append(np.cos(x))
        names.append('cos(x)')
        cols.append(np.sin(2*x))
        names.append('sin(2x)')
        cols.append(np.cos(2*x))
        names.append('cos(2x)')
        cols.append(np.sin(3*x))
        names.append('sin(3x)')
        cols.append(np.cos(3*x))
        names.append('cos(3x)')
        
        # Exponential
        for c in [0.5, 1, 2, -0.5, -1]:
            cols.append(np.exp(c * x))
            names.append(f'exp({c}*x)')
        
        # Log
        safe_x = np.where(x > 0, x, 1e-10)
        cols.append(np.log(safe_x))
        names.append('log(x)')
        
        # Abs, sqrt
        cols.append(np.abs(x))
        names.append('|x|')
        cols.append(np.sign(x))
        names.append('sign(x)')
        
        # x*exp(x), x*sin(x), etc.
        cols.append(x * np.exp(x))
        names.append('x*exp(x)')
        cols.append(x * np.sin(x))
        names.append('x*sin(x)')
        cols.append(x * np.cos(x))
        names.append('x*cos(x)')
        cols.append(x * np.abs(x))
        names.append('x*|x|')
        
        # Higher order interactions
        cols.append(x ** 2 * np.sin(x))
        names.append('x^2*sin(x)')
        cols.append(x ** 2 * np.cos(x))
        names.append('x^2*cos(x)')
        
        # Rational-like
        cols.append(1.0 / (1.0 + np.exp(-x)))
        names.append('sigmoid(x)')
        cols.append(1.0 / (1.0 + np.exp(-2*x)))
        names.append('sigmoid(2x)')
        
        # Tanh
        cols.append(np.tanh(x))
        names.append('tanh(x)')
        cols.append(np.tanh(2*x))
        names.append('tanh(2x)')
        
        return np.column_stack(cols), names
    
    basis, names = make_basis(X)
    n_features = basis.shape[1]
    
    # Remove constant or near-constant columns
    valid_mask = []
    for j in range(n_features):
        col = basis[:, j]
        std = np.std(col)
        if std > 1e-8 and np.all(np.isfinite(col)):
            valid_mask.append(j)
    
    if not valid_mask:
        valid_mask = [0]
    
    basis = basis[:, valid_mask]
    names = [names[j] for j in valid_mask]
    n_features = basis.shape[1]
    
    # Precompute Gram matrix for fast selection
    # Use ridge for stability
    def solve_lstsq(B, y):
        """Solve least squares with ridge regularization for stability."""
        try:
            w, residuals, rank, sv = np.linalg.lstsq(B, y, rcond=1e-10)
            if not np.all(np.isfinite(w)):
                # Fall back to ridge
                lam = 1e-8
                w = np.linalg.solve(B.T @ B + lam * np.eye(len(w)), B.T @ y)
            return w
        except:
            lam = 1e-6
            w = np.linalg.solve(B.T @ B + lam * np.eye(B.shape[1]), B.T @ y)
            return w
    
    def mse_with_features(B, y, indices):
        """Compute MSE using only specified feature indices."""
        B_sub = B[:, indices]
        w = solve_lstsq(B_sub, y)
        pred = B_sub @ w
        return np.mean((pred - y) ** 2)
    
    # Forward selection with multiple restarts
    best_mse = np.inf
    best_indices = [0]
    
    # Try several random initial seeds for forward selection
    n_restarts = 8
    max_features = min(15, n_features)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        if restart == 0:
            selected = [0]  # start with constant
        else:
            idx = np.random.randint(0, n_features)
            selected = [idx]
        
        current_mse = mse_with_features(basis, y, selected)
        
        for step in range(max_features - len(selected)):
            if time.time() > deadline:
                break
            
            best_gain = -np.inf
            best_idx = None
            candidates = [j for j in range(n_features) if j not in selected]
            
            # Evaluate each candidate feature
            for j in candidates:
                new_indices = selected + [j]
                mse_j = mse_with_features(basis, y, new_indices)
                gain = current_mse - mse_j
                if gain > best_gain:
                    best_gain = gain
                    best_idx = j
            
            if best_idx is None or best_gain < -1e-10:
                break
            
            selected.append(best_idx)
            current_mse = mse_with_features(basis, y, selected)
        
        if current_mse < best_mse:
            best_mse = current_mse
            best_indices = selected[:]
    
    # Also try pure polynomial fits of various degrees
    for deg in range(1, 12):
        if time.time() > deadline:
            break
        poly_idx = list(range(deg + 1))  # 1, x, x^2, ..., x^deg
        if all(i < n_features for i in poly_idx):
            mse_p = mse_with_features(basis, y, poly_idx)
            if mse_p < best_mse:
                best_mse = mse_p
                best_indices = poly_idx[:]
    
    # Final fit with best features
    B_final = basis[:, best_indices]
    w_final = solve_lstsq(B_final, y)
    
    # Verify finiteness
    if not np.all(np.isfinite(w_final)):
        # Fallback: simple polynomial
        B_fb = basis[:, :5]
        w_final = solve_lstsq(B_fb, y)
    
    def predict(X_test):
        Xt = X_test[:, 0] if X_test.ndim == 2 else X_test
        Bt, _ = make_basis(Xt)
        # Apply same valid mask
        Bt = Bt[:, valid_mask]
        return Bt[:, best_indices] @ w_final
    
    return predict
# EVOLVE-BLOCK-END
