import numpy as np
import time

def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 325.0  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd
    
    def make_features(Xraw):
        Xr = np.asarray(Xraw, dtype=np.float64)
        Xrn = (Xr - mu) / sd
        cols = [np.ones(len(Xr))]
        
        # Linear
        for j in range(d):
            cols.append(Xrn[:, j])
            
        # Polynomial terms up to degree 4
        # Generate all unique monomials
        # To keep it efficient and not too high dimensional, we limit the total degree.
        # For d=3, degree 4 is manageable.
        # We'll generate x_i^k and combinations.
        # Simpler: just add pairwise and triple products and powers.
        
        # Squares
        for j in range(d):
            cols.append(Xrn[:, j] ** 2)
            
        # Pairwise products (i < j)
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(Xrn[:, i] * Xrn[:, j])
                
        # Cubes
        for j in range(d):
            cols.append(Xrn[:, j] ** 3)
            
        # Triple products (i < j < k)
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    cols.append(Xrn[:, i] * Xrn[:, j] * Xrn[:, k])
                    
        # Quartics (just powers of 4 for simplicity, and some mixed)
        for j in range(d):
            cols.append(Xrn[:, j] ** 4)
            
        # Trig functions
        for j in range(d):
            cols.append(np.sin(Xrn[:, j]))
            cols.append(np.cos(Xrn[:, j]))
            
        # Exp functions (careful with overflow, but standardized X is usually safe)
        # Clip to avoid extreme values if standardization fails or range is large
        Xrn_clipped = np.clip(Xrn, -10, 10)
        for j in range(d):
            cols.append(np.exp(Xrn_clipped[:, j]))
            cols.append(np.exp(-Xrn_clipped[:, j]))
            
        return np.column_stack(cols)
    
    A = make_features(X_train)
    
    # Grid search for lambda
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_w = None
    best_val_err = np.inf
    
    # Simple K-Fold CV (K=5)
    k = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    for lam in lambdas:
        val_errors = []
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            
            A_tr = A[train_idx]
            y_tr = y[train_idx]
            A_val = A[val_idx]
            y_val = y[val_idx]
            
            # Solve (A^T A + lam * I) w = A^T y
            # Use lstsq for robustness
            # Or solve explicitly
            try:
                # Add regularization to diagonal
                # A is n x p
                # A^T A is p x p
                # p might be > n, so A^T A is singular without reg.
                # We solve the regularized system.
                # Efficient way: np.linalg.lstsq with added noise? 
                # Better: solve (A.T @ A + lam * np.eye(A.shape[1])) @ w = A.T @ y
                p = A.shape[1]
                ATA = A_tr.T @ A_tr
                ATy = A_tr.T @ y_tr
                reg = lam * np.eye(p)
                w = np.linalg.solve(ATA + reg, ATy)
                
                y_pred_val = A_val @ w
                mse = np.mean((y_pred_val - y_val) ** 2)
                val_errors.append(mse)
            except np.linalg.LinAlgError:
                # Fallback to lstsq if solve fails (shouldn't happen with reg)
                w, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
                y_pred_val = A_val @ w
                mse = np.mean((y_pred_val - y_val) ** 2)
                val_errors.append(mse)
                
        mean_val_err = np.mean(val_errors)
        if mean_val_err < best_val_err:
            best_val_err = mean_val_err
            # Refit on all data with best lambda
            p = A.shape[1]
            ATA = A.T @ A
            ATy = A.T @ y
            reg = lam * np.eye(p)
            try:
                best_w = np.linalg.solve(ATA + reg, ATy)
            except np.linalg.LinAlgError:
                best_w, *_ = np.linalg.lstsq(A, y, rcond=None)
                
            # Check time
            if time.time() > DEADLINE:
                break
                
    # If no lambda was selected (e.g. due to time), use a default
    if best_w is None:
        p = A.shape[1]
        ATA = A.T @ A
        ATy = A.T @ y
        reg = 1e-3 * np.eye(p)
        best_w = np.linalg.solve(ATA + reg, ATy)
        
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        A_pred = make_features(X)
        return A_pred @ best_w
        
    return predict
