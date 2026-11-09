import numpy as np
import time
from scipy.linalg import cho_solve, cho_factor

def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    X_std = (X - mu) / sd
    
    def rbf_kernel(X1, X2, length_scale):
        """RBF kernel between two sets of points."""
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-sq_dists / (2 * length_scale**2))
    
    def composite_kernel(X1, X2, length_scale, poly_degree=2):
        """Composite kernel: RBF + linear + polynomial."""
        K_rbf = rbf_kernel(X1, X2, length_scale)
        K_lin = X1 @ X2.T
        # Polynomial kernel (degree 2, with bias)
        K_poly = (1 + X1 @ X2.T)**poly_degree
        # Combine: RBF captures local structure, linear captures global trend, poly adds some flexibility
        return K_rbf + 0.1 * K_lin + 0.01 * K_poly
    
    def solve_krr(X_std, y, length_scale, lam):
        """Solve KRR and return alpha coefficients."""
        K = composite_kernel(X_std, X_std, length_scale)
        # Add small jitter for numerical stability
        K_reg = K + lam * np.eye(n) + 1e-12 * np.eye(n)
        try:
            # Use Cholesky for efficiency and stability
            L = np.linalg.cholesky(K_reg)
            alpha = cho_solve((L, True), y)
        except np.linalg.LinAlgError:
            # Fallback to lstsq if Cholesky fails
            alpha = np.linalg.lstsq(K_reg, y, rcond=None)[0]
        return alpha
    
    def loo_cv_score(X_std, y, length_scale, lam):
        """Leave-One-Out CV score using the closed-form KRR formula."""
        K = composite_kernel(X_std, X_std, length_scale)
        K_reg = K + lam * np.eye(n) + 1e-12 * np.eye(n)
        try:
            L = np.linalg.cholesky(K_reg)
            # LOO prediction: y_i_hat = y_i - (K_reg^{-1} y)_i / (K_reg^{-1})_{ii}
            # = y_i - alpha_i / (K_reg^{-1})_{ii}
            # (K_reg^{-1})_{ii} = 1 / L_{ii}^2 (from Cholesky)
            # Actually, (K_reg^{-1})_{ii} = 1 / (L^T L)_{ii}... let me use a different approach
            # K_reg^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}
            # (K_reg^{-1})_{ii} = sum_j (L^{-1})_{ji}^2
            # Simpler: compute alpha = K_reg^{-1} y, then LOO error
            alpha = cho_solve((L, True), y)
            # Diagonal of K_reg^{-1}:
            # (K_reg^{-1})_{ii} = 1 / (L^{-1} L^{-T})_{ii}... 
            # Actually for K_reg = L L^T, K_reg^{-1} = L^{-T} L^{-1}
            # (K_reg^{-1})_{ii} = ||L^{-1} e_i||^2 = sum_j (L^{-1})_{ji}^2
            # This is expensive. Alternative: use the formula
            # LOO MSE = mean((y_i - y_i_hat)^2) where y_i_hat = y_i - alpha_i / (K_reg^{-1})_{ii}
            # Let's just compute it directly for n=200
            K_inv_diag = np.zeros(n)
            # Solve K_reg * v = e_i for each i? Too expensive.
            # Use: (K_reg^{-1})_{ii} = 1 / (L^T L)_{ii}... no.
            # For K_reg = L L^T, the diagonal of K_reg^{-1} can be computed as:
            # Let W = L^{-1} (upper triangular), then K_reg^{-1} = W^T W
            # (K_reg^{-1})_{ii} = ||W[:, i]||^2
            W = np.linalg.solve(L, np.eye(n))  # L W = I, so W = L^{-1}
            K_inv_diag = np.sum(W**2, axis=0)  # Wait, this gives diagonal of W^T W? No.
            # K_reg^{-1} = L^{-T} L^{-1} = (L^{-1})^T L^{-1}
            # If W = L^{-1}, then K_reg^{-1} = W^T W
            # (W^T W)_{ii} = sum_k W_{ki}^2 = ||W[:, i]||^2
            # But W = L^{-1}, so W[:, i] is the i-th column of L^{-1}
            # np.linalg.solve(L, I) gives W such that L @ W = I, so W = L^{-1}
            # (W^T W)_{ii} = sum_k W[k, i]^2
            K_inv_diag = np.sum(W**2, axis=0)  # This is wrong dimensionally
            # Let me reconsider. L is lower triangular, n x n.
            # W = L^{-1} is also lower triangular, n x n.
            # K_reg^{-1} = L^{-T} L^{-1} = W^T W where W = L^{-1}
            # (W^T W)_{ii} = dot(W[:, i], W[:, i]) = sum_k W[k, i]^2
            # W[:, i] is the i-th column of W.
            # Since W is lower triangular, W[k, i] = 0 for k < i.
            # So (W^T W)_{ii} = sum_{k=i}^{n-1} W[k, i]^2
            # This is just np.sum(W**2, axis=0) which gives sum over rows for each column.
            # Yes, that's correct.
            y_hat_loo = y - alpha / K_inv_diag
            mse = np.mean((y - y_hat_loo)**2)
        except np.linalg.LinAlgError:
            # Fallback: just use training MSE as a proxy
            alpha = np.linalg.lstsq(K_reg, y, rcond=None)[0]
            y_pred = K @ alpha
            mse = np.mean((y_pred - y)**2)
        return mse
    
    # Grid search for hyperparameters
    length_scales = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    best_score = np.inf
    best_params = (1.0, 1e-2)  # default: length_scale=1.0, lambda=1e-2
    best_alpha = None
    
    # Sort lambdas to try smaller ones first (less regularization, better fit)
    # Actually, let's do a proper grid search
    for ls in length_scales:
        if time.time() > DEADLINE - 50:
            break
        for lam in lambdas:
            if time.time() > DEADLINE - 50:
                break
            try:
                score = loo_cv_score(X_std, y, ls, lam)
                if score < best_score:
                    best_score = score
                    best_params = (ls, lam)
                    # Store the alpha for this configuration
                    best_alpha = solve_krr(X_std, y, ls, lam)
            except Exception:
                continue
    
    # If KRR didn't work well, fall back to polynomial feature basis
    # Check if the KRR model is good enough
    if best_alpha is not None:
        # Evaluate on training data
        K_train = composite_kernel(X_std, X_std, best_params[0])
        y_pred_train = K_train @ best_alpha
        train_mse = np.mean((y_pred_train - y)**2)
        y_var = np.var(y)
        normalized_mse = train_mse / y_var if y_var > 1e-12 else train_mse
        
        if normalized_mse < 0.1:
            # KRR is working well, use it
            def predict(X):
                X = np.asarray(X, dtype=np.float64)
                X_std_pred = (X - mu) / sd
                K_pred = composite_kernel(X_std_pred, X_std, best_params[0])
                return K_pred @ best_alpha
            return predict
    
    # Fallback: polynomial feature basis with Ridge regression
    def make_features(Xraw):
        Xr = np.asarray(Xraw, dtype=np.float64)
        Xrn = (Xr - mu) / sd
        cols = [np.ones(len(Xr))]
        
        # Linear
        for j in range(d):
            cols.append(Xrn[:, j])
            
        # Squares
        for j in range(d):
            cols.append(Xrn[:, j] ** 2)
            
        # Pairwise products
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(Xrn[:, i] * Xrn[:, j])
                
        # Cubes
        for j in range(d):
            cols.append(Xrn[:, j] ** 3)
            
        # Triple products
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    cols.append(Xrn[:, i] * Xrn[:, j] * Xrn[:, k])
                    
        # Quartics
        for j in range(d):
            cols.append(Xrn[:, j] ** 4)
            
        # Trig
        for j in range(d):
            cols.append(np.sin(Xrn[:, j]))
            cols.append(np.cos(Xrn[:, j]))
            
        # Exp (clipped)
        Xrn_clipped = np.clip(Xrn, -10, 10)
        for j in range(d):
            cols.append(np.exp(Xrn_clipped[:, j]))
            cols.append(np.exp(-Xrn_clipped[:, j]))
            
        return np.column_stack(cols)
    
    A = make_features(X_train)
    p = A.shape[1]
    
    # Quick ridge regression with a few lambda values
    lambdas_fallback = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_w = None
    best_train_mse = np.inf
    
    for lam in lambdas_fallback:
        if time.time() > DEADLINE:
            break
        ATA = A.T @ A + lam * np.eye(p)
        ATy = A.T @ y
        try:
            w = np.linalg.solve(ATA, ATy)
            y_pred = A @ w
            mse = np.mean((y_pred - y)**2)
            if mse < best_train_mse:
                best_train_mse = mse
                best_w = w
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, y, rcond=None)[0]
            y_pred = A @ w
            mse = np.mean((y_pred - y)**2)
            if mse < best_train_mse:
                best_train_mse = mse
                best_w = w
    
    if best_w is None:
        # Last resort: just predict the mean
        y_mean = np.mean(y)
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            return np.full(len(X), y_mean)
        return predict
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        A_pred = make_features(X)
        return A_pred @ best_w
        
    return predict
