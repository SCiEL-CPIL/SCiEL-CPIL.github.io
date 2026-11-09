import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 290.0  # Safety margin
    
    n_samples, n_features = X_train.shape
    
    # 1. Feature Engineering
    # We generate a rich set of basis functions to capture potential symbolic structures.
    # Inputs are in various ranges, so we should be careful with exponentials.
    # We will standardize X for feature generation to keep values bounded, 
    # but for trig/exponentials, the original scale might matter for periodicity.
    # However, the ranges are [1,3], [1,2], [1,5], [1,10]. 
    # Let's use the raw X for trig/exp to preserve frequency info, 
    # but normalize polynomial terms to avoid huge values.
    
    def generate_features(X):
        # X: (N, d)
        N, d = X.shape
        feats = [np.ones((N, 1))]
        
        # Polynomial features (up to degree 4)
        # Use normalized X for polynomials to prevent overflow and scale issues
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std < 1e-8] = 1.0
        X_norm = (X - X_mean) / X_std
        
        for i in range(d):
            x = X[:, i:i+1]
            xn = X_norm[:, i:i+1]
            
            # Linear
            feats.append(x)
            feats.append(xn)
            
            # Quadratic
            feats.append(x**2)
            feats.append(xn**2)
            
            # Cubic
            feats.append(x**3)
            feats.append(xn**3)
            
            # Quartic
            feats.append(x**4)
            feats.append(xn**4)
            
            # Trigonometric (use raw x to capture actual period)
            feats.append(np.sin(x))
            feats.append(np.cos(x))
            feats.append(np.sin(2*x))
            feats.append(np.cos(2*x))
            
            # Exponential (use raw x, but dampen for large x)
            # Ranges are small (max 10), so exp(-x) and exp(x) are safe-ish
            feats.append(np.exp(-0.5 * x))
            feats.append(np.exp(-0.1 * x))
            feats.append(np.exp(0.1 * x))
            
            # Abs
            feats.append(np.abs(x))
            
        # Interaction terms (pairwise)
        if d > 1:
            for i in range(d):
                for j in range(i+1, d):
                    xi = X[:, i:i+1]
                    xj = X[:, j:j+1]
                    xni = X_norm[:, i:i+1]
                    xnj = X_norm[:, j:j+1]
                    
                    feats.append(xi * xj)
                    feats.append(xni * xnj)
                    feats.append(np.sin(xi) * np.cos(xj))
                    feats.append(np.cos(xi) * np.sin(xj))
                    feats.append(xni * np.sin(xj))
                    feats.append(xnj * np.sin(xi))
                    
        # Triple interactions if d >= 3
        if d >= 3:
            for i in range(d):
                for j in range(i+1, d):
                    for k in range(j+1, d):
                        xi = X[:, i:i+1]
                        xj = X[:, j:j+1]
                        xk = X[:, k:k+1]
                        xni = X_norm[:, i:i+1]
                        xnj = X_norm[:, j:j+1]
                        xnk = X_norm[:, k:k+1]
                        
                        feats.append(xni * xnj * xnk)
                        feats.append(np.sin(xi) * np.sin(xj) * np.sin(xk))
                        
        Phi = np.hstack(feats)
        return Phi, X_mean, X_std

    # Generate features for training data
    Phi_train, X_mean, X_std = generate_features(X_train)
    
    # Standardize the feature matrix for regression stability
    # We do this to make the regularization penalty $\alpha$ apply uniformly
    Phi_mean = Phi_train.mean(axis=0)
    Phi_std = Phi_train.std(axis=0)
    Phi_std[Phi_std < 1e-8] = 1.0
    Phi_train_norm = (Phi_train - Phi_mean) / Phi_std
    
    # Standardize y
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_std_val = float(y_std) if y_std > 1e-8 else 1.0
    y_train_norm = (y_train - y_mean) / y_std_val
    
    N, M = Phi_train_norm.shape
    
    # 2. Hyperparameter Search (Ridge Regression)
    # We search for the optimal regularization parameter alpha.
    # Since N=200, we can afford to compute LOOCV scores for a grid of alphas.
    
    def ridge_loocv_score(Phi, y, alpha):
        """
        Compute LOOCV MSE for Ridge Regression.
        Uses the hat matrix formula:
        LOOCV_i = (y_i - y_hat_i) / (1 - H_ii)
        where H = Phi (Phi^T Phi + alpha I)^-1 Phi^T
        This is equivalent to:
        y_hat = H y
        LOOCV errors = (y - H y) / (1 - diag(H))
        """
        # Efficient computation of hat matrix diagonal
        # H = Phi (Phi^T Phi + alpha I)^-1 Phi^T
        # Let A = Phi^T Phi + alpha I
        # H_ii = Phi_i A^-1 Phi_i^T
        # We can solve A X = Phi^T to get X, then H_ii = sum(Phi_i * X_i)
        
        PhiT_Phi = Phi.T @ Phi
        A = PhiT_Phi + alpha * np.eye(M)
        
        try:
            # Solve A X = Phi^T
            # X is (M, N)
            X_sol = np.linalg.solve(A, Phi.T)
            # H_ii = sum over j of Phi[i, j] * X[j, i]
            H_diag = np.sum(Phi * X_sol.T, axis=1)
        except np.linalg.LinAlgError:
            # Fallback to lstsq if singular (unlikely with alpha > 0)
            X_sol, _, _, _ = np.linalg.lstsq(A, Phi.T, rcond=None)
            H_diag = np.sum(Phi * X_sol.T, axis=1)
            
        # y_hat = Phi X_sol
        y_hat = Phi @ X_sol
        # Actually, y_hat = H y = Phi (Phi^T Phi + alpha I)^-1 Phi^T y
        # Wait, the solution for Ridge is w = (Phi^T Phi + alpha I)^-1 Phi^T y
        # So y_hat = Phi w = Phi (Phi^T Phi + alpha I)^-1 Phi^T y
        # The matrix X_sol above is (Phi^T Phi + alpha I)^-1 Phi^T
        # So y_hat = Phi @ X_sol
        
        # LOOCV prediction for point i:
        # y_hat_i_loocv = (y_i - y_hat_i) / (1 - H_ii) + y_hat_i ? No.
        # Formula: y_hat_i_loocv = (y_hat_i - H_ii y_i) / (1 - H_ii)
        # Error = y_i - y_hat_i_loocv = (y_i - y_hat_i) / (1 - H_ii)
        
        errors = (y - y_hat) / (1 - H_diag)
        mse = np.mean(errors**2)
        return mse

    # Grid search for alpha
    # Log-spaced alpha
    alphas = np.logspace(-4, 4, 100)
    
    best_alpha = 1.0
    best_score = np.inf
    
    for alpha in alphas:
        if time.time() > deadline - 5:
            break
        try:
            score = ridge_loocv_score(Phi_train_norm, y_train_norm, alpha)
            if score < best_score:
                best_score = score
                best_alpha = alpha
        except Exception:
            continue
            
    # 3. Final Fit
    # Solve for weights using best_alpha
    A = Phi_train_norm.T @ Phi_train_norm + best_alpha * np.eye(M)
    try:
        w = np.linalg.solve(A, Phi_train_norm.T @ y_train_norm)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, Phi_train_norm.T @ y_train_norm, rcond=None)[0]
        
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # Generate features for test data using training stats
        Phi_test, _, _ = generate_features(X)
        
        # Standardize features using training stats
        Phi_test_norm = (Phi_test - Phi_mean) / Phi_std
        
        # Predict
        y_pred_norm = Phi_test_norm @ w
        y_pred = y_pred_norm * y_std_val + y_mean
        
        return y_pred

    return predict
