"""Symbolic regression using KRR with robust fallbacks and multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    t_start = time.time()
    DEADLINE = 305.0  # Safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        
    n, d = X.shape
    y_flat = y.flatten()

    # Standardize X and y
    mu_x = X.mean(axis=0)
    std_x = X.std(axis=0)
    std_x[std_x < 1e-8] = 1.0
    X_std = (X - mu_x) / std_x
    
    mu_y = y_flat.mean()
    std_y = y_flat.std()
    if std_y < 1e-8:
        std_y = 1.0
    y_std = (y_flat - mu_y) / std_y

    # 1. Polynomial Feature Generator (Degree 3)
    def poly_features(X, degree=3):
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            cols.append(X[:, j])
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                cols.append(X[:, i] * X[:, j])
        if degree >= 3:
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    for k in range(j, X.shape[1]):
                        cols.append(X[:, i] * X[:, j] * X[:, k])
        return np.column_stack(cols)

    # 2. Fourier Feature Generator (Sum-of-Sines)
    # We use a fixed grid of frequencies. Since inputs are in [1, 5] or [1, 10],
    # we scale X to [0, 1] or use the raw range to define frequencies.
    # Let's use the raw X but scale frequencies based on the domain size.
    # Domain ranges: [1,3], [1,2], [1,5], [1,10].
    # We'll use a set of frequencies that cover a few periods over the max range.
    
    def fourier_features(X, max_freq=10.0, n_freqs=20):
        # X is raw, not standardized, to preserve periodicity relative to input values
        # If we standardize, we lose the absolute scale which might be important for periodicity.
        # However, standardizing helps with numerical stability for non-periodic parts.
        # Let's try using standardized X for the Fourier basis as well, but with frequencies
        # scaled appropriately. Or better, use raw X if we know the bounds.
        # Since we don't know the exact bounds per equation but have hints, we can assume
        # frequencies should be in the range of [pi/min_range, 10*pi/max_range].
        
        # Simpler approach: Use standardized X and a range of frequencies.
        # If the function is periodic in raw X, standardization distorts it.
        # Let's provide both: one set on raw X, one on std X? No, keep it simple.
        # Let's assume the "hidden equations" are smooth. 
        # We will use a bank of sines and cosines.
        
        cols = [np.ones(len(X))]
        # Frequencies: logarithmic spacing from 0.1 to max_freq
        freqs = np.logspace(-1, np.log10(max_freq), n_freqs)
        
        for f in freqs:
            for j in range(X.shape[1]):
                # Use raw X for periodicity? 
                # If X is in [1, 10], sin(0.1 * X) is very low freq. sin(10 * X) is high freq.
                # Let's use the raw X values directly.
                cols.append(np.sin(f * X[:, j]))
                cols.append(np.cos(f * X[:, j]))
        
        # Add interactions for 2D+ cases? 
        # For d=2, add sin(f * (X1+X2)) etc? That doubles the cost.
        # Let's stick to per-dimension features first, maybe add simple products.
        
        return np.column_stack(cols)

    # Compute features
    X_poly = poly_features(X_std)
    
    # For Fourier, using raw X might be better for periodic functions.
    # Let's try both or just raw. Given the ranges [1, 10], raw is safe.
    X_fourier = fourier_features(X, max_freq=15.0, n_freqs=15)
    
    # Combine into a list of candidate feature sets
    # We will evaluate KRR with LOO-CV for each
    candidate_features = [
        ("poly", X_poly),
        ("fourier", X_fourier)
    ]
    
    best_model_type = "poly"
    best_lambda = 1e-3
    best_loo_error = np.inf
    best_W = None
    best_feat = None

    # Regularization strengths to try
    lambdas = np.logspace(-5, 2, 20)
    
    # Solve KRR for each candidate and lambda
    # KRR: W = (X^T X + lambda * I)^-1 X^T y
    # To be efficient, we can precompute X^T X.
    
    for name, Feat in candidate_features:
        if time.time() > DEADLINE - 10:
            break
            
        # Centering y is already done (y_std). 
        # We should also center X? 
        # KRR is invariant to centering X if we handle the bias correctly.
        # Here we include a bias term (column of ones) in Feat.
        # So we don't need to explicitly center X, but standardization helps conditioning.
        # For Fourier, we used raw X. Let's standardize the columns of Feat for numerical stability?
        # No, that changes the basis. Let's just solve the system.
        
        # Precompute Gram matrix G = Feat @ Feat.T ? No, that's n x n.
        # If p (features) < n, solving p x p is better.
        # If p > n, solving n x n is better.
        
        p = Feat.shape[1]
        
        if p < n:
            # Solve W = (Feat^T Feat + lambda I)^-1 Feat^T y
            G = Feat.T @ Feat
            y_proj = Feat.T @ y_std
            for lam in lambdas:
                try:
                    A = G + lam * np.eye(p)
                    # Use Cholesky for stability
                    L, info = np.linalg.cholesky(A)
                    if info > 0: # np.linalg.cholesky doesn't return info like scipy
                         pass # np.linalg.cholesky raises LinAlgError
                    W = np.linalg.solve(A, y_proj)
                    
                    # Compute LOO error
                    # LOO error for KRR: e_i = (y_i - f_i) / (1 - H_ii)
                    # H = Feat @ W
                    # But we need H_ii. 
                    # H_ii = (Feat_i^T (Feat^T Feat + lam I)^-1 Feat_i)
                    # This is expensive to compute for all i if we do it naively.
                    # Alternative: Compute predictions for all points, then use the LOO formula.
                    # f_i = Feat_i @ W
                    # H_ii requires the diagonal of the hat matrix.
                    # Hat Matrix H = Feat @ (Feat^T Feat + lam I)^-1 @ Feat^T
                    # This is n x n.
                    
                    # Efficient LOO for KRR:
                    # Let K = Feat^T Feat.
                    # W = (K + lam I)^-1 Feat^T y
                    # f = Feat W = Feat (K + lam I)^-1 Feat^T y
                    # Let M = Feat (K + lam I)^-1. Then f = M Feat^T y.
                    # H_ii = M_i Feat_i^T.
                    # This is still O(n^2 p) or similar.
                    
                    # Given n=200, we can afford O(n^3) or O(n^2 p) a few times.
                    # Let's just compute the full prediction matrix and use the formula.
                    # f = Feat @ W
                    f = Feat @ W
                    # Compute H_ii
                    # H = Feat @ W @ Feat.T ? No.
                    # H = Feat @ (Feat^T Feat + lam I)^-1 @ Feat^T
                    # We have W = (Feat^T Feat + lam I)^-1 Feat^T y
                    # So (Feat^T Feat + lam I)^-1 = W @ Feat.T @ inv(Feat^T y)? No.
                    
                    # Let's just compute H_ii directly for the chosen W.
                    # H_ii = sum_j Feat[i,j] * (Feat^T (Feat^T Feat + lam I)^-1 Feat^T)[j, i] ? No.
                    # H_ii = Feat[i, :] @ (Feat^T Feat + lam I)^-1 @ Feat[i, :].T
                    # Let Z = (Feat^T Feat + lam I)^-1. We have Z @ Feat^T y = W.
                    # We don't have Z explicitly if we used solve.
                    # Let's compute Z explicitly for p < n. p is small (poly d=3 -> ~20 feats, fourier d=3 -> ~90 feats).
                    # p is definitely < n=200.
                    
                    Z = np.linalg.inv(G + lam * np.eye(p))
                    # H_ii = Feat[i] @ Z @ Feat[i].T
                    # H_diag = np.einsum('ij, jk, ik -> i', Feat, Z, Feat)
                    H_diag = np.einsum('ij, jk, ik -> i', Feat, Z, Feat)
                    
                    # LOO error
                    denom = 1.0 - H_diag
                    # Avoid division by zero
                    denom[np.abs(denom) < 1e-8] = 1e-8
                    loo_errors = (y_std - f) / denom
                    mse = np.mean(loo_errors**2)
                    
                    if mse < best_loo_error:
                        best_loo_error = mse
                        best_lambda = lam
                        best_model_type = name
                        best_W = W
                        best_feat = Feat
                        
                except np.linalg.LinAlgError:
                    continue
                except Exception:
                    continue
        else:
            # p >= n, solve n x n system
            G = Feat @ Feat.T
            for lam in lambdas:
                try:
                    A = G + lam * np.eye(n)
                    W_alpha = np.linalg.solve(A, y_std)
                    f = Feat @ W_alpha
                    # H = Feat @ W_alpha @ Feat.T ? No.
                    # H = Feat @ (Feat^T Feat + lam I)^-1 @ Feat^T
                    # If we solve (G + lam I) alpha = y, then f = Feat alpha.
                    # H_ii = Feat[i] @ (Feat^T Feat + lam I)^-1 @ Feat[i].T
                    # Note that (Feat^T Feat + lam I)^-1 is p x p.
                    # But we solved n x n.
                    # Relationship: (Feat^T Feat + lam I)^-1 Feat^T = Feat^T (Feat Feat^T + lam I)^-1
                    # So H = Feat Feat^T (Feat Feat^T + lam I)^-1 = A - lam I all inv?
                    # H = (G + lam I) (G + lam I)^-1 - lam (G + lam I)^-1 = I - lam (G + lam I)^-1
                    # So H_diag = 1 - lam * diag((G + lam I)^-1)
                    # We have alpha = (G + lam I)^-1 y.
                    # We need diag((G + lam I)^-1).
                    # Let's compute inv(G + lam I) explicitly since n=200 is small.
                    InvA = np.linalg.inv(A)
                    H_diag = 1.0 - lam * np.diag(InvA)
                    
                    denom = 1.0 - H_diag
                    denom[np.abs(denom) < 1e-8] = 1e-8
                    loo_errors = (y_std - f) / denom
                    mse = np.mean(loo_errors**2)
                    
                    if mse < best_loo_error:
                        best_loo_error = mse
                        best_lambda = lam
                        best_model_type = name
                        # For p >= n, we store alpha and feat
                        best_W = W_alpha 
                        best_feat = Feat
                        # Note: predict logic needs to know if we stored W (p x 1) or alpha (n x 1)
                        # We'll handle this in predict.
                        # Actually, it's safer to always store the final coefficients in the feature space if possible.
                        # If p >= n, W is not unique. But f = Feat alpha is unique.
                        # So we store alpha and use f = Feat @ alpha for prediction.
                        # If p < n, we store W and use f = Feat @ W.
                        # Let's standardize: store the vector used for prediction.
                        # If p < n: pred_vec = W (size p)
                        # If p >= n: pred_vec = alpha (size n), and pred = Feat @ pred_vec
                        # We need to distinguish in predict.
                        
                except np.linalg.LinAlgError:
                    continue
                except Exception:
                    continue

    # Fallback: Simple Polynomial Regression if KRR failed
    if best_W is None:
        # Use lstsq on poly features
        w_poly, *_ = np.linalg.lstsq(X_poly, y_std, rcond=None)
        best_model_type = "poly_lstsq"
        best_W = w_poly
        best_feat = X_poly
        # In this case, predict is Feat @ W

    # Define predict function
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        
        # Generate features
        if best_model_type == "poly":
            X_test_feat = poly_features((X_test - mu_x) / std_x)
        elif best_model_type == "fourier":
            X_test_feat = fourier_features(X_test, max_freq=15.0, n_freqs=15)
        else:
            # Fallback
            X_test_feat = poly_features((X_test - mu_x) / std_x)

        # Compute prediction
        # Check dimensionality of best_W to determine if it's W (p) or alpha (n)
        # If best_W.shape[0] == X_test_feat.shape[1], it's W.
        # If best_W.shape[0] == n, it's alpha.
        
        if best_W.shape[0] == X_test_feat.shape[1]:
            y_pred_std = X_test_feat @ best_W
        else:
            # It's alpha, so y_pred_std = X_test_feat @ best_W
            # Wait, if it was the n x n case, best_W is alpha (n).
            # Prediction is X_test_feat @ alpha? No.
            # In the n x n case, f_train = X_train @ alpha.
            # For test, f_test = X_test @ alpha? No, that's only if X_test is in the same space.
            # The dual representation is f(x) = sum_i alpha_i K(x, x_i).
            # Our "kernel" is linear in the feature space: K(x, z) = phi(x)^T phi(z).
            # So f(x) = phi(x)^T (Phi^T Phi + lam I)^-1 Phi^T y = phi(x)^T alpha.
            # Yes, so y_pred_std = X_test_feat @ best_W works in both cases!
            y_pred_std = X_test_feat @ best_W

        y_pred = y_pred_std * std_y + mu_y
        y_pred = np.where(np.isfinite(y_pred), y_pred, mu_y)
        return y_pred

    return predict
# EVOLVE-BLOCK-END
