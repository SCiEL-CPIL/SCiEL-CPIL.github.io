"""
Strategy: Adaptive Ridge Regression with Rich Basis Expansion.

This approach expands the 1D input into a large set of basis functions:
1. Polynomials up to degree 15.
2. Chebyshev polynomials up to degree 25 (scaled to [-1, 1]).
3. Sine and Cosine terms up to frequency 10.

It then performs Ridge Regression with L2 regularization.
The regularization strength lambda is selected via Leave-One-Out Cross-Validation (LOOCV),
which is computationally efficient for this problem size (n=200).
This allows the model to adapt to the specific complexity of each hidden equation,
capturing non-linear, periodic, and polynomial behaviors without overfitting.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _chebyshev(x, degree):
    """
    Compute Chebyshev polynomials of the first kind up to 'degree' for input x.
    x should be in [-1, 1].
    Returns a matrix of shape (len(x), degree+1).
    """
    T = np.zeros((len(x), degree + 1))
    T[:, 0] = 1
    if degree >= 1:
        T[:, 1] = x
        for k in range(2, degree + 1):
            T[:, k] = 2 * x * T[:, k - 1] - T[:, k - 2]
    return T

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    # Ensure X is 2D column vector
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    x = X_train[:, 0]
    n = len(x)
    
    # Scale x to [-1, 1] for Chebyshev
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max == x_min:
        x_scaled = np.zeros_like(x)
    else:
        x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1
    
    # Generate Basis Features
    # 1. Polynomials (up to degree 15)
    poly_deg = 15
    poly_feats = np.vander(x, N=poly_deg + 1, increasing=True)
    
    # 2. Chebyshev (up to degree 25)
    cheb_deg = 25
    cheb_feats = _chebyshev(x_scaled, cheb_deg)
    
    # 3. Fourier (sin/cos up to freq 10)
    # Use original x for frequency, as it preserves physical scale
    freqs = np.arange(1, 11)
    sin_feats = np.column_stack([np.sin(f * x) for f in freqs])
    cos_feats = np.column_stack([np.cos(f * x) for f in freqs])
    
    # Combine all features
    # Note: poly_feats includes constant term (col 0). Chebyshev T_0 is also constant.
    # Fourier cos(0) is constant, but we started at freq 1.
    # We have redundancy in constants. Ridge regression handles this, but 
    # it's better to remove exact duplicates or let regularization handle it.
    # Let's just stack them. The high dimensionality (16+26+20=62) is small.
    Phi = np.column_stack([poly_feats, cheb_feats, sin_feats, cos_feats])
    
    # Standardize features? 
    # Ridge is scale-invariant if we tune lambda appropriately, 
    # but standardizing helps with the conditioning of the matrix.
    # However, the constant term should probably not be standardized or handled carefully.
    # Let's standardize all columns except the first one (which is constant 1 in poly).
    # Actually, let's just center and scale the non-constant features.
    # Simpler: Just use the raw features. The matrix is small.
    
    # Grid search for Lambda
    # Lambda values: from small to large
    # Since features are not standardized, the scale of lambda matters.
    # Let's estimate a typical variance of features to set a reasonable range.
    feat_var = np.var(Phi, axis=0)
    max_var = np.max(feat_var)
    # Lambda range: 1e-6 to 1e6 * max_var?
    # Let's try a log-spaced grid.
    lambdas = np.logspace(-6, 8, num=45)
    
    best_lambda = 0.0
    best_loocv = np.inf
    best_w = None
    
    # Precompute Phi.T Phi
    PhiTPhi = Phi.T @ Phi
    
    for lam in lambdas:
        if time.time() > deadline:
            break
            
        # Ridge: (Phi.T Phi + lam * I) w = Phi.T y
        # We add lam * I. 
        # To make it scale invariant, we could add lam * diag(feat_var), 
        # but standard Ridge adds lam * I.
        # Let's use standard Ridge.
        A = PhiTPhi + lam * np.eye(Phi.shape[1])
        
        try:
            # Solve for w
            w = np.linalg.solve(A, Phi.T @ y_train)
        except np.linalg.LinAlgError:
            continue
            
        # Calculate LOOCV error
        # H = Phi (Phi.T Phi + lam I)^-1 Phi.T
        # H_ii = row_i(Phi) @ inv(A) @ col_i(Phi)
        # We can compute inv(A) once per lambda
        try:
            inv_A = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue
            
        # H_ii = sum_j Phi_ij * (inv_A @ Phi.T)[:, i]
        # Let M = inv_A @ Phi.T. Shape (M, n).
        # Then H_ii = sum_j Phi_ij * M[j, i] = dot(Phi[:, i], M[:, i])
        M = inv_A @ Phi.T
        H_diag = np.sum(Phi * M, axis=0)
        
        # LOOCV formula:
        # E = 1/n * sum ( (y_i - yhat_i) / (1 - H_ii) )^2
        y_hat = Phi @ w
        residuals = y_train - y_hat
        denominators = 1 - H_diag
        
        # Avoid division by zero or near-zero
        # If H_ii is close to 1, the leverage is high.
        # In practice, with regularization, H_ii < 1.
        # Clip to be safe.
        denominators = np.clip(denominators, 1e-6, None)
        
        loocv_errors = (residuals / denominators) ** 2
        mean_loocv = np.mean(loocv_errors)
        
        if mean_loocv < best_loocv:
            best_loocv = mean_loocv
            best_lambda = lam
            best_w = w
    
    if best_w is None:
        # Fallback to simple mean if something went wrong
        mean_y = np.mean(y_train)
        def predict(X):
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.full(X.shape[0], mean_y)
        return predict

    w_final = best_w
    Phi_final = Phi
    x_min_final = x_min
    x_max_final = x_max
    
    def predict(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        x_pred = X[:, 0]
        
        # Scale to [-1, 1]
        if x_max_final == x_min_final:
            x_scaled_pred = np.zeros_like(x_pred)
        else:
            x_scaled_pred = 2 * (x_pred - x_min_final) / (x_max_final - x_min_final) - 1
            
        # Recreate features
        poly_feats = np.vander(x_pred, N=poly_deg + 1, increasing=True)
        cheb_feats = _chebyshev(x_scaled_pred, cheb_deg)
        
        freqs = np.arange(1, 11)
        sin_feats = np.column_stack([np.sin(f * x_pred) for f in freqs])
        cos_feats = np.column_stack([np.cos(f * x_pred) for f in freqs])
        
        Phi_pred = np.column_stack([poly_feats, cheb_feats, sin_feats, cos_feats])
        
        return Phi_pred @ w_final

    return predict
# EVOLVE-BLOCK-END
