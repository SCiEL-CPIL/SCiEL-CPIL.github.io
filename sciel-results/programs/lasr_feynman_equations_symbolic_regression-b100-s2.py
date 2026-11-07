import numpy as np
import time

def _build_poly_features(X, degree):
    """Build polynomial features of a given degree."""
    n, d = X.shape
    features = [np.ones((n, 1))]
    for deg in range(1, degree + 1):
        # Use itertools to generate combinations with replacement for powers
        # But for small d and degree, we can just do it manually or use a helper
        # Here we assume d is small (1-3)
        # Generate all monomials of degree 'deg'
        # Simple approach: expand X and take powers? No, that's not right.
        # We need all combinations x_i^a1 ... x_j^ak such that sum(ai)=deg
        # For d=1: x^deg
        # For d=2: x1^deg, x1^(deg-1)x2, ..., x2^deg
        # We will use a recursive or iterative method
        pass
        
    # Efficient way for small d:
    # Let's just generate all monomials up to 'degree'
    # This might be overkill, but d is small.
    # Actually, let's just use a fixed set of common features if degree is not specified, 
    # or implement a simple monomial generator.
    
    # For simplicity and speed, let's just use powers of each variable and their products up to degree 2 or 3
    # Since d <= 3, the number of features is manageable.
    
    # Let's implement a proper monomial generator for d variables up to max_degree
    def get_monomials(d, max_degree):
        monomials = []
        # Recursive helper
        def gen(prefix, remaining_deg, start_var):
            if remaining_deg == 0:
                monomials.append(prefix)
                return
            for i in range(start_var, d):
                # Add variable i with power 1..remaining_deg?
                # No, we need to distribute 'remaining_deg' among variables i..d-1
                # This is stars and bars.
                # Easier: just iterate all possible exponent vectors
                pass
        # Simpler: just iterate all exponent vectors e1..ed such that sum(e) <= max_degree
        # For d=3, max_degree=4, this is small.
        # Let's just hardcode the expansion for d=1,2,3 to be safe and fast.
        
        if d == 1:
            for i in range(1, max_degree+1):
                monomials.append((i,))
        elif d == 2:
            for i in range(1, max_degree+1):
                for j in range(0, i+1):
                    if i == 0 and j == 0: continue
                    monomials.append((i, j))
        elif d == 3:
            for i in range(0, max_degree+1):
                for j in range(0, max_degree+1-i):
                    k = max_degree - i - j
                    # We want sum <= max_degree, so this loop is for sum == max_degree.
                    # We need all sums <= max_degree.
                    # This is getting complicated.
                    pass
        return monomials

    # Let's use a simpler strategy: 
    # Features: x1, x2, x3, x1^2, x2^2, x3^2, x1x2, x1x3, x2x3, x1^3, ...
    # We will just construct a basis of all monomials of degree 1 to 4 for d<=3.
    
    # Precomputed indices for d=1,2,3
    # d=1: [1], [2], [3], [4]
    # d=2: [1,0], [0,1], [2,0], [0,2], [1,1], [3,0], [0,3], [2,1], [1,2], [4,0]...
    
    # To keep it simple and fast, we will only use degree up to 3.
    # Features for d=1: x, x^2, x^3
    # Features for d=2: x1, x2, x1^2, x2^2, x1x2, x1^3, x1^2x2, x1x2^2, x2^3
    # Features for d=3: x1, x2, x3, x1^2, x2^2, x3^2, x1x2, x1x3, x2x3, ... (degree 3 terms)
    
    # Let's just generate them dynamically for the given d.
    from itertools import product
    
    # Generate all exponent tuples (e1, ..., ed) such that 1 <= sum(ei) <= degree
    exp_tuples = []
    # This is a stars and bars problem, but for small d and degree, brute force is fine.
    # Max degree 4, d=3 -> 3^4 = 81 checks.
    for exps in product(range(degree + 1), repeat=d):
        if sum(exps) > 0 and sum(exps) <= degree:
            exp_tuples.append(exps)
            
    features = []
    for exps in exp_tuples:
        feat = np.ones(n)
        for i in range(d):
            if exps[i] > 0:
                feat = feat * (X[:, i] ** exps[i])
        features.append(feat)
        
    if features:
        return np.column_stack(features)
    else:
        return np.ones((n, 1))

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    X = np.array(X_train, dtype=float)
    y = np.array(y_train, dtype=float)
    n, d = X.shape
    
    # Normalize X
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    X_norm = (X - x_mean) / x_std
    
    # Normalize y
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-8:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    best_score = np.inf
    best_predict_fn = None

    # 1. Polynomial Regression
    for degree in [1, 2, 3, 4]:
        if time.time() > deadline - 20:
            break
            
        try:
            Phi = _build_poly_features(X_norm, degree)
            # Add intercept
            Phi = np.column_stack([np.ones(n), Phi])
            
            # Ridge Regression to handle potential collinearity
            # Use small lambda for stability
            lam = 1e-6
            A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
            b = Phi.T @ y_norm
            w = np.linalg.solve(A, b)
            
            # LOOCV score
            # H = Phi (Phi^T Phi + lam I)^-1 Phi^T
            # diag(H) = sum_i Phi_ij^2 (Phi^T Phi + lam I)^-1_ij ... 
            # Actually, diag(H) = np.einsum('ij,ji->i', Phi, np.linalg.inv(A) @ Phi.T) is not quite right because of the lam.
            # H = Phi (Phi^T Phi + lam I)^-1 Phi^T
            # Let's compute H_diag directly
            # H_diag[i] = Phi[i] @ inv(A) @ Phi[i].T
            # This is O(n * p^2) which is fine for small n, p.
            inv_A = np.linalg.inv(A)
            H_diag = np.einsum('ij,ji->i', Phi, inv_A @ Phi.T)
            
            # LOOCV error: (y_i - y_hat_i) / (1 - H_ii)
            y_hat = Phi @ w
            denom = 1 - H_diag
            # Clip denom to avoid division by zero
            denom[denom < 1e-8] = 1e-8
            denom[denom > 1 - 1e-8] = 1 - 1e-8
            
            loocv_errors = (y_norm - y_hat) / denom
            mse = np.mean(loocv_errors**2)
            
            if mse < best_score:
                best_score = mse
                # Create predict function
                w_poly = w.copy()
                degree_poly = degree
                x_mean_poly = x_mean
                x_std_poly = x_std
                y_mean_poly = y_mean
                y_std_poly = y_std
                
                def poly_predict(X_test):
                    X_test = np.array(X_test, dtype=float)
                    if X_test.ndim == 1:
                        X_test = X_test.reshape(1, -1)
                    X_test_norm = (X_test - x_mean_poly) / x_std_poly
                    Phi_test = _build_poly_features(X_test_norm, degree_poly)
                    Phi_test = np.column_stack([np.ones(len(X_test)), Phi_test])
                    y_pred_norm = Phi_test @ w_poly
                    return y_pred_norm * y_std_poly + y_mean_poly
                
                best_predict_fn = poly_predict
                
        except Exception:
            continue

    # 2. Random Fourier Features (Approximating Gaussian Kernel)
    # This is useful if the function is smooth but not polynomial.
    if time.time() < deadline - 20:
        # Try a few different scales and dimensions
        for n_features in [16, 32, 64]:
            for sigma in [0.5, 1.0, 2.0]:
                if time.time() > deadline - 10:
                    break
                
                try:
                    # Generate random Fourier features
                    # w ~ N(0, I), b ~ U(0, 2pi)
                    # Phi(x) = [cos(w_1 x + b_1), sin(w_1 x + b_1), ...]
                    # Scale w by 1/sigma to match kernel exp(-||x-x'||^2 / (2 sigma^2))
                    # Actually, standard RFF for Gaussian kernel uses w ~ N(0, 1/sigma^2)
                    
                    w_rff = np.random.randn(d, n_features) / sigma
                    b_rff = np.random.rand(n_features) * 2 * np.pi
                    
                    # Train features
                    sin_part = np.sin(X_norm @ w_rff + b_rff)
                    cos_part = np.cos(X_norm @ w_rff + b_rff)
                    Phi_rff = np.column_stack([np.ones(n), sin_part, cos_part])
                    
                    lam = 1e-6
                    A = Phi_rff.T @ Phi_rff + lam * np.eye(Phi_rff.shape[1])
                    b = Phi_rff.T @ y_norm
                    w_rff_weights = np.linalg.solve(A, b)
                    
                    # LOOCV
                    inv_A = np.linalg.inv(A)
                    H_diag = np.einsum('ij,ji->i', Phi_rff, inv_A @ Phi_rff.T)
                    y_hat = Phi_rff @ w_rff_weights
                    denom = 1 - H_diag
                    denom[denom < 1e-8] = 1e-8
                    denom[denom > 1 - 1e-8] = 1 - 1e-8
                    loocv_errors = (y_norm - y_hat) / denom
                    mse = np.mean(loocv_errors**2)
                    
                    if mse < best_score:
                        best_score = mse
                        w_rff_store = w_rff.copy()
                        b_rff_store = b_rff.copy()
                        w_rff_weights_store = w_rff_weights.copy()
                        x_mean_rff = x_mean
                        x_std_rff = x_std
                        y_mean_rff = y_mean
                        y_std_rff = y_std
                        n_features_rff = n_features
                        
                        def rff_predict(X_test):
                            X_test = np.array(X_test, dtype=float)
                            if X_test.ndim == 1:
                                X_test = X_test.reshape(1, -1)
                            X_test_norm = (X_test - x_mean_rff) / x_std_rff
                            sin_part = np.sin(X_test_norm @ w_rff_store + b_rff_store)
                            cos_part = np.cos(X_test_norm @ w_rff_store + b_rff_store)
                            Phi_test = np.column_stack([np.ones(len(X_test)), sin_part, cos_part])
                            y_pred_norm = Phi_test @ w_rff_weights_store
                            return y_pred_norm * y_std_rff + y_mean_rff
                        
                        best_predict_fn = rff_predict
                        
                except Exception:
                    continue

    # Fallback
    if best_predict_fn is None:
        def fallback_predict(X_test):
            return np.full(len(X_test), y_mean)
        best_predict_fn = fallback_predict

    return best_predict_fn
