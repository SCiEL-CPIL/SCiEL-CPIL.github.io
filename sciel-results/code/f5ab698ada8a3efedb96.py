import numpy as np
import time
import warnings

def fit(X_train, y_train):
    """
    Fits a regression model using a dense sum-of-sines basis with interactions.
    This approach uses a rich set of periodic features to capture smooth
    nonlinear patterns and interactions, solved via Ridge Regression.
    """
    t0 = time.time()
    DEADLINE = 320.0  # Safety margin

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # Standardize inputs to [0, 1] or similar scale for better frequency matching
    # Using min-max scaling to [0, 1] helps define frequencies relative to the range
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges < 1e-12] = 1.0
    X_norm = (X - min_vals) / ranges

    # Define frequency grid
    # We use a range of frequencies to capture various oscillation modes.
    # Since inputs are scaled to [0,1], frequencies represent cycles per unit range.
    # Let's try frequencies from 1 to 10.
    freqs = np.arange(1, 11, 1.0)  # 1, 2, ..., 10

    basis_list = []
    basis_names = []

    # Add constant term
    basis_list.append(np.ones(n))
    basis_names.append("1")

    # Add single-variable sine and cosine terms
    for i in range(d):
        xi = X_norm[:, i]
        for f in freqs:
            sin_val = np.sin(2 * np.pi * f * xi)
            cos_val = np.cos(2 * np.pi * f * xi)
            basis_list.append(sin_val)
            basis_names.append(f"sin({f}x{i})")
            basis_list.append(cos_val)
            basis_names.append(f"cos({f}x{i})")

    # Add interaction terms (products of sine/cosine from different variables)
    # To keep dimensionality manageable, we limit interactions to pairs.
    # We only use low frequencies for interactions to avoid explosion.
    inter_freqs = freqs[:5]  # 1, 2, 3, 4, 5

    for i in range(d):
        for j in range(i + 1, d):
            xi = X_norm[:, i]
            xj = X_norm[:, j]
            for fi in inter_freqs:
                for fj in inter_freqs:
                    # sin(xi) * sin(xj), sin(xi) * cos(xj), etc.
                    # This creates a rich interaction space.
                    s_i_sin = np.sin(2 * np.pi * fi * xi)
                    s_i_cos = np.cos(2 * np.pi * fi * xi)
                    s_j_sin = np.sin(2 * np.pi * fj * xj)
                    s_j_cos = np.cos(2 * np.pi * fj * xj)

                    basis_list.append(s_i_sin * s_j_sin)
                    basis_names.append(f"sin({fi}x{i})*sin({fj}x{j})")
                    basis_list.append(s_i_sin * s_j_cos)
                    basis_names.append(f"sin({fi}x{i})*cos({fj}x{j})")
                    basis_list.append(s_i_cos * s_j_sin)
                    basis_names.append(f"cos({fi}x{i})*sin({fj}x{j})")
                    basis_list.append(s_i_cos * s_j_cos)
                    basis_names.append(f"cos({fi}x{i})*cos({fj}x{j})")

    # Add triple interactions for d >= 3, using very low frequencies
    if d >= 3:
        triple_freqs = freqs[:2]  # 1, 2
        for i in range(d):
            for j in range(i + 1, d):
                for k in range(j + 1, d):
                    xi = X_norm[:, i]
                    xj = X_norm[:, j]
                    xk = X_norm[:, k]
                    for fi in triple_freqs:
                        for fj in triple_freqs:
                            for fk in triple_freqs:
                                s_i = np.sin(2 * np.pi * fi * xi)
                                s_j = np.sin(2 * np.pi * fj * xj)
                                s_k = np.sin(2 * np.pi * fk * xk)
                                basis_list.append(s_i * s_j * s_k)
                                basis_names.append(f"sin({fi}x{i})*sin({fj}x{j})*sin({fk}x{k})")

    # Stack basis matrix
    Phi = np.column_stack(basis_list)
    M = Phi.shape[1]

    # Check for non-finite values (shouldn't happen with sin/cos, but just in case)
    if not np.all(np.isfinite(Phi)):
        Phi = np.nan_to_num(Phi, nan=0.0, posinf=0.0, neginf=0.0)

    # Solve Ridge Regression
    # lambda is the regularization parameter
    # We can try a few values or use a fixed one.
    # A small lambda is good if the basis is not too overcomplete relative to n.
    # M is likely around 100-200 for d=3. n=200.
    # Let's use lambda = 1e-4 to 1e-2.
    
    best_lambda = 1e-3
    best_mse = np.inf
    best_coeff = None

    # Try a few regularization strengths
    for lam in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        if time.time() > DEADLINE:
            break
            
        # Solve (Phi.T Phi + lam * I) coeff = Phi.T y
        try:
            A = Phi.T @ Phi + lam * np.eye(M)
            b = Phi.T @ y
            coeff = np.linalg.solve(A, b)
            
            pred = Phi @ coeff
            residual = y - pred
            mse = np.mean(residual**2)
            
            if mse < best_mse:
                best_mse = mse
                best_coeff = coeff
                best_lambda = lam
                
        except np.linalg.LinAlgError:
            continue

    if best_coeff is None:
        # Fallback to linear regression if Ridge failed
        try:
            Phi_design = np.column_stack([np.ones(n), X_norm])
            best_coeff_full, *_ = np.linalg.lstsq(Phi_design, y, rcond=None)
            # Map back to our basis? No, just use this simple model.
            # But we need to return a predict that uses the SAME logic.
            # Let's just return the mean if everything fails.
            mean_y = np.mean(y)
            def predict(X_test):
                return np.full(X_test.shape[0], mean_y)
            return predict
        except:
            mean_y = np.mean(y)
            def predict(X_test):
                return np.full(X_test.shape[0], mean_y)
            return predict

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - min_vals) / ranges

        # Reconstruct basis
        test_basis_list = []
        
        # Constant
        test_basis_list.append(np.ones(X_test.shape[0]))

        # Single variables
        for i in range(d):
            xi = X_test_norm[:, i]
            for f in freqs:
                test_basis_list.append(np.sin(2 * np.pi * f * xi))
                test_basis_list.append(np.cos(2 * np.pi * f * xi))

        # Interactions
        for i in range(d):
            for j in range(i + 1, d):
                xi = X_test_norm[:, i]
                xj = X_test_norm[:, j]
                for fi in inter_freqs:
                    for fj in inter_freqs:
                        s_i_sin = np.sin(2 * np.pi * fi * xi)
                        s_i_cos = np.cos(2 * np.pi * fi * xi)
                        s_j_sin = np.sin(2 * np.pi * fj * xj)
                        s_j_cos = np.cos(2 * np.pi * fj * xj)

                        test_basis_list.append(s_i_sin * s_j_sin)
                        test_basis_list.append(s_i_sin * s_j_cos)
                        test_basis_list.append(s_i_cos * s_j_sin)
                        test_basis_list.append(s_i_cos * s_j_cos)

        # Triple interactions
        if d >= 3:
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        xi = X_test_norm[:, i]
                        xj = X_test_norm[:, j]
                        xk = X_test_norm[:, k]
                        for fi in triple_freqs:
                            for fj in triple_freqs:
                                for fk in triple_freqs:
                                    s_i = np.sin(2 * np.pi * fi * xi)
                                    s_j = np.sin(2 * np.pi * fj * xj)
                                    s_k = np.sin(2 * np.pi * fk * xk)
                                    test_basis_list.append(s_i * s_j * s_k)

        Phi_test = np.column_stack(test_basis_list)
        
        # Ensure shapes match (M should be same)
        if Phi_test.shape[1] != len(best_coeff):
            # Fallback if something went wrong with dimensionality
            return np.full(X_test.shape[0], np.mean(y))
            
        preds = Phi_test @ best_coeff
        # Ensure finite output
        return np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

    return predict
