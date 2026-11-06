"""Improved program for the suite task; uses k-fold CV to select ridge regularization for odd-power polynomial closure."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Ensure inputs are float64 numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle edge case of empty or tiny training sets
    if len(x) < 6:
        return np.zeros(6)
        
    # Check time
    if time.time() - start_time > deadline:
        return np.zeros(6)
        
    try:
        # Strategy: Fit an odd-power polynomial (exploiting antisymmetry) with ridge regularization.
        # The regularization strength lambda is selected via internal k-fold cross-validation
        # to balance bias and variance, improving generalization to the unseen F=10 regime.
        
        # Design matrix for odd powers: [x, x^3, x^5]
        A = np.column_stack((x, x**3, x**5))
        
        # If the matrix is ill-conditioned or has issues, fallback to simple least squares
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(y)):
            return np.zeros(6)
            
        # Check time
        if time.time() - start_time > deadline:
            # Fallback: simple least squares on odd powers
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            c = np.zeros(6)
            c[1] = coeffs[0]
            c[3] = coeffs[1]
            c[5] = coeffs[2]
            return c
        
        # Perform SVD of A for efficient ridge regression
        # A = U S Vt
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        
        # Precompute U^T y
        Ut_y = U.T @ y
        
        # Define a range of lambda values to test via cross-validation
        # Lambda scales with the largest singular value
        S_max = S[0] if len(S) > 0 else 1.0
        # Log-spaced lambdas from very small (LS) to large (strong regularization)
        lambdas = S_max * 10**np.linspace(-6, 2, 25)
        
        # K-fold cross-validation to select best lambda
        n = len(x)
        k = 5
        if n < k * 10:
            k = max(2, n // 10)
        
        # Initialize fold indices
        indices = np.arange(n)
        # Shuffle for better CV (use a fixed seed for determinism if needed, but random is fine here)
        np.random.seed(42)
        indices = np.random.permutation(indices)
        fold_sizes = np.full(k, n // k, dtype=int)
        fold_sizes[:n % k] += 1
        folds = [indices[i*fold_sizes[i]:(i+1)*fold_sizes[i]] for i in range(k)]
        
        best_lambda = lambdas[0]
        best_cv_error = np.inf
        
        for lam in lambdas:
            if time.time() - start_time > deadline:
                break
                
            total_error = 0.0
            for i in range(k):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                
                A_train = A[train_idx]
                y_train = y[train_idx]
                A_val = A[val_idx]
                y_val = y[val_idx]
                
                # Ridge solution: c = V (S / (S^2 + lambda)) U^T y_train
                # Note: We use the SVD of the full A, but for ridge on a subset, 
                # strictly we should recompute SVD or use the normal equations.
                # However, re-computing SVD for each fold/lambda is expensive.
                # Alternative: Use the formula c = (A^T A + lambda I)^-1 A^T y
                # Let's precompute A^T A and A^T y for the full set? No, that's for the full fit.
                # For CV, we need the fit on the training subset.
                
                # Efficient ridge via normal equations for small d=3
                # c = inv(A_train.T @ A_train + lam * I) @ (A_train.T @ y_train)
                AtA = A_train.T @ A_train
                Aty = A_train.T @ y_train
                I = np.eye(3)
                
                try:
                    c_val = np.linalg.solve(AtA + lam * I, Aty)
                except np.linalg.LinAlgError:
                    c_val = np.linalg.lstsq(AtA + lam * I, Aty, rcond=None)[0]
                
                y_pred = A_val @ c_val
                error = np.sum((y_val - y_pred)**2)
                total_error += error
                
            mean_error = total_error / k
            if mean_error < best_cv_error:
                best_cv_error = mean_error
                best_lambda = lam
        
        # Final fit with the best lambda using the full training set
        # c = (A^T A + best_lambda * I)^-1 A^T y
        AtA_full = A.T @ A
        Aty_full = A.T @ y
        I = np.eye(3)
        
        try:
            coeffs = np.linalg.solve(AtA_full + best_lambda * I, Aty_full)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(AtA_full + best_lambda * I, Aty_full, rcond=None)[0]
        
        # Map to ascending power coefficients
        c = np.zeros(6)
        c[1] = coeffs[0]
        c[3] = coeffs[1]
        c[5] = coeffs[2]
        
        # Ensure coefficients are finite
        if not np.all(np.isfinite(c)):
            # Fallback to odd-power least squares
            coeffs_ls, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            c = np.zeros(6)
            c[1] = coeffs_ls[0]
            c[3] = coeffs_ls[1]
            c[5] = coeffs_ls[2]
            
        return c
        
    except Exception:
        # Fallback to odd-power least squares in case of any errors
        try:
            A = np.column_stack((x, x**3, x**5))
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            c = np.zeros(6)
            c[1] = coeffs[0]
            c[3] = coeffs[1]
            c[5] = coeffs[2]
            return c
        except Exception:
            return np.zeros(6)
# EVOLVE-BLOCK-END
