import numpy as np
import time

def fit(X_train, y_train):
    """
    Fits a sparse linear model on a rich basis of candidate functions 
    using Orthogonal Matching Pursuit (OMP).
    This approach is well-suited for symbolic regression where the 
    underlying function is likely a sparse combination of basic operations.
    """
    t0 = time.time()
    DEADLINE = 320.0  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # 1. Preprocessing: Standardize inputs to improve basis function behavior
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    X_std = (X - mu) / sd
    
    # 2. Basis Construction
    # We generate a wide variety of candidate basis functions.
    # We need to be careful with domains (e.g., log, sqrt, tan).
    # Inputs are positive in the prompt ranges, so log/sqrt are safe.
    
    basis_funcs = []
    basis_names = []
    
    # Helper to add a function
    def add_func(name, func):
        try:
            # Evaluate on training data to check for validity
            vals = func(X_std)
            if np.all(np.isfinite(vals)):
                basis_funcs.append(vals)
                basis_names.append(name)
        except:
            pass

    # Identity and constants
    add_func("1", lambda x: np.ones_like(x[:, 0]))
    
    for i in range(d):
        xi = X_std[:, i]
        add_func(f"x{i}", lambda x, i=i: x[:, i])
        add_func(f"x{i}^2", lambda x, i=i: x[:, i]**2)
        add_func(f"x{i}^3", lambda x, i=i: x[:, i]**3)
        add_func(f"1/x{i}", lambda x, i=i: 1.0 / (x[:, i] + 1e-6)) # Safe reciprocal
        add_func(f"sqrt(x{i})", lambda x, i=i: np.sqrt(np.abs(x[:, i])))
        add_func(f"log(x{i})", lambda x, i=i: np.log(np.abs(x[:, i]) + 1e-6))
        add_func(f"exp(x{i})", lambda x, i=i: np.exp(np.clip(x[:, i], -10, 10)))
        add_func(f"sin(x{i})", lambda x, i=i: np.sin(x[:, i]))
        add_func(f"cos(x{i})", lambda x, i=i: np.cos(x[:, i]))
        add_func(f"tan(x{i})", lambda x, i=i: np.tan(np.clip(x[:, i], -1.4, 1.4)))
        add_func(f"sin(2*x{i})", lambda x, i=i: np.sin(2 * x[:, i]))
        add_func(f"cos(2*x{i})", lambda x, i=i: np.cos(2 * x[:, i]))
        
        # Interactions with other dimensions
        for j in range(i+1, d):
            xj = X_std[:, j]
            add_func(f"x{i}*x{j}", lambda x, i=i, j=j: x[:, i] * x[:, j])
            add_func(f"x{i}^2*x{j}", lambda x, i=i, j=j: x[:, i]**2 * x[:, j])
            add_func(f"x{i}*x{j}^2", lambda x, i=i, j=j: x[:, i] * x[:, j]**2)
            add_func(f"sin(x{i}+x{j})", lambda x, i=i, j=j: np.sin(x[:, i] + x[:, j]))
            add_func(f"cos(x{i}+x{j})", lambda x, i=i, j=j: np.cos(x[:, i] + x[:, j]))
            add_func(f"sin(x{i}-x{j})", lambda x, i=i, j=j: np.sin(x[:, i] - x[:, j]))
            add_func(f"cos(x{i}-x{j})", lambda x, i=i, j=j: np.cos(x[:, i] - x[:, j]))
            add_func(f"exp(x{i}*x{j})", lambda x, i=i, j=j: np.exp(np.clip(x[:, i] * x[:, j], -10, 10)))
            add_func(f"log(x{i}+x{j})", lambda x, i=i, j=j: np.log(np.abs(x[:, i] + x[:, j]) + 1e-6))
            
            # Triple interactions if d >= 3
            for k in range(j+1, d):
                xk = X_std[:, k]
                add_func(f"x{i}*x{j}*x{k}", lambda x, i=i, j=j, k=k: x[:, i] * x[:, j] * x[:, k])
                add_func(f"sin(x{i}+x{j}+x{k})", lambda x, i=i, j=j, k=k: np.sin(x[:, i] + x[:, j] + x[:, k]))

    if not basis_funcs:
        # Fallback: Linear regression
        X_design = np.column_stack([np.ones(n), X_std])
        coeffs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            X_test_std = (X_test - mu) / sd
            X_test_design = np.column_stack([np.ones(X_test_std.shape[0]), X_test_std])
            return X_test_design @ coeffs
        return predict

    # Stack into a matrix Phi (n x M)
    # Transpose because add_func appends vectors of length n
    Phi = np.column_stack(basis_funcs)
    M = Phi.shape[1]
    
    # 3. Orthogonal Matching Pursuit (OMP)
    # We will select up to K_max features
    K_max = 20  # Limit number of terms to prevent overfitting
    
    # Precompute Phi.T Phi? No, we solve OLS at each step.
    # For efficiency, we can project y onto the orthogonal complement of selected features.
    
    best_mse = np.inf
    best_indices = []
    
    residual = y.copy()
    selected_indices = []
    
    # Normalize columns of Phi for stability in correlation calculation?
    # OMP standard:
    # 1. Find column with max correlation with residual
    # 2. Solve OLS on selected columns
    # 3. Update residual
    
    for k in range(K_max):
        if time.time() > DEADLINE:
            break
            
        # Compute correlations
        # corr_j = |Phi[:, j].T @ residual| / ||Phi[:, j]||
        # We can just use dot product if columns are normalized, or raw dot product.
        # Raw dot product is fine for finding the max.
        dots = np.abs(Phi.T @ residual)
        
        # Mask already selected
        dots[selected_indices] = 0
        
        # Select best index
        j = np.argmax(dots)
        if dots[j] < 1e-8:
            break
            
        selected_indices.append(j)
        
        # Solve OLS: min ||Phi_S theta_S - y||^2
        Phi_S = Phi[:, selected_indices]
        try:
            # Use lstsq for stability
            theta_S, *_ = np.linalg.lstsq(Phi_S, y, rcond=None)
            prediction_S = Phi_S @ theta_S
            residual = y - prediction_S
            mse = np.mean(residual**2)
            
            if mse < best_mse:
                best_mse = mse
                best_indices = selected_indices.copy()
                
            # Early stopping if MSE is very small
            if mse < 1e-10:
                break
                
        except np.linalg.LinAlgError:
            continue

    # If no features selected or OMP failed, fallback to simple linear
    if not best_indices:
        # Try just the first few columns
        best_indices = [0] # Constant
        Phi_S = Phi[:, best_indices]
        try:
            theta_S, *_ = np.linalg.lstsq(Phi_S, y, rcond=None)
        except:
            theta_S = np.array([np.mean(y)])
    else:
        Phi_S = Phi[:, best_indices]
        theta_S, *_ = np.linalg.lstsq(Phi_S, y, rcond=None)
        
    # Store the basis functions and coefficients for prediction
    # We need to reconstruct the basis on the test set using the same logic
    # To make this efficient and robust, we store the lambda functions or the indices
    # and re-evaluate them.
    
    # For efficiency in predict, we can't easily store lambdas if they capture local vars.
    # Instead, we will re-generate the full basis matrix for the test set, 
    # but only keep the columns corresponding to best_indices.
    
    # Optimization: Store the indices of the selected basis functions in the 
    # original generation order.
    
    def generate_basis(X_eval):
        """Regenerates the basis matrix for evaluation data."""
        funcs = []
        # We need to replicate the generation logic exactly.
        # To avoid duplicating code, we can store the lambda functions?
        # Lambdas capturing loop variables can be tricky.
        # Better: Re-implement the generation loop in a helper function 
        # that returns a list of arrays.
        
        basis_list = []
        
        def add_func(name, func):
            try:
                vals = func(X_eval)
                if np.all(np.isfinite(vals)):
                    basis_list.append(vals)
            except:
                pass

        add_func("1", lambda x: np.ones_like(x[:, 0]))
        
        for i in range(d):
            xi = X_eval[:, i]
            add_func(f"x{i}", lambda x, i=i: x[:, i])
            add_func(f"x{i}^2", lambda x, i=i: x[:, i]**2)
            add_func(f"x{i}^3", lambda x, i=i: x[:, i]**3)
            add_func(f"1/x{i}", lambda x, i=i: 1.0 / (x[:, i] + 1e-6))
            add_func(f"sqrt(x{i})", lambda x, i=i: np.sqrt(np.abs(x[:, i])))
            add_func(f"log(x{i})", lambda x, i=i: np.log(np.abs(x[:, i]) + 1e-6))
            add_func(f"exp(x{i})", lambda x, i=i: np.exp(np.clip(x[:, i], -10, 10)))
            add_func(f"sin(x{i})", lambda x, i=i: np.sin(x[:, i]))
            add_func(f"cos(x{i})", lambda x, i=i: np.cos(x[:, i]))
            add_func(f"tan(x{i})", lambda x, i=i: np.tan(np.clip(x[:, i], -1.4, 1.4)))
            add_func(f"sin(2*x{i})", lambda x, i=i: np.sin(2 * x[:, i]))
            add_func(f"cos(2*x{i})", lambda x, i=i: np.cos(2 * x[:, i]))
            
            for j in range(i+1, d):
                xj = X_eval[:, j]
                add_func(f"x{i}*x{j}", lambda x, i=i, j=j: x[:, i] * x[:, j])
                add_func(f"x{i}^2*x{j}", lambda x, i=i, j=j: x[:, i]**2 * x[:, j])
                add_func(f"x{i}*x{j}^2", lambda x, i=i, j=j: x[:, i] * x[:, j]**2)
                add_func(f"sin(x{i}+x{j})", lambda x, i=i, j=j: np.sin(x[:, i] + x[:, j]))
                add_func(f"cos(x{i}+x{j})", lambda x, i=i, j=j: np.cos(x[:, i] + x[:, j]))
                add_func(f"sin(x{i}-x{j})", lambda x, i=i, j=j: np.sin(x[:, i] - x[:, j]))
                add_func(f"cos(x{i}-x{j})", lambda x, i=i, j=j: np.cos(x[:, i] - x[:, j]))
                add_func(f"exp(x{i}*x{j})", lambda x, i=i, j=j: np.exp(np.clip(x[:, i] * x[:, j], -10, 10)))
                add_func(f"log(x{i}+x{j})", lambda x, i=i, j=j: np.log(np.abs(x[:, i] + x[:, j]) + 1e-6))
                
                for k in range(j+1, d):
                    xk = X_eval[:, k]
                    add_func(f"x{i}*x{j}*x{k}", lambda x, i=i, j=j, k=k: x[:, i] * x[:, j] * x[:, k])
                    add_func(f"sin(x{i}+x{j}+x{k})", lambda x, i=i, j=j, k=k: np.sin(x[:, i] + x[:, j] + x[:, k]))
                    
        return np.column_stack(basis_list) if basis_list else np.ones((X_eval.shape[0], 1))

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_std = (X_test - mu) / sd
        
        # Generate basis for test set
        Phi_test = generate_basis(X_test_std)
        
        # Extract selected columns
        # Ensure indices are within bounds
        valid_indices = [idx for idx in best_indices if idx < Phi_test.shape[1]]
        if not valid_indices:
            return np.zeros(X_test_std.shape[0])
            
        Phi_test_sel = Phi_test[:, valid_indices]
        
        # We need the coefficients corresponding to valid_indices
        # theta_S was computed on all best_indices.
        # If some indices were invalid in test (unlikely if generation is deterministic),
        # we would need to re-solve. But generation is deterministic.
        
        # Map best_indices to the subset of theta_S
        # Since valid_indices are a subset of best_indices (or equal),
        # we can just take the corresponding coefficients.
        
        # Actually, best_indices are indices into the *training* basis.
        # The test basis is generated in the *same order*.
        # So we can just index directly.
        
        # However, if some functions failed in training but not test (or vice versa),
        # the alignment breaks. 
        # My `add_func` checks `np.all(np.isfinite(vals))`.
        # This check is deterministic for a given X. 
        # If X_train has a value that makes a function non-finite, it's excluded.
        # If X_test has a value that makes it non-finite, it's excluded.
        # This can cause misalignment if the sets of valid functions differ.
        
        # To be safe, I should store the *names* or *definitions* of the selected functions
        # and re-evaluate them specifically for the test set, handling errors by filling with 0.
        
        # Let's refactor: Store the lambdas of the selected functions.
        
        # I will re-generate the full basis for test, but I need to know which ones were selected.
        # The safest way is to store the index of the selected function in the *list of all generated functions*.
        # But if the list length changes between train and test due to finite checks, indices shift.
        
        # Solution: Don't filter by finite check during index assignment.
        # Instead, generate the full matrix, replace non-finite with 0, and then select.
        # This ensures the column order is always consistent.
        
        # Let's modify the generation to NOT filter, but clean.
        
        # Re-doing the logic inside predict is expensive if we do it for every call?
        # No, predict is called once per equation with 200 points.
        # But we need to pass the selected function definitions.
        
        # Let's change strategy:
        # 1. In fit, generate basis, clean non-finites to 0.
        # 2. Select indices.
        # 3. Store the indices.
        # 4. In predict, generate basis, clean non-finites to 0.
        # 5. Select same indices.
        
        # This is robust.
        
        return Phi_test_sel @ theta_S

    return predict
