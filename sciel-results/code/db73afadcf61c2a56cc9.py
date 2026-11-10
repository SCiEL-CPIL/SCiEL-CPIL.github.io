"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_basis(x):
    """Build physics-informed basis: x, x^3, x*mean(x), x^5, x^2*mean(x)
    These capture advection (cubic), mean-field coupling, and higher-order terms."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    x2 = x ** 2
    x3 = x ** 3
    x5 = x ** 5
    mean_x = np.full(n, x.mean())
    
    # Basis functions:
    # b0: x (linear advection)
    # b1: x^3 (cubic advection/nonlinearity)
    # b2: x * mean(x) (mean-field coupling)
    # b3: x^5 (higher-order nonlinearity)
    # b4: x^2 * mean(x) (quadratic-mean coupling)
    # b5: x^2 (quadratic term for flexibility)
    
    basis = np.column_stack([
        x,
        x3,
        x * mean_x,
        x5,
        x2 * mean_x,
        x2
    ])
    return basis

def _l1_solve(basis, y, lam):
    """Simple L1-regularized least squares via coordinate descent."""
    n, p = basis.shape
    beta = np.zeros(p)
    
    # Precompute norms
    norms = np.sum(basis ** 2, axis=0)
    norms[norms < 1e-10] = 1e-10
    
    # Coordinate descent
    for iteration in range(200):
        for j in range(p):
            # Compute residual without current coefficient
            residual = y - basis @ beta
            residual[j] += basis[:, j] * beta[j]
            
            # Soft thresholding
            rho = np.dot(basis[:, j], residual) / norms[j]
            if rho > lam:
                beta[j] = (rho - lam) / (1.0)
            elif rho < -lam:
                beta[j] = (rho + lam) / (1.0)
            else:
                beta[j] = 0.0
    
    return beta

def _sparse_to_poly(coeff_sparse, x):
    """Convert sparse basis coefficients to standard polynomial coefficients.
    
    The basis is: [x, x^3, x*mean_x, x^5, x^2*mean_x, x^2]
    Since mean_x is a constant for a given dataset, we need to fit
    a standard polynomial that best approximates the sparse solution.
    
    We'll do a final least-squares fit of the sparse prediction to a degree-5 polynomial.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    basis = _build_basis(x)
    y_pred = basis @ coeff_sparse
    
    # Fit degree-5 polynomial to the sparse prediction
    try:
        poly_coeff = np.polynomial.polynomial.polyfit(x, y_pred, 5)
    except:
        # Fallback: just use linear terms
        poly_coeff = np.zeros(6)
        if len(x) > 1:
            A = np.column_stack([np.ones_like(x), x])
            try:
                sol, _, _, _ = np.linalg.lstsq(A, y_pred, rcond=None)
                poly_coeff[:2] = sol
            except:
                pass
    
    return poly_coeff

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    training_x = np.asarray(training_x, dtype=np.float64).ravel()
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(training_x)
    
    # Normalize for numerical stability
    x_mean = training_x.mean()
    x_std = training_x.std()
    if x_std < 1e-10:
        x_std = 1.0
    
    x_norm = (training_x - x_mean) / x_std
    y = training_subgrid_tendency
    
    # Build normalized basis
    basis_norm = _build_basis(x_norm)
    
    # Try multiple lambda values with L1 regularization
    lambdas = np.logspace(-2, 2, 20)
    best_score = np.inf
    best_coeff = np.zeros(6)
    
    # Use cross-validation to select lambda
    n_cv = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_cv)
    
    for lam in lambdas:
        if time.time() > deadline - 5.0:
            break
        
        cv_errors = []
        for i in range(n_cv):
            val_idx = folds[i]
            train_idx = np.setdiff1d(indices, val_idx)
            
            try:
                beta = _l1_solve(basis_norm[train_idx], y[train_idx], lam)
                pred_val = basis_norm[val_idx] @ beta
                mse = np.mean((pred_val - y[val_idx]) ** 2)
                cv_errors.append(mse)
            except:
                cv_errors.append(np.inf)
        
        mean_cv = np.mean(cv_errors)
        if mean_cv < best_score:
            best_score = mean_cv
            # Fit on all data with best lambda
            try:
                beta_best = _l1_solve(basis_norm, y, lam)
                # Convert to polynomial
                best_coeff = _sparse_to_poly(beta_best, x_norm)
            except:
                pass
    
    # If L1 didn't work well, fall back to ridge regression
    if best_score == np.inf or np.any(~np.isfinite(best_coeff)):
        if time.time() > deadline - 3.0:
            # Return simple linear fit
            try:
                A = np.column_stack([np.ones_like(x_norm), x_norm])
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                best_coeff = np.zeros(6)
                best_coeff[:2] = sol
            except:
                best_coeff = np.zeros(6)
        else:
            # Try ridge regression
            try:
                A = np.polynomial.polynomial.polyvander(x_norm, 5).T
                lam_ridge = 1e-3
                A_reg = A + lam_ridge * np.eye(6)
                sol = np.linalg.solve(A_reg.T @ A_reg, A_reg.T @ y)
                best_coeff = sol
            except:
                try:
                    A = np.column_stack([np.ones_like(x_norm), x_norm])
                    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                    best_coeff = np.zeros(6)
                    best_coeff[:2] = sol
                except:
                    best_coeff = np.zeros(6)
    
    # Convert from normalized to original x basis
    # If we fit on x_norm = (x - x_mean) / x_std, we need to convert
    # P(x_norm) = P((x - x_mean)/x_std)
    # We'll do a final refit on original x to get proper coefficients
    
    if time.time() < deadline - 2.0:
        # Refit on original scale using the sparse model as guide
        try:
            basis_orig = _build_basis(training_x)
            y_pred_norm = basis_norm @ _l1_solve(basis_norm, y, 0.1)
            
            # Fit degree-5 polynomial on original x to match predictions
            A_orig = np.polynomial.polynomial.polyvander(training_x, 5).T
            sol, _, _, _ = np.linalg.lstsq(A_orig, y_pred_norm, rcond=None)
            final_coeff = sol
        except:
            final_coeff = best_coeff
    else:
        final_coeff = best_coeff
    
    # Ensure finite coefficients
    final_coeff = np.where(np.isfinite(final_coeff), final_coeff, 0.0)
    
    return final_coeff
# EVOLVE-BLOCK-END
