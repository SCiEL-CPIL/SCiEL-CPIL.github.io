"""Tikhonov-regularized odd-degree polynomial fit exploiting antisymmetry."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def construct(training_x, training_subgrid_tendency):
    """Fit a shared odd-degree (1,3,5) polynomial to the unresolved tendency.
    
    Exploits the antisymmetry of the unresolved forcing with respect to the slow state X.
    Uses Tikhonov (L2) regularization with lambda selected by internal k-fold cross-validation.
    """
    t0 = time.time()
    DEADLINE = 120.0
    SAFETY_MARGIN = 5.0

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = len(x)

    if n < 3:
        return np.zeros(6)

    # Scale x to have max absolute value 1 for numerical stability
    x_scale = np.max(np.abs(x))
    if x_scale < 1e-12:
        x_scale = 1.0
    z = x / x_scale

    # Odd power basis: z, z^3, z^5
    V = np.column_stack((z, z**3, z**5))

    # Cross-validation setup
    n_folds = 10
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    # Precompute V^T V and V^T y for each training fold
    fold_data = []
    for f in range(n_folds):
        val_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        
        V_train = V[train_idx]
        y_train = y[train_idx]
        V_val = V[val_idx]
        y_val = y[val_idx]
        
        # Precompute Gram matrix and cross-product
        VtV = V_train.T @ V_train  # 3x3
        Vty = V_train.T @ y_train  # 3x1
        
        fold_data.append((VtV, Vty, V_val, y_val))

    def solve_ridge(VtV, Vty, lam):
        """Solve ridge regression in odd-power space."""
        A = VtV + lam * np.eye(3)
        try:
            return np.linalg.solve(A, Vty)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, Vty, rcond=None)[0]

    def compute_cv_score(lam):
        """Compute mean CV MSE for a given lambda."""
        cv_errors = []
        for VtV, Vty, V_val, y_val in fold_data:
            c_train = solve_ridge(VtV, Vty, lam)
            pred_val = V_val @ c_train
            cv_errors.append(np.mean((pred_val - y_val) ** 2))
        if cv_errors:
            return np.mean(cv_errors)
        return np.inf

    # Coarse lambda search
    lambdas_coarse = np.concatenate([
        [0.0],
        np.logspace(-12, 8, 60)
    ])

    best_lambda = 0.0
    best_cv_score = np.inf

    for lam in lambdas_coarse:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        cv_score = compute_cv_score(lam)
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lambda = lam

    # Fine lambda search around best_lambda
    if best_lambda > 0:
        lower = max(best_lambda * 0.3, 1e-15)
        upper = best_lambda * 3.0
        lambdas_fine = np.logspace(np.log10(lower), np.log10(upper), 30)
    else:
        lambdas_fine = np.logspace(-12, -1, 30)

    for lam in lambdas_fine:
        if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
            break
        cv_score = compute_cv_score(lam)
        if cv_score < best_cv_score:
            best_cv_score = cv_score
            best_lambda = lam

    # Final fit on all data
    VtV_full = V.T @ V
    Vty_full = V.T @ y

    if best_lambda < 1e-10:
        A = VtV_full + 1e-14 * np.eye(3)
    else:
        A = VtV_full + best_lambda * np.eye(3)

    try:
        coeff_z = np.linalg.solve(A, Vty_full)
    except np.linalg.LinAlgError:
        coeff_z = np.linalg.lstsq(A, Vty_full, rcond=None)[0]

    # coeff_z = [a1, a3, a5] for y = a1*z + a3*z^3 + a5*z^5
    # z = x / x_scale
    # y = a1*(x/x_scale) + a3*(x/x_scale)^3 + a5*(x/x_scale)^5
    # y = (a1/x_scale)*x + (a3/x_scale^3)*x^3 + (a5/x_scale^5)*x^5
    
    c = np.zeros(6)
    c[1] = coeff_z[0] / x_scale
    c[3] = coeff_z[1] / (x_scale ** 3)
    c[5] = coeff_z[2] / (x_scale ** 5)

    if not np.all(np.isfinite(c)):
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

    return c
# EVOLVE-BLOCK-END
