"""Combined: piecewise cubic spline proxy + Chebyshev-basis Ridge regression with k-fold CV."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved subgrid tendency.
    
    Args:
        training_x: flattened array of slow-state values
        training_subgrid_tendency: flattened array of corresponding subgrid tendencies
    
    Returns:
        6 finite real coefficients in ascending powers of x
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n == 0:
        return np.zeros(6)

    # Remove any NaN/Inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n == 0:
        return np.zeros(6)

    if n < 6:
        if n >= 2:
            try:
                coeffs = np.polyfit(x, y, 1)
                out = np.zeros(6)
                out[0] = coeffs[1]
                out[1] = coeffs[0]
                return out
            except:
                return np.zeros(6)
        return np.zeros(6)

    # Map x to [-1, 1] for Chebyshev basis
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        return np.zeros(6)

    a = 2.0 / (x_max - x_min)
    b = -(x_min + x_max) / (x_max - x_min)
    t = a * x + b

    # Build Chebyshev basis matrix (degree 5)
    def chebyshev_basis(t, degree=5):
        """Build Chebyshev basis matrix T_0(t), T_1(t), ..., T_degree(t)."""
        n = len(t)
        C = np.zeros((n, degree + 1))
        C[:, 0] = 1.0
        if degree >= 1:
            C[:, 1] = t
        for k in range(2, degree + 1):
            C[:, k] = 2.0 * t * C[:, k-1] - C[:, k-2]
        return C

    C = chebyshev_basis(t, 5)

    # K-fold cross-validation for lambda selection
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)

    if n >= 50:
        n_folds = 5
    elif n >= 20:
        n_folds = 4
    elif n >= 10:
        n_folds = 3
    else:
        n_folds = 2

    fold_size = n // n_folds
    folds = []
    for f in range(n_folds):
        start = f * fold_size
        end = start + fold_size if f < n_folds - 1 else n
        folds.append(indices[start:end])

    # Pre-compute Gram matrices and target vectors for each fold
    gram_matrices = []
    target_vectors = []
    val_C = []
    val_y = []

    for f in range(n_folds):
        val_idx = folds[f]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])

        C_train = C[train_idx]
        y_train = y[train_idx]
        C_val = C[val_idx]
        y_val = y[val_idx]

        G_f = C_train.T @ C_train
        h_f = C_train.T @ y_train

        gram_matrices.append(G_f)
        target_vectors.append(h_f)
        val_C.append(C_val)
        val_y.append(y_val)

    # Regularization: penalize higher-order Chebyshev terms more heavily
    # This encourages smoothness and prevents extrapolation blow-up
    reg_weights = np.array([1.0, 1.0, 5.0, 25.0, 125.0, 625.0])
    reg_matrix = np.diag(reg_weights)

    # Log-spaced regularization values
    lambdas = list(np.logspace(-8, 4, 40))

    best_mse = np.inf
    best_lambda = 0.0

    for lam in lambdas:
        if time.time() - start_time > deadline - 5:
            break
        try:
            mse_sum = 0.0
            valid_fold = True
            for f in range(n_folds):
                A = gram_matrices[f] + lam * reg_matrix
                try:
                    coeffs_lam = np.linalg.solve(A, target_vectors[f])
                except np.linalg.LinAlgError:
                    coeffs_lam, _, _, _ = np.linalg.lstsq(A, target_vectors[f], rcond=None)

                if not np.all(np.isfinite(coeffs_lam)):
                    valid_fold = False
                    break

                y_val_pred = val_C[f] @ coeffs_lam
                mse_sum += np.mean((y_val_pred - val_y[f]) ** 2)

            if not valid_fold:
                continue

            avg_mse = mse_sum / n_folds
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_lambda = lam
        except:
            continue

    # Final fit on all data using best_lambda
    CtC = C.T @ C
    Cty = C.T @ y
    A = CtC + best_lambda * reg_matrix

    try:
        coeffs_cheb = np.linalg.solve(A, Cty)
    except np.linalg.LinAlgError:
        coeffs_cheb, _, _, _ = np.linalg.lstsq(A, Cty, rcond=None)

    if not np.all(np.isfinite(coeffs_cheb)):
        # Fallback to OLS
        try:
            coeffs_cheb, _, _, _ = np.linalg.lstsq(C, y, rcond=None)
            if not np.all(np.isfinite(coeffs_cheb)):
                return np.zeros(6)
        except:
            return np.zeros(6)

    # Convert Chebyshev coefficients to monomial coefficients
    # P(x) = sum_{k=0}^5 c_k * T_k(a*x + b)
    # We need to express T_k(a*x + b) as sum_{j=0}^k M[k,j] * x^j
    # Then coeff_orig[j] = sum_{k=j}^5 c_k * M[k,j]

    # Build conversion matrix M where M[k, j] = coeff of x^j in T_k(a*x + b)
    M = np.zeros((6, 6))
    M[0, 0] = 1.0
    if 1 < 6:
        M[1, 1] = a
        M[1, 0] = b

    for k in range(2, 6):
        # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
        # 2*t*T_{k-1}(t) = 2*(a*x+b) * sum_j M[k-1,j]*x^j
        #                  = sum_j [2*a*M[k-1,j]*x^{j+1} + 2*b*M[k-1,j]*x^j]
        for j in range(k):
            M[k, j] += 2.0 * b * M[k-1, j]
        for j in range(1, k+1):
            M[k, j] += 2.0 * a * M[k-1, j-1]
        # Subtract T_{k-2}
        for j in range(k-1):
            M[k, j] -= M[k-2, j]

    # Now monomial_coeffs[j] = sum_k M[k, j] * coeffs_cheb[k]
    coeff_orig = np.zeros(6)
    for j in range(6):
        for k in range(j, 6):
            coeff_orig[j] += M[k, j] * coeffs_cheb[k]

    # Ensure all coefficients are finite
    if not np.all(np.isfinite(coeff_orig)):
        try:
            c_lin = np.polyfit(x, y, 1)
            coeff_orig = np.zeros(6)
            coeff_orig[0] = c_lin[1]
            coeff_orig[1] = c_lin[0]
        except:
            coeff_orig = np.zeros(6)

    if not np.all(np.isfinite(coeff_orig)):
        return np.zeros(6)

    return coeff_orig
# EVOLVE-BLOCK-END
