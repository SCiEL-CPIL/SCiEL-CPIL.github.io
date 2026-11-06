"""Strategy: Regularized polynomial fit with degree selection and standardization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Standardize x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-12:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    # Try different degrees and regularization strengths
    best_coeff = np.zeros(6)
    best_score = np.inf

    # Internal validation split
    # Use a simple deterministic split
    n_val = max(1, n // 10)
    n_train = n - n_val

    # Shuffle deterministically based on index
    indices = np.arange(n)
    # Simple deterministic shuffle
    np.random.seed(42)
    np.random.shuffle(indices)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    x_train = x_norm[train_idx]
    y_train = y[train_idx]
    x_val = x_norm[val_idx]
    y_val = y[val_idx]

    # Try degrees 1 through 5
    for degree in range(1, 6):
        if time.time() > deadline - 20:
            break

        # Build Vandermonde matrix for training
        X_train = np.vander(x_train, N=degree + 1, increasing=True)
        X_val = np.vander(x_val, N=degree + 1, increasing=True)

        # Try several regularization strengths
        lambdas = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]

        for lam in lambdas:
            if time.time() > deadline - 10:
                break

            try:
                if lam == 0.0:
                    # OLS
                    coeff_train, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
                else:
                    # Ridge
                    A = X_train.T @ X_train + lam * np.eye(degree + 1)
                    b = X_train.T @ y_train
                    # Ensure diagonal dominance for stability
                    coeff_train = np.linalg.solve(A, b)

                # Evaluate on validation
                pred_val = X_val @ coeff_train
                mse_val = np.mean((pred_val - y_val) ** 2)

                if mse_val < best_score:
                    best_score = mse_val
                    best_coeff = coeff_train.copy()

            except np.linalg.LinAlgError:
                continue

    # If we found a good coefficient set, expand to degree 5
    final_coeff = np.zeros(6)
    max_degree = min(len(best_coeff), 6)
    final_coeff[:max_degree] = best_coeff[:max_degree]

    # Now transform back from standardized x to original x
    # We have: y = sum_{i=0}^{d} c_i * ((x - x_mean)/x_std)^i
    # We need: y = sum_{j=0}^{5} a_j * x^j
    # Use polynomial expansion
    if max_degree > 0:
        # Build the transformation using polynomial arithmetic
        # (x - mu)/sigma = (1/sigma)*x - mu/sigma
        # Let t = (x - mu)/sigma, so x = sigma*t + mu
        # We want to express sum c_i * t^i in terms of x
        # t = (x - mu)/sigma
        # t^i = ((x - mu)/sigma)^i

        # Use numpy polynomial operations
        # t as polynomial in x: t = (1/sigma)*x - mu/sigma
        t_poly = np.array([-x_mean / x_std, 1.0 / x_std])  # constant, linear coefficient in x

        # Compute sum c_i * t^i
        result_poly = np.zeros(max_degree + 1)  # coefficients in x, ascending
        for i in range(max_degree):
            if i == 0:
                t_pow = np.array([1.0])
            elif i == 1:
                t_pow = t_poly.copy()
            else:
                # Multiply previous t^(i-1) by t_poly
                t_pow = np.convolve(t_pow, t_poly)

            # Scale by c_i and add to result
            result_poly[:len(t_pow)] += best_coeff[i] * t_pow

        # Truncate to degree 5
        if len(result_poly) > 6:
            result_poly = result_poly[:6]
        else:
            final_coeff = np.zeros(6)
            final_coeff[:len(result_poly)] = result_poly

    # Check if all coefficients are finite
    if not np.all(np.isfinite(final_coeff)):
        final_coeff = np.zeros(6)

    return final_coeff
# EVOLVE-BLOCK-END
