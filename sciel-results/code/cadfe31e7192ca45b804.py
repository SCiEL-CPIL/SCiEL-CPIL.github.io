import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return np.zeros(6)

    n = x.size

    if n < 10:
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeff = np.zeros(6)
            coeff[:2] = c_lin
            if not np.all(np.isfinite(coeff)):
                return np.zeros(6)
            return coeff
        except Exception:
            return np.zeros(6)

    # Scale x for numerical stability
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-10:
        sigma = 1.0
    u = (x - mu) / sigma

    # Use odd-power polynomial: the subgrid forcing in Lorenz-96 has antisymmetric structure
    # Fit: y ≈ a1*u + a3*u^3 + a5*u^5
    # This exploits the physics: the mean of Y over the fast ring is an odd function of X
    # due to the periodic boundary conditions and the structure of the fast equations.

    # Build design matrix with odd powers only
    A = np.column_stack([u, u**3, u**5])

    # K-fold cross-validation to select regularization
    K = 5
    if n > 500:
        K = 10

    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    fold_sizes = np.full(K, n // K, dtype=int)
    fold_sizes[:n % K] += 1
    folds = np.array_split(indices, K)

    # Regularization strengths for Tikhonov
    alphas = np.logspace(-8, 6, 15)

    best_cv_err = np.inf
    best_alpha = 1.0

    for alpha in alphas:
        if time.time() > deadline - 10.0:
            break

        cv_errors = []

        for i in range(K):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(K) if j != i])

            A_train = A[train_idx]
            A_test = A[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            try:
                ATA = A_train.T @ A_train
                ATy = A_train.T @ y_train

                reg = alpha * np.eye(3)
                coeff = np.linalg.solve(ATA + reg, ATy)

                pred_test = A_test @ coeff
                err = np.mean((pred_test - y_test)**2)
                cv_errors.append(err)

            except np.linalg.LinAlgError:
                continue

        if cv_errors:
            mean_err = np.mean(cv_errors)
            if mean_err < best_cv_err:
                best_cv_err = mean_err
                best_alpha = alpha

    # Final fit on all training data
    A_all = A
    ATA = A_all.T @ A_all
    ATy = A_all.T @ y

    reg = best_alpha * np.eye(3)
    try:
        coeff_u = np.linalg.solve(ATA + reg, ATy)
    except np.linalg.LinAlgError:
        coeff_u = np.linalg.lstsq(A_all, y, rcond=None)[0]

    # coeff_u = [a1, a3, a5] for y = a1*u + a3*u^3 + a5*u^5
    # Convert to polynomial in x: y = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*x^5
    # u = (x - mu) / sigma
    # u^i = (1/sigma^i) * (x - mu)^i
    # (x - mu)^i = sum_{k=0}^i binom(i, k) * x^k * (-mu)^(i-k)

    c_x = np.zeros(6)

    for power, a_i in zip([1, 3, 5], coeff_u):
        scale_i = a_i / (sigma ** power)

        # Expand (x - mu)^power
        for k in range(power + 1):
            import math
            binom_val = math.comb(power, k)
            term_coeff = binom_val * ((-mu) ** (power - k))
            c_x[k] += scale_i * term_coeff

    # Ensure finite
    if not np.all(np.isfinite(c_x)):
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            c_x = np.zeros(6)
            c_x[:2] = c_lin
        except Exception:
            c_x = np.zeros(6)

    return c_x
