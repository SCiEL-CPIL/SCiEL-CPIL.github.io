"""Rewritten program for unresolved-tendency closure task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import comb


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing.
    Uses Chebyshev-basis fitting with weighted regularization and k-fold CV.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter out any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Determine the range of x for Chebyshev mapping
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        x_max = x_min + 1.0

    # Map x to [-1, 1] for Chebyshev basis
    x_cheb = 2.0 * (x - x_min) / (x_max - x_min) - 1.0

    def cheb_to_monomial(cheb_coeffs, x_min, x_max):
        """Convert Chebyshev coefficients to monomial coefficients."""
        from numpy.polynomial.chebyshev import Chebyshev
        from numpy.polynomial.polynomial import Polynomial

        cheb_poly = Chebyshev(cheb_coeffs, domain=[-1, 1], window=[-1, 1])
        poly_t = cheb_poly.convert(kind=Polynomial)

        a = 2.0 / (x_max - x_min)
        b = -1.0 - 2.0 * x_min / (x_max - x_min)

        monomial_coef = np.zeros(6)
        for k in range(6):
            ck = poly_t.coef[k]
            if abs(ck) < 1e-15:
                continue
            for j in range(k + 1):
                monomial_coef[j] += ck * comb(k, j) * (a ** j) * (b ** (k - j))

        return monomial_coef

    def fit_with_reg(x_cheb_data, y_data, lam, weight_vec):
        """Fit Chebyshev coefficients with weighted regularization."""
        V = np.polynomial.chebyshev.chebvander(x_cheb_data, 5)
        VT_W_V = (V * weight_vec[None, :]).T @ V
        VT_y = V.T @ y_data
        A = VT_W_V + lam * np.diag(weight_vec)
        try:
            coeff_cheb = np.linalg.solve(A, VT_y)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(coeff_cheb)):
            return None
        return coeff_cheb

    def predict_monomial(monomial_coeff, x_data):
        """Evaluate polynomial sum c_j * x^j at x_data."""
        result = np.zeros_like(x_data)
        for j in range(6):
            result += monomial_coeff[j] * (x_data ** j)
        return result

    # Weight vector: penalize higher-order terms more strongly
    weight_vec = np.array([1.0, 1.0, 5.0, 20.0, 80.0, 300.0])

    # K-fold CV for lambda selection
    use_cv = n > 50 and (time.time() - start_time) < (DEADLINE - 30.0)

    if use_cv:
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        n_folds = min(5, max(2, n // 20))
        fold_size = n // n_folds

        lambdas = np.logspace(-10, 2, 20)

        best_coeff = None
        best_val_mse = np.inf

        for lam in lambdas:
            if time.time() > DEADLINE:
                break

            fold_mses = []
            valid_fold = True
            for fold in range(n_folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size
                if fold == n_folds - 1:
                    val_end = n

                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                x_cheb_train = x_cheb[train_idx]
                y_train = y[train_idx]
                x_val = x[val_idx]
                y_val = y[val_idx]

                coeff_cheb = fit_with_reg(x_cheb_train, y_train, lam, weight_vec)
                if coeff_cheb is None:
                    valid_fold = False
                    break

                monomial_coeff = cheb_to_monomial(coeff_cheb, x_min, x_max)
                if not np.all(np.isfinite(monomial_coeff)):
                    valid_fold = False
                    break

                pred = predict_monomial(monomial_coeff, x_val)
                mse = np.mean((pred - y_val) ** 2)
                fold_mses.append(mse)

            if valid_fold and len(fold_mses) == n_folds:
                mean_mse = np.mean(fold_mses)
                if mean_mse < best_val_mse:
                    best_val_mse = mean_mse
                    coeff_cheb = fit_with_reg(x_cheb, y, lam, weight_vec)
                    if coeff_cheb is not None:
                        mc = cheb_to_monomial(coeff_cheb, x_min, x_max)
                        if np.all(np.isfinite(mc)):
                            best_coeff = mc

        if best_coeff is not None:
            return best_coeff

    # Fallback: fit on all data with a moderate lambda
    for lam in [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]:
        if time.time() > DEADLINE:
            break
        coeff_cheb = fit_with_reg(x_cheb, y, lam, weight_vec)
        if coeff_cheb is not None:
            monomial_coeff = cheb_to_monomial(coeff_cheb, x_min, x_max)
            if np.all(np.isfinite(monomial_coeff)):
                return monomial_coeff

    # Final fallback: simple least squares in Chebyshev basis
    try:
        V = np.polynomial.chebyshev.chebvander(x_cheb, 5)
        coeff_cheb, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
        monomial_coeff = cheb_to_monomial(coeff_cheb, x_min, x_max)
        if np.all(np.isfinite(monomial_coeff)):
            return monomial_coeff
    except Exception:
        pass

    return np.zeros(6)
# EVOLVE-BLOCK-END
