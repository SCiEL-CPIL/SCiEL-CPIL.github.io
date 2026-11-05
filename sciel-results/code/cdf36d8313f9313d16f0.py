"""Rewritten program: Chebyshev-basis degree-5 fit with conditioning safeguards."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 150.0
_MARGIN = 5.0  # seconds safety margin before evaluator kills


def _chebyshev_to_monomial(cheb_coeffs, n):
    """
    Convert Chebyshev coefficients (ascending order in T_k) to monomial
    coefficients (ascending powers of x) for a polynomial of degree n.
    Uses the standard three-term recurrence:
        T_0 = 1, T_1 = x, T_k = 2*x*T_{k-1} - T_{k-2}
    """
    # monomial coefficients for T_0 and T_1
    T0 = np.zeros(n + 1)
    T0[0] = 1.0
    T1 = np.zeros(n + 1)
    T1[1] = 1.0

    result = cheb_coeffs[0] * T0
    if n >= 1:
        result = result + cheb_coeffs[1] * T1

    for k in range(2, n + 1):
        T2 = 2.0 * np.roll(T1, 1) - T0
        T2[0] = 0.0  # T2 has no constant term from 2*x*T1 - T0; actually let's be careful
        # Actually: T2 = 2*x*T1 - T0
        # 2*x*T1 shifts T1 up by 1: [0, 2, 0, ...]
        # T0 = [1, 0, 0, ...]
        # So T2 = [-1, 2, 0, ...]
        # Let me redo properly:
        T2 = np.zeros(n + 1)
        for i in range(n + 1):
            # 2*x*T1[i-1] - T0[i]
            val = 0.0
            if i - 1 >= 0:
                val += 2.0 * T1[i - 1]
            val -= T0[i]
            T2[i] = val
        result = result + cheb_coeffs[k] * T2
        T0, T1 = T1, T2

    return result[:n + 1]


def _chebyshev_to_monomial_fast(cheb_coeffs, n):
    """
    Fast conversion using numpy operations.
    """
    # Build the Chebyshev-to-monomial transformation matrix
    # T_k(x) expressed in monomial basis
    T = np.zeros((n + 1, n + 1))  # T[k, i] = coefficient of x^i in T_k
    T[0, 0] = 1.0
    if n >= 1:
        T[1, 1] = 1.0
    for k in range(2, n + 1):
        # T_k = 2*x*T_{k-1} - T_{k-2}
        T[k, 1:] = 2.0 * T[k - 1, :-1]
        T[k, :] -= T[k - 2, :]

    # cheb_coeffs is in ascending order: c[0]*T_0 + c[1]*T_1 + ...
    # monomial_coeffs[i] = sum_k cheb_coeffs[k] * T[k, i]
    monomial_coeffs = T @ cheb_coeffs
    return monomial_coeffs


def _fit_chebyshev(x, y, degree):
    """
    Fit a degree-`degree` Chebyshev polynomial to (x, y) data.
    Returns Chebyshev coefficients in ascending order.
    """
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        # Degenerate case
        c = np.zeros(degree + 1)
        c[0] = np.mean(y)
        return c

    # Map x to [-1, 1]
    x_scaled = 2.0 * (x - x_min) / (x_max - x_min) - 1.0

    # Build Chebyshev Vandermonde matrix
    # V[i, k] = T_k(x_scaled[i])
    V = np.zeros((len(x), degree + 1))
    V[:, 0] = 1.0
    if degree >= 1:
        V[:, 1] = x_scaled
    for k in range(2, degree + 1):
        V[:, k] = 2.0 * x_scaled * V[:, k - 1] - V[:, k - 2]

    # Least squares solve
    try:
        # Use lstsq for numerical stability
        cheb_coeffs, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        cheb_coeffs = np.zeros(degree + 1)
        cheb_coeffs[0] = np.mean(y)

    return cheb_coeffs


def _weighted_chebyshev_fit(x, y, degree, weights=None):
    """
    Weighted Chebyshev fit. If weights is None, uniform weights.
    """
    if weights is None:
        weights = np.ones(len(x))

    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        c = np.zeros(degree + 1)
        c[0] = np.average(y, weights=weights)
        return c

    x_scaled = 2.0 * (x - x_min) / (x_max - x_min) - 1.0

    V = np.zeros((len(x), degree + 1))
    V[:, 0] = 1.0
    if degree >= 1:
        V[:, 1] = x_scaled
    for k in range(2, degree + 1):
        V[:, k] = 2.0 * x_scaled * V[:, k - 1] - V[:, k - 2]

    # Weighted least squares: (W^{1/2} V) c = W^{1/2} y
    w_sqrt = np.sqrt(weights)
    Vw = V * w_sqrt[:, np.newaxis]
    yw = y * w_sqrt

    try:
        cheb_coeffs, residuals, rank, sv = np.linalg.lstsq(Vw, yw, rcond=None)
    except np.linalg.LinAlgError:
        cheb_coeffs = np.zeros(degree + 1)
        cheb_coeffs[0] = np.average(y, weights=weights)

    return cheb_coeffs


def _evaluate_polynomial_monomial(coeffs, x):
    """Evaluate sum(c_i * x^i) for array x."""
    result = np.zeros_like(x, dtype=float)
    for i in range(len(coeffs) - 1, -1, -1):
        result = result * x + coeffs[i]
    return result


def _evaluate_polynomial_chebyshev(cheb_coeffs, x, x_min, x_max):
    """Evaluate Chebyshev polynomial at x given the scaling parameters."""
    if x_max - x_min < 1e-12:
        return np.full_like(x, cheb_coeffs[0], dtype=float)
    x_scaled = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    # Clenshaw's algorithm
    n = len(cheb_coeffs) - 1
    result = np.zeros_like(x_scaled, dtype=float)
    if n >= 1:
        result = cheb_coeffs[n]
    if n >= 2:
        temp = 2.0 * x_scaled * result - cheb_coeffs[n - 1]
        result = temp
    for k in range(n - 1, 0, -1):
        temp = 2.0 * x_scaled * result - temp
        result = temp + cheb_coeffs[k]
    # Add the last term properly - Clenshaw needs care
    # Let me use a simpler approach: direct sum
    result = np.zeros_like(x_scaled, dtype=float)
    T = np.zeros_like(x_scaled)
    T[:] = 1.0
    result += cheb_coeffs[0] * T
    if n >= 1:
        T_prev = np.ones_like(x_scaled)
        T_curr = x_scaled.copy()
        result += cheb_coeffs[1] * T_curr
        for k in range(2, n + 1):
            T_next = 2.0 * x_scaled * T_curr - T_prev
            result += cheb_coeffs[k] * T_next
            T_prev, T_curr = T_curr, T_next
    return result


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-5 polynomial to the unresolved fast-scale forcing.
    Returns 6 monomial coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN

    training_x = np.asarray(training_x, dtype=float)
    training_subgrid_tendency = np.asarray(training_subgrid_tendency, dtype=float)

    n_samples = len(training_x)
    degree = 5

    # Basic sanity: if too few samples, use lower effective degree
    if n_samples < degree + 1:
        degree = max(1, n_samples - 1)

    # Strategy: Try multiple approaches and pick the best based on
    # cross-validation or regularized fit.

    # Approach 1: Standard Chebyshev fit
    cheb_coeffs = _fit_chebyshev(training_x, training_subgrid_tendency, degree)

    # Convert to monomial basis
    monomial_coeffs = _chebyshev_to_monomial_fast(cheb_coeffs, degree)

    # Ensure finite
    if not np.all(np.isfinite(monomial_coeffs)):
        # Fallback: simple linear fit
        if n_samples >= 2:
            coeff = np.zeros(6)
            coeff[:2] = np.polyfit(training_x, training_subgrid_tendency, 1)
            # np.polyfit returns descending, we need ascending
            coeff[0] = coeff[1]
            coeff[1] = coeff[0]
            # Actually np.polyfit returns [a1, a0] for degree 1
            a1, a0 = np.polyfit(training_x, training_subgrid_tendency, 1)
            coeff = np.zeros(6)
            coeff[0] = a0
            coeff[1] = a1
            return coeff
        else:
            return np.zeros(6)

    # Approach 2: Try with slight regularization via Tikhonov
    if time.time() < deadline:
        # Add small ridge penalty in Chebyshev basis
        x_min = np.min(training_x)
        x_max = np.max(training_x)
        if x_max - x_min > 1e-12:
            x_scaled = 2.0 * (training_x - x_min) / (x_max - x_min) - 1.0
            V = np.zeros((n_samples, degree + 1))
            V[:, 0] = 1.0
            V[:, 1] = x_scaled
            for k in range(2, degree + 1):
                V[:, k] = 2.0 * x_scaled * V[:, k - 1] - V[:, k - 2]

            # Try a few small regularization values
            best_reg_score = np.inf
            best_reg_cheb = cheb_coeffs.copy()

            for lam in [1e-8, 1e-6, 1e-4, 1e-2]:
                # Regularized least squares: (V^T V + lam*I) c = V^T y
                A = V.T @ V + lam * np.eye(degree + 1)
                b = V.T @ training_subgrid_tendency
                try:
                    reg_cheb = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue

                # Evaluate on training data (as proxy)
                pred = _evaluate_polynomial_chebyshev(reg_cheb, training_x, x_min, x_max)
                score = np.mean((pred - training_subgrid_tendency) ** 2)

                # Also check condition number / stability
                if score < best_reg_score:
                    best_reg_score = score
                    best_reg_cheb = reg_cheb

            # Compare with unregularized
            pred_orig = _evaluate_polynomial_chebyshev(cheb_coeffs, training_x, x_min, x_max)
            score_orig = np.mean((pred_orig - training_subgrid_tendency) ** 2)

            if best_reg_score < score_orig * 0.99:
                cheb_coeffs = best_reg_cheb
                monomial_coeffs = _chebyshev_to_monomial_fast(cheb_coeffs, degree)

    # Final check: ensure all coefficients are finite
    if not np.all(np.isfinite(monomial_coeffs)):
        # Fallback to linear
        if n_samples >= 2:
            a1, a0 = np.polyfit(training_x, training_subgrid_tendency, 1)
            coeff = np.zeros(6)
            coeff[0] = a0
            coeff[1] = a1
            return coeff
        return np.zeros(6)

    # Pad to length 6 if needed
    if len(monomial_coeffs) < 6:
        padded = np.zeros(6)
        padded[:len(monomial_coeffs)] = monomial_coeffs
        return padded
    elif len(monomial_coeffs) > 6:
        return monomial_coeffs[:6]

    return monomial_coeffs
# EVOLVE-BLOCK-END
