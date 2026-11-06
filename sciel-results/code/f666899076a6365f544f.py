"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Scale x to [-1, 1] for stable Legendre evaluation
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        # Degenerate: return near-zero polynomial
        return np.zeros(6)
    x_scaled = (x - (x_max + x_min) / 2.0) / (x_range / 2.0)

    # Build Legendre design matrix (degree 0..5)
    n = len(x)
    V = np.zeros((n, 6))
    V[:, 0] = np.polynomial.legendre.legval(x_scaled, [1])
    for d in range(1, 6):
        V[:, d] = np.polynomial.legendre.legval(x_scaled, np.eye(6)[:, d])

    # Search over regularization strengths in Legendre coefficient space
    # Penalty: sum_{d=0}^{5} lambda_d * a_d^2, with lambda_d increasing with degree
    best_score = np.inf
    best_a = None
    best_lambda = None

    # Try a grid of base regularization strengths
    base_lambdas = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    # Degree-dependent scaling: higher degrees get more penalty
    degree_weights = np.array([1.0, 1.0, 5.0, 20.0, 80.0, 300.0])

    for base_lam in base_lambdas:
        if time.time() > deadline:
            break
        lam = base_lam * degree_weights
        # Regularized least squares in Legendre basis
        # Minimize ||V @ a - y||^2 + a^T @ diag(lam) @ a
        # (V^T V + diag(lam)) a = V^T y
        VtV = V.T @ V
        Vty = V.T @ y
        A = VtV + np.diag(lam)
        try:
            a = np.linalg.solve(A, Vty)
        except np.linalg.LinAlgError:
            a = np.linalg.lstsq(A, Vty, rcond=None)[0]

        # Convert Legendre coefficients to monomial coefficients
        # We need to convert from Legendre basis on [-1,1] to monomial basis on original x
        # First, get monomial coefficients in the scaled variable z = (x - c)/s
        # P_d(z) in monomial form, then substitute z = (x-c)/s
        # Simpler: use numpy's polynomial conversion
        # leg2poly gives monomial coefficients in z
        mono_z = np.polynomial.legendre.leg2poly(a)  # degree <= 5 in z

        # Now convert from z = (x - c)/s to x
        # z = (x - c)/s => x = s*z + c
        # p(z) = p((x-c)/s)
        c_val = (x_max + x_min) / 2.0
        s_val = x_range / 2.0

        # Substitute: for each monomial z^k, replace with ((x-c)/s)^k
        # This is a polynomial composition. Use Horner-like approach.
        # p(z) = sum_{k=0}^{5} mono_z[k] * z^k
        # z = (x - c)/s
        # We can build this iteratively.
        # Start with result = 0
        # For k from 5 down to 0: result = result * z + mono_z[k]
        # But z is linear in x, so we can track coefficients.

        # Let's build the polynomial in x by repeated substitution.
        # Represent polynomial as coefficient array in ascending powers of x.
        # Start with the constant term of p(z): mono_z[0]
        # Then for each higher term, multiply by z and add.

        # More efficient: build step by step
        # poly_x starts as [mono_z[5]] (the leading coefficient for z^5)
        # Actually, let's use a different approach: evaluate the transformation matrix.

        # Build transformation: for each degree k, z^k expressed in x.
        # z = (x - c)/s = (1/s)*x - c/s
        # Let a1 = 1/s, a0 = -c/s
        # z^k = (a1*x + a0)^k
        # We can compute the monomial coefficients of (a1*x + a0)^k for k=0..5

        a1 = 1.0 / s_val
        a0 = -c_val / s_val

        # Transform matrix T: T[k, j] = coefficient of x^j in (a1*x + a0)^k
        T = np.zeros((6, 6))
        T[0, 0] = 1.0
        for k in range(1, 6):
            # (a1*x + a0)^k = (a1*x + a0) * (a1*x + a0)^(k-1)
            # If (a1*x+a0)^(k-1) = sum_j T[k-1,j] * x^j
            # Then (a1*x+a0)^k = sum_j T[k-1,j] * (a1*x+a0) * x^j
            # = sum_j T[k-1,j] * (a1*x^(j+1) + a0*x^j)
            for j in range(6):
                if T[k-1, j] != 0:
                    T[k, j] += T[k-1, j] * a0
                    if j + 1 < 6:
                        T[k, j+1] += T[k-1, j] * a1

        # monomial coefficients in x: coeff_x = T @ mono_z
        coeff_x = T @ mono_z

        # Evaluate on training data to compute MSE
        pred = np.polynomial.polynomial.polyval(x, coeff_x)
        mse = np.mean((pred - y) ** 2)

        # Use training MSE as proxy (we don't have validation data)
        # But we want to minimize MSE, so lower is better
        if mse < best_score:
            best_score = mse
            best_a = a.copy()
            best_lambda = lam.copy()
            best_coeff_x = coeff_x.copy()

    # If we found a best, return it. Otherwise fall back to simple fit.
    if best_coeff_x is not None and np.all(np.isfinite(best_coeff_x)):
        return best_coeff_x

    # Fallback: linear fit
    coeff = np.zeros(6)
    coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
    return coeff
# EVOLVE-BLOCK-END
