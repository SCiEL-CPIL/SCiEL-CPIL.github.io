"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s limit

    x = np.asarray(training_x, dtype=np.float64).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).flatten()

    n_samples = len(x)

    if n_samples < 6:
        if n_samples < 2:
            return np.zeros(6)
        coeff = np.zeros(6)
        coeff[:2] = np.polyfit(x, y, 1)[::-1]
        return coeff

    # Scale x to [0, 1] for better conditioning
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    if x_range < 1e-12:
        mean_y = np.mean(y)
        coeff = np.zeros(6)
        coeff[0] = mean_y
        return coeff

    x_scaled = (x - x_min) / x_range

    # Strategy: Use a Fourier-like basis transformed to monomial form
    # The idea: fit in a well-conditioned basis, then convert to monomial coefficients
    
    # Build Fourier design matrix: 1, cos(2πt), sin(2πt), cos(4πt), sin(4πt), cos(6πt)
    # This gives 6 basis functions that are orthogonal and well-conditioned
    t = x_scaled
    fourier_basis = np.column_stack([
        np.ones_like(t),
        np.cos(2 * np.pi * t),
        np.sin(2 * np.pi * t),
        np.cos(4 * np.pi * t),
        np.sin(4 * np.pi * t),
        np.cos(6 * np.pi * t),
    ])
    
    # Fit in Fourier space with light regularization
    lambda_reg = 1e-6 * np.eye(6)
    try:
        A_f = fourier_basis.T @ fourier_basis + lambda_reg
        b_f = fourier_basis.T @ y
        fourier_coeffs = np.linalg.solve(A_f, b_f)
    except np.linalg.LinAlgError:
        fourier_coeffs = np.linalg.lstsq(fourier_basis, y, rcond=None)[0]
    
    # Now we need to convert the Fourier fit to a degree-5 polynomial in x
    # The Fourier fit is: f(t) = a0 + a1*cos(2πt) + a2*sin(2πt) + a3*cos(4πt) + a4*sin(4πt) + a5*cos(6πt)
    # We want to find a polynomial p(x) = sum_{i=0}^{5} c_i * x^i that best approximates f
    # on the training data (or equivalently, project the Fourier fit onto the polynomial space)
    
    # Build monomial design matrix on scaled x
    V_mono = np.vander(x_scaled, 6, increasing=True)
    
    # Evaluate the Fourier fit at training points
    y_fourier = fourier_basis @ fourier_coeffs
    
    # Now fit a degree-5 polynomial to the Fourier-evaluated values
    # This is a two-stage approach: first fit Fourier, then project to polynomial
    lambda_reg2 = 1e-4 * np.eye(6)
    try:
        A_m = V_mono.T @ V_mono + lambda_reg2
        b_m = V_mono.T @ y_fourier
        coeff_scaled = np.linalg.solve(A_m, b_m)
    except np.linalg.LinAlgError:
        coeff_scaled = np.linalg.lstsq(V_mono, y_fourier, rcond=None)[0]
    
    # Transform coefficients back to original x scale
    # If p(t) = sum c_i * t^i where t = (x - x_min)/x_range
    # Then p(x) = sum c_i * ((x - x_min)/x_range)^i
    coeff_orig = np.zeros(6)
    for i in range(6):
        if abs(coeff_scaled[i]) < 1e-15:
            continue
        for j in range(i + 1):
            binom = 1.0
            for k in range(j):
                binom = binom * (i - k) / (k + 1)
            term_coeff = coeff_scaled[i] * binom * ((-x_min) ** (i - j)) / (x_range ** i)
            coeff_orig[j] += term_coeff
    
    # Check time and validity
    if not np.all(np.isfinite(coeff_orig)):
        # Fallback: direct regularized polynomial fit
        V = np.vander(x, 6, increasing=True)
        lam = 1e-4 * np.eye(6)
        try:
            A = V.T @ V + lam
            b = V.T @ y
            coeff_orig = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coeff_orig = np.zeros(6)
            if n_samples >= 2:
                coeff_orig[:2] = np.polyfit(x, y, 1)[::-1]
    
    # Final check
    if not np.all(np.isfinite(coeff_orig)):
        coeff_orig = np.zeros(6)
    
    return coeff_orig
# EVOLVE-BLOCK-END
