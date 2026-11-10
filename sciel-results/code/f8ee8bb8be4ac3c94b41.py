"""Rewritten program: robust degree-5 polynomial closure with adaptive basis scaling,
regularization, and a time-bounded refinement loop."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def construct(training_x, training_subgrid_tendency):
    """Fit a shared degree-5 polynomial to the unresolved tendency.

    Strategy:
      1. Scale x to a stable range to avoid ill-conditioned Vandermonde.
      2. Fit degree-5 polynomial via least squares with light Tikhonov regularization.
      3. If time permits, refine with a higher-precision solve or iterative reweighting.
      4. Return coefficients in ascending powers of the ORIGINAL x.
    """
    t_start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    n = len(x)
    if n < 6:
        # Not enough points; return zeros
        return np.zeros(6)

    # --- Scale x for numerical stability ---
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-12:
        # All x are the same; constant fit
        c0 = np.mean(y)
        coeff = np.zeros(6)
        coeff[0] = c0
        return coeff

    # Map x to [-1, 1] for stable polynomial fitting
    x_scaled = (x - (x_min + x_max) / 2.0) / (x_range / 2.0)

    # Build Vandermonde matrix for degree 5 in scaled variable
    # V[i, j] = x_scaled[i]^j
    V = np.vander(x_scaled, N=6, increasing=True)

    # Light Tikhonov regularization to guard against overfitting/ill-conditioning
    # Scale lambda relative to the data magnitude
    y_var = np.var(y)
    if y_var < 1e-20:
        y_var = 1.0
    lam = 1e-6 * y_var

    # Solve (V^T V + lam*I) c = V^T y
    VtV = V.T @ V
    Vty = V.T @ y
    reg = lam * np.eye(6)
    try:
        c_scaled = np.linalg.solve(VtV + reg, Vty)
    except np.linalg.LinAlgError:
        c_scaled = np.linalg.lstsq(V, y, rcond=None)[0]

    # Convert coefficients from scaled basis to original x basis
    # x_scaled = (x - mid) / half, where mid = (x_min+x_max)/2, half = x_range/2
    # We need: sum_i c_scaled[i] * x_scaled^i = sum_j C[j] * x^j
    # x_scaled = (x - mid)/half
    # Use polynomial composition / change of variable
    # We'll compute the transformed coefficients by evaluating the basis transformation.
    # For a degree-5 poly, we can do this via repeated substitution or via matrix.
    # Build the transformation: express x_scaled^k in terms of x.
    # x_scaled = (x - mid)/half = a*x + b, where a = 1/half, b = -mid/half
    a = 1.0 / (x_range / 2.0)
    b = -(x_min + x_max) / 2.0 / (x_range / 2.0)

    # Coefficients of (a*x + b)^k for k=0..5 in powers of x
    # We'll build a 6x6 transformation matrix T such that:
    # [x_scaled^0, x_scaled^1, ..., x_scaled^5]^T = T @ [1, x, x^2, x^3, x^4, x^5]^T
    # Actually, x_scaled^k = (a*x+b)^k = sum_{j=0}^k binom(k,j) a^j b^(k-j) x^j
    # So the coefficient of x^j in x_scaled^k is binom(k,j) * a^j * b^(k-j) for j<=k, else 0.
    # The full polynomial is sum_k c_scaled[k] * x_scaled^k = sum_j C[j] x^j
    # C[j] = sum_{k=j}^5 c_scaled[k] * binom(k,j) * a^j * b^(k-j)

    C = np.zeros(6)
    for k in range(6):
        for j in range(k + 1):
            # binom(k, j)
            binom_val = 1.0
            for mm in range(j):
                binom_val = binom_val * (k - mm) / (mm + 1)
            C[j] += c_scaled[k] * binom_val * (a ** j) * (b ** (k - j))

    # Check time
    elapsed = time.time() - t_start
    if elapsed > DEADLINE - 10.0:
        return C

    # --- Refinement pass: try a weighted least squares to reduce outliers' influence ---
    # Compute residuals in original space
    pred_scaled = V @ c_scaled
    residuals = y - pred_scaled

    # Use soft weighting: down-weight large residuals
    # sigma = std(residuals)
    sigma = np.std(residuals)
    if sigma > 1e-12:
        weights = 1.0 / (1.0 + (residuals / sigma) ** 2)
        # Re-solve with weights
        W_sqrt = np.sqrt(weights)
        Vw = V * W_sqrt[:, None]
        yw = y * W_sqrt
        try:
            c_scaled_w = np.linalg.lstsq(Vw, yw, rcond=None)[0]
        except Exception:
            c_scaled_w = c_scaled

        # Recompute C from c_scaled_w
        C_w = np.zeros(6)
        for k in range(6):
            for j in range(k + 1):
                binom_val = 1.0
                for mm in range(j):
                    binom_val = binom_val * (k - mm) / (mm + 1)
                C_w[j] += c_scaled_w[k] * binom_val * (a ** j) * (b ** (k - j))

        # Choose the one with lower training MSE
        pred_orig = np.zeros(n)
        for j in range(6):
            pred_orig += C[j] * (x ** j)
        mse_orig = np.mean((y - pred_orig) ** 2)

        pred_orig_w = np.zeros(n)
        for j in range(6):
            pred_orig_w += C_w[j] * (x ** j)
        mse_orig_w = np.mean((y - pred_orig_w) ** 2)

        if mse_orig_w < mse_orig:
            C = C_w

    # Final check
    elapsed = time.time() - t_start
    if elapsed > DEADLINE:
        return C

    # Ensure finite
    if not np.all(np.isfinite(C)):
        # Fallback to simple linear fit
        try:
            c_lin = np.polyfit(x, y, 1)  # descending order
            C_fb = np.zeros(6)
            C_fb[0] = c_lin[1]
            C_fb[1] = c_lin[0]
            return C_fb
        except Exception:
            return np.zeros(6)

    return C
# EVOLVE-BLOCK-END
