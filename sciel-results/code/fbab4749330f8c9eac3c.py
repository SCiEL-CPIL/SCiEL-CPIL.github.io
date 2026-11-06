import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin out of 120s

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Remove any NaN/Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # --- Nadaraya-Watson smoothing to denoise the target ---
    # Estimate bandwidth using MAD
    median_x = np.median(x)
    mad = np.median(np.abs(x - median_x))
    if mad < 1e-12:
        mad = np.std(x)
    if mad < 1e-12:
        mad = 1.0
    # Silverman-like bandwidth scaled for the data range
    bw = 1.06 * max(mad, np.std(x) * 0.9) * n ** (-0.2)
    bw = max(bw, (x.max() - x.min()) / 20.0)  # ensure at least 20 bins worth

    # Evaluate NW kernel at each training point to get smoothed y
    # Use Gaussian kernel: K(u) = exp(-0.5 * u^2)
    # To avoid O(n^2) for very large n, subsample if needed
    if n > 50000:
        # Subsample for bandwidth evaluation, but use full data for regression
        idx = np.random.default_rng(42).choice(n, size=50000, replace=False)
        x_sub = x[idx]
        y_sub = y[idx]
    else:
        x_sub = x
        y_sub = y

    # Compute smoothed values using subsampled points
    # Process in chunks to manage memory
    chunk_size = 5000
    y_smoothed = np.empty(n)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x_chunk = x[start:end]
        # u = (x_sub - x_chunk[:, None]) / bw
        u = (x_sub[None, :] - x_chunk[:, None]) / bw
        weights = np.exp(-0.5 * u * u)
        w_sum = np.sum(weights, axis=1)
        w_sum = np.where(w_sum < 1e-300, 1e-300, w_sum)
        y_smoothed[start:end] = np.sum(weights * y_sub[None, :], axis=1) / w_sum

    # Check time
    if time.time() > deadline - 10.0:
        # Fallback: linear fit on smoothed data
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y_smoothed, 1)
        except Exception:
            coeff[1] = np.mean(y_smoothed) / np.mean(x) if np.mean(x) != 0 else 0
        return coeff

    # --- Degree-5 polynomial fit on smoothed data using Legendre basis for stability ---
    # Scale x to [-1, 1] for numerical stability in Legendre fit
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min

    if x_range < 1e-12:
        # Degenerate case: all x are the same
        c0 = np.mean(y_smoothed)
        return np.array([c0, 0.0, 0.0, 0.0, 0.0, 0.0])

    x_scaled = (x - (x_min + x_max) / 2.0) / (x_range / 2.0)

    # Fit Legendre polynomial of degree 5
    leg_coeffs = np.polynomial.legendre.legfit(x_scaled, y_smoothed, 5)

    # Convert Legendre coefficients to standard monomial coefficients
    # First, convert Legendre coeffs to monomial coeffs in the scaled variable u
    monomial_u = np.polynomial.legendre.leg2poly(leg_coeffs)

    # Now we have y ≈ sum_{i=0}^5 a_i * u^i
    # u = (x - mid) / half
    # We need to expand sum a_i * ((x - mid) / half)^i into sum c_j * x^j
    # Let's build the transformation matrix T such that monomial_x = T @ monomial_u
    # u = A * x + B
    A = 1.0 / (x_range / 2.0)
    B = -(x_min + x_max) / 2.0 / (x_range / 2.0)

    # We want to compute sum_{i=0}^5 a_i * (A*x + B)^i
    # Let's compute powers of (A*x+B)
    # power_0 = 1
    # power_1 = A*x + B
    # power_k = power_{k-1} * (A*x+B)

    powers = []
    current_power = np.array([1.0])  # constant 1
    for i in range(6):
        powers.append(current_power.copy())
        if i < 5:
            # Multiply current_power (which is coeff for u^i) by (A*x+B)
            new_len = len(current_power) + 1
            new_power = np.zeros(new_len)
            new_power[0] = current_power[0] * B
            for k in range(1, len(current_power)):
                new_power[k] = current_power[k - 1] * A + current_power[k] * B
            new_power[-1] = current_power[-1] * A
            current_power = new_power

    # Now sum a_i * powers[i]
    c_x = np.zeros(6)
    for i in range(6):
        p = powers[i]
        # Pad p to length 6
        padded_p = np.zeros(6)
        padded_p[:len(p)] = p
        c_x += monomial_u[i] * padded_p

    # Check time
    if time.time() > deadline:
        # Fallback to linear if somehow slow (unlikely)
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y_smoothed, 1)
        except Exception:
            coeff[0] = np.mean(y_smoothed)
        return coeff

    # Ensure finite coefficients
    c_x = np.where(np.isfinite(c_x), c_x, 0.0)

    return c_x
