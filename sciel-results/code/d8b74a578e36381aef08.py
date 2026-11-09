"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial using kernel-weighted denoising
    with weighted least squares on confidence-weighted pseudo-labels.
    
    Strategy:
    1. Use Gaussian kernel weighting to estimate E[Y|X] at each training point.
    2. Compute confidence weights from kernel weight sums.
    3. Fit a degree-5 polynomial via weighted least squares on centered/scaled x.
    4. Return coefficients in ascending powers of original x.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit (10s buffer)

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n == 0:
        return np.zeros(6)

    if n < 10:
        coeff = np.zeros(6)
        if n >= 2:
            try:
                coeff[:2] = np.polynomial.polynomial.polyfit(x, y, min(1, n - 1))
            except Exception:
                pass
        return coeff

    # If time is already tight, fall back to simple polyfit
    if time.time() > deadline - 10:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
            return np.where(np.isfinite(coeff), coeff, 0.0)
        except Exception:
            return np.zeros(6)

    # Stage 1: Kernel-weighted denoising to estimate E[Y|X]
    # Compute bandwidth using Silverman's rule
    x_range = np.ptp(x)
    if x_range < 1e-10:
        bandwidth = 1.0
    else:
        bandwidth = 1.06 * np.std(x) * (n ** (-1.0 / 5.0))
        if bandwidth < 1e-6:
            bandwidth = x_range * 0.1
        bandwidth = max(bandwidth, x_range * 0.03)

    # Subsample if n is very large for efficiency
    max_knn = 5000
    if n > max_knn:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=max_knn, replace=False)
        x_sub = x[idx]
        y_sub = y[idx]
    else:
        x_sub = x
        y_sub = y

    # Evaluate kernel-weighted mean at each training point
    chunk_size = 2000
    y_smoothed = np.zeros(n)
    weights_confidence = np.zeros(n)

    for i in range(0, n, chunk_size):
        if time.time() > deadline - 20:
            # Fallback to simple polyfit
            try:
                coeff = np.polynomial.polynomial.polyfit(x, y, 5)
                return np.where(np.isfinite(coeff), coeff, 0.0)
            except Exception:
                return np.zeros(6)

        i_end = min(i + chunk_size, n)
        x_chunk = x[i:i_end]

        # Compute kernel weights
        diff = (x_chunk[:, np.newaxis] - x_sub[np.newaxis, :]) / bandwidth
        raw_weights = np.exp(-0.5 * diff ** 2)

        # Normalize weights for averaging
        weight_sums = np.sum(raw_weights, axis=1, keepdims=True)
        weight_sums_safe = np.where(weight_sums < 1e-12, 1.0, weight_sums)
        normalized_weights = raw_weights / weight_sums_safe

        # Compute weighted average
        y_smoothed[i:i_end] = normalized_weights @ y_sub

        # Store confidence (sum of raw weights)
        weights_confidence[i:i_end] = np.sum(raw_weights, axis=1)

    # Stage 2: Weighted least squares polynomial fit
    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_scaled = np.zeros(n)
    else:
        x_scaled = (x - x_mean) / x_std

    # Build Vandermonde matrix for degree 5 on scaled x
    # A[:, j] = x_scaled^j
    A = np.vander(x_scaled, N=6, increasing=True)

    # Weighted least squares: minimize sum_i w_i * (A_i @ c - y_i)^2
    # c = (A^T W A)^{-1} A^T W y
    # W is diagonal with weights_confidence
    # Use sqrt of weights for numerical stability
    w = weights_confidence
    # Normalize weights to avoid overflow
    w_max = np.max(w)
    if w_max > 0:
        w = w / w_max
    else:
        w = np.ones(n)

    sqrt_w = np.sqrt(w)
    A_w = A * sqrt_w[:, np.newaxis]
    y_w = y_smoothed * sqrt_w

    try:
        # Solve the normal equations
        ATA = A_w.T @ A_w
        ATy = A_w.T @ y_w

        # Add small regularization for numerical stability
        reg = 1e-10 * np.trace(ATA) / 6
        ATA += reg * np.eye(6)

        c_scaled = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        try:
            c_scaled = np.linalg.lstsq(A_w, y_w, rcond=None)[0]
        except Exception:
            # Fallback to unweighted polyfit
            try:
                c_scaled = np.polynomial.polynomial.polyfit(x_scaled, y_smoothed, 5)
            except Exception:
                c_scaled = np.zeros(6)

    # Convert from scaled coefficients to original x coefficients
    # If p(x) = sum c_j * ((x - x_mean)/x_std)^j
    # We need q(x) = sum d_k * x^k such that q(x) = p(x)
    # Use polynomial composition: p(scaled_x) where scaled_x = (x - x_mean)/x_std
    # Let t = (x - x_mean)/x_std, so x = x_std * t + x_mean
    # p(t) = sum c_j * t^j
    # We want to express p((x - x_mean)/x_std) as a polynomial in x
    # t = (x - x_mean)/x_std
    # t^j = (x - x_mean)^j / x_std^j
    # (x - x_mean)^j = sum_{k=0}^{j} binom(j, k) * x^k * (-x_mean)^(j-k)
    # So p(t) = sum_j c_j * sum_{k=0}^{j} binom(j,k) * x^k * (-x_mean)^(j-k) / x_std^j
    # d_k = sum_{j=k}^{5} c_j * binom(j,k) * (-x_mean)^(j-k) / x_std^j

    if x_std < 1e-10:
        coeff = c_scaled.copy()
    else:
        coeff = np.zeros(6)
        for j in range(6):
            for k in range(j + 1):
                binom = 1.0
                # Compute binom(j, k)
                for m in range(k):
                    binom *= (j - m) / (m + 1)
                coeff[k] += c_scaled[j] * binom * ((-x_mean) ** (j - k)) / (x_std ** j)

    # Ensure all coefficients are finite
    coeff = np.where(np.isfinite(coeff), coeff, 0.0)

    return coeff
# EVOLVE-BLOCK-END
