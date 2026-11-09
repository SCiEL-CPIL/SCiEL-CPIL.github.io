import numpy as np
import scipy.optimize as optimize
import time

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    timeout = 580  # safety margin before 630s limit

    N = 8

    # Precompute quadrature points
    c_vals, w_vals = np.polynomial.legendre.leggauss(120)
    c_vals = 0.5 * (c_vals + 1.0)
    w_vals = 0.5 * w_vals

    # Precompute interval merge for speed: vectorized merge of N intervals
    # For each c, we have N intervals [X_i + c*i/N, X_i + c*i/N + 1/N]
    # We need the total length of their union.

    def compute_volume(X):
        """
        Compute the union volume of the 64 tubes.
        X is an array of shape (N,) representing the base x-coordinates for each i.
        The y-coordinates are the same as x-coordinates (symmetry).
        Volume = integral over c of (merged_length_x(c) * merged_length_y(c)) dc
        Since X is symmetric, merged_length_x = merged_length_y, so volume = integral of length^2 dc.
        """
        # For each quadrature point c, compute the union length of intervals
        # starts_i = X_i + c * i / N, ends_i = starts_i + 1/N
        # We need the total length of union of these N intervals.

        # Vectorized approach: for each c, compute all starts and ends, sort, merge
        # Since N is small (8), we can do this efficiently.

        # Precompute: for each c_idx, starts = X + c_vals[idx] * arange(N)/N
        # This is a matrix of shape (n_quad, N)
        c_col = c_vals[:, np.newaxis]  # (n_quad, 1)
        i_row = np.arange(N)[np.newaxis, :]  # (1, N)
        starts = X[np.newaxis, :] + c_col * i_row / N  # (n_quad, N)
        ends = starts + 1.0 / N  # (n_quad, N)

        # For each row (each c), sort by start and merge
        # Sort indices
        order = np.argsort(starts, axis=1)  # (n_quad, N)
        s_sorted = np.take_along_axis(starts, order, axis=1)  # (n_quad, N)
        e_sorted = np.take_along_axis(ends, order, axis=1)  # (n_quad, N)

        # Merge: iterate over intervals
        # cur_s = s_sorted[:, 0], cur_e = e_sorted[:, 0]
        # For k in 1..N-1: if s_sorted[:, k] <= cur_e: cur_e = max(cur_e, e_sorted[:, k])
        # else: length += cur_e - cur_s; cur_s = s_sorted[:, k]; cur_e = e_sorted[:, k]
        # length += cur_e - cur_s

        cur_s = s_sorted[:, 0].copy()
        cur_e = e_sorted[:, 0].copy()
        length = np.zeros(len(c_vals))

        for k in range(1, N):
            s_k = s_sorted[:, k]
            e_k = e_sorted[:, k]
            # Where s_k <= cur_e, extend cur_e; otherwise, close current interval
            mask = s_k <= cur_e
            # Extend cur_e where mask is true
            cur_e = np.where(mask, np.maximum(cur_e, e_k), cur_e)
            # Where mask is false, add current interval length and reset
            length += np.where(mask, 0.0, cur_e - cur_s)
            cur_s = np.where(mask, cur_s, s_k)
            cur_e = np.where(mask, cur_e, e_k)

        # Add last interval
        length += cur_e - cur_s

        # Volume = sum of length^2 * w
        vol = np.sum(length ** 2 * w_vals)
        return vol

    # Parametric representation using Fourier modes
    # X_i = a0 + sum_{m=1}^{M} (a_m * cos(2*pi*m*i/N) + b_m * sin(2*pi*m*i/N))
    # This gives a smooth deformation field with 2*M+1 parameters
    M = 4  # Number of Fourier modes (8 parameters + 1 DC = 9 total)

    i_vals = np.arange(N)
    # Precompute Fourier basis
    fourier_basis = np.zeros((N, 2 * M + 1))
    fourier_basis[:, 0] = 1.0  # DC term
    for m in range(1, M + 1):
        fourier_basis[:, 2 * m - 1] = np.cos(2 * np.pi * m * i_vals / N)
        fourier_basis[:, 2 * m] = np.sin(2 * np.pi * m * i_vals / N)

    def params_to_X(params):
        """Convert Fourier parameters to X array."""
        # params has shape (2*M+1,)
        X = fourier_basis @ params
        # Center the array (subtract mean) to remove translational invariance
        X = X - np.mean(X)
        return X

    def obj(params):
        X = params_to_X(params)
        return compute_volume(X)

    # Initial guess: linear shearing corresponds to a specific Fourier mode
    # Linear ramp X_i = -i/N has a particular Fourier representation
    # But let's start with a few different initializations in parameter space

    rng = np.random.default_rng(42)

    # Linear shearing: X_i = -i/N
    # In Fourier: this is not a single mode, but we can project it
    X_linear = -i_vals / N
    X_linear_centered = X_linear - np.mean(X_linear)
    # Project onto Fourier basis (least squares)
    params_linear = np.linalg.lstsq(fourier_basis, X_linear_centered, rcond=None)[0]

    # Zero initialization
    params_zero = np.zeros(2 * M + 1)

    # Random initializations
    params_random1 = rng.uniform(-0.3, 0.3, 2 * M + 1)
    params_random2 = rng.uniform(-0.5, 0.5, 2 * M + 1)
    params_random3 = rng.uniform(-0.2, 0.2, 2 * M + 1)

    initializations = [
        params_linear,
        params_zero,
        params_random1,
        params_random2,
        params_random3,
    ]

    best_params = params_linear.copy()
    best_val = obj(best_params)

    for init in initializations:
        if time.time() - start_time > timeout:
            break

        # Powell's method on the reduced parameter space
        try:
            res = optimize.minimize(obj, init, method='Powell',
                                    options={'maxiter': 3000, 'xtol': 1e-9, 'ftol': 1e-12})
            if time.time() - start_time > timeout:
                break
            if res.fun < best_val:
                best_params = res.x.copy()
                best_val = res.fun
        except Exception:
            pass

        # Nelder-Mead polish
        if time.time() - start_time > timeout:
            break
        try:
            res2 = optimize.minimize(obj, best_params, method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-13})
            if time.time() - start_time > timeout:
                break
            if res2.fun < best_val:
                best_params = res2.x.copy()
                best_val = res2.fun
        except Exception:
            pass

    # Final high-precision polish
    if time.time() - start_time < timeout - 30:
        try:
            res3 = optimize.minimize(obj, best_params, method='Powell',
                                     options={'maxiter': 5000, 'xtol': 1e-11, 'ftol': 1e-14})
            if time.time() - start_time > timeout:
                pass
            elif res3.fun < best_val:
                best_params = res3.x.copy()
                best_val = res3.fun
        except Exception:
            pass

    # Convert best parameters to X
    X_best = params_to_X(best_params)

    # Construct positions
    positions = np.zeros((N * N, 2))
    for k in range(N * N):
        i = k // N
        j = k % N
        positions[k, 0] = X_best[i]
        positions[k, 1] = X_best[j]

    return positions
