"""Strategy: Nadaraya-Watson smoothing denoising followed by degree-5 polynomial fit with efficient kernel evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    n = len(x)
    if n < 10:
        return np.zeros(6)

    # Remove any NaNs or Infs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n < 10:
        return np.zeros(6)

    # Sort by x for efficient binning
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Determine x range
    x_min = x_sorted[0]
    x_max = x_sorted[-1]
    x_range = x_max - x_min
    if x_range < 1e-12:
        # Constant x, just return mean
        coeff = np.zeros(6)
        coeff[0] = np.mean(y_sorted)
        return coeff

    # Create evaluation grid
    n_grid = min(500, max(100, n // 50))
    x_grid = np.linspace(x_min, x_max, n_grid)

    # Try different bandwidths
    base_bw = x_range * 0.05
    bandwidths = [base_bw * 0.3, base_bw * 0.5, base_bw, base_bw * 1.5, base_bw * 2.0, base_bw * 3.0]

    best_coeff = None
    best_cv_error = np.inf

    # Simple cross-validation: split into 5 folds
    n_folds = 5
    fold_size = n // n_folds

    for bw in bandwidths:
        if time.time() > deadline - 15:
            break

        # Compute smoothed values on the grid using Nadaraya-Watson
        # Use efficient vectorized computation
        # y_smooth(x) = sum(w_i * y_i) / sum(w_i) where w_i = exp(-0.5 * ((x - x_i)/bw)^2)

        # For efficiency, compute smoothed values at grid points using matrix operations
        # d = (x_sorted[None, :] - x_grid[:, None]) / bw
        # w = np.exp(-0.5 * d * d)
        # y_smooth = np.sum(w * y_sorted[None, :], axis=1) / np.sum(w, axis=1)

        # This creates an n x n_grid matrix which might be large. Let's chunk it.
        y_smooth = np.zeros_like(x_grid)
        chunk_size = min(100, len(x_grid))
        
        for start_idx in range(0, len(x_grid), chunk_size):
            end_idx = min(start_idx + chunk_size, len(x_grid))
            x_chunk = x_grid[start_idx:end_idx]
            # d shape: (chunk_size, n)
            d = (x_sorted[None, :] - x_chunk[:, None]) / bw
            w = np.exp(-0.5 * d * d)
            w_sum = np.sum(w, axis=1)
            # Avoid division by zero
            valid = w_sum > 1e-12
            y_smooth[start_idx:end_idx][valid] = np.sum(w[valid] * y_sorted[None, :], axis=1) / w_sum[valid]
            if np.any(~valid):
                y_smooth[start_idx:end_idx][~valid] = np.mean(y_sorted)

        # Fit degree-5 polynomial to smoothed data
        try:
            coeff_candidate = np.polynomial.polynomial.polyfit(x_grid, y_smooth, 5)
            if not np.all(np.isfinite(coeff_candidate)):
                continue
        except Exception:
            continue

        # Cross-validate: evaluate polynomial on held-out data
        cv_errors = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n

            # Train on all except validation fold
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_start:val_end] = False

            x_train = x_sorted[train_mask]
            y_train = y_sorted[train_mask]
            x_val = x_sorted[val_start:val_end]
            y_val = y_sorted[val_start:val_end]

            if len(x_train) < 20 or len(x_val) < 5:
                continue

            # Smooth training data at validation points
            y_smooth_val = np.zeros_like(x_val)
            # Chunked computation for validation
            val_chunk_size = min(50, len(x_val))
            for v_start in range(0, len(x_val), val_chunk_size):
                v_end = min(v_start + val_chunk_size, len(x_val))
                x_v_chunk = x_val[v_start:v_end]
                d_v = (x_train[None, :] - x_v_chunk[:, None]) / bw
                w_v = np.exp(-0.5 * d_v * d_v)
                w_sum_v = np.sum(w_v, axis=1)
                valid_v = w_sum_v > 1e-12
                y_smooth_val[v_start:v_end][valid_v] = np.sum(w_v[valid_v] * y_train[None, :], axis=1) / w_sum_v[valid_v]
                if np.any(~valid_v):
                    y_smooth_val[v_start:v_end][~valid_v] = np.mean(y_train)

            # Fit polynomial to smoothed training data at validation points
            try:
                coeff_cv = np.polynomial.polynomial.polyfit(x_val, y_smooth_val, 5)
                if not np.all(np.isfinite(coeff_cv)):
                    continue
            except Exception:
                continue

            # Evaluate on validation set (compare polynomial prediction to raw y_val)
            pred = np.polynomial.polynomial.polyval(x_val, coeff_cv)
            mse = np.mean((pred - y_val) ** 2)
            cv_errors.append(mse)

        if cv_errors:
            mean_cv_error = np.mean(cv_errors)
            if mean_cv_error < best_cv_error:
                best_cv_error = mean_cv_error
                best_coeff = coeff_candidate
                best_bw = bw

    if time.time() > deadline - 5:
        # Return best found so far, or fallback
        if best_coeff is not None:
            return best_coeff
        # Fallback: simple linear fit
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            return coeff
        except Exception:
            return np.zeros(6)

    if best_coeff is not None:
        # Final high-precision polish: refit on all data with the chosen bandwidth
        # Re-smooth all data and refit
        y_smooth_final = np.zeros_like(x_grid)
        chunk_size = min(100, len(x_grid))
        for start_idx in range(0, len(x_grid), chunk_size):
            end_idx = min(start_idx + chunk_size, len(x_grid))
            x_chunk = x_grid[start_idx:end_idx]
            d = (x_sorted[None, :] - x_chunk[:, None]) / best_bw
            w = np.exp(-0.5 * d * d)
            w_sum = np.sum(w, axis=1)
            valid = w_sum > 1e-12
            y_smooth_final[start_idx:end_idx][valid] = np.sum(w[valid] * y_sorted[None, :], axis=1) / w_sum[valid]
            if np.any(~valid):
                y_smooth_final[start_idx:end_idx][~valid] = np.mean(y_sorted)

        try:
            final_coeff = np.polynomial.polynomial.polyfit(x_grid, y_smooth_final, 5)
            if np.all(np.isfinite(final_coeff)):
                return final_coeff
        except Exception:
            pass
        
        return best_coeff

    # Fallback: direct polynomial fit with regularization
    try:
        # Use weighted least squares with a small regularization
        V = np.vander(x, 6, increasing=True)
        # Add small ridge regularization
        lam = 1e-4
        A = V.T @ V + lam * np.eye(6)
        b_vec = V.T @ y
        coeff = np.linalg.solve(A, b_vec)
        return coeff
    except Exception:
        try:
            coeff = np.zeros(6)
            coeff[:2] = np.polynomial.polynomial.polyfit(x, y, 1)
            return coeff
        except Exception:
            return np.zeros(6)

# EVOLVE-BLOCK-END
