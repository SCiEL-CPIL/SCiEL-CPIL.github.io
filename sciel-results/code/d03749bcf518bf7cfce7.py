"""Strategy: Nadaraya-Watson smoothing denoising followed by degree-5 polynomial fit."""
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

    # Try multiple smoothing approaches and pick the best via cross-validation
    # Approach: Nadaraya-Watson kernel smoothing with Gaussian kernel

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
        if time.time() > deadline - 10:
            break

        # Compute smoothed values on the grid using Nadaraya-Watson
        # y_smooth(x) = sum(w_i * y_i) / sum(w_i) where w_i = exp(-0.5 * ((x - x_i)/bw)^2)

        # For efficiency, compute smoothed values at grid points
        # Then fit polynomial to the smoothed (x_grid, y_smooth) pairs

        y_smooth = np.zeros_like(x_grid)
        for i in range(len(x_grid)):
            xi = x_grid[i]
            # Compute weights
            d = (x_sorted - xi) / bw
            w = np.exp(-0.5 * d * d)
            w_sum = np.sum(w)
            if w_sum > 1e-12:
                y_smooth[i] = np.sum(w * y_sorted) / w_sum
            else:
                y_smooth[i] = np.mean(y_sorted)

        # Fit degree-5 polynomial to smoothed data
        try:
            coeff_candidate = np.polynomial.polynomial.polyfit(x_grid, y_smooth, 5)
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

            # Smooth training data
            y_smooth_train = np.zeros_like(x_val)
            for j in range(len(x_val)):
                xj = x_val[j]
                d = (x_train - xj) / bw
                w = np.exp(-0.5 * d * d)
                w_sum = np.sum(w)
                if w_sum > 1e-12:
                    y_smooth_train[j] = np.sum(w * y_train) / w_sum
                else:
                    y_smooth_train[j] = np.mean(y_train)

            # Fit polynomial to smoothed training data
            try:
                coeff_cv = np.polynomial.polynomial.polyfit(x_val, y_smooth_train, 5)
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
        bw_best = bandwidths[np.argmin([
            # Recompute to find which bw was best - just refit with best_coeff
            0
        ])] if len(bandwidths) > 0 else base_bw

        # Actually, let's just do a final fit on all smoothed data
        # Reconstruct which bandwidth gave best result
        # Simpler: just use the best_coeff we found
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
