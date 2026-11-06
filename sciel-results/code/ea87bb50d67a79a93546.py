"""Strategy: Binned-mean denoising with adaptive bin width followed by regularized degree-5 polynomial fit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit

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
    n_grid = min(800, max(200, n // 30))
    x_grid = np.linspace(x_min, x_max, n_grid)

    # Try different bin widths
    base_bw = x_range * 0.08
    bin_widths = [base_bw * 0.3, base_bw * 0.5, base_bw, base_bw * 1.5, base_bw * 2.0, base_bw * 3.0]

    best_coeff = None
    best_cv_error = np.inf

    # Simple cross-validation: split into 5 folds
    n_folds = 5
    fold_size = n // n_folds

    for bw in bin_widths:
        if time.time() > deadline - 15:
            break

        # Compute binned-mean smoothed values on the grid
        y_smooth = np.zeros_like(x_grid)
        for i, xg in enumerate(x_grid):
            # Find points within +/- bw of xg
            mask_bin = np.abs(x_sorted - xg) < bw
            if np.any(mask_bin):
                y_smooth[i] = np.mean(y_sorted[mask_bin])
            else:
                y_smooth[i] = np.mean(y_sorted)

        # Fit degree-5 polynomial to smoothed data with regularization
        try:
            V = np.vander(x_grid, 6, increasing=True)
            lam = 1e-3
            A = V.T @ V + lam * np.eye(6)
            b_vec = V.T @ y_smooth
            coeff_candidate = np.linalg.solve(A, b_vec)
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

            # Smooth training data at validation points using binned mean
            y_smooth_val = np.zeros_like(x_val)
            for i, xv in enumerate(x_val):
                mask_bin = np.abs(x_train - xv) < bw
                if np.any(mask_bin):
                    y_smooth_val[i] = np.mean(y_train[mask_bin])
                else:
                    y_smooth_val[i] = np.mean(y_train)

            # Fit polynomial to smoothed training data at validation points
            try:
                V_val = np.vander(x_val, 6, increasing=True)
                lam_val = 1e-3
                A_val = V_val.T @ V_val + lam_val * np.eye(6)
                b_vec_val = V_val.T @ y_smooth_val
                coeff_cv = np.linalg.solve(A_val, b_vec_val)
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
        # Final high-precision polish: refit on all data with the chosen bin width
        y_smooth_final = np.zeros_like(x_grid)
        for i, xg in enumerate(x_grid):
            mask_bin = np.abs(x_sorted - xg) < best_bw
            if np.any(mask_bin):
                y_smooth_final[i] = np.mean(y_sorted[mask_bin])
            else:
                y_smooth_final[i] = np.mean(y_sorted)

        try:
            V = np.vander(x_grid, 6, increasing=True)
            lam = 1e-3
            A = V.T @ V + lam * np.eye(6)
            b_vec = V.T @ y_smooth_final
            final_coeff = np.linalg.solve(A, b_vec)
            if np.all(np.isfinite(final_coeff)):
                return final_coeff
        except Exception:
            pass

        return best_coeff

    # Fallback: direct polynomial fit with regularization
    try:
        V = np.vander(x, 6, increasing=True)
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
