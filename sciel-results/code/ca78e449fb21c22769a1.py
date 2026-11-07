import numpy as np
import time
from scipy.interpolate import PchipInterpolator, CubicSpline, interp1d

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 315.0  # safety margin before 330s kill

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape

    # Handle constant y
    if np.var(y) < 1e-12:
        mean_y = float(np.mean(y))
        def predict_const(X_new):
            X_new = np.asarray(X_new, dtype=float)
            return np.full(X_new.shape, mean_y)
        return predict_const

    # If only 1 input dimension, use powerful 1D methods
    if d == 1:
        x = X.flatten()
        y_flat = y.flatten()

        # Sort by x
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y_flat[sort_idx]

        # Handle duplicates
        unique_x, inverse_indices, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
        if len(unique_x) != len(x_sorted):
            y_sum = np.zeros_like(unique_x, dtype=float)
            np.add.at(y_sum, inverse_indices, y_sorted)
            y_sorted = y_sum / counts
            x_sorted = unique_x

        if len(x_sorted) < 2:
            mean_y = float(np.mean(y_flat))
            def predict_single(X_new):
                X_new = np.asarray(X_new, dtype=float)
                return np.full(X_new.shape, mean_y)
            return predict_single

        x_min = float(np.min(x_sorted))
        x_max = float(np.max(x_sorted))
        x_range = x_max - x_min

        # Build candidate models. Each candidate is (name, F_matrix_or_None, interpolator_or_None)
        # For linear-algebra candidates, F is the design matrix.
        # For interpolator candidates, F is None and we store the interpolator.
        candidates = []

        # 1. Polynomial basis (degrees 2-10)
        for deg in [2, 3, 4, 5, 6, 7, 8, 10]:
            try:
                F = np.vander(x_sorted, N=deg+1, increasing=True)
                candidates.append(('poly_%d' % deg, F, None))
            except Exception:
                pass

        # 2. Fourier basis
        for nf in [4, 6, 8, 10, 12, 15, 20]:
            try:
                cols = [np.ones(len(x_sorted))]
                for k in range(1, nf + 1):
                    cols.append(np.sin(k * x_sorted))
                    cols.append(np.cos(k * x_sorted))
                F = np.column_stack(cols)
                candidates.append(('fourier_%d' % nf, F, None))
            except Exception:
                pass

        # 3. RBF basis with various centers and bandwidths
        if x_range > 0:
            for nc in [10, 15, 20, 25, 30, 40, 50]:
                for bw_mult in [1.0, 1.5, 2.0]:
                    try:
                        centers = np.linspace(x_min, x_max, nc)
                        bandwidth = x_range / nc * bw_mult
                        cols = [np.ones(len(x_sorted))]
                        for c in centers:
                            cols.append(np.exp(-((x_sorted - c) ** 2) / (2 * bandwidth ** 2)))
                        F = np.column_stack(cols)
                        candidates.append(('rbf_%d_bw%.1f' % (nc, bw_mult), F, None))
                    except Exception:
                        pass

        # 4. B-spline-like piecewise linear basis
        for nk in [10, 15, 20, 25, 30]:
            try:
                knots = np.linspace(x_min, x_max, nk + 1)
                cols = [np.ones(len(x_sorted))]
                for i in range(len(knots) - 1):
                    left = knots[i]
                    right = knots[i + 1]
                    center = (left + right) / 2
                    width = (right - left) / 2
                    if width > 0:
                        feat = np.clip(1 - np.abs(x_sorted - center) / width, 0, 1)
                        cols.append(feat)
                F = np.column_stack(cols)
                candidates.append(('spline_%d' % nk, F, None))
            except Exception:
                pass

        # 5. PCHIP interpolator (excellent for smooth non-polynomial functions)
        try:
            if np.all(np.diff(x_sorted) > 0):
                pchip_interp = PchipInterpolator(x_sorted, y_sorted)
                # Verify it produces finite values
                test_pts = x_sorted[::max(1, len(x_sorted)//10)]
                test_vals = pchip_interp(test_pts)
                if np.all(np.isfinite(test_vals)):
                    candidates.append(('pchip', None, pchip_interp))
        except Exception:
            pass

        # 6. Cubic Spline interpolator
        try:
            if np.all(np.diff(x_sorted) > 0):
                cs_interp = CubicSpline(x_sorted, y_sorted)
                test_pts = x_sorted[::max(1, len(x_sorted)//10)]
                test_vals = cs_interp(test_pts)
                if np.all(np.isfinite(test_vals)):
                    candidates.append(('cubic_spline', None, cs_interp))
        except Exception:
            pass

        # 7. Linear interpolation (very robust fallback)
        try:
            lin_interp = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
            candidates.append(('linear_interp', None, lin_interp))
        except Exception:
            pass

        # Cross-validation to select best candidate
        best_mse = np.inf
        best_candidate = None

        n_folds = 5
        indices = np.arange(len(x_sorted))
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = len(indices) // n_folds

        for name, F, interp in candidates:
            if time.time() > deadline - 10:
                break
            try:
                mse_sum = 0.0
                valid = True
                for i in range(n_folds):
                    val_start = i * fold_size
                    val_end = val_start + fold_size if i < n_folds - 1 else len(indices)
                    val_idx = indices[val_start:val_end]
                    train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                    if interp is not None:
                        # Interpolator candidate: refit on train fold
                        x_tr = x_sorted[train_idx]
                        y_tr = y_sorted[train_idx]
                        x_va = x_sorted[val_idx]
                        y_va = y_sorted[val_idx]

                        # Rebuild interpolator for this fold
                        try:
                            if name == 'pchip':
                                if len(x_tr) < 2 or not np.all(np.diff(x_tr) > 0):
                                    valid = False
                                    break
                                fold_interp = PchipInterpolator(x_tr, y_tr)
                            elif name == 'cubic_spline':
                                if len(x_tr) < 4 or not np.all(np.diff(x_tr) > 0):
                                    valid = False
                                    break
                                fold_interp = CubicSpline(x_tr, y_tr)
                            elif name == 'linear_interp':
                                if len(x_tr) < 2:
                                    valid = False
                                    break
                                fold_interp = interp1d(x_tr, y_tr, kind='linear', fill_value='extrapolate')
                            else:
                                valid = False
                                break

                            y_pred_cv = fold_interp(x_va)
                            if not np.all(np.isfinite(y_pred_cv)):
                                valid = False
                                break
                            mse_sum += np.mean((y_pred_cv - y_va) ** 2)
                        except Exception:
                            valid = False
                            break
                    else:
                        # Linear-algebra candidate
                        X_train_cv = F[train_idx]
                        y_train_cv = y_sorted[train_idx]
                        X_val_cv = F[val_idx]
                        y_val_cv = y_sorted[val_idx]

                        p = X_train_cv.shape[1]
                        # Use lstsq which is more robust; add small regularization via SVD
                        try:
                            w_cv, residuals, rank, sv = np.linalg.lstsq(X_train_cv, y_train_cv, rcond=1e-12)
                            y_pred_cv = X_val_cv @ w_cv
                        except Exception:
                            # Fallback: ridge
                            reg = 1e-6
                            A = X_train_cv.T @ X_train_cv + reg * np.eye(p)
                            b = X_train_cv.T @ y_train_cv
                            try:
                                w_cv = np.linalg.solve(A, b)
                                y_pred_cv = X_val_cv @ w_cv
                            except Exception:
                                valid = False
                                break

                        if not np.all(np.isfinite(y_pred_cv)):
                            valid = False
                            break
                        mse_sum += np.mean((y_pred_cv - y_val_cv) ** 2)

                if valid:
                    mse_avg = mse_sum / n_folds
                    if mse_avg < best_mse:
                        best_mse = mse_avg
                        best_candidate = (name, F, interp)
            except Exception:
                continue

        # If no candidate worked, fall back to linear interpolation
        if best_candidate is None:
            try:
                lin_interp = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
                def predict_fallback(X_new):
                    X_new = np.asarray(X_new, dtype=float)
                    x_eval = X_new.flatten()
                    y_pred = lin_interp(x_eval)
                    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                    return y_pred.reshape(X_new.shape)
                return predict_fallback
            except Exception:
                mean_y = float(np.mean(y_flat))
                def predict_mean(X_new):
                    X_new = np.asarray(X_new, dtype=float)
                    return np.full(X_new.shape, mean_y)
                return predict_mean

        best_name, best_F, best_interp = best_candidate

        # Final fit on all data
        if best_interp is not None:
            # Interpolator-based model
            if best_name == 'pchip':
                final_interp = PchipInterpolator(x_sorted, y_sorted)
            elif best_name == 'cubic_spline':
                final_interp = CubicSpline(x_sorted, y_sorted)
            elif best_name == 'linear_interp':
                final_interp = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
            else:
                final_interp = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')

            def predict_interp(X_new):
                X_new = np.asarray(X_new, dtype=float)
                x_eval = X_new.flatten()
                try:
                    y_pred = final_interp(x_eval)
                except Exception:
                    y_pred = np.full_like(x_eval, float(np.mean(y_sorted)))
                y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                return y_pred.reshape(X_new.shape)

            return predict_interp
        else:
            # Linear-algebra based model
            p = best_F.shape[1]
            try:
                best_w, residuals, rank, sv = np.linalg.lstsq(best_F, y_sorted, rcond=1e-12)
            except Exception:
                reg = 1e-6
                A = best_F.T @ best_F + reg * np.eye(p)
                b = best_F.T @ y_sorted
                try:
                    best_w = np.linalg.solve(A, b)
                except Exception:
                    best_w = np.zeros(p)

            # Store parameters for prediction
            x_min_pred = x_min
            x_max_pred = x_max
            x_range_pred = x_range

            def predict_la(X_new):
                X_new = np.asarray(X_new, dtype=float)
                x_eval = X_new.flatten()
                try:
                    if 'poly' in best_name:
                        deg = int(best_name.split('_')[1])
                        F_new = np.vander(x_eval, N=deg+1, increasing=True)
                    elif 'fourier' in best_name:
                        nf = int(best_name.split('_')[1])
                        cols = [np.ones(len(x_eval))]
                        for k in range(1, nf + 1):
                            cols.append(np.sin(k * x_eval))
                            cols.append(np.cos(k * x_eval))
                        F_new = np.column_stack(cols)
                    elif 'rbf' in best_name:
                        parts = best_name.split('_')
                        nc = int(parts[1])
                        bw_mult = float(parts[2].replace('bw', ''))
                        centers = np.linspace(x_min_pred, x_max_pred, nc)
                        bandwidth = x_range_pred / nc * bw_mult
                        cols = [np.ones(len(x_eval))]
                        for c in centers:
                            cols.append(np.exp(-((x_eval - c) ** 2) / (2 * bandwidth ** 2)))
                        F_new = np.column_stack(cols)
                    elif 'spline' in best_name:
                        nk = int(best_name.split('_')[1])
                        knots = np.linspace(x_min_pred, x_max_pred, nk + 1)
                        cols = [np.ones(len(x_eval))]
                        for i in range(len(knots) - 1):
                            left = knots[i]
                            right = knots[i + 1]
                            center = (left + right) / 2
                            width = (right - left) / 2
                            if width > 0:
                                feat = np.clip(1 - np.abs(x_eval - center) / width, 0, 1)
                                cols.append(feat)
                        F_new = np.column_stack(cols)
                    else:
                        F_new = np.column_stack([np.ones(len(x_eval)), x_eval])

                    y_pred = F_new @ best_w
                    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                    return y_pred.reshape(X_new.shape)
                except Exception:
                    return np.full(X_new.shape, float(np.mean(y_sorted)))

            return predict_la

    # Multi-dimensional fallback: polynomial regression with CV
    best_mse = np.inf
    best_w = None
    best_degree = 2

    for deg in [2, 3, 4]:
        try:
            cols = [np.ones(n)]
            for i in range(d):
                cols.append(X[:, i])
            if deg >= 2:
                for i in range(d):
                    for j in range(i, d):
                        cols.append(X[:, i] * X[:, j])
            if deg >= 3:
                for i in range(d):
                    for j in range(d):
                        for k in range(d):
                            cols.append(X[:, i] * X[:, j] * X[:, k])

            F = np.column_stack(cols)

            n_folds = 5
            indices = np.arange(n)
            np.random.seed(42)
            np.random.shuffle(indices)
            fold_size = n // n_folds

            mse_sum = 0.0
            for i in range(n_folds):
                val_start = i * fold_size
                val_end = val_start + fold_size if i < n_folds - 1 else n
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                X_train_cv = F[train_idx]
                y_train_cv = y[train_idx]
                X_val_cv = F[val_idx]
                y_val_cv = y[val_idx]

                p = X_train_cv.shape[1]
                try:
                    w_cv, _, _, _ = np.linalg.lstsq(X_train_cv, y_train_cv, rcond=1e-12)
                    y_pred_cv = X_val_cv @ w_cv
                except Exception:
                    reg = 1e-6
                    A = X_train_cv.T @ X_train_cv + reg * np.eye(p)
                    b = X_train_cv.T @ y_train_cv
                    w_cv = np.linalg.solve(A, b)
                    y_pred_cv = X_val_cv @ w_cv

                mse_sum += np.mean((y_pred_cv - y_val_cv) ** 2)

            mse_avg = mse_sum / n_folds
            if mse_avg < best_mse:
                best_mse = mse_avg
                best_degree = deg
        except Exception:
            continue

    # Final fit
    cols = [np.ones(n)]
    for i in range(d):
        cols.append(X[:, i])
    if best_degree >= 2:
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
    if best_degree >= 3:
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])

    F = np.column_stack(cols)
    p = F.shape[1]
    try:
        best_w, _, _, _ = np.linalg.lstsq(F, y, rcond=1e-12)
    except Exception:
        reg = 1e-6
        A = F.T @ F + reg * np.eye(p)
        b = F.T @ y
        best_w = np.linalg.solve(A, b)

    def predict_md(X_new):
        X_new = np.asarray(X_new, dtype=float)
        cols = [np.ones(X_new.shape[0])]
        for i in range(d):
            cols.append(X_new[:, i])
        if best_degree >= 2:
            for i in range(d):
                for j in range(i, d):
                    cols.append(X_new[:, i] * X_new[:, j])
        if best_degree >= 3:
            for i in range(d):
                for j in range(d):
                    for k in range(d):
                        cols.append(X_new[:, i] * X_new[:, j] * X_new[:, k])
        F_new = np.column_stack(cols)
        y_pred = F_new @ best_w
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        return y_pred

    return predict_md
