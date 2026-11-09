"""Symbolic regression over 10 hidden equations."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = start_time + 285.0  # safety margin within 330s

    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)

    def predict_const(X):
        return np.full(len(X), y_mean)

    if y_var < 1e-14:
        return predict_const

    def time_ok():
        return time.time() < DEADLINE

    # Build diverse feature sets
    def build_features(X):
        """Build a rich feature matrix."""
        n_samples = X.shape[0]
        d = X.shape[1]
        cols = [np.ones(n_samples)]

        # Linear terms
        for j in range(d):
            cols.append(X[:, j])

        # Quadratic terms (including cross terms)
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])

        # Cubic terms (diagonal only to keep size manageable)
        for j in range(d):
            cols.append(X[:, j] ** 3)

        # Trigonometric
        for j in range(d):
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))

        # Exponential (clipped to avoid overflow)
        for j in range(d):
            x_clipped = np.clip(X[:, j], -50, 50)
            cols.append(np.exp(x_clipped))
            cols.append(np.exp(-x_clipped))

        # Logarithmic (only for positive values)
        for j in range(d):
            x_pos = np.maximum(X[:, j], 1e-10)
            cols.append(np.log(x_pos))

        # Square root
        for j in range(d):
            x_pos = np.maximum(X[:, j], 1e-10)
            cols.append(np.sqrt(x_pos))

        # Reciprocal
        for j in range(d):
            x_safe = np.where(np.abs(X[:, j]) > 1e-10, X[:, j], 1e-10)
            cols.append(1.0 / x_safe)

        # Pairwise interactions with trig
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    cols.append(np.sin(X[:, i]) * X[:, j])
                    cols.append(np.cos(X[:, i]) * X[:, j])
                    cols.append(X[:, i] * np.sin(X[:, j]))
                    cols.append(X[:, i] * np.cos(X[:, j]))

        return np.column_stack(cols)

    # Simple linear model as fallback
    def fit_linear(X, y):
        A = np.column_stack([np.ones(len(X)), X])
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        return lambda X_: np.column_stack([np.ones(len(X_)), X_]) @ w

    # Ridge regression
    def fit_ridge(X, y, alpha=1.0):
        A = np.column_stack([np.ones(len(X)), X])
        n_feat = A.shape[1]
        I = np.eye(n_feat)
        I[0, 0] = 0  # don't regularize intercept
        w = np.linalg.solve(A.T @ A + alpha * I, A.T @ y)
        return lambda X_: np.column_stack([np.ones(len(X_)), X_]) @ w

    # Try different approaches and pick the best
    best_predict = predict_const
    best_score = np.inf

    # First try simple linear
    try:
        pred_linear = fit_linear(X_train, y_train)
        train_score = np.mean((pred_linear(X_train) - y_train) ** 2)
        if train_score < best_score:
            best_score = train_score
            best_predict = pred_linear
    except:
        pass

    if not time_ok():
        return best_predict

    # Build rich features
    F_train = None
    try:
        F_train = build_features(X_train)
        # Remove columns with NaN/Inf
        mask = np.all(np.isfinite(F_train), axis=0)
        F_train = F_train[:, mask]

        # Use ridge regression on rich features
        pred_ridge = fit_ridge(X_train, y_train, alpha=1.0)
        train_score_ridge = np.mean((pred_ridge(X_train) - y_train) ** 2)
        if train_score_ridge < best_score:
            best_score = train_score_ridge
            best_predict = pred_ridge
    except:
        pass

    if not time_ok():
        return best_predict

    # Try LASSO-like via iterative feature selection
    if F_train is not None:
        try:
            # Forward feature selection
            n_features = F_train.shape[1]
            selected = [0]  # always include intercept
            current_score = np.mean((F_train[:, selected] @ np.linalg.lstsq(F_train[:, selected], y_train, rcond=None)[0] - y_train) ** 2)

            max_features = min(25, n_features)
            for step in range(max_features - 1):
                if not time_ok():
                    break

                best_feat = -1
                best_step_score = current_score

                for f in range(n_features):
                    if f in selected:
                        continue

                    new_selected = selected + [f]
                    try:
                        w, *_ = np.linalg.lstsq(F_train[:, new_selected], y_train, rcond=None)
                        pred = F_train[:, new_selected] @ w
                        score = np.mean((pred - y_train) ** 2)
                        if score < best_step_score - 1e-10:
                            best_step_score = score
                            best_feat = f
                    except:
                        continue

                if best_feat == -1:
                    break

                selected.append(best_feat)
                current_score = best_step_score

            # Fit final model with selected features
            w, *_ = np.linalg.lstsq(F_train[:, selected], y_train, rcond=None)

            # Store selected feature indices for prediction
            selected_indices = selected

            def predict_selected(X):
                F = build_features(X)
                mask = np.all(np.isfinite(F), axis=0)
                F = F[:, mask]
                return F[:, selected_indices] @ w

            train_score_selected = np.mean((predict_selected(X_train) - y_train) ** 2)
            if train_score_selected < best_score:
                best_score = train_score_selected
                best_predict = predict_selected
        except:
            pass

    if not time_ok():
        return best_predict

    # Try different ridge alphas
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        if not time_ok():
            break
        try:
            pred = fit_ridge(X_train, y_train, alpha=alpha)
            score = np.mean((pred(X_train) - y_train) ** 2)
            if score < best_score:
                best_score = score
                best_predict = pred
        except:
            continue

    # Try a neural network-like approach using multiple random RBF centers
    if time_ok():
        try:
            # Random RBF kernel approach
            n_centers = 50
            # Sample centers from training data
            center_indices = np.random.choice(n, n_centers, replace=False)
            centers = X_train[center_indices]

            # Compute bandwidth
            dists = np.linalg.norm(X_train[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            median_dist = np.median(dists)
            gamma = 1.0 / (median_dist ** 2 + 1e-10)

            def rbf_features(X):
                d = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
                return np.exp(-gamma * d ** 2)

            F_rbf_train = rbf_features(X_train)
            # Add bias term
            F_rbf_train = np.column_stack([np.ones(n), F_rbf_train])

            # Ridge regression on RBF features
            alpha_rbf = 1.0
            I = np.eye(F_rbf_train.shape[1])
            I[0, 0] = 0
            w_rbf = np.linalg.solve(F_rbf_train.T @ F_rbf_train + alpha_rbf * I, F_rbf_train.T @ y_train)

            def predict_rbf(X):
                F = rbf_features(X)
                F = np.column_stack([np.ones(len(X)), F])
                return F @ w_rbf

            train_score_rbf = np.mean((predict_rbf(X_train) - y_train) ** 2)
            if train_score_rbf < best_score:
                best_score = train_score_rbf
                best_predict = predict_rbf
        except:
            pass

    return best_predict
# EVOLVE-BLOCK-END
