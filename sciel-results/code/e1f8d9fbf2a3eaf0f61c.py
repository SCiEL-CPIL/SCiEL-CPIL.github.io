"""Symbolic regression over 4 hidden 1-D equations using adaptive basis + local smoothing."""
import numpy as np
import time
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 330.0
    margin = 10.0  # safety margin in seconds

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    d = X.shape[1]

    # If d > 1, try to reduce to 1D by checking collinearity
    if d > 1:
        # Check if all columns are nearly identical or linearly related
        # Use PCA to see if one component explains most variance
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        if S[0] / (S.sum() + 1e-12) > 0.99:
            # Project onto first principal component
            X = (X_centered @ Vt[0]).reshape(-1, 1)
        else:
            # Use first column as proxy (risky but better than nothing)
            X = X[:, :1]

    x = X.flatten()

    # Sort by x for interpolation-based methods
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Build candidate basis functions and select via greedy forward selection
    # with early stopping based on validation-like criterion (use training MSE with regularization)

    def make_basis(x_vals, n_points=200):
        """Generate a diverse set of candidate basis functions."""
        bases = []
        names = []

        # Polynomials up to degree 8
        for deg in range(9):
            bases.append(x_vals ** deg)
            names.append(f'poly_{deg}')

        # Sine/cosine with various frequencies
        for freq in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
            bases.append(np.sin(freq * x_vals))
            names.append(f'sin_{freq}')
            bases.append(np.cos(freq * x_vals))
            names.append(f'cos_{freq}')

        # Exponentials
        for rate in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
            bases.append(np.exp(rate * x_vals))
            names.append(f'exp_{rate}')

        # Rational-like: 1/(1+x^2), etc.
        bases.append(1.0 / (1.0 + x_vals ** 2))
        names.append('rational1')
        bases.append(1.0 / (1.0 + x_vals ** 4))
        names.append('rational2')

        # Sigmoid-like
        for c in [-2, -1, 0, 1, 2]:
            bases.append(1.0 / (1.0 + np.exp(-x_vals + c)))
            names.append(f'sigmoid_{c}')

        # Gaussian bumps
        for mu in np.linspace(-3, 3, 7):
            for sigma in [0.5, 1.0, 2.0]:
                bases.append(np.exp(-0.5 * ((x_vals - mu) / sigma) ** 2))
                names.append(f'gauss_mu{mu}_sig{sigma}')

        # Tanh
        for scale in [0.5, 1.0, 2.0]:
            bases.append(np.tanh(scale * x_vals))
            names.append(f'tanh_{scale}')

        # Absolute value and sign
        bases.append(np.abs(x_vals))
        names.append('abs')
        bases.append(np.sign(x_vals))
        names.append('sign')

        # x * sin(x), x * cos(x)
        bases.append(x_vals * np.sin(x_vals))
        names.append('x_sin_x')
        bases.append(x_vals * np.cos(x_vals))
        names.append('x_cos_x')

        # x^2 * sin(x), etc.
        bases.append(x_vals ** 2 * np.sin(x_vals))
        names.append('x2_sin_x')

        # Log-like (careful with domain)
        safe_x = np.where(np.abs(x_vals) > 0.01, x_vals, np.sign(x_vals) * 0.01)
        bases.append(np.log(np.abs(safe_x) + 1e-10))
        names.append('log_abs')

        return np.column_stack(bases), names

    # Generate basis
    B, basis_names = make_basis(x_sorted)

    # Handle any NaN/Inf in basis
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize basis columns for better conditioning
    B_mean = B.mean(axis=0, keepdims=True)
    B_std = B.std(axis=0, keepdims=True)
    B_std[B_std < 1e-12] = 1.0
    B_std = B_std  # keep original for now, standardize in solve

    # Greedy forward selection with ridge regularization
    # Use a small ridge to avoid overfitting
    ridge = 1e-6

    def compute_mse(weights, X_b, y_true):
        pred = X_b @ weights
        return np.mean((pred - y_true) ** 2)

    # Start with intercept
    selected = [0]  # index 0 is poly_0 = ones
    best_mse = np.inf
    max_features = 12
    min_mse_improvement = 1e-10

    # Precompute B.T @ B and B.T @ y for efficiency
    # But with greedy selection, we need to recompute as we add features
    # For 200 points and ~50 basis functions, this is fine

    for step in range(max_features):
        best_feat = -1
        best_mse_step = best_mse
        best_w_step = None

        current_B = B[:, selected]

        for feat_idx in range(B.shape[1]):
            if feat_idx in selected:
                continue

            candidate_selected = selected + [feat_idx]
            cand_B = B[:, candidate_selected]

            # Solve ridge regression
            A = cand_B.T @ cand_B + ridge * np.eye(len(candidate_selected))
            b = cand_B.T @ y_sorted
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue

            pred = cand_B @ w
            mse = np.mean((pred - y_sorted) ** 2)

            if mse < best_mse_step - min_mse_improvement:
                best_mse_step = mse
                best_feat = feat_idx
                best_w_step = w

        if best_feat == -1:
            break

        selected.append(best_feat)
        best_mse = best_mse_step
        best_w = best_w_step

        # Check time
        if time.time() > deadline - margin:
            break

    # Final model using selected features
    final_B = B[:, selected]
    A = final_B.T @ final_B + ridge * np.eye(len(selected))
    b = final_B.T @ y_sorted
    try:
        final_w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback: use just intercept
        final_w = np.array([np.mean(y_sorted)])
        final_B = np.ones((len(x_sorted), 1))

    # Also try a spline-based approach and pick the better one
    spline_pred_train = None
    if time.time() < deadline - margin:
        try:
            # Use UnivariateSpline with moderate smoothing
            s = UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted) * 0.1, k=4)
            spline_pred_train = s(x_sorted)
            spline_mse = np.mean((spline_pred_train - y_sorted) ** 2)
        except Exception:
            spline_pred_train = None

    # Choose between polynomial/basis model and spline
    basis_pred_train = final_B @ final_w
    basis_mse = np.mean((basis_pred_train - y_sorted) ** 2)

    use_spline = False
    if spline_pred_train is not None:
        spline_mse = np.mean((spline_pred_train - y_sorted) ** 2)
        if spline_mse < basis_mse:
            use_spline = True

    # Create prediction function
    if use_spline:
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test.flatten()
            return s(x_test)
    else:
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            if X_test.ndim == 1:
                X_test = X_test.reshape(-1, 1)
            x_test = X_test.flatten()

            # Build basis for test points
            B_test, _ = make_basis(x_test)
            B_test = np.nan_to_num(B_test, nan=0.0, posinf=0.0, neginf=0.0)

            # Use only the selected columns
            B_test_sel = B_test[:, selected]
            return B_test_sel @ final_w

    return predict
