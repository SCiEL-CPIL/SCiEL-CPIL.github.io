import numpy as np
import time

# EVOLVE-BLOCK-START

def fit(X_train, y_train):
    start_time = time.time()
    TIME_LIMIT = 300.0
    SAFETY_MARGIN = 15.0

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    n, d = X_train.shape
    u_train = X_train[:, 0]

    # Sort by x for spline and potential other uses
    sort_idx = np.argsort(u_train)
    u_sorted = u_train[sort_idx]
    y_sorted = y_train[sort_idx]

    # We will evaluate a set of candidate models and pick the best one based on
    # 5-fold cross-validation MSE.
    # We use a "recipe" approach: each candidate is a function that takes (u, y)
    # and returns a predict function.

    candidates = []

    # 1. Polynomial Regression (Degrees 1-12)
    # High degrees can capture complex shapes if the true function is smooth.
    for deg in range(1, 13):
        def make_poly_fitter(deg):
            def fit_func(u, y):
                # Use normalized x for better conditioning?
                # Let's try raw x first, but if it fails, normalize.
                # Actually, for polynomials, normalization helps a lot.
                # But the domain is known.
                # Let's use a simple centering/scaling if range is large.
                # For now, use raw x.
                V = np.vander(u, N=deg + 1, increasing=True)
                # Ridge regression for stability
                lam = 1e-6
                A = V.T @ V + lam * np.eye(deg + 1)
                b = V.T @ y
                try:
                    coeffs = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    coeffs = np.linalg.lstsq(A, b, rcond=None)[0]

                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    u = X[:, 0]
                    V_test = np.vander(u, N=deg + 1, increasing=True)
                    return V_test @ coeffs
                return predict
            return fit_func
        candidates.append(make_poly_fitter(deg))

    # 2. Fourier Series
    # Try different fundamental frequencies
    base_freqs = [1.0, 0.5, 2.0, np.pi, 2 * np.pi, 1.0 / (2 * np.pi), 0.1, 5.0]
    for base_f in base_freqs:
        for max_k in range(1, 15):
            def make_fourier_fitter(base_f, max_k):
                def fit_func(u, y):
                    cols = [np.ones(len(u))]
                    for k in range(1, max_k + 1):
                        cols.append(np.cos(k * base_f * u))
                        cols.append(np.sin(k * base_f * u))
                    V = np.column_stack(cols)
                    lam = 1e-6
                    A = V.T @ V + lam * np.eye(V.shape[1])
                    b = V.T @ y
                    try:
                        coeffs = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]

                    def predict(X):
                        X = np.asarray(X, dtype=np.float64)
                        if X.ndim == 1:
                            X = X.reshape(-1, 1)
                        u = X[:, 0]
                        cols = [np.ones(len(u))]
                        for k in range(1, max_k + 1):
                            cols.append(np.cos(k * base_f * u))
                            cols.append(np.sin(k * base_f * u))
                        V_test = np.column_stack(cols)
                        return V_test @ coeffs
                    return predict
                return fit_func
            candidates.append(make_fourier_fitter(base_f, max_k))

    # 3. Exponential / Logarithmic / Power / Rational
    # These are specific functional forms that might match hidden equations.

    # 3a. Exponential y = a * exp(b * x) + c
    # Linearize: y - c = a * exp(b * x)
    # Grid search for b and c? Too expensive.
    # Try fixed c=0 first, then maybe c=mean(y).
    # Or use a single basis: exp(b*x).
    # Let's try y = a * exp(b * x) + c with grid search on b, and linear fit for a, c.
    b_vals = np.linspace(-3, 3, 40)
    for b in b_vals:
        def make_exp_fitter(b):
            def fit_func(u, y):
                # y = a * exp(b*u) + c
                # Design matrix: [exp(b*u), 1]
                V = np.column_stack([np.exp(b * u), np.ones(len(u))])
                lam = 1e-6
                A = V.T @ V + lam * np.eye(2)
                vec = V.T @ y
                try:
                    coeffs = np.linalg.solve(A, vec)
                except np.linalg.LinAlgError:
                    coeffs = np.linalg.lstsq(A, vec, rcond=None)[0]
                a, c = coeffs

                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    u = X[:, 0]
                    return a * np.exp(b * u) + c
                return predict
            return fit_func
        candidates.append(make_exp_fitter(b))

    # 3b. Power Law y = a * x^b + c
    # Only if x > 0
    if np.min(u_train) > 0:
        b_vals = np.linspace(-3, 3, 40)
        for b in b_vals:
            def make_pow_fitter(b):
                def fit_func(u, y):
                    # y = a * u^b + c
                    V = np.column_stack([u ** b, np.ones(len(u))])
                    if not np.all(np.isfinite(V)):
                        return lambda X: np.zeros(len(X))
                    lam = 1e-6
                    A = V.T @ V + lam * np.eye(2)
                    vec = V.T @ y
                    try:
                        coeffs = np.linalg.solve(A, vec)
                    except np.linalg.LinAlgError:
                        coeffs = np.linalg.lstsq(A, vec, rcond=None)[0]
                    a, c = coeffs

                    def predict(X):
                        X = np.asarray(X, dtype=np.float64)
                        if X.ndim == 1:
                            X = X.reshape(-1, 1)
                        u = X[:, 0]
                        return a * (u ** b) + c
                    return predict
                return fit_func
            candidates.append(make_pow_fitter(b))

    # 3c. Logarithmic y = a * log(x + b) + c
    # b must be > -min(x)
    min_x = np.min(u_train)
    b_vals = np.linspace(min_x + 0.1, min_x + 5.0, 20)
    for b in b_vals:
        def make_log_fitter(b):
            def fit_func(u, y):
                # y = a * log(u + b) + c
                V = np.column_stack([np.log(u + b), np.ones(len(u))])
                if not np.all(np.isfinite(V)):
                    return lambda X: np.zeros(len(X))
                lam = 1e-6
                A = V.T @ V + lam * np.eye(2)
                vec = V.T @ y
                try:
                    coeffs = np.linalg.solve(A, vec)
                except np.linalg.LinAlgError:
                    coeffs = np.linalg.lstsq(A, vec, rcond=None)[0]
                a, c = coeffs

                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    u = X[:, 0]
                    return a * np.log(u + b) + c
                return predict
            return fit_func
        candidates.append(make_log_fitter(b))

    # 3d. Rational y = a / (x + b) + c
    b_vals = np.linspace(min_x + 0.1, min_x + 5.0, 20)
    for b in b_vals:
        def make_rat_fitter(b):
            def fit_func(u, y):
                # y = a / (u + b) + c
                V = np.column_stack([1.0 / (u + b), np.ones(len(u))])
                if not np.all(np.isfinite(V)):
                    return lambda X: np.zeros(len(X))
                lam = 1e-6
                A = V.T @ V + lam * np.eye(2)
                vec = V.T @ y
                try:
                    coeffs = np.linalg.solve(A, vec)
                except np.linalg.LinAlgError:
                    coeffs = np.linalg.lstsq(A, vec, rcond=None)[0]
                a, c = coeffs

                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    u = X[:, 0]
                    return a / (u + b) + c
                return predict
            return fit_func
        candidates.append(make_rat_fitter(b))

    # 4. Spline Interpolation
    # Use UnivariateSpline with different smoothing factors s
    from scipy.interpolate import UnivariateSpline
    s_vals = [0, 1, 10, 100, 1000, 10000, 100000]
    for s in s_vals:
        def make_spline_fitter(s):
            def fit_func(u, y):
                try:
                    sort_idx = np.argsort(u)
                    u_s = u[sort_idx]
                    y_s = y[sort_idx]
                    spl = UnivariateSpline(u_s, y_s, s=s, k=3)
                    def predict(X):
                        X = np.asarray(X, dtype=np.float64)
                        if X.ndim == 1:
                            X = X.reshape(-1, 1)
                        u = X[:, 0]
                        return spl(u)
                    return predict
                except Exception:
                    return lambda X: np.zeros(len(X))
            return fit_func
        candidates.append(make_spline_fitter(s))

    # 5. Gaussian Process / RBF Kernel Regression
    # y(x) = sum_i alpha_i * exp(-gamma * (x - x_i)^2)
    # Solve for alpha using kernel matrix K.
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for gamma in gammas:
        def make_rbf_fitter(gamma):
            def fit_func(u, y):
                diff = u[:, np.newaxis] - u[np.newaxis, :]
                K = np.exp(-gamma * diff ** 2)
                lam = 1e-6
                A = K + lam * np.eye(len(u))
                try:
                    alpha = np.linalg.solve(A, y)
                except np.linalg.LinAlgError:
                    alpha = np.linalg.lstsq(A, y, rcond=None)[0]

                u_train_local = u
                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    u = X[:, 0]
                    diff = u[:, np.newaxis] - u_train_local[np.newaxis, :]
                    K_test = np.exp(-gamma * diff ** 2)
                    return K_test @ alpha
                return predict
            return fit_func
        candidates.append(make_rbf_fitter(gamma))

    # 6. Matérn 5/2 Kernel GP (as per directive)
    # K(x, x') = (1 + sqrt(5)*d + 5/3*d^2) * exp(-sqrt(5)*d)
    # where d = |x - x'| / l
    # We will grid search over length-scale l and signal variance sigma_f.
    # For simplicity, we can fix sigma_f=1 and search l, or search both.
    # Let's search l in log space.
    l_vals = np.logspace(-2, 2, 10)
    for l in l_vals:
        def make_matern_fitter(l):
            def fit_func(u, y):
                def kernel(u1, u2):
                    d = np.abs(u1[:, np.newaxis] - u2[np.newaxis, :]) / l
                    return (1 + np.sqrt(5) * d + 5 / 3 * d ** 2) * np.exp(-np.sqrt(5) * d)

                K = kernel(u, u)
                lam = 1e-6
                A = K + lam * np.eye(len(u))
                try:
                    alpha = np.linalg.solve(A, y)
                except np.linalg.LinAlgError:
                    alpha = np.linalg.lstsq(A, y, rcond=None)[0]

                u_train_local = u
                def predict(X):
                    X = np.asarray(X, dtype=np.float64)
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    u = X[:, 0]
                    K_test = kernel(u, u_train_local)
                    return K_test @ alpha
                return predict
            return fit_func
        candidates.append(make_matern_fitter(l))

    # Evaluate candidates using 5-Fold CV
    rng = np.random.RandomState(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, 5)

    best_mse = np.inf
    best_fit_func = None

    # To speed up, we can pre-compute some things, but let's just loop.
    # If we run out of time, we break and use the best so far.
    # If no model was evaluated, we use a simple linear regression or mean.

    # Fallback model: Mean
    def make_mean_fitter():
        def fit_func(u, y):
            mean_y = np.mean(y)
            def predict(X):
                X = np.asarray(X, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                return np.full(X.shape[0], mean_y)
            return predict
        return fit_func
    candidates.append(make_mean_fitter())

    # Evaluate
    for i, fit_func in enumerate(candidates):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break

        fold_mses = []
        for fold_idx in range(5):
            test_idx = folds[fold_idx]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != fold_idx])

            u_tr = u_train[train_idx]
            y_tr = y_train[train_idx]
            u_te = u_train[test_idx]
            y_te = y_train[test_idx]

            try:
                predict_func = fit_func(u_tr, y_tr)
                y_pred = predict_func(u_te)
                if not np.all(np.isfinite(y_pred)):
                    mse = np.inf
                else:
                    mse = np.mean((y_pred - y_te) ** 2)
                fold_mses.append(mse)
            except Exception:
                fold_mses.append(np.inf)

        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_fit_func = fit_func

    # Refit best model on all data
    if best_fit_func is not None:
        final_predict = best_fit_func(u_train, y_train)
    else:
        # Fallback
        mean_y = np.mean(y_train)
        final_predict = lambda X: np.full(len(X), mean_y)

    # Wrap final predict to ensure 2D input handling
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        u = X[:, 0]
        y_pred = final_predict(u)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
        return y_pred

    return predict

# EVOLVE-BLOCK-END
