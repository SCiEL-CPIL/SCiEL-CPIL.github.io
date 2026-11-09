# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    # 330s kill, 10 equations total, so ~33s per equation.
    # We will use a strict deadline to ensure we return something.
    # Let's allow up to 25 seconds of computation per equation to be safe.
    deadline = start_time + 25.0
    
    n, d = X_train.shape
    y_var = np.var(y_train)
    if y_var < 1e-12:
        y_mean = np.mean(y_train)
        return lambda X: np.full(len(X), y_mean)

    def time_ok():
        return time.time() < deadline

    # Candidate feature generators
    # We want a rich set of features to capture potential nonlinearities.
    
    def gen_polynomial_features(X, degree):
        """Generate polynomial features up to 'degree'."""
        cols = [np.ones(len(X))]
        # Linear
        for j in range(X.shape[1]):
            cols.append(X[:, j])
        # Higher orders
        if degree >= 2:
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
        if degree >= 3:
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    for k in range(j, X.shape[1]):
                        cols.append(X[:, i] * X[:, j] * X[:, k])
        return np.column_stack(cols)

    def gen_trig_features(X):
        """Generate sine/cosine features."""
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            cols.append(X[:, j])
            cols.append(np.sin(X[:, j]))
            cols.append(np.cos(X[:, j]))
            # Maybe higher freq?
            cols.append(np.sin(2 * X[:, j]))
            cols.append(np.cos(2 * X[:, j]))
        # Interactions of linear terms
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                cols.append(X[:, i] * X[:, j])
        return np.column_stack(cols)

    def gen_mixed_features(X):
        """Mix of polynomial and trig."""
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            x = X[:, j]
            cols.append(x)
            cols.append(x**2)
            cols.append(np.sin(x))
            cols.append(np.cos(x))
            cols.append(np.sqrt(x)) # Input ranges are [1, ...] so sqrt is safe
            cols.append(np.log1p(x))
        # Interactions
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                cols.append(X[:, i] * X[:, j])
        return np.column_stack(cols)

    def gen_rbf_features(X, centers, bandwidths):
        """Radial Basis Functions. Centers picked from quantiles."""
        cols = []
        for j in range(X.shape[1]):
            x = X[:, j]
            # Pick 3 centers: 25%, 50%, 75% quantiles
            centers = np.quantile(x, [0.25, 0.5, 0.75])
            # Use a fixed bandwidth relative to range
            bandwidth = (np.max(x) - np.min(x)) / 3.0
            if bandwidth < 1e-6:
                bandwidth = 1.0
            for c in centers:
                cols.append(np.exp(-((x - c) ** 2) / (2 * bandwidth ** 2)))
        if not cols:
             cols = [np.ones(len(X))]
        return np.column_stack(cols)

    # List of strategies: (name, feature_func)
    strategies = [
        ("poly2", lambda X: gen_polynomial_features(X, 2)),
        ("poly3", lambda X: gen_polynomial_features(X, 3)),
        ("trig", gen_trig_features),
        ("mixed", gen_mixed_features),
        ("rbf", gen_rbf_features),
        # Linear only as a baseline
        ("linear", lambda X: np.column_stack([np.ones(len(X)), X])),
    ]

    best_mse = np.inf
    best_predict = None
    best_name = "linear"

    # Sort strategies? Maybe simpler ones first?
    # Let's just iterate.

    for name, func in strategies:
        if not time_ok():
            break
        
        try:
            F_train = func(X_train)
            
            # Check for numerical issues
            if not np.all(np.isfinite(F_train)):
                continue
                
            # Solve least squares
            # Use ridge regression for stability if matrix is ill-conditioned
            # np.linalg.lstsq is usually fine, but let's add a tiny ridge if needed?
            # Actually, lstsq with rcond=None is robust.
            
            w, residuals, rank, sv = np.linalg.lstsq(F_train, y_train, rcond=None)
            
            pred_train = F_train @ w
            mse = np.mean((pred_train - y_train) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_name = name
                # Capture w and func for the closure
                # Note: func needs to be callable with X later.
                # We need to be careful with closures in loops.
                # We can store the function object itself.
                best_predict = lambda X, f=func, w=w: f(X) @ w
                
        except Exception as e:
            # Skip if it fails
            continue

    # If we ran out of time or all failed, fallback to simple linear
    if best_predict is None:
        try:
            F = np.column_stack([np.ones(n), X_train])
            w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
            best_predict = lambda X: np.column_stack([np.ones(len(X)), X]) @ w
        except:
            mean_y = np.mean(y_train)
            best_predict = lambda X: np.full(len(X), mean_y)

    # Wrap in a safe predictor
    def safe_predict(X):
        try:
            result = best_predict(X)
            result = np.asarray(result, dtype=float)
            if not np.all(np.isfinite(result)):
                # Replace non-finite with mean
                result = np.nan_to_num(result, nan=np.mean(y_train), posinf=np.max(y_train), neginf=np.min(y_train))
            return result
        except Exception:
            return np.full(len(X), np.mean(y_train))

    return safe_predict
# EVOLVE-BLOCK-END
