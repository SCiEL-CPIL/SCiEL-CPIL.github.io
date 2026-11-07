"""Symbolic regression over 4 hidden equations using a hybrid kernel + basis portfolio."""
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0


def _rbf_kernel(X1, X2, gamma):
    """Compute RBF kernel matrix between X1 and X2."""
    sqdist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    sqdist = np.maximum(sqdist, 0)
    return np.exp(-gamma * sqdist)


def _fit_ridge_kernel(X, y, kernel_type, params):
    """
    Fit kernel ridge regression. Returns (predict_func, train_mse).
    kernel_type: 'rbf', 'linear', 'poly'
    """
    n = X.shape[0]
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    Xn = (X - X_mean) / X_std
    y_mean = y.mean()
    yn = y - y_mean

    if kernel_type == 'rbf':
        gamma = params.get('gamma', 1.0)
        K = _rbf_kernel(Xn, Xn, gamma)
    elif kernel_type == 'linear':
        K = Xn @ Xn.T
    elif kernel_type == 'poly':
        degree = params.get('degree', 3)
        K = (Xn @ Xn.T + 1) ** degree
    else:
        return None, np.inf

    lam = params.get('lam', 1e-3)
    K_reg = K + lam * np.eye(n)

    try:
        alpha = np.linalg.solve(K_reg, yn)
    except np.linalg.LinAlgError:
        try:
            alpha = np.linalg.lstsq(K_reg, yn, rcond=None)[0]
        except:
            return None, np.inf

    train_pred = K @ alpha + y_mean
    train_mse = np.mean((train_pred - y) ** 2)

    def predict(X_test):
        Xn_test = (X_test - X_mean) / X_std
        if kernel_type == 'rbf':
            K_test = _rbf_kernel(Xn_test, Xn, gamma)
        elif kernel_type == 'linear':
            K_test = Xn_test @ Xn.T
        elif kernel_type == 'poly':
            degree = params.get('degree', 3)
            K_test = (Xn_test @ Xn.T + 1) ** degree
        else:
            return np.zeros(len(X_test))
        return K_test @ alpha + y_mean

    return predict, train_mse


def _fit_basis(X, y, basis_type, params):
    """
    Fit basis function regression. Returns (predict_func, train_mse).
    """
    n, d = X.shape
    x = X[:, 0] if d == 1 else X[:, 0]

    if basis_type == 'poly':
        deg = params.get('deg', 5)
        cols = [np.ones(n)]
        for k in range(1, deg + 1):
            cols.append(x ** k)
        feat = np.column_stack(cols)
        deg = params.get('deg', 5)
    elif basis_type == 'trig':
        max_k = params.get('max_k', 6)
        cols = [np.ones(n)]
        for k in range(1, max_k + 1):
            cols.append(np.sin(k * x))
            cols.append(np.cos(k * x))
        feat = np.column_stack(cols)
        max_k = params.get('max_k', 6)
    elif basis_type == 'exp_log':
        cols = [np.ones(n), np.exp(x), np.exp(-x)]
        pos_mask = x > 0
        safe_x = np.where(pos_mask, x, 1.0)
        cols.append(np.log(safe_x))
        cols.append(np.log1p(np.abs(x)))
        feat = np.column_stack(cols)
    elif basis_type == 'rational':
        safe_x = np.where(np.abs(x) > 1e-6, x, 1e-6)
        feat = np.column_stack([np.ones(n), x, 1.0 / safe_x, x ** 2, x ** 3])
    elif basis_type == 'tanh':
        feat = np.column_stack([np.ones(n), np.tanh(x), np.tanh(2 * x), np.tanh(0.5 * x), 1.0 / (1.0 + np.exp(-x))])
    elif basis_type == 'hinge':
        thresholds = params.get('thresholds', [-2, -1, 0, 1, 2])
        cols = [np.ones(n)]
        for t in thresholds:
            cols.append(np.maximum(0, x - t))
            cols.append(np.maximum(0, t - x))
        feat = np.column_stack(cols)
        thresholds = params.get('thresholds', [-2, -1, 0, 1, 2])
    else:
        return None, np.inf

    lam = params.get('lam', 1e-4)
    XtX = feat.T @ feat
    Xty = feat.T @ y
    A = XtX + lam * np.eye(feat.shape[1])
    try:
        w = np.linalg.solve(A, Xty)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, Xty, rcond=None)[0]

    train_pred = feat @ w
    train_mse = np.mean((train_pred - y) ** 2)

    def predict(X_test):
        n_test = X_test.shape[0]
        x_test = X_test[:, 0]
        if basis_type == 'poly':
            cols = [np.ones(n_test)]
            for k in range(1, deg + 1):
                cols.append(x_test ** k)
            feat_test = np.column_stack(cols)
        elif basis_type == 'trig':
            cols = [np.ones(n_test)]
            for k in range(1, max_k + 1):
                cols.append(np.sin(k * x_test))
                cols.append(np.cos(k * x_test))
            feat_test = np.column_stack(cols)
        elif basis_type == 'exp_log':
            cols = [np.ones(n_test), np.exp(x_test), np.exp(-x_test)]
            pos_mask = x_test > 0
            safe_x = np.where(pos_mask, x_test, 1.0)
            cols.append(np.log(safe_x))
            cols.append(np.log1p(np.abs(x_test)))
            feat_test = np.column_stack(cols)
        elif basis_type == 'rational':
            safe_x = np.where(np.abs(x_test) > 1e-6, x_test, 1e-6)
            feat_test = np.column_stack([np.ones(n_test), x_test, 1.0 / safe_x, x_test ** 2, x_test ** 3])
        elif basis_type == 'tanh':
            feat_test = np.column_stack([np.ones(n_test), np.tanh(x_test), np.tanh(2 * x_test), np.tanh(0.5 * x_test), 1.0 / (1.0 + np.exp(-x_test))])
        elif basis_type == 'hinge':
            cols = [np.ones(n_test)]
            for t in thresholds:
                cols.append(np.maximum(0, x_test - t))
                cols.append(np.maximum(0, t - x_test))
            feat_test = np.column_stack(cols)
        else:
            return np.zeros(n_test)
        return feat_test @ w

    return predict, train_mse


def fit(X_train, y_train):
    t_start = time.time()
    X = X_train
    y = y_train
    n, d = X.shape

    def time_left():
        return TIME_LIMIT - SAFETY_MARGIN - (time.time() - t_start)

    # Portfolio of candidate models
    candidates = []

    # Kernel ridge regression candidates
    for gamma in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for lam in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]:
            candidates.append(('kernel', 'rbf', {'gamma': gamma, 'lam': lam}))

    for degree in [2, 3, 4, 5]:
        for lam in [1e-4, 1e-3, 1e-2]:
            candidates.append(('kernel', 'poly', {'degree': degree, 'lam': lam}))

    for lam in [1e-4, 1e-3, 1e-2]:
        candidates.append(('kernel', 'linear', {'lam': lam}))

    # Basis function candidates
    for deg in [2, 3, 4, 5, 6, 8, 10]:
        for lam in [0.0, 1e-6, 1e-4, 1e-2]:
            candidates.append(('basis', 'poly', {'deg': deg, 'lam': lam}))

    for max_k in [3, 4, 5, 6, 8]:
        for lam in [0.0, 1e-6, 1e-4]:
            candidates.append(('basis', 'trig', {'max_k': max_k, 'lam': lam}))

    for lam in [0.0, 1e-4, 1e-2]:
        candidates.append(('basis', 'exp_log', {'lam': lam}))
        candidates.append(('basis', 'rational', {'lam': lam}))
        candidates.append(('basis', 'tanh', {'lam': lam}))
        candidates.append(('basis', 'hinge', {'thresholds': [-2, -1, 0, 1, 2], 'lam': lam}))

    # Fit all candidates and keep the best by training MSE
    best_mse = np.inf
    best_predictor = None

    for i, (model_type, func_type, params) in enumerate(candidates):
        if time_left() < 5.0:
            break

        try:
            if model_type == 'kernel':
                predict_func, train_mse = _fit_ridge_kernel(X, y, func_type, params)
            else:
                predict_func, train_mse = _fit_basis(X, y, func_type, params)

            if predict_func is not None and np.isfinite(train_mse) and train_mse < best_mse:
                best_mse = train_mse
                best_predictor = predict_func
        except Exception:
            continue

    # Fallback: simple linear regression
    if best_predictor is None:
        A = np.column_stack([np.ones(n)] + [X[:, j] for j in range(d)])
        w = np.linalg.lstsq(A, y, rcond=None)[0]
        return lambda X_test: (np.column_stack([np.ones(len(X_test))] + [X_test[:, j] for j in range(d)]) @ w)

    return best_predictor
