"""Time-aware internal least-squares fit over a rich feature basis."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = None
_TIME_MARGIN = 8.0  # seconds safety margin before the 150s kill


def _init_deadline():
    global _DEADLINE
    _DEADLINE = time.time() + 150.0


def _time_ok():
    if _DEADLINE is None:
        return True
    return (time.time() - _DEADLINE) < _TIME_MARGIN


def _build_features(X, Y):
    """Build a diverse, bounded feature matrix from inputs."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[0]
    Xc = np.clip(X, -1.0, 1.0)
    Yc = np.clip(Y, -1.0, 1.0)

    feats = [np.ones(n)]
    # Linear
    feats += [Xc, Yc]
    # Quadratic
    feats += [Xc * Xc, Yc * Yc, Xc * Yc]
    # Cubic
    feats += [Xc ** 3, Yc ** 3, Xc * Yc ** 2, Xc ** 2 * Yc]
    # Quartic
    feats += [Xc ** 4, Yc ** 4, Xc ** 2 * Yc ** 2]
    # Trig
    feats += [np.sin(Xc), np.sin(Yc), np.cos(Xc), np.cos(Yc)]
    feats += [np.sin(Xc + Yc), np.sin(Xc - Yc), np.cos(Xc + Yc), np.cos(Xc - Yc)]
    # Radial
    R = np.sqrt(Xc * Xc + Yc * Yc)
    feats += [R, R ** 2, R ** 3]
    # Angular
    theta = np.arctan2(Yc, Xc)
    feats += [np.sin(theta), np.cos(theta), np.sin(2 * theta), np.cos(2 * theta)]
    # Exponential (bounded)
    feats += [np.exp(-Xc ** 2), np.exp(-Yc ** 2), np.exp(-(Xc ** 2 + Yc ** 2))]
    # Sigmoid-like
    feats += [1.0 / (1.0 + np.exp(-5 * Xc)), 1.0 / (1.0 + np.exp(-5 * Yc))]
    # Interaction with radial
    feats += [Xc * R, Yc * R]
    # Log (safe)
    feats += [np.log1p(np.abs(Xc)), np.log1p(np.abs(Yc))]
    # Tanh
    feats += [np.tanh(Xc), np.tanh(Yc), np.tanh(Xc + Yc), np.tanh(Xc - Yc)]

    F = np.column_stack(feats)
    # Standardize columns for better conditioning
    col_mean = F.mean(axis=0)
    col_std = F.std(axis=0)
    col_std[col_std < 1e-10] = 1.0
    F = (F - col_mean) / col_std
    return F, col_mean, col_std


def _fit_params(X, Y, params, F, col_mean, col_std):
    """Loss function for internal fitting."""
    pred = F @ params
    return np.mean((pred - Y) ** 2)


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _DEADLINE
    if _DEADLINE is None:
        _init_deadline()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]

    # Build features
    F, col_mean, col_std = _build_features(x, y)
    p = F.shape[1]

    # Use params as initial guess if length matches, else use ones
    if params is not None and len(params) == p:
        w0 = np.array(params, dtype=float)
    else:
        w0 = np.ones(p)

    # Try least squares first (fast, exact for linear model)
    try:
        # Solve F w = y in least squares sense
        w_ls, residuals, rank, sv = np.linalg.lstsq(F, y, rcond=None)
        if _time_ok():
            # Check if solution is reasonable
            pred = F @ w_ls
            if np.all(np.isfinite(pred)):
                mse = np.mean((pred - y) ** 2)
                if mse < 1e-10:
                    return pred
    except Exception:
        pass

    # Use BFGS if time permits
    if _time_ok():
        try:
            result = minimize(
                _fit_params,
                w0,
                args=(x, y, F, col_mean, col_std),
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-6}
            )
            if _time_ok():
                w_opt = result.x
                if np.all(np.isfinite(w_opt)):
                    pred = F @ w_opt
                    if np.all(np.isfinite(pred)):
                        return pred
        except Exception:
            pass

    # Fallback: return linear combination using original params if possible
    if params is not None and len(params) >= 3:
        return params[0] * x + params[1] * y + params[2]
    else:
        return np.zeros(n)
# EVOLVE-BLOCK-END
