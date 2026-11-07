"""Rewritten program: matched 10-param degree-3 bivariate polynomial with robust fitting."""
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: keep a safety margin before the 150s wall-clock limit
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0  # seconds to leave spare


def _build_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Build a 10-feature basis for a degree-3 bivariate polynomial in x, y.
    Features (degree-3 complete polynomial in 2 variables has C(3+2,2)=10 terms):
        1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    """
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    x2y = x2 * y
    xy2 = x * y2
    xy = x * y

    # 10 columns
    F = np.column_stack([
        np.ones_like(x),
        x,
        y,
        x2,
        xy,
        y2,
        x3,
        x2y,
        xy2,
        y3,
    ])
    return F


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form prediction using a degree-3 bivariate polynomial with 10 parameters.
    params: array of length 10, the polynomial coefficients.
    """
    F = _build_features(x, y)
    return F @ params


def _objective(params, F, y_target):
    """MSE objective for fitting."""
    pred = F @ params
    residual = pred - y_target
    return np.mean(residual ** 2)


def _fit_and_predict(x_train, y_train, x_test, y_test):
    """
    Fit the 10-parameter polynomial on training data using BFGS from all-ones start,
    then predict on test data.
    """
    F_train = _build_features(x_train, y_train)
    F_test = _build_features(x_test, y_test)

    n_params = 10
    x0 = np.ones(n_params)

    start_time = time.time()
    best_params = x0.copy()
    best_loss = np.inf

    # Primary BFGS fit
    try:
        result = minimize(
            _objective,
            x0,
            args=(F_train, y_train),
            method="BFGS",
            options={"maxiter": 1000, "gtol": 1e-8},
        )
        if result.fun < best_loss and np.isfinite(result.fun):
            best_loss = result.fun
            best_params = result.x.copy()
    except Exception:
        pass

    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        pred = F_test @ best_params
        if np.all(np.isfinite(pred)):
            return pred
        return np.zeros_like(y_test)

    # If the initial fit didn't converge well, try a few more from different starts
    # but only if we have time budget
    n_extra = 3
    for i in range(n_extra):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        # Perturbed starts
        rng = np.random.RandomState(42 + i)
        x0_perturb = x0 + rng.randn(n_params) * 0.1
        try:
            result = minimize(
                _objective,
                x0_perturb,
                args=(F_train, y_train),
                method="BFGS",
                options={"maxiter": 500, "gtol": 1e-8},
            )
            if result.fun < best_loss and np.isfinite(result.fun):
                best_loss = result.fun
                best_params = result.x.copy()
        except Exception:
            pass

    pred = F_test @ best_params
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    return pred


def main():
    """
    Entry point used by the evaluator.
    The evaluator calls equation(x, y, params) where x, y are the test inputs
    and params are the fitted parameters. But looking at the task contract:

    "The evaluator fits params, an array of length 10, with scipy.optimize.minimize(method='BFGS')
     starting from all ones, minimizing the training MSE."

    So the evaluator does the fitting itself. The equation function is just the forward pass.
    We just need to return F @ params.
    """
    pass


# The actual entry point as required by the task:
# equation(x, y, params) -> np.ndarray
# The evaluator will call this with training data to compute losses during fitting,
# and with test data for final prediction.

# Since the evaluator does the fitting, we just need the forward function.
# The current design already has this. Let's keep it clean.

# Re-define equation to be the clean forward pass (the evaluator handles fitting)
def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Forward pass: degree-3 bivariate polynomial with 10 parameters.
    x, y: input arrays of shape (n,)
    params: array of shape (10,)
    Returns: array of shape (n,)
    """
    F = _build_features(x, y)
    return F @ params
