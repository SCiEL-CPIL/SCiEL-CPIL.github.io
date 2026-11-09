"""Nonlinear oscillator symbolic regression with Chebyshev basis and multi-start optimization."""
import numpy as np
import time
from scipy.optimize import minimize

# Known ranges from task description
T_MIN, T_MAX = 30.0, 50.0
X_MIN, X_MAX = -0.2056, 0.1749
V_MIN, V_MAX = -0.3391, 0.3624

# Precompute Chebyshev nodes and weights for integration (not directly used, but for reference)
# We'll use Chebyshev polynomials evaluated at normalized coordinates

def _chebyshev_values(z, order):
    """Evaluate Chebyshev polynomials T_0(z), T_1(z), ..., T_order(z) at points z.
    
    z: array of shape (N,) in [-1, 1]
    order: int
    Returns: array of shape (N, order+1)
    """
    N = len(z)
    T = np.zeros((N, order + 1))
    T[:, 0] = 1.0
    if order >= 1:
        T[:, 1] = z
    for k in range(2, order + 1):
        T[:, k] = 2.0 * z * T[:, k - 1] - T[:, k - 2]
    return T


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration model using Chebyshev polynomial basis in normalized (x, v)
    with a sinusoidal driving term in t.
    
    Model:
      a = drive(t) + sum_{i,j} c_{ij} * T_i(x_norm) * T_j(v_norm)
    
    where:
      x_norm = (2*(x - X_MIN)/(X_MAX - X_MIN)) - 1
      v_norm = (2*(v - V_MIN)/(V_MAX - V_MIN)) - 1
      drive(t) = p0 * sin(p1 * t + p2)
    
    params layout (10 total):
      p0: drive amplitude
      p1: drive frequency  
      p2: drive phase
      p3-p9: 7 coefficients for the Chebyshev basis terms
    
    The 7 basis terms used (low-order Chebyshev product basis):
      T0(x)*T0(v), T1(x)*T0(v), T0(x)*T1(v), T2(x)*T0(v), 
      T0(x)*T2(v), T1(x)*T1(v), T2(x)*T1(v)
    
    This gives a flexible nonlinear approximation in (x,v) with good conditioning.
    """
    p0, p1, p2, c0, c1, c2, c3, c4, c5, c6 = params
    
    # Normalize x and v to [-1, 1]
    x_norm = 2.0 * (x - X_MIN) / (X_MAX - X_MIN) - 1.0
    v_norm = 2.0 * (v - V_MIN) / (V_MAX - V_MIN) - 1.0
    
    # Clip to avoid numerical issues if slightly out of range
    x_norm = np.clip(x_norm, -1.0, 1.0)
    v_norm = np.clip(v_norm, -1.0, 1.0)
    
    # Evaluate Chebyshev polynomials up to order 2 for x and v
    # T0 = 1, T1 = z, T2 = 2z^2 - 1
    T0x = np.ones_like(x_norm)
    T1x = x_norm
    T2x = 2.0 * x_norm * x_norm - 1.0
    
    T0v = np.ones_like(v_norm)
    T1v = v_norm
    T2v = 2.0 * v_norm * v_norm - 1.0
    
    # Sinusoidal drive
    drive = p0 * np.sin(p1 * t + p2)
    
    # Chebyshev product basis terms (7 terms)
    term0 = c0 * T0x * T0v
    term1 = c1 * T1x * T0v
    term2 = c2 * T0x * T1v
    term3 = c3 * T2x * T0v
    term4 = c4 * T0x * T2v
    term5 = c5 * T1x * T1v
    term6 = c6 * T2x * T1v
    
    return drive + term0 + term1 + term2 + term3 + term4 + term5 + term6


# Multi-start optimization to avoid local minima
def _fit_params(t, x, v, y, n_starts=5):
    """Fit params using multi-start BFGS."""
    N = len(y)
    y_mean = np.mean(y)
    y_var = np.var(y)
    if y_var < 1e-12:
        y_var = 1e-12
    
    best_params = np.ones(10)
    best_mse = np.inf
    
    def objective(params):
        pred = equation(t, x, v, params)
        if not np.all(np.isfinite(pred)):
            return 1e30
        mse = np.mean((pred - y) ** 2)
        return mse
    
    # Starting points
    starts = [np.ones(10)]
    
    # Add some diverse starting points
    starts.append(np.array([0.1, 0.5, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0]))
    starts.append(np.array([0.0, 0.0, 0.0, 0.0, -2.0, -2.0, 0.0, 0.0, 0.0, 0.0]))
    starts.append(np.array([0.1, 1.0, 0.0, 0.0, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0]))
    starts.append(np.array([0.05, 0.3, 0.5, 0.0, -1.5, -1.5, 0.1, 0.0, 0.0, 0.0]))
    
    for i in range(min(n_starts, len(starts))):
        res = minimize(objective, starts[i], method='BFGS', 
                       options={'maxiter': 500, 'gtol': 1e-8})
        if res.success or res.fun < best_mse:
            if np.isfinite(res.fun) and res.fun < best_mse:
                best_mse = res.fun
                best_params = res.x.copy()
    
    return best_params


# This block is not executed during evaluation; it's for local testing
if __name__ == "__main__":
    # Generate synthetic test data to verify the function works
    np.random.seed(42)
    N = 1000
    t = np.random.uniform(T_MIN, T_MAX, N)
    x = np.random.uniform(X_MIN, X_MAX, N)
    v = np.random.uniform(V_MIN, V_MAX, N)
    
    # Create synthetic target from a known nonlinear oscillator
    true_params = np.array([0.5, 1.0, 0.3, 0.1, -1.0, -0.8, 0.05, 0.0, 0.0, 0.0])
    y = equation(t, x, v, true_params) + np.random.normal(0, 0.01, N)
    
    # Test that equation returns correct shape
    params_test = np.ones(10)
    pred = equation(t, x, v, params_test)
    assert pred.shape == (N,), f"Shape mismatch: {pred.shape}"
    assert np.all(np.isfinite(pred)), "Non-finite predictions"
    
    # Run a quick fit
    start_time = time.time()
    fitted_params = _fit_params(t, x, v, y, n_starts=3)
    elapsed = time.time() - start_time
    print(f"Fitting took {elapsed:.2f}s")
    print(f"Fitted params: {fitted_params}")
    pred_fitted = equation(t, x, v, fitted_params)
    mse = np.mean((pred_fitted - y) ** 2)
    print(f"MSE: {mse:.6f}")
