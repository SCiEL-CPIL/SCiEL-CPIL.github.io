import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / 1200.0
X = np.linspace(-0.25, 0.25, N, endpoint=False)

# Fine grid for accurate convolution evaluation
FINE_N = 8000
X_FINE = np.linspace(-0.25, 0.25, FINE_N, endpoint=False)
DX = 0.5 / FINE_N
CELL = np.clip(((X_FINE + 0.25) / W).astype(np.int64), 0, N - 1)
N_FFT = 1 << (2 * FINE_N - 1).bit_length()

# B-spline basis: use a small set of shifted, scaled B-splines (cubic)
# Define B-spline basis functions on [-0.25, 0.25]
# We'll use M cubic B-splines centered at different locations with different widths
M = 12  # number of basis functions

# Precompute B-spline basis values on the fine grid
# Cubic B-spline: support [-2,2] in normalized coords, value depends on distance
def cubic_bspline(t):
    """Cubic B-spline: support [-2, 2], peak 1 at 0."""
    t = np.abs(t)
    b = np.zeros_like(t)
    mask = t < 2
    b[mask] = (2 - t[mask]) ** 3 / 6
    mask2 = (t >= 1) & (t < 2)
    b[mask2] += (t[mask2] - 1) ** 3 / 2
    mask3 = t >= 2
    b[mask3] = 0
    return b

# Define basis: centers and widths
centers = np.linspace(-0.22, 0.22, M)
widths = np.array([0.04, 0.06, 0.08, 0.10, 0.05, 0.07, 0.09, 0.11, 0.04, 0.06, 0.08, 0.10])

# Precompute basis matrix B[i, j] = value of basis j at fine grid point i
Basis = np.zeros((FINE_N, M))
for j in range(M):
    t = (X_FINE - centers[j]) / widths[j]
    Basis[:, j] = cubic_bspline(t)

# Also precompute on step cells for conversion
Basis_step = np.zeros((N, M))
for j in range(M):
    t = (X - centers[j]) / widths[j]
    Basis_step[:, j] = cubic_bspline(t)


def coeffs_to_f(coeffs):
    """Convert nonnegative coefficients to fine-grid function."""
    coeffs = np.clip(coeffs, 0.0, None)
    return Basis @ coeffs


def coeffs_to_h(coeffs):
    """Convert nonnegative coefficients to step heights."""
    coeffs = np.clip(coeffs, 0.0, None)
    return Basis_step @ coeffs


def ratio_from_f(f):
    """Compute the peak autoconvolution / L2 norm ratio."""
    if np.sum(f) < 1e-15:
        return float('inf')
    F = np.fft.rfft(f, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1] * DX
    l2 = np.sum(f * f) * DX
    if l2 < 1e-15:
        return float('inf')
    return float(np.max(conv) / l2)


def objective(coeffs):
    """Objective: minimize peak autoconvolution / L2 norm."""
    f = coeffs_to_f(coeffs)
    return ratio_from_f(f)


def gradient(coeffs):
    """Compute gradient numerically via finite differences (cheap enough for M=12)."""
    eps = 1e-6
    grad = np.zeros(M)
    f0 = objective(coeffs)
    for i in range(M):
        c_plus = coeffs.copy()
        c_plus[i] += eps
        f_plus = objective(c_plus)
        c_minus = coeffs.copy()
        c_minus[i] = max(0.0, coeffs[i] - eps)
        f_minus = objective(c_minus)
        if coeffs[i] > eps:
            grad[i] = (f_plus - f_minus) / (2 * eps)
        else:
            grad[i] = (f_plus - f0) / eps
    return grad


def construct():
    deadline = time.time() + 140.0  # safety margin

    # Start with uniform coefficients
    best_coeffs = np.ones(M) * 0.5
    best_val = objective(best_coeffs)

    # Phase 1: L-BFGS-B optimization
    if time.time() < deadline - 30.0:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='L-BFGS-B',
                bounds=[(0.0, None)] * M,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass

    # Phase 2: Try multiple random restarts with L-BFGS-B
    if time.time() < deadline - 20.0:
        rng = np.random.default_rng(42)
        for restart in range(30):
            if time.time() > deadline - 20.0:
                break
            # Random initialization
            c_init = np.abs(rng.normal(0.5, 0.3, M))
            c_init = np.clip(c_init, 0.0, None)
            try:
                result = minimize(
                    objective,
                    c_init,
                    method='L-BFGS-B',
                    bounds=[(0.0, None)] * M,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_coeffs = result.x
            except Exception:
                pass

    # Phase 3: Fine-grained local polish using coordinate descent
    if time.time() < deadline - 5.0:
        cur_coeffs = best_coeffs.copy()
        cur_val = best_val
        for it in range(5000):
            if time.time() > deadline - 5.0:
                break
            i = np.random.randint(M)
            delta = np.random.normal(0, 0.02)
            trial = cur_coeffs.copy()
            trial[i] = max(0.0, cur_coeffs[i] + delta)
            v = objective(trial)
            if v < cur_val:
                cur_coeffs = trial
                cur_val = v
                if cur_val < best_val:
                    best_val = cur_val
                    best_coeffs = cur_coeffs

    # Convert best coefficients to step heights
    best_h = coeffs_to_h(best_coeffs)
    best_h = np.clip(best_h, 0.0, None)

    # Final verification
    final_val = ratio_from_f(coeffs_to_f(best_coeffs))
    if final_val < best_val:
        best_val = final_val

    return list(best_h)
