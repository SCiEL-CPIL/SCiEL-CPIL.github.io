"""Symbolic regression: nonlinear oscillator acceleration via local RBF-linearization basis.

Structural change: instead of a global polynomial/trig basis selected by surrogate fitting,
we use a fixed set of Gaussian RBF centers on a 2D grid in (x, v). Each RBF is multiplied
by a linear term (1, x, v) to locally approximate the acceleration field. This gives 10
features (bias + 3 RBFs * 3 linear terms) that capture non-additive coupling in the phase
plane. The model is linear in the 10 params, so the evaluator's BFGS fit recovers the
least-squares optimum.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 130.0  # explicit wall-clock safety margin (< 150 s kill)


def _make_rbf_basis(t, x, v, centers, widths):
    """Build a 10-column feature matrix using RBF centers on (x, v) with linear modulation.
    
    Features:
      col 0: bias (ones)
      cols 1-3: RBF_i * 1 for i=0,1,2
      cols 4-6: RBF_i * x for i=0,1,2
      cols 7-9: RBF_i * v for i=0,1,2
    
    This gives local linearization around each center in the (x, v) phase plane.
    """
    n = len(x)
    cols = [np.ones(n)]
    
    for i in range(len(centers)):
        cx, cv = centers[i]
        wx, wv = widths[i]
        
        # Gaussian RBF in (x, v)
        dx = (x - cx) / wx
        dv = (v - cv) / wv
        rbf = np.exp(-0.5 * (dx * dx + dv * dv))
        
        cols.append(rbf)          # RBF * 1
        cols.append(rbf * x)      # RBF * x
        cols.append(rbf * v)      # RBF * v
    
    return np.column_stack(cols)


# ---- Define RBF centers and widths on a 3x3 grid in (x, v) ----
# x range: [-0.2056, 0.1749], v range: [-0.3391, 0.3624]
# Use 3 centers to keep total features at 10 (1 bias + 3*3 RBF-linear terms)
# Centers spread across the phase space

_x_min, _x_max = -0.2056, 0.1749
_v_min, _v_max = -0.3391, 0.3624

# 3 centers: corners of a rectangle covering most of the phase space
# Center 1: lower-left (negative x, negative v)
# Center 2: upper-right (positive x, positive v)  
# Center 3: center (near origin)
_CENTERS = [
    ((_x_min + _x_max) * 0.25, (_v_min + _v_max) * 0.25),   # lower-left region
    ((_x_min + _x_max) * 0.75, (_v_min + _v_max) * 0.75),   # upper-right region
    ((_x_min + _x_max) * 0.5,  (_v_min + _v_max) * 0.5),    # center
]

# Widths: cover about half the range in each direction
_x_range = _x_max - _x_min
_v_range = _v_max - _v_min
_WIDTHS = [
    (_x_range * 0.35, _v_range * 0.35),
    (_x_range * 0.35, _v_range * 0.35),
    (_x_range * 0.35, _v_range * 0.35),
]


def _build_surrogate():
    """Synthesize plausible nonlinear-oscillator acceleration data for validation."""
    rng = np.random.default_rng(42)
    n = 20000
    t = rng.uniform(30.0, 50.0, n)
    x = rng.uniform(-0.2056, 0.1749, n)
    v = rng.uniform(-0.3391, 0.3624, n)
    
    # Driven Duffing with nonlinear damping
    drive = 0.4 * np.sin(t) + 0.2 * np.cos(t)
    stiff = 2.0 * x + 5.0 * x * x * x
    damp = 0.5 * v - 1.0 * v * v * v
    couple = 0.6 * x * v + 0.3 * x * x * v
    y = -stiff - damp + drive + couple
    return t, x, v, y


# Validate the RBF basis on surrogate data at import time
def _validate_basis():
    """Check that the RBF basis can reasonably approximate the surrogate target."""
    t, x, v, y = _build_surrogate()
    F = _make_rbf_basis(t, x, v, _CENTERS, _WIDTHS)
    
    if not (F.shape == (len(t), 10) and np.all(np.isfinite(F))):
        return False
    
    try:
        c, *_ = np.linalg.lstsq(F, y, rcond=None)
        r = y - F @ c
        mse = float(np.mean(r * r))
        var_y = float(np.var(y))
        nrmse = mse / var_y if var_y > 0 else np.inf
        # Accept if normalized MSE is reasonable (below 1.0 means better than mean)
        return np.isfinite(nrmse) and nrmse < 1.0
    except Exception:
        return False


# Fallback basis if RBF validation fails: simple polynomial
def _make_fallback_basis(t, x, v):
    """Simple polynomial basis as fallback."""
    cols = [
        np.ones_like(t),
        np.sin(t),
        np.cos(t),
        x,
        x * x,
        x * x * x,
        v,
        v * v,
        x * v,
        v * v * v,
    ]
    return np.column_stack(cols)


_USE_RBF = _validate_basis()


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction using RBF-linearization or fallback polynomial basis.
    
    Returns an array with one prediction per row.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if _USE_RBF:
        F = _make_rbf_basis(t, x, v, _CENTERS, _WIDTHS)
    else:
        F = _make_fallback_basis(t, x, v)
    
    out = F @ params
    if out.ndim != 1:
        out = out.reshape(-1)
    
    # Ensure finite output
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)
    
    return out

# EVOLVE-BLOCK-END
