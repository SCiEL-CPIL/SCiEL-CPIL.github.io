"""Symbolic regression: nonlinear oscillator acceleration via pure RBF interpolation.

Structural change from local-linearization: use pure Gaussian RBF centers (no local
linear multipliers) for more flexible nonlinear function approximation. Each basis
function is a single Gaussian kernel centered at a fixed point in the (x, v) phase
plane. The 10 parameters are the RBF weights plus a global bias. This captures
arbitrary smooth nonlinearities in the acceleration field without the constraint of
local linearity.

At import time, we search over RBF center placements and bandwidths using a surrogate
driven-oscillator model to select the best configuration.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 130.0   # explicit wall-clock safety margin (< 150 s kill)

# Training ranges for x and v (from task contract)
_X_MIN, _X_MAX = -0.2056, 0.1749
_V_MIN, _V_MAX = -0.3391, 0.3624


def _make_rbf_pure_basis(t, x, v, centers, sigma):
    """Build a 10-column feature matrix using pure RBF interpolation.
    
    Features: [bias, phi_1(x,v), phi_2(x,v), ..., phi_9(x,v)]
    where phi_i = exp(-((x-xi)^2 + (v-vi)^2) / (2*sigma^2))
    
    9 RBF centers + 1 global bias = 10 features.
    Returns a (n, 10) array.
    """
    n = len(x)
    cols = [np.ones(n)]  # bias
    for (cx, cv) in centers:
        dx = x - cx
        dv = v - cv
        phi = np.exp(-(dx * dx + dv * dv) / (2.0 * sigma * sigma))
        cols.append(phi)
    # Ensure exactly 10 columns
    if len(cols) > 10:
        cols = cols[:10]
    elif len(cols) < 10:
        while len(cols) < 10:
            cols.append(np.zeros(n))
    return np.column_stack(cols)


def _build_surrogate():
    """Synthesize plausible nonlinear-oscillator acceleration data for basis selection.
    
    We build a representative surrogate from a driven nonlinear oscillator model.
    It is only used to rank RBF configurations by approximation quality; the true
    fit is done by the evaluator.
    """
    rng = np.random.default_rng(1234)
    n = 20000
    t = rng.uniform(30.0, 50.0, n)
    x = rng.uniform(_X_MIN, _X_MAX, n)
    v = rng.uniform(_V_MIN, _V_MAX, n)

    # Ground-truth-like acceleration: driven Duffing with nonlinear damping + coupling
    drive = 0.3 * np.sin(t) + 0.15 * np.cos(t)
    stiff = 1.5 * x + 3.0 * x * x * x          # linear + Duffing stiffness
    damp = 0.4 * v - 0.8 * v * v * v          # linear + nonlinear damping
    couple = 0.5 * x * v + 0.2 * x * x * v    # coupling terms
    # Add local nonlinearity that a global polynomial would miss
    local = 0.3 * np.exp(-((x - 0.1)**2 + (v + 0.1)**2) / (2 * 0.12**2)) * x
    local2 = 0.2 * np.exp(-((x + 0.1)**2 + (v - 0.1)**2) / (2 * 0.12**2)) * v
    y = -stiff - damp + drive + couple + local + local2
    return t, x, v, y


def _fit_linear(F, y):
    """Least-squares fit; return (coefficients, in-sample residual std)."""
    try:
        c, *_ = np.linalg.lstsq(F, y, rcond=None)
        r = y - F @ c
        return c, float(np.sqrt(np.mean(r * r)))
    except Exception:
        return None, np.inf


def _generate_rbf_centers(nx, nv, pattern="grid"):
    """Generate RBF centers over the (x, v) phase plane.
    
    Supports different patterns:
    - 'grid': uniform grid
    - 'diagonal': centers along main diagonal
    - 'cross': centers in a cross pattern
    """
    if pattern == "grid":
        xs = np.linspace(_X_MIN, _X_MAX, nx)
        vs = np.linspace(_V_MIN, _V_MAX, nv)
        centers = [(float(x), float(v)) for x in xs for v in vs]
    elif pattern == "diagonal":
        # Place centers along the diagonal from (X_MIN, V_MIN) to (X_MAX, V_MAX)
        n_total = nx * nv
        xs = np.linspace(_X_MIN, _X_MAX, n_total)
        vs = np.linspace(_V_MIN, _V_MAX, n_total)
        centers = [(float(x), float(v)) for x, v in zip(xs, vs)]
    elif pattern == "cross":
        # Cross pattern: some along x-axis, some along v-axis
        centers = []
        n_x = min(nx, 5)
        n_v = min(nv, 5)
        for i in range(n_x):
            cx = np.linspace(_X_MIN, _X_MAX, n_x)[i]
            centers.append((float(cx), 0.0))
        for j in range(n_v):
            cv = np.linspace(_V_MIN, _V_MAX, n_v)[j]
            if abs(cv) > 0.01:  # avoid duplicate at origin
                centers.append((0.0, float(cv)))
        # Trim or pad to get exactly 9
        while len(centers) > 9:
            centers.pop()
        while len(centers) < 9:
            centers.append((0.0, 0.0))
    else:
        xs = np.linspace(_X_MIN, _X_MAX, nx)
        vs = np.linspace(_V_MIN, _V_MAX, nv)
        centers = [(float(x), float(v)) for x in xs for v in vs]
    
    # Ensure exactly 9 centers
    while len(centers) > 9:
        centers.pop()
    while len(centers) < 9:
        centers.append((0.0, 0.0))
    return centers


def _select_best_rbf_config():
    """Search over RBF configurations and select the best by validation error."""
    t, x, v, y = _build_surrogate()
    # Hold out a portion to avoid overfitting the config choice to noise
    idx = np.arange(len(t))
    np.random.default_rng(7).shuffle(idx)
    split = int(0.8 * len(t))
    tr, te = idx[:split], idx[split:]

    best_config = None
    best_score = np.inf

    # Candidate configurations: (nx, nv, sigma, pattern)
    # We need exactly 9 RBF centers + 1 bias = 10 features
    configs = []
    
    # Grid patterns with 9 centers (3x3)
    for sigma in [0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
        configs.append((3, 3, sigma, "grid"))
    
    # Diagonal patterns
    for sigma in [0.10, 0.15, 0.20, 0.25]:
        configs.append((3, 3, sigma, "diagonal"))
    
    # Cross patterns
    for sigma in [0.10, 0.15, 0.20, 0.25, 0.30]:
        configs.append((3, 3, sigma, "cross"))
    
    # Different grid spacings (non-uniform)
    for sigma in [0.12, 0.15, 0.20]:
        configs.append((3, 3, sigma, "grid"))
    
    seen = set()
    for nx, nv, sigma, pattern in configs:
        key = (nx, nv, round(sigma, 3), pattern)
        if key in seen:
            continue
        seen.add(key)
        if time.time() > _DEADLINE:
            break
        try:
            centers = _generate_rbf_centers(nx, nv, pattern)
            F = _make_rbf_pure_basis(t, x, v, centers, sigma)
        except Exception:
            continue
        if not (F.shape == (len(t), 10) and np.all(np.isfinite(F))):
            continue
        _, err_tr = _fit_linear(F[tr], y[tr])
        if not np.isfinite(err_tr):
            continue
        _, err_te = _fit_linear(F[te], y[te])
        if not np.isfinite(err_te):
            continue
        gap = err_te - err_tr
        score = err_te + 0.1 * max(0.0, gap)
        if score < best_score:
            best_score = score
            best_config = (nx, nv, sigma, pattern)

    if best_config is None:
        # Fallback: 3x3 grid with sigma=0.15
        best_config = (3, 3, 0.15, "grid")

    nx, nv, sigma, pattern = best_config
    centers = _generate_rbf_centers(nx, nv, pattern)
    return centers, sigma


try:
    _BEST_CENTERS, _BEST_SIGMA = _select_best_rbf_config()
except Exception:
    # Fallback: 3x3 grid with sigma=0.15
    _BEST_CENTERS = _generate_rbf_centers(3, 3, "grid")
    _BEST_SIGMA = 0.15

# EVOLVE-BLOCK-END


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction: pure RBF interpolation model.

    Returns an array with one prediction per row.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    F = _make_rbf_pure_basis(t, x, v, _BEST_CENTERS, _BEST_SIGMA)
    out = F @ params
    if out.ndim != 1:
        out = out.reshape(-1)
    return out
