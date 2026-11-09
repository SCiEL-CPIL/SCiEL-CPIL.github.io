# EVOLVE-BLOCK-START
import time
import numpy as np

_TOL = 1e-7
_DEADLINE = 590.0  # wall-clock seconds, safety margin under the 630s kill


def _objective_and_grad(cx, cy, s):
    """Return (score, grad_cx, grad_cy, grad_s) for sum of sides.

    score = sum(s) - penalty, where penalty penalizes:
      - overlap of interiors (pairwise L1 separation < (s_i+s_j)/2)
      - leaving the unit square
    Uses smooth soft-min via log-sum-exp for the overlap term.
    """
    n = len(s)
    iu, ju = np.triu_indices(n, k=1)
    dx = cx[iu] - cx[ju]
    dy = cy[iu] - cy[ju]
    t = 0.5 * (s[iu] + s[ju]) - _TOL

    # Required separations: |dx| >= t and |dy| >= t.
    # Smooth violation per pair: soft-min of (t - |dx|, t - |dy|)
    a = t - np.abs(dx)
    b = t - np.abs(dy)
    M = np.maximum(a, b)
    # soft-min = -logsumexp(-a, -b)
    logsum = M + np.log(np.exp(-a - M) + np.exp(-b - M))
    softmin = -logsum
    # violation = max(0, softmin); use smooth clamp
    viol = np.log1p(np.exp(softmin))  # softplus
    # gradient of softplus(softmin) wrt a,b
    sig_a = 1.0 / (1.0 + np.exp(-softmin))
    # d softmin / da:
    wa = np.exp(-a - M)
    wb = np.exp(-b - M)
    wsum = wa + wb
    dsoftmin_da = wa / wsum
    dsoftmin_db = wb / wsum

    grad_cx = np.zeros(n)
    grad_cy = np.zeros(n)
    grad_s = np.zeros(n)

    # dviol/dcx_i for pair (i,j): sig_a * dsoftmin_da * d(a)/d|dx| * d|dx|/dcx
    # a = t - |dx|, da/d|dx| = -1, d|dx|/dcx_i = sign(dx)
    # grad_cx[i] += -sig_a * dsoftmin_da * sign(dx)  ... but sign(dx) for i vs j
    # Let's be careful: a = t - |cx_i - cx_j|.
    # d a / d cx_i = -sign(cx_i - cx_j) = -sign(dx)
    # d a / d cx_j = +sign(cx_i - cx_j) = +sign(dx)
    # d viol / d cx_i = sig_a * dsoftmin_da * (-sign(dx))
    # d viol / d cx_j = sig_a * dsoftmin_da * (+sign(dx))
    # Same for y.
    # d a / d s_k = 0.5 for k in {i,j} (since t = 0.5(s_i+s_j))
    # d viol / d s_k = sig_a * dsoftmin_da * 0.5

    sign_dx = np.sign(dx)
    sign_dy = np.sign(dy)

    np.add.at(grad_cx, iu, -sig_a * dsoftmin_da * sign_dx)
    np.add.at(grad_cx, ju, +sig_a * dsoftmin_da * sign_dx)
    np.add.at(grad_cy, iu, -sig_a * dsoftmin_db * sign_dy)
    np.add.at(grad_cy, ju, +sig_a * dsoftmin_db * sign_dy)

    # For each pair, both squares get 0.5 * sig_a * dsoftmin_da added to grad_s
    np.add.at(grad_s, iu, 0.5 * sig_a * dsoftmin_da)
    np.add.at(grad_s, ju, 0.5 * sig_a * dsoftmin_da)

    # Boundary penalties: square i must satisfy s_i/2 <= cx_i <= 1 - s_i/2 (and y)
    # Use smooth softplus of violation.
    # x_low_viol = s_i/2 - cx_i + tol  (want <= 0)
    # x_high_viol = cx_i + s_i/2 - 1 - tol
    # We penalize max(0, x_low_viol) and max(0, x_high_viol) via softplus.
    x_low = 0.5 * s - cx + _TOL
    x_high = cx + 0.5 * s - 1.0 - _TOL
    y_low = 0.5 * s - cy + _TOL
    y_high = cy + 0.5 * s - 1.0 - _TOL

    # softplus(v) = log1p(exp(v)); d/dv = sigmoid(v)
    sp_xl = np.log1p(np.exp(x_low))
    sp_xh = np.log1p(np.exp(x_high))
    sp_yl = np.log1p(np.exp(y_low))
    sp_yh = np.log1p(np.exp(y_high))

    penalty = np.sum(viol) + np.sum(sp_xl) + np.sum(sp_xh) + np.sum(sp_yl) + np.sum(sp_yh)

    # gradients of boundary terms
    sig_xl = 1.0 / (1.0 + np.exp(-x_low))
    sig_xh = 1.0 / (1.0 + np.exp(-x_high))
    sig_yl = 1.0 / (1.0 + np.exp(-y_low))
    sig_yh = 1.0 / (1.0 + np.exp(-y_high))

    # d x_low / dcx_i = -1, d x_low / ds_i = 0.5
    grad_cx -= sig_xl
    grad_s += 0.5 * sig_xl
    # d x_high / dcx_i = +1, d x_high / ds_i = 0.5
    grad_cx += sig_xh
    grad_s += 0.5 * sig_xh
    grad_cy -= sig_yl
    grad_s += 0.5 * sig_yl
    grad_cy += sig_yh
    grad_s += 0.5 * sig_yh

    # Objective: maximize sum(s) - penalty
    score = np.sum(s) - penalty
    # gradient of (sum(s) - penalty):
    grad_s += 1.0
    grad_cx = -grad_cx
    grad_cy = -grad_cy
    grad_s = -grad_s

    return score, grad_cx, grad_cy, grad_s


def _is_feasible(cx, cy, s, tol=_TOL):
    """Check all constraints with the given tolerance."""
    n = len(s)
    if np.any(~np.isfinite(cx)) or np.any(~np.isfinite(cy)) or np.any(~np.isfinite(s)):
        return False
    if np.any(s <= 0):
        return False
    # inside [-tol, 1+tol]
    if np.any(cx - 0.5 * s < -tol) or np.any(cx + 0.5 * s > 1.0 + tol):
        return False
    if np.any(cy - 0.5 * s < -tol) or np.any(cy + 0.5 * s > 1.0 + tol):
        return False
    # pairwise
    iu, ju = np.triu_indices(n, k=1)
    dx = np.abs(cx[iu] - cx[ju])
    dy = np.abs(cy[iu] - cy[ju])
    t = 0.5 * (s[iu] + s[ju]) - tol
    if np.any((dx < t) & (dy < t)):
        return False
    return True


def _grid_packing(n=10):
    """Deterministic grid packing: 4x4 grid, side = 0.25."""
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def _lbfgs_step(cx, cy, s, grad_cx, grad_cy, grad_s, max_iter=2000, tol=1e-12):
    """Simple L-BFGS with two-loop recursion, line search via backtracking."""
    n = len(s)
    m = 10  # L-BFGS memory
    s_hist = []
    y_hist = []
    rho_hist = []
    x = np.concatenate([cx, cy, s])
    g = np.concatenate([grad_cx, grad_cy, grad_s])

    def obj(xv):
        cv = xv[:n].copy()
        cyv = xv[n:2*n].copy()
        sv = xv[2*n:].copy()
        sc, gcx, gcy, gs = _objective_and_grad(cv, cyv, sv)
        return sc, np.concatenate([gcx, gcy, gs])

    f, g = obj(x)
    for it in range(max_iter):
        # two-loop recursion to compute direction
        q = g.copy()
        alphas = np.zeros(m)
        for i in range(len(s_hist) - 1, -1, -1):
            alphas[i] = rho_hist[i] * np.dot(s_hist[i], q)
            q = q - alphas[i] * y_hist[i]
        r = q.copy()
        for i in range(len(s_hist)):
            beta = rho_hist[i] * np.dot(y_hist[i], r)
            r = r + (alphas[i] - beta) * s_hist[i]
        d = -r
        # line search
        step = 1.0
        for _ in range(30):
            x_new = x + step * d
            f_new, g_new = obj(x_new)
            if f_new < f + 1e-4 * step * np.dot(g, d):
                break
            step *= 0.5
        else:
            break
        s_new = x_new - x
        y_new = g_new - g
        sy = np.dot(s_new, y_new)
        if sy > 1e-10:
            if len(s_hist) >= m:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(s_new)
            y_hist.append(y_new)
            rho_hist.append(1.0 / sy)
        x = x_new
        f, g = f_new, g_new
        if np.linalg.norm(g) < tol:
            break
    cx2 = x[:n].copy()
    cy2 = x[n:2*n].copy()
    s2 = x[2*n:].copy()
    return cx2, cy2, s2


def squares_in_square10() -> tuple:
    n = 10
    t0 = time.time()

    # Start from grid packing
    centers, sides = _grid_packing(n)
    cx = centers[:, 0].copy()
    cy = centers[:, 1].copy()
    s = sides.copy()

    best_score = float('-inf')
    best_cx, best_cy, best_s = cx.copy(), cy.copy(), s.copy()

    # L-BFGS local search
    for restart in range(5):
        if time.time() - t0 > _DEADLINE - 10:
            break
        if restart == 0:
            cx, cy, s = cx.copy(), cy.copy(), s.copy()
        else:
            # perturb
            cx = best_cx + 0.01 * np.random.randn(n)
            cy = best_cy + 0.01 * np.random.randn(n)
            s = best_s * (1 + 0.01 * np.random.randn(n))
            s = np.clip(s, 0.01, 1.0)

        score, gcx, gcy, gs = _objective_and_grad(cx, cy, s)
        cx, cy, s = _lbfgs_step(cx, cy, s, gcx, gcy, gs)

        # project s to be positive and within bounds
        s = np.clip(s, 1e-6, 1.0)
        cx = np.clip(cx, 0.5 * s, 1.0 - 0.5 * s)
        cy = np.clip(cy, 0.5 * s, 1.0 - 0.5 * s)

        score, gcx, gcy, gs = _objective_and_grad(cx, cy, s)
        if score > best_score:
            best_score = score
            best_cx, best_cy, best_s = cx.copy(), cy.copy(), s.copy()

    # Use best found
    cx, cy, s = best_cx, best_cy, best_s

    # Final polish: try to increase sides slightly while maintaining feasibility
    # by solving a small local problem. We'll do a few more L-BFGS iterations
    # with tighter tolerance.
    if time.time() - t0 < _DEADLINE - 5:
        score, gcx, gcy, gs = _objective_and_grad(cx, cy, s)
        cx, cy, s = _lbfgs_step(cx, cy, s, gcx, gcy, gs, max_iter=5000, tol=1e-14)
        s = np.clip(s, 1e-6, 1.0)
        cx = np.clip(cx, 0.5 * s, 1.0 - 0.5 * s)
        cy = np.clip(cy, 0.5 * s, 1.0 - 0.5 * s)

    # Ensure feasibility; if not, fall back to grid packing
    if not _is_feasible(cx, cy, s):
        # Try to repair: shrink sides slightly
        for shrink in [0.99, 0.98, 0.95, 0.9]:
            s_try = s * shrink
            if _is_feasible(cx, cy, s_try):
                s = s_try
                break
        else:
            # fall back to grid
            centers, sides = _grid_packing(n)
            return centers, sides

    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    return centers, sides


# EVOLVE-BLOCK-END
