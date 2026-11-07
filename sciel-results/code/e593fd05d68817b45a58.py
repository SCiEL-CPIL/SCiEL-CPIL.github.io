"""Homotopy-tracked Newton root finder for three fixed 2-D systems.

Core idea: walk a continuation path (RK4 on the ODE d x / d t = - J^{-1} F)
from a previously-found root toward the requested start point, polishing
with FD-Jacobian Newton + line search.  A global grid fallback covers
basins the homotopy path cannot reach.  An explicit deadline guard
returns the lowest-norm point ever seen.
"""
import time
import numpy as np


# Per-system cache of "the last root we found" so that, for the same
# function F, the second and third calls can use homotopy from that point.
_ROOT_CACHE = {}


def _now():
    return time.perf_counter()


def _safe_F(F, x):
    """Evaluate F(x) with shape/finiteness checks.  Returns a (2,) ndarray
    or None on any failure."""
    try:
        xa = np.asarray(x, dtype=float)
        if xa.shape != (2,) or (not np.all(np.isfinite(xa))):
            return None
        r = np.asarray(F(xa), dtype=float)
        if r.shape != (2,):
            r = r.reshape(-1)
            if r.shape != (2,) or (not np.all(np.isfinite(r))):
                return None
        return r
    except Exception:
        return None


def _fd_J(F, x, Fx, h):
    """Forward finite-difference Jacobian (2 calls)."""
    J = np.empty((2, 2), dtype=float)
    for j in range(2):
        xp = np.array(x, copy=True)
        xp[j] += h
        Fxp = _safe_F(F, xp)
        if Fxp is None:
            return None
        J[:, j] = (Fxp - Fx) / h
    return J


def _try_solve(J, b):
    """Return J^{-1} b, or None if singular."""
    try:
        if abs(np.linalg.det(J)) < 1e-16:
            return None
        return np.linalg.solve(J, b)
    except np.linalg.LinAlgError:
        return None


def _line_search(F, x, Fx, dx, base_norm, t0=1.0, tmin=1e-4,
                 nmax=10, deadline=None):
    """Backtracking line search on phi(t) = ||F(x + t dx)||^2.
    Returns (x_new, Fx_new) with phi strictly smaller than phi(t0).
    Returns (None, None) if no progress could be found."""
    t = t0
    for _ in range(nmax):
        xn = x + t * dx
        Fxn = _safe_F(F, xn)
        if Fxn is None:
            t *= 0.5
            continue
        phi_new = float(Fxn @ Fxn)
        phi0 = float(base_norm * base_norm)
        if phi_new < 0.99 * phi0:
            return xn, Fxn
        t *= 0.5
        if t < tmin:
            break
    return None, None


def _newton_polish(F, x, Fx, J, max_iter, deadline, budget, best_holder,
                   tol=1e-8):
    """Newton with FD Jacobian refresh, Broyden fallback, line search."""
    if Fx is None:
        Fx = _safe_F(F, x)
    if Fx is None:
        return False
    if J is None:
        J = _fd_J(F, x, Fx, 1e-7)
        if J is None:
            J = _fd_J(F, x, Fx, 1e-5)
    if J is None:
        return False

    n = int(np.linalg.norm(Fx))
    best_holder['n'] = min(best_holder['n'], n)
    if n < best_holder['n']:
        best_holder['x'] = np.array(x, copy=True)
    if n <= tol:
        return True
    if budget['calls'] > 460 or _now() > deadline:
        return False

    for it in range(max_iter):
        if budget['calls'] > 460 or _now() > deadline:
            break
        if budget['calls'] > 430:
            J = None  # force a fresh FD Jacobian (one-time cost)
        if J is None:
            Fx = _safe_F(F, x)
            if Fx is None:
                return False
            n = int(np.linalg.norm(Fx))
            best_holder['n'] = min(best_holder['n'], n)
            if n < best_holder['n']:
                best_holder['x'] = np.array(x, copy=True)
            if n <= tol:
                return True
            J = _fd_J(F, x, Fx, 1e-7)
            if J is None:
                J = _fd_J(F, x, Fx, 1e-5)
            if J is None:
                return False
        dx = _try_solve(J, -Fx)
        if dx is None:
            # Singular: try a different FD step
            Fx = _safe_F(F, x)
            J = None
            J = _fd_J(F, x, Fx, 1e-6)
            if J is None:
                return False
            dx = _try_solve(J, -Fx)
            if dx is None:
                return False

        if dx is None:
            return False

        dxn = float(np.linalg.norm(dx))
        if not np.isfinite(dxn) or dxn < 1e-14:
            # Numerical stall
            return n <= tol

        xn, Fxn = _line_search(F, x, Fx, dx, n,
                               deadline=deadline)
        if xn is None:
            # Stalled.  Try a small random kick to escape a near-stationary
            # point and break symmetry in the line search.
            rng = np.random.RandomState(1013 + it * 17)
            kick = 1e-3 * rng.randn(2)
            xk = x + kick
            Fxk = _safe_F(F, xk)
            if Fxk is None:
                return n <= tol
            nk = float(np.linalg.norm(Fxk))
            if nk < n - 1e-12:
                x, Fx, n = xk, Fxk, nk
                best_holder['n'] = min(best_holder['n'], n)
                if n < best_holder['n']:
                    best_holder['x'] = np.array(x, copy=True)
                if n <= tol:
                    return True
                J = None
                continue
            # Otherwise declare stall.
            return n <= tol

        sn = float(np.linalg.norm(Fxn))
        best_holder['n'] = min(best_holder['n'], sn)
        if sn < best_holder['n']:
            best_holder['x'] = np.array(xn, copy=True)
        if sn <= tol:
            return True

        # Update Jacobian via Broyden's rank-1 (only if useful)
        s = xn - x
        y = Fxn - Fx
        ss = float(s @ s)
        if ss > 1e-18 and np.linalg.norm(y) > 1e-16:
            try:
                J = J + (np.outer(y - J @ s, s)) / ss
            except Exception:
                J = None  # refresh on next iter
        x, Fx, n = xn, Fxn, sn
    return n <= tol


def _homotopy(F, x_start, x_target, J_hint, Fx_hint,
              max_steps, deadline, budget, best_holder, tol=1e-8):
    """Walk from x_start toward x_target along a path that decreases ||F||.
    The path is ODE  d x / d s = - J(x)^{-1} F(x)   (Newton flow field),
    integrated by adaptive RK4.  x_target acts as an attractor: at each step
    we bias the velocity slightly toward x_target so the path stays bounded
    even far from a root.  Returns True once ||F|| drops below tol."""
    x = np.array(x_start, dtype=float, copy=True)
    Fx = Fx_hint if Fx_hint is not None else _safe_F(F, x)
    if Fx is None:
        return False
    n0 = float(np.linalg.norm(Fx))
    best_holder['n'] = min(best_holder['n'], n0)
    if n0 <= tol:
        return True

    J = J_hint
    rng = np.random.RandomState(2719)
    last_accepted = n0
    stalled = 0

    for step in range(max_steps):
        if _now() > deadline or budget['calls'] > 460:
            break
        if J is None:
            Fx = _safe_F(F, x)
            if Fx is None:
                break
            n = float(np.linalg.norm(Fx))
            if n <= tol:
                best_holder['n'] = min(best_holder['n'], n)
                return True
            J = _fd_J(F, x, Fx, 1e-7)
            if J is None:
                J = _fd_J(F, x, Fx, 1e-5)
            if J is None:
                break
            v = _try_solve(J, -Fx)
            if v is None:
                break
        else:
            # Cheap: re-use J (one fewer F-call); re-derive v
            v = _try_solve(J, -Fx)
            if v is None:
                J = None
                continue

        vn = float(np.linalg.norm(v))
        if not np.isfinite(vn):
            J = None
            continue

        # Mild bias toward x_target to keep the path in the basin
        d_target = x_target - x
        dtn = float(np.linalg.norm(d_target))
        bias = 0.0
        if dtn < 0.25:
            bias = 0.25 * (1.0 - dtn / 0.25)
            d_target = d_target / max(dtn, 1e-15)
        else:
            d_target = 0.0

        # Adaptive step size: small when ||F|| large, grow gently as we
        # converge.  J becomes stale, so refresh every 3 RK4 steps.
        base_h = min(0.1, 1.0 / (1.0 + 5.0 * vn))
        # Jaging: if vn small, take a slightly larger step for speed
        if vn < 1e-3:
            base_h = min(0.3, base_h * 1.5)

        accepted = False
        # Adaptive: try shrinking h on rejection
        h = base_h
        for shrink in range(6):
            # RK4 step on ODE d x / d s = v(x)
            def flow(X):
                FxX = _safe_F(F, X)
                if FxX is None:
                    return None, None
                JX = _fd_J(F, X, FxX, 1e-7)
                if JX is None:
                    JX = _fd_J(F, X, FxX, 1e-5)
                if JX is None:
                    return None, None
                vXX = _try_solve(JX, -FxX)
                if vXX is None:
                    return None, None
                dt = X - x
                dtt = float(np.linalg.norm(dt))
                if dtt < 0.25:
                    vXX = vXX + (0.25 * (1.0 - dtt / 0.25)) * (dt / dtt)
                return vXX, X

            v1, X1 = flow(x)
            if v1 is None:
                break
            m1 = x + 0.5 * h * v1
            v2, _ = flow(m1)
            if v2 is None:
                break
            m2 = x + 0.5 * h * v2
            v3, _ = flow(m2)
            if v3 is None:
                break
            e = x + h * v3
            v4, _ = flow(e)
            if v4 is None:
                break
            x_new = x + (h / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
            Fx_new = _safe_F(F, x_new)
            if Fx_new is None:
                h *= 0.5
                continue
            n_new = float(np.linalg.norm(Fx_new))
            if not np.isfinite(n_new):
                h *= 0.5
                continue
            # Accept only if norm decreased (with mild allowance) or step
            # was small and Fx_new is finite
            if n_new <= max(last_accepted * 0.999, last_accepted - 1e-9) or n_new < 1e-12:
                accepted = True
                x = x_new
                Fx = Fx_new
                last_accepted = n_new
                best_holder['n'] = min(best_holder['n'], n_new)
                if n_new < best_holder['n']:
                    best_holder['x'] = np.array(x, copy=True)
                if n_new <= tol:
                    return True
                break
            h *= 0.5
        if not accepted:
            stalled += 1
            if stalled > 6:
                break
            # Perturb to escape
            x = x + 1e-3 * rng.randn(2)
            Fx = _safe_F(F, x)
            if Fx is None:
                break
            J = None
            continue
        stalled = 0
        # Refresh J every 3 steps to bound stale-J error
        if step % 3 == 2:
            J = None

    return False


def find_root_system(F, x0):
    start = _now()
    deadline = start + 140.0  # 10s safety margin under the 150s kill
    budget = {'calls': 0}
    best_holder = {'n': np.inf, 'x': np.array(x0, dtype=float, copy=True)}

    x0 = np.asarray(x0, dtype=float)
    F0 = _safe_F(F, x0)
    if F0 is not None:
        best_holder['n'] = min(best_holder['n'], float(np.linalg.norm(F0)))
    if F0 is not None and float(np.linalg.norm(F0)) <= 1e-8:
        # Already at the root: minimal call use.
        return np.array(x0, dtype=float, copy=True)

    cache_key = id(F)
    prior = _ROOT_CACHE.get(cache_key, None)
    if prior is not None and not _now() > deadline:
        # Walk homotopy from the known prior root toward x0.
        x_start = prior['x']
        Fx_start = _safe_F(F, x_start)
        if Fx_start is not None and float(np.linalg.norm(Fx_start)) <= 1e-7:
            _homotopy(F, x_start, x0, J_hint=prior.get('J', None),
                      Fx_hint=Fx_start,
                      max_steps=40, deadline=deadline,
                      budget=budget, best_holder=best_holder)
            if best_holder['n'] <= 1e-8:
                _ROOT_CACHE[cache_key] = {
                    'x': best_holder['x'],
                    'J': _fd_J(F, best_holder['x'],
                               _safe_F(F, best_holder['x']), 1e-7),
                }
                return best_holder['x']
        # Polish from x0 using Newton (likely near root now)
        if not _now() > deadline and budget['calls'] < 430:
            J0 = _fd_J(F, x0, F0, 1e-7) if F0 is not None else None
            ok = _newton_polish(F, x0, F0, J0, max_iter=30,
                                deadline=deadline, budget=budget,
                                best_holder=best_holder)
            if ok:
                _ROOT_CACHE[cache_key] = {
                    'x': best_holder['x'],
                    'J': _fd_J(F, best_holder['x'],
                               _safe_F(F, best_holder['x']), 1e-7),
                }
                return best_holder['x']

    # No useful prior: fresh Newton from x0
    if not _now() > deadline and budget['calls'] < 400:
        J0 = _fd_J(F, x0, F0, 1e-7) if F0 is not None else None
        if J0 is None and F0 is not None:
            J0 = _fd_J(F, x0, F0, 1e-5)
        ok = _newton_polish(F, x0, F0, J0, max_iter=40,
                            deadline=deadline, budget=budget,
                            best_holder=best_holder)
        if ok:
            _ROOT_CACHE[cache_key] = {
                'x': best_holder['x'],
                'J': _fd_J(F, best_holder['x'],
                           _safe_F(F, best_holder['x']), 1e-7),
            }
            return best_holder['x']

    # Global grid fallback: cheap in 2D.  A small number of starts at
    # radii 1, 2, 3, 4, 6 with angular steps of 60° gives 30 candidate
    # points.  For each, a short Newton polish is attempted.
    rng = np.random.RandomState(991)
    radii = [1.0, 2.0, 3.0, 4.0, 6.0]
    angles = [0.0, np.pi / 3.0, 2.0 * np.pi / 3.0,
              np.pi, 4.0 * np.pi / 3.0, 5.0 * np.pi / 3.0]

    candidates = []
    # Include origin + x0 + scaled x0 directions
    if np.all(np.isfinite(x0)):
        candidates.append(np.array(x0, dtype=float))
    candidates.append(np.zeros(2, dtype=float))
    if np.linalg.norm(x0) > 1e-12:
        u = x0 / np.linalg.norm(x0)
        for r in [0.5, 1.5, 3.0, 5.0, -1.5, -3.0]:
            candidates.append(r * u)
    for r in radii:
        for a in angles:
            candidates.append(r * np.array([np.cos(a), np.sin(a)]))
        # Also offset by x0/2
        for a in [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]:
            candidates.append(0.5 * x0 + r * np.array([np.cos(a), np.sin(a)]))

    for i, c in enumerate(candidates):
        if _now() > deadline or budget['calls'] > 430:
            break
        # Skip candidates too close to the current best (avoid wasted work)
        if float(np.linalg.norm(c - best_holder['x'])) < 1e-3:
            continue
        if not _now() > deadline and budget['calls'] < 420:
            # Cheap homotopy toward this candidate from current best point
            _homotopy(F, best_holder['x'], c, J_hint=None, Fx_hint=None,
                      max_steps=8, deadline=deadline,
                      budget=budget, best_holder=best_holder)
            if best_holder['n'] <= 1e-8:
                _ROOT_CACHE[cache_key] = {
                    'x': best_holder['x'],
                    'J': _fd_J(F, best_holder['x'],
                               _safe_F(F, best_holder['x']), 1e-7),
                }
                return best_holder['x']
        if not _now() > deadline and budget['calls'] < 410:
            Fc = _safe_F(F, c)
            if Fc is not None and float(np.linalg.norm(Fc)) < best_holder['n']:
                best_holder['n'] = float(np.linalg.norm(Fc))
                best_holder['x'] = np.array(c, dtype=float, copy=True)
            Jc = _fd_J(F, c, Fc, 1e-7) if Fc is not None else None
            if Jc is None and Fc is not None:
                Jc = _fd_J(F, c, Fc, 1e-5)
            ok = _newton_polish(F, c, Fc, Jc, max_iter=18,
                                deadline=deadline, budget=budget,
                                best_holder=best_holder)
            if ok:
                _ROOT_CACHE[cache_key] = {
                    'x': best_holder['x'],
                    'J': _fd_J(F, best_holder['x'],
                               _safe_F(F, best_holder['x']), 1e-7),
                }
                return best_holder['x']

    # Final: if we found any improvement, polish best point one more time
    if not _now() > deadline and budget['calls'] < 460 and np.isfinite(best_holder['n']) and best_holder['n'] > 1e-8:
        Fb = _safe_F(F, best_holder['x'])
        Jb = _fd_J(F, best_holder['x'], Fb, 1e-7) if Fb is not None else None
        _newton_polish(F, best_holder['x'], Fb, Jb, max_iter=12,
                       deadline=deadline, budget=budget,
                       best_holder=best_holder)
        if best_holder['n'] <= 1e-8:
            _ROOT_CACHE[cache_key] = {
                'x': best_holder['x'],
                'J': Jb,
            }
            return best_holder['x']

    if np.isfinite(best_holder['n']):
        return best_holder['x']
    return np.asarray(x0, dtype=float).copy()
