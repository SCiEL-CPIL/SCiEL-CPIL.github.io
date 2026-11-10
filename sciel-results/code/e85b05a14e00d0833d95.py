"""
Minimize logarithmic energy of 150 points on the unit sphere.
E = sum_{i<j} log(1/|x_i - x_j|)  (natural log, chordal distances)

Strategy (combined):
  - Global: spherical-coordinate L-BFGS-B from diverse starts (fast, robust)
  - Local: Riemannian L-BFGS directly on the sphere (no projection artifacts)
  - Polish: adaptive tangent gradient descent + Newton on tangent space
  - Final: ultra-tight Riemannian L-BFGS + CG to drive local minimum
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_DEADLINE = 150.0
_START = time.time()
_SAFETY = 8.0
_TIME_LIMIT = _DEADLINE - _SAFETY


def _time_left():
    return _TIME_LIMIT - (time.time() - _START)


def _retract(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return X / n


def _energy_and_grad(X, eps=1e-14):
    """Vectorized energy and Euclidean gradient. Returns (E, G)."""
    diff = X[:, None, :] - X[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, eps * eps)
    d = np.sqrt(d2)
    log_d = np.log(d)
    np.fill_diagonal(log_d, 0.0)
    E = -0.5 * float(np.sum(log_d))
    inv = 1.0 / d2
    np.fill_diagonal(inv, 0.0)
    S1 = inv @ X
    S2 = (inv.sum(axis=1, keepdims=True)) * X
    G = S2 - S1
    return E, G


def _project_tangent(G, X):
    a = np.einsum('ij,ij->i', G, X)
    return G - a[:, None] * X


def _bfgs_two_loop(v, s_list, y_list, rho_list, m):
    n = len(v)
    q = v.copy()
    alphas = np.zeros(m)
    for k in range(m - 1, -1, -1):
        s = s_list[k]
        y = y_list[k]
        rho = rho_list[k]
        alphas[k] = rho * float(np.dot(s, q))
        q = q - alphas[k] * y
    r = np.zeros(n)
    if y_list:
        yy = float(np.dot(y_list[-1], y_list[-1]))
        if yy > 1e-20:
            gamma = float(np.dot(s_list[-1], y_list[-1])) / yy
            r = gamma * q
    for k in range(m):
        s = s_list[k]
        y = y_list[k]
        rho = rho_list[k]
        beta = rho * float(np.dot(y, r))
        r = r + (alphas[k] - beta) * s
    return q + r


def _lbfgs_sphere(X0, time_budget, m=15, max_iter=5000, tol=1e-10):
    """Riemannian L-BFGS on the sphere with retraction and strong Wolfe line search."""
    X = _retract(X0.copy())
    E, G = _energy_and_grad(X)
    g = _project_tangent(G, X)
    flat = g.ravel()

    s_list, y_list, rho_list = [], [], []
    it = 0
    best_E = E
    best_X = X.copy()

    while _time_left() > 2.0 and it < max_iter:
        p_flat = -_bfgs_two_loop(flat, s_list, y_list, rho_list, min(len(s_list), m))
        p = p_flat.reshape(N, 3)
        p = _project_tangent(p, X)
        pnorm = float(np.linalg.norm(p))
        if pnorm < 1e-13:
            break

        gnorm = float(np.linalg.norm(flat))
        if gnorm < tol:
            break

        c1 = 1e-4
        c2 = 0.9
        slope = float(np.dot(flat, p_flat))
        if slope >= 0:
            p = -g
            p = _project_tangent(p, X)
            p_flat = p.ravel()
            slope = float(np.dot(flat, p_flat))
            if slope >= 0:
                break

        alpha = 1.0
        Xn, En, Gn, gn = X, E, G, g
        for _ in range(60):
            if _time_left() < 3.0:
                break
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En <= E + c1 * alpha * slope:
                if float(np.dot(gn.ravel(), p_flat)) <= c2 * alpha * slope:
                    break
            alpha *= 0.5

        if En >= E:
            alpha = max(alpha * 0.1, 1e-8)
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En >= E:
                break

        dX = (Xn - X).ravel()
        dy = gn.ravel() - flat
        sy = float(np.dot(dX, dy))
        if sy > 1e-14:
            s_list.append(dX)
            y_list.append(dy)
            rho_list.append(1.0 / sy)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)

        X = Xn
        E = En
        G = Gn
        g = gn
        flat = g.ravel()
        it += 1

        if E < best_E:
            best_E = E
            best_X = X.copy()

    return best_X, best_E


def _cg_sphere(X0, time_budget, max_iter=2000, tol=1e-10):
    """Riemannian conjugate gradient on the sphere with backtracking line search."""
    X = _retract(X0.copy())
    E, G = _energy_and_grad(X)
    g = _project_tangent(G, X)
    p = -g
    best_E = E
    best_X = X.copy()
    beta = 0.0
    it = 0

    while _time_left() > 2.0 and it < max_iter:
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            break

        p = _project_tangent(p, X)
        pnorm = float(np.linalg.norm(p))
        if pnorm < 1e-14:
            break

        slope = float(np.dot(g.ravel(), p.ravel()))
        if slope >= 0:
            p = -g
            p = _project_tangent(p, X)
            slope = float(np.dot(g.ravel(), p.ravel()))
            if slope >= 0:
                break
            beta = 0.0

        alpha = 1.0
        Xn, En, Gn, gn = X, E, G, g
        for _ in range(80):
            if _time_left() < 3.0:
                break
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En <= E + 1e-4 * alpha * slope:
                break
            alpha *= 0.5

        if En >= E:
            alpha = max(alpha * 0.1, 1e-8)
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En >= E:
                break

        gnew2 = float(np.dot(gn.ravel(), gn.ravel()))
        if gnorm > 0:
            beta = gnew2 / (gnorm * gnorm)
            if beta > 1.0:
                beta = 1.0

        p = -gn + beta * p
        p = _project_tangent(p, X)

        X = Xn
        E = En
        G = Gn
        g = gn
        it += 1

        if E < best_E:
            best_E = E
            best_X = X.copy()

    return best_X, best_E


def _fibonacci_spiral(offset=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N, 3))
    return _retract(v)


def _structured_grid():
    n_lat, n_lon = 12, 13
    points = []
    for i in range(n_lat):
        theta = (i + 0.5) * math.pi / n_lat
        for j in range(n_lon):
            phi = (j + 0.5) * 2 * math.pi / n_lon
            points.append([math.sin(theta) * math.cos(phi),
                           math.sin(theta) * math.sin(phi),
                           math.cos(theta)])
    while len(points) < N:
        rng = np.random.default_rng(len(points) * 7919)
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        points.append(v)
    return _retract(np.array(points[:N]))


def _perturb(X, seed, sigma):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=X.shape) * sigma
    return _retract(X + noise)


# --- Spherical-coordinate optimizer (fast global search) ---
def _cartesian_to_spherical(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    theta = np.arctan2(y, x)
    phi = np.arccos(np.clip(z, -1.0, 1.0))
    return np.column_stack([theta, phi])


def _spherical_to_cartesian(params):
    theta, phi = params[:, 0], params[:, 1]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def _energy_grad_spherical(params):
    X = _spherical_to_cartesian(params)
    diff = X[:, None, :] - X[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-12)
    dist = np.sqrt(dist_sq)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    e = -np.sum(np.log(dist[mask]))
    inv = 1.0 / dist_sq
    np.fill_diagonal(inv, 0.0)
    grad_cart = -np.sum(diff * inv[:, :, None], axis=1)
    theta, phi = params[:, 0], params[:, 1]
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    gtheta = -sp * st * grad_cart[:, 0] + sp * ct * grad_cart[:, 1]
    gphi = cp * ct * grad_cart[:, 0] + cp * st * grad_cart[:, 1] - sp * grad_cart[:, 2]
    gtheta[sp < 1e-6] = 0.0
    return e, np.column_stack([gtheta, gphi]).flatten()


def _optimize_spherical(X_init, max_iter=500):
    params = _cartesian_to_spherical(X_init).flatten()

    def fun(x):
        e, _ = _energy_grad_spherical(x.reshape(N, 2))
        return e

    def jac(x):
        _, g = _energy_grad_spherical(x.reshape(N, 2))
        return g

    try:
        res = minimize(fun, params, jac=jac, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        X = _spherical_to_cartesian(res.x.reshape(N, 2))
        X = _retract(X)
        return X, _energy_and_grad(X)[0]
    except Exception:
        return X_init, _energy_and_grad(X_init)[0]


# --- Adaptive tangent gradient-descent polish ---
def _gd_polish(X0, time_budget, lr_init=1e-4, min_lr=1e-9):
    X = _retract(X0.copy())
    E, _ = _energy_and_grad(X)
    best_X, best_E = X.copy(), E
    lr = lr_init
    patience = 0
    end = time.time() + time_budget
    while time.time() < end and _time_left() > 2.0:
        _, G = _energy_and_grad(X)
        g = _project_tangent(G, X)
        Xn = _retract(X - lr * g)
        En, _ = _energy_and_grad(Xn)
        if En < best_E - 1e-15:
            best_X, best_E = Xn.copy(), En
            patience = 0
            lr = min(lr * 1.03, 0.05)
        else:
            patience += 1
            lr = max(lr * 0.9, min_lr)
        X = Xn
        if patience > 120:
            break
    return best_X, best_E


# --- Newton polish on tangent space ---
def _hessian_tangent(X):
    diff = X[:, None, :] - X[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-14)
    W = 1.0 / d2
    S = W.sum(axis=1)
    Wx = W @ X
    H = np.zeros((N * 3, N * 3))
    eye3 = np.eye(3)
    for i in range(N):
        xi = X[i]
        H[i*3:(i+1)*3, i*3:(i+1)*3] = S[i] * eye3 - 2.0 * np.outer(xi, Wx[i])
    return H


def _newton_polish(X0, time_budget, max_iter=60):
    X = _retract(X0.copy())
    E, G = _energy_and_grad(X)
    g = _project_tangent(G, X)
    best_X, best_E = X.copy(), E
    end = time.time() + time_budget
    it = 0
    while time.time() < end and _time_left() > 2.0 and it < max_iter:
        gnorm = float(np.linalg.norm(g))
        if gnorm < 1e-12:
            break
        try:
            H = _hessian_tangent(X)
            rhs = -g.ravel()
            lam = 1e-8
            try:
                p_flat = np.linalg.solve(H + lam * np.eye(N * 3), rhs)
            except np.linalg.LinAlgError:
                p_flat = np.linalg.lstsq(H + lam * np.eye(N * 3), rhs, rcond=None)[0]
            p = p_flat.reshape(N, 3)
            p = _project_tangent(p, X)
            pn = float(np.linalg.norm(p))
            if pn < 1e-14:
                break
            slope = float(np.dot(g.ravel(), p.ravel()))
            alpha = 1.0
            Xn, En, Gnn = X, E, G
            gn = g
            improved = False
            for _ in range(40):
                Xn = _retract(X + alpha * p)
                En, Gnn = _energy_and_grad(Xn)
                gn = _project_tangent(Gnn, Xn)
                if En <= E + 1e-4 * alpha * slope:
                    improved = True
                    break
                alpha *= 0.5
            if not improved or En >= E:
                p = -g
                p = _project_tangent(p, X)
                alpha = 1e-3
                Xn = _retract(X + alpha * p)
                En, Gnn = _energy_and_grad(Xn)
                gn = _project_tangent(Gnn, Xn)
                if En >= E:
                    break
            X, E, G, g = Xn, En, Gnn, gn
            if E < best_E:
                best_E, best_X = E, X.copy()
            it += 1
        except Exception:
            break
    return best_X, best_E


def _validate(X):
    if not np.all(np.isfinite(X)):
        return False
    norms = np.linalg.norm(X, axis=1)
    if not np.all(np.abs(norms - 1.0) < 1e-6):
        return False
    diff = X[:, None, :] - X[None, :, :]
    dd = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(dd, 2.0)
    if float(dd.min()) < 1e-9:
        return False
    return True


def construct():
    best_X = _fibonacci_spiral()
    best_E = _energy_and_grad(best_X)[0]

    # Phase 1: Fast global search with spherical-coordinate L-BFGS-B
    # Use diverse starting points; this is faster per restart than Riemannian L-BFGS
    starts = []
    for offset in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        starts.append(_fibonacci_spiral(offset))
    starts.append(_structured_grid())
    for s in [42, 123, 456, 789, 1024, 2048, 3000, 4096, 5000, 6000, 7000, 8000, 9000, 10000]:
        starts.append(_random_points(s))

    for start in starts:
        if _time_left() > 8.0:
            X, E = _optimize_spherical(start, max_iter=400)
            if E < best_E:
                best_E, best_X = E, X

    # Phase 2: Riemannian L-BFGS polish on the best from global search
    if _time_left() > 15.0:
        X, E = _lbfgs_sphere(best_X, 12.0, m=15, max_iter=5000, tol=1e-11)
        if E < best_E:
            best_E, best_X = E, X

    # Phase 3: Diversifying restarts around the current best
    if _time_left() > 25.0:
        for s in [11, 22, 33, 44, 55, 66, 77]:
            if _time_left() < 6.0:
                break
            start = _perturb(best_X, s * 7919, 0.12)
            X, E = _optimize_spherical(start, max_iter=300)
            if E < best_E:
                best_E, best_X = E, X
            # Polish with Riemannian L-BFGS
            if _time_left() > 10.0:
                X2, E2 = _lbfgs_sphere(X, 8.0, m=15, max_iter=3000, tol=1e-11)
                if E2 < best_E:
                    best_E, best_X = E2, X2

    # Phase 4: Multi-scale local perturbations
    for sigma in [0.25, 0.15, 0.08, 0.04, 0.02]:
        if _time_left() > 8.0:
            for trial in range(3):
                if _time_left() < 4.0:
                    break
                start = _perturb(best_X, trial * 1000 + int(sigma * 10000), sigma)
                X, E = _lbfgs_sphere(start, 6.0, m=15, max_iter=3000, tol=1e-11)
                if E < best_E:
                    best_E, best_X = E, X

    # Phase 5: Riemannian CG polish (different path)
    if _time_left() > 12.0:
        X, E = _cg_sphere(best_X, 10.0, max_iter=2000, tol=1e-12)
        if E < best_E:
            best_E, best_X = E, X

    # Phase 6: High-precision Riemannian L-BFGS
    if _time_left() > 15.0:
        X, E = _lbfgs_sphere(best_X, 12.0, m=20, max_iter=5000, tol=1e-13)
        if E < best_E:
            best_E, best_X = E, X

    # Phase 7: Newton polish for tightest local minimum
    if _time_left() > 12.0:
        X, E = _newton_polish(best_X, 10.0, max_iter=60)
        if E < best_E:
            best_E, best_X = E, X

    # Phase 8: Final ultra-tight passes
    if _time_left() > 8.0:
        X2, E2 = _lbfgs_sphere(best_X, 6.0, m=25, max_iter=3000, tol=1e-15)
        if E2 < best_E:
            best_E, best_X = E2, X2
    if _time_left() > 5.0:
        X3, E3 = _cg_sphere(best_X, 4.0, max_iter=1000, tol=1e-15)
        if E3 < best_E:
            best_E, best_X = E3, X3

    # Final gradient descent polish for any remaining gradient
    if _time_left() > 4.0:
        X4, E4 = _gd_polish(best_X, 3.0, lr_init=1e-5, min_lr=1e-12)
        if E4 < best_E:
            best_E, best_X = E4, X4

    best_X = _retract(best_X)
    if not _validate(best_X):
        best_X = _fibonacci_spiral()
    return best_X
