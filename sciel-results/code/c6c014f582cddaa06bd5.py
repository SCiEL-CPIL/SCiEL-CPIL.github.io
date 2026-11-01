"""Fresh design: Halton init + L-BFGS-B + Riemannian CG + restarts, with deadline guard."""
import time
import math
import numpy as np

N = 150
DEADLINE = 150.0
SAFETY_MARGIN = 8.0  # seconds before the hard limit


def _log_energy(X):
    """Sum over i<j of log(1/d) with chordal distances, via math.fsum."""
    n = X.shape[0]
    terms = []
    for i in range(n):
        d = X[i] - X[i + 1:]
        dist = np.sqrt(np.sum(d * d, axis=1))
        terms.extend(math.log(1.0 / di) for di in dist)
    return math.fsum(terms)


def _energy_and_grad(X):
    """Energy and gradient (in ambient R^3) for log(1/d) sum over pairs."""
    n = X.shape[0]
    diff = X[:, None, :] - X[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    # energy
    E = 0.5 * np.sum(np.log(1.0 / dist))
    # grad: d/dX_i sum_{j != i} log(1/|Xi-Xj|) = sum_{j != i} (Xi - Xj)/|Xi-Xj|^2
    inv = np.zeros_like(dist)
    inv[dist > 0] = 1.0 / (dist[dist > 0] ** 2)
    np.fill_diagonal(inv, 0.0)
    G = np.sum(diff * inv[:, :, None], axis=1)
    return E, G


def _project_to_sphere(X):
    """Normalize each row to unit norm."""
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm < 1e-15] = 1.0
    return X / nrm


def _halton_init(num_dims, skip=1):
    """Halton sequence in [0,1)^num_dims, then map to sphere via Fibonacci-like."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    pts = []
    for i in range(skip, skip + num_dims):
        pts.append([halton(i, b) for b in (2, 3, 5)])
    pts = np.array(pts)
    # Map [0,1)^3 to sphere: use a spiral parameterization
    i = np.arange(num_dims)
    z = 1.0 - 2.0 * (i + 0.5) / num_dims
    # add a small deterministic perturbation from halton to break symmetry
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + 2.0 * np.pi * pts[:, 0] * 0.01
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _fibonacci_init():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _lbfgs_refine(X, time_budget):
    """L-BFGS-B on ambient coords with periodic re-projection."""
    from scipy.optimize import minimize
    best = X.copy()
    best_E = _log_energy(best)
    t0 = time.time()
    it = [0]
    callback = None

    def fun(g):
        it[0] += 1
        Xc = _project_to_sphere(g.reshape(N, 3))
        E, _ = _energy_and_grad(Xc)
        return E

    def jac(g):
        Xc = _project_to_sphere(g.reshape(N, 3))
        _, G = _energy_and_grad(Xc)
        # project gradient to tangent: G - (G.x) x
        proj = G - np.sum(G * Xc, axis=1, keepdims=True) * Xc
        return proj.reshape(-1)

    res = minimize(fun, X.reshape(-1), jac=jac, method='L-BFGS-B',
                   options={'maxiter': 400, 'maxfun': 800, 'maxls': 40})
    Xr = _project_to_sphere(res.x.reshape(N, 3))
    Er = _log_energy(Xr)
    if Er < best_E:
        best = Xr
        best_E = Er
    return best, best_E


def _riemannian_cg(X, time_budget, max_iter=2000):
    """Riemannian conjugate gradient on the sphere (tangent space)."""
    best = X.copy()
    best_E = _log_energy(best)
    t0 = time.time()
    Xc = X.copy()
    E, G = _energy_and_grad(Xc)
    proj = G - np.sum(G * Xc, axis=1, keepdims=True) * Xc
    d = -proj
    beta = 0.0
    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        # backtracking line search in tangent space
        alpha = 0.5
        prev_E = E
        for ls in range(30):
            Xn = Xc + alpha * d
            Xn = _project_to_sphere(Xn)
            En, Gn = _energy_and_grad(Xn)
            if En < prev_E or ls == 29:
                break
            alpha *= 0.5
        if En >= prev_E and ls > 0:
            # accept smallest step that improved, else break
            if En < best_E:
                best = Xn
                best_E = En
            break
        Xc, E = Xn, En
        if E < best_E:
            best = Xc.copy()
            best_E = E
        proj = Gn - np.sum(Gn * Xc, axis=1, keepdims=True) * Xc
        if np.linalg.norm(proj) < 1e-10:
            break
        beta = np.sum(proj * proj) / max(np.sum(Gn * Gn), 1e-30)
        d = -proj + beta * d
    return best, best_E


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY_MARGIN

    # multiple initializations
    inits = []
    inits.append(_fibonacci_init())
    inits.append(_halton_init(N))
    # a perturbed fibonacci
    rng = np.random.default_rng(12345)
    inits.append(_project_to_sphere(_fibonacci_init() + 0.05 * rng.standard_normal((N, 3))))
    inits.append(_project_to_sphere(_fibonacci_init() + 0.2 * rng.standard_normal((N, 3))))

    best = inits[0]
    best_E = _log_energy(best)

    # refine each init with both methods, tracking deadline
    for k, X0 in enumerate(inits):
        if time.time() > deadline:
            break
        rem = max(0.5, deadline - time.time())
        # split budget between methods
        Xa, Ea = _lbfgs_refine(X0, rem * 0.5)
        if time.time() > deadline:
            if Ea < best_E:
                best, best_E = Xa, Ea
            break
        Xb, Eb = _riemannian_cg(X0, max(0.5, deadline - time.time()))
        for Xc, Ec in ((Xa, Ea), (Xb, Eb)):
            if Ec < best_E:
                best, best_E = Xc, Ec

    # final polish with L-BFGS-B on the best if time remains
    if time.time() < deadline:
        Xp, Ep = _lbfgs_refine(best, max(0.5, deadline - time.time()))
        if Ep < best_E:
            best, best_E = Xp, Ep

    # ensure valid: finite, normalized, no near-duplicates
    best = _project_to_sphere(best)
    if not np.all(np.isfinite(best)):
        best = _fibonacci_init()
    return best
