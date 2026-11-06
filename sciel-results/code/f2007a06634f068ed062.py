"""Universal 8-node positive quadrature for dimensionless luminosity distance."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _cpl_g(x, w0, wa):
    """Model-dependent factor g(x)=1/E(1-x) for CPL. x = z/(1+z), a=1-x."""
    a = 1.0 - x
    wa_eff = wa / a
    g = np.power(a, 3.0 * (1.0 + w0 + wa_eff))
    return np.where(np.isfinite(g) & (g > 0), g, 1.0)


def _project_positive(w, tol=1e-10):
    """Project weights onto the cone w_i>0, sum w = 1 (Euclidean projection)."""
    w = np.asarray(w, dtype=float)
    n = w.size
    for _ in range(500):
        lam = max(0.0, (w.sum() - 1.0) / n)
        w_new = np.maximum(w - lam, tol)
        if np.abs(w_new.sum() - 1.0) < 1e-13 and np.all(w_new >= tol):
            w = w_new
            break
        w = w_new
    w = np.maximum(w, tol)
    w = w / w.sum()
    return w


def _basis_matrix(x, degree):
    """Chebyshev basis T_k(2x-1) on x in [0,1]."""
    t = 2.0 * x - 1.0
    K = degree + 1
    B = np.empty((x.size, K))
    B[:, 0] = 1.0
    if K > 1:
        B[:, 1] = t
    for k in range(2, K):
        B[:, k] = 2.0 * t * B[:, k - 1] - B[:, k - 2]
    return B


def _build_library(w0_grid, wa_grid, M=6000):
    """Integrands f_m(x)=x*g_m(x) and their true integrals over the model grid."""
    n_models = len(w0_grid) * len(wa_grid)
    x = np.linspace(1e-4, 1.0 - 1e-4, M)
    f_true = np.empty(n_models)
    lib = np.empty((n_models, M))
    k = 0
    for w0 in w0_grid:
        for wa in wa_grid:
            g = _cpl_g(x, w0, wa)
            f = x * g
            lib[k] = f
            f_true[k] = np.trapz(f, x)
            k += 1
    f_true = np.maximum(f_true, 1e-30)
    return x, lib, f_true


def _solve(w0_grid, wa_grid):
    x, lib, f_true = _build_library(w0_grid, wa_grid, M=6000)
    n_models = lib.shape[0]
    n_nodes = 8

    # --- Step 1: weighted least-squares fit of the integrands to a Chebyshev basis.
    # Choose a degree that spans the dark-energy variation across the CPL grid.
    degree = 7
    B = _basis_matrix(x, degree)            # (M, K)
    # Weight by 1/f_true to balance relative error (small integrals -> large rel err).
    wgt = 1.0 / np.sqrt(np.maximum(f_true, 1e-3))
    wgt = wgt / wgt.max()
    A = lib * wgt[:, None]                  # (m, M)
    b = f_true * wgt                        # (m,)
    # Fit: A ≈ C @ B^T  =>  C (m, K) minimises ||A - C B^T||_F
    # Solve (B B^T) C^T = B A^T
    G = B @ B.T                             # (K, K)
    rhs = B @ A.T                           # (K, m)
    C = np.linalg.solve(G, rhs).T           # (m, K)
    # Basis coefficients of the "average" integrand (mean over models).
    c_mean = C.mean(axis=0)                 # (K,)
    # Also keep the per-model coefficients for a richer node search.

    # --- Step 2: choose nodes as the zeros of the weighted Chebyshev polynomial,
    # then refine by a global+local search on the worst relative error.
    # Start from the (shifted) Chebyshev nodes, which are well-distributed on [0,1].
    kth = np.arange(1, n_nodes + 1)
    t0 = np.cos(np.pi * (kth - 0.5) / n_nodes)   # in (-1,1)
    nodes = (t0 + 1.0) / 2.0
    nodes = np.sort(np.clip(nodes, 1e-3, 1 - 1e-3))

    # Weights from least squares on the mean integrand (positively projected).
    def _weights_from_nodes(nd):
        Bn = _basis_matrix(nd, degree)          # (8, K)
        # Fit c_mean ≈ Bn^T w  =>  w = (Bn Bn^T)^-1 Bn c_mean
        Gw = Bn.T @ Bn
        try:
            w = np.linalg.solve(Gw, Bn @ c_mean)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(Bn.T, c_mean, rcond=None)[0]
        return _project_positive(w)

    weights = _weights_from_nodes(nodes)

    # --- Step 3: joint refinement. Minimize worst relative error over the model
    # library with respect to nodes (weights re-solved at each step).
    def _worst(nodes_in, weights_in):
        S = np.empty((n_models, n_nodes))
        for i in range(n_nodes):
            S[:, i] = _cpl_g(nodes_in[i], 0.0, 0.0)  # placeholder, recomputed below
        # recompute per model
        S = np.array([_cpl_g(nodes_in, w0, wa) for w0, wa in
                      zip(*np.meshgrid(w0_grid, wa_grid, indexing='ij')).T]).reshape(n_models, n_nodes)
        S = S * nodes_in
        Q = S @ weights_in
        err = np.abs(Q - f_true) / f_true
        return err.max()

    def objective(v):
        nd = np.sort(np.clip(v, 1e-3, 1 - 1e-3))
        for i in range(1, n_nodes):
            if nd[i] <= nd[i - 1] + 1e-6:
                nd[i] = nd[i - 1] + 1e-6
        w = _weights_from_nodes(nd)
        return _worst(nd, w)

    best_nodes = nodes.copy()
    best_weights = weights.copy()
    best_val = objective(nodes)

    # Multi-start local refinement.
    starts = [nodes.copy()]
    # Add a few perturbations around the Chebyshev start.
    rng = np.random.default_rng(0)
    for _ in range(6):
        p = nodes + rng.normal(0.0, 0.02, size=n_nodes)
        p = np.sort(np.clip(p, 1e-3, 1 - 1e-3))
        for i in range(1, n_nodes):
            if p[i] <= p[i - 1] + 1e-6:
                p[i] = p[i - 1] + 1e-6
        starts.append(p)

    for s in starts:
        res = minimize(objective, s, method="L-BFGS-B",
                       bounds=[(1e-3, 1 - 1e-3)] * n_nodes,
                       options={"maxiter": 400, "ftol": 1e-14})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_nodes = np.sort(np.clip(res.x, 1e-3, 1 - 1e-3))
            for i in range(1, n_nodes):
                if best_nodes[i] <= best_nodes[i - 1] + 1e-6:
                    best_nodes[i] = best_nodes[i - 1] + 1e-6
            best_weights = _weights_from_nodes(best_nodes)

    # Nelder-Mead polish on the best.
    res3 = minimize(objective, best_nodes, method="Nelder-Mead",
                    options={"maxiter": 4000, "xatol": 1e-12, "fatol": 1e-15})
    if res3.fun < best_val:
        best_val = res3.fun
        best_nodes = np.sort(np.clip(res3.x, 1e-3, 1 - 1e-3))
        for i in range(1, n_nodes):
            if best_nodes[i] <= best_nodes[i - 1] + 1e-6:
                best_nodes[i] = best_nodes[i - 1] + 1e-6
        best_weights = _weights_from_nodes(best_nodes)

    best_nodes = np.clip(best_nodes, 1e-3, 1 - 1e-3)
    best_weights = _project_positive(best_weights)
    return {"nodes": best_nodes, "weights": best_weights}


def construct():
    w0_grid = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    wa_grid = np.array([-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0])
    try:
        rule = _solve(w0_grid, wa_grid)
    except Exception:
        n = 8
        rule = {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}

    nodes = np.asarray(rule["nodes"], dtype=float)
    weights = np.asarray(rule["weights"], dtype=float)
    nodes = np.clip(np.sort(nodes), 1e-9, 1 - 1e-9)
    for i in range(1, 8):
        if nodes[i] <= nodes[i - 1] + 1e-9:
            nodes[i] = nodes[i - 1] + 1e-9
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
