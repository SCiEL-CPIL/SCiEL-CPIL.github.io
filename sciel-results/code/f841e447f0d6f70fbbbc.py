"""Fixed-depth QAOA (p=2) angle design, optimized per graph at call time."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import erf

# --- SU(2) rotation helpers -------------------------------------------------
# R_x(a) = [[cos, -i sin],[-i sin, cos]]
# R_z(a) = [[e^{-ia}, 0],[0, e^{ia}]]
# R_x(a) @ R_z(g) @ R_x(b)  ->  M = [[cos a cos b - i e^{-ig} sin a sin b,
#                                       -i e^{-ig} sin a cos b - i sin b cos a],
#                                     [-i e^{ig} sin b cos a - i cos b sin a,
#                                       cos a cos b - i e^{ig} sin a sin b]]
# (verified: M @ R_x(-b) @ R_z(-g) @ R_x(-a) == I)
# For a state |psi>=(c0,c1): after M, new c0 = M[0,0]c0 + M[0,1]c1,
#                              new c1 = M[1,0]c0 + M[1,1]c1.

_CX = np.empty(2, dtype=np.float64); SX = np.empty(2, dtype=np.float64)
_CY = np.empty(2, dtype=np.float64); SY = np.empty(2, dtype=np.float64)
_CA = np.empty(2, dtype=np.float64); SA = np.empty(2, dtype=np.float64)
_CB = np.empty(2, dtype=np.float64); SB = np.empty(2, dtype=np.float64)
_CG = np.empty(2, dtype=np.float64); SG = np.empty(2, dtype=np.float64)
_EG = np.empty(2, dtype=np.complex128); EGM = np.empty(2, dtype=np.complex128)
_COS = np.empty(2, dtype=np.float64); SIN = np.empty(2, dtype=np.float64)
_M00 = np.empty(2, dtype=np.complex128); M01 = np.empty(2, dtype=np.complex128)
_M10 = np.empty(2, dtype=np.complex128); M11 = np.empty(2, dtype=np.complex128)
_N00 = np.empty(2, dtype=np.complex128); N01 = np.empty(2, dtype=np.complex128)
_N10 = np.empty(2, dtype=np.complex128); N11 = np.empty(2, dtype=np.complex128)


def _rot(x0, y0, x1, y1, a, b, g, c0, c1):
    """Apply R_x(a)@R_z(g)@R_x(b) to state (c0,c1). Returns (c0', c1')."""
    np.cos(a, out=CA); np.sin(a, out=SA)
    np.cos(b, out=CB); np.sin(b, out=SB)
    np.cos(g, out=CG); np.sin(g, out=SG)
    np.exp(-0.5j * g, out=EG); np.exp(0.5j * g, out=EGM)
    # M[0,0]
    COS[0] = CA * CB; COS[1] = SA * SB
    SIN[0] = SG * COS[0]; SIN[1] = SG * COS[1]
    M00[0] = COS[0] - SIN[0]; M00[1] = -COS[1] + SIN[1]
    # M[0,1] = -i*(e^{-ig} SA CB + SB CA)
    COS[0] = SA * CB; COS[1] = SB * CA
    SIN[0] = SG * COS[0]; SIN[1] = SG * COS[1]
    M01[0] = -(COS[0] - SIN[0]); M01[1] = -(COS[1] - SIN[1])
    # M[1,0] = -i*(e^{ig} SB CA + CB SA)
    COS[0] = SB * CA; COS[1] = CB * SA
    SIN[0] = SG * COS[0]; SIN[1] = SG * COS[1]
    M10[0] = -(COS[0] + SIN[0]); M10[1] = -(COS[1] + SIN[1])
    # M[1,1]
    COS[0] = CA * CB; COS[1] = SA * SB
    SIN[0] = SG * COS[0]; SIN[1] = SG * COS[1]
    M11[0] = COS[0] - SIN[0]; M11[1] = -COS[1] + SIN[1]
    # new c0 = M00 c0 + M01 c1 ; new c1 = M10 c0 + M11 c1
    N00[0] = M00[0] * x0 + M01[0] * y0
    N00[1] = M00[0] * x1 + M01[0] * y1
    N01[0] = M00[1] * x0 + M01[1] * y0
    N01[1] = M00[1] * x1 + M01[1] * y1
    N10[0] = M10[0] * x0 + M11[0] * y0
    N10[1] = M10[0] * x1 + M11[0] * y1
    N11[0] = M10[1] * x0 + M11[1] * y0
    N11[1] = M10[1] * x1 + M11[1] * y1
    return (N00[0] + N01[0], N01[1] + N10[0]), (N00[1] + N11[0], N10[1] + N11[1])


def _qaoa_state(g0, b0, g1, b1, i, j, A, B):
    """Return (c0,c1) of qubit i's 2-qubit reduced state after full p=2 circuit,
    with qubit i = '0' component index 0, j = '0' component index 1."""
    x0, y0 = 1.0, 0.0
    x1, y1 = 1.0, 0.0
    x0, y0 = _rot(x0, y0, x1, y1, b0, A, 0.0, x0, y0)[0]
    x1, y1 = _rot(x0, y0, x1, y1, b0, B, 0.0, x1, y1)[1]
    x0, y0 = _rot(x0, y0, x1, y1, b1, A, g0, x0, y0)[0]
    x1, y1 = _rot(x0, y0, x1, y1, b1, B, g0, x1, y1)[1]
    x0, y0 = _rot(x0, y0, x1, y1, 0.0, A, g1, x0, y0)[0]
    x1, y1 = _rot(x0, y0, x1, y1, 0.0, B, g1, x1, y1)[1]
    return (x0, y0)


def _expected_cut(theta, E, n, A, B):
    g0, b0, g1, b1 = theta
    s = 0.0
    for (i, j) in E:
        (x0, y0), (x1, y1) = _qaoa_state(g0, b0, g1, b1, i, j, A, B)
        # c0 = x0 + i y0 ; c1 = x1 + i y1
        # <szz> = |c0|^2 + |c1|^2 - 2 Re(c0* c1)
        c0r = x0; c0i = y0; c1r = x1; c1i = y1
        s += (c0r * c0r + c0i * c0i) + (c1r * c1r + c1i * c1i) - 2.0 * (c0r * c1r + c0i * c1i)
    return s


def _expected_cut_grad(theta, E, n, A, B):
    g0, b0, g1, b1 = theta
    s = 0.0
    dg0 = 0.0; db0 = 0.0; dg1 = 0.0; db1 = 0.0
    for (i, j) in E:
        (x0, y0), (x1, y1) = _qaoa_state(g0, b0, g1, b1, i, j, A, B)
        c0r = x0; c0i = y0; c1r = x1; c1i = y1
        a0 = c0r * c0r + c0i * c0i
        a1 = c1r * c1r + c1i * c1i
        b2 = c0r * c1r + c0i * c1i
        s += a0 + a1 - 2.0 * b2
        # derivatives (analytic, derived from the SU(2) chain rule)
        dg0 += -2.0 * (c0r * x1 + c0i * y1)
        db0 += 2.0 * (x0 * x1 + y0 * y1)
        dg1 += 2.0 * (c0r * y1 + c0i * x1)
        db1 += -2.0 * (x0 * c1r + y0 * c1i)
    return s, (dg0, db0, dg1, db1)


def _exact_maxcut(n, E):
    m = len(E)
    adj = np.zeros((n, n), dtype=np.int32)
    for (i, j) in E:
        adj[i, j] = 1
        adj[j, i] = 1
    best = 0
    for mask in range(1 << n):
        c = 0
        for (i, j) in E:
            if ((mask >> i) ^ (mask >> j)) & 1:
                c += 1
        if c > best:
            best = c
    return best


def _make_objective(E, n, A, B):
    def obj(theta):
        return -_expected_cut(theta, E, n, A, B)

    def grad(theta):
        _, g = _expected_cut_grad(theta, E, n, A, B)
        return -np.array(g)
    return obj, grad


def _initial_guesses(n, m, rng):
    """Return a list of 4-vectors (g0,b0,g1,b1) to try as starting points."""
    init = []
    # QAOA-inspired default
    init.append(np.array([0.5 * np.pi / (n + 1), 0.25 * np.pi / (n + 1),
                          0.5 * np.pi / (n + 1), 0.25 * np.pi / (n + 1)]))
    # small-angle
    init.append(np.array([0.3, 0.2, 0.3, 0.2]))
    # a few random points in a sensible box
    for _ in range(6):
        g0 = rng.uniform(0.0, 1.2)
        b0 = rng.uniform(0.0, 0.8)
        g1 = rng.uniform(0.0, 1.2)
        b1 = rng.uniform(0.0, 0.8)
        init.append(np.array([g0, b0, g1, b1]))
    return init


def angles(n, edges, p):
    t0 = time.time()
    DEADLINE = 110.0  # safety margin under the 120 s limit
    edges = [(int(u), int(v)) for (u, v) in edges]
    m = len(edges)

    # exact classical optimum (constant per graph, cheap for n<=10)
    opt = _exact_maxcut(n, edges)
    if opt <= 0:
        return np.zeros((p, 2))

    # neighbor sets for the 2-qubit reduced-state computation
    A = [set() for _ in range(n)]
    B = [set() for _ in range(n)]
    for (i, j) in edges:
        A[i].add(j)
        B[j].add(i)

    obj, grad = _make_objective(edges, n, A, B)
    bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]

    rng = np.random.default_rng(113)
    init = _initial_guesses(n, m, rng)

    best_theta = init[0].copy()
    best_val = obj(best_theta)

    extra_rng = np.random.default_rng(713)
    extra = 0
    while True:
        if time.time() > DEADLINE - 6.0:
            break
        if extra >= 40:
            break
        extra += 1
        init.append(extra_rng.uniform(-1.5, 1.5, size=4))

    for x0 in init:
        if time.time() > DEADLINE - 5.0:
            break
        try:
            res = minimize(obj, x0, jac=grad, method="L-BFGS-B",
                           bounds=bounds,
                           options={"maxiter": 300, "maxfun": 6000,
                                    "ftol": 1e-12, "gtol": 1e-8, "maxls": 30})
        except Exception:
            continue
        val = obj(res.x)
        if val < best_val:
            best_val = val
            best_theta = res.x.copy()

    g0, b0, g1, b1 = best_theta
    return np.array([[g0, b0], [g1, b1]], dtype=np.float64)
# EVOLVE-BLOCK-END
