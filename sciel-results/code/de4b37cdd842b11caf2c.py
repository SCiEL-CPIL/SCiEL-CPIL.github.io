import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

_START = time.time()
_DEADLINE = _START + 170.0  # safety margin under 210s

def _time_left():
    return _DEADLINE - time.time()

# ---- Hermite (physicists') even polynomials, coeff dict {power: coeff} ----
_H0  = {0:1}
_H4  = {0:12, 2:-48, 4:16}
_H8  = {0:112, 2:-1344, 4:2688, 6:-1280, 8:256}
_H12 = {0:9216, 2:-96768, 4:253440, 6:-282240, 8:153600, 10:-39936, 12:4096}

def _b_coeffs(c0, c4, c8):
    c12 = -(c0 + 12*c4 + 112*c8) / 9216.0
    b = np.zeros(7)  # b[j] = coeff of x^{2j}
    b[0] = 0.0  # x^0 term cancels by construction (g(0)=0)
    b[1] = c4*(-48) + c8*(-1344) + c12*(-96768)
    b[2] = c4*16 + c8*2688 + c12*253440
    b[3] = c8*(-1280) + c12*(-282240)
    b[4] = c8*256 + c12*153600
    b[5] = c12*(-39936)
    b[6] = c12*4096
    return b

# ---- P_k polynomials (in xi), P_k[i] = coeff of xi^i ----
def _build_P():
    P = [[1.0]]
    for k in range(12):
        Pk = P[k]
        dpk = [i*Pk[i] for i in range(1, len(Pk))]
        xPk = [0.0] + Pk
        L = max(len(dpk), len(xPk))
        dpk = dpk + [0.0]*(L-len(dpk))
        xPk = xPk + [0.0]*(L-len(xPk))
        P.append([dpk[i]-xPk[i] for i in range(L)])
    return P
_P = _build_P()

def _R_coeffs_u(b):
    R = np.zeros(13)
    for j in range(7):
        sign = (-1.0)**j
        Pk = np.array(_P[2*j])
        R[:len(Pk)] += b[j]*sign*Pk
    return R[0::2]  # even coeffs -> u form

from scipy.special import gamma
def _poly4_integral(coeffs_u):
    d = np.convolve(np.convolve(coeffs_u, coeffs_u), np.convolve(coeffs_u, coeffs_u))
    n = np.arange(len(d))
    vals = gamma(n + 0.5) / (2.0 ** (n + 0.5))
    return float(np.dot(d, vals))

def _raw_value(c0, c4, c8):
    b = _b_coeffs(c0, c4, c8)
    return _poly4_integral(b) * _poly4_integral(_R_coeffs_u(b))

_SEED = np.array([-113/100, 1/25, 1/3240])
_RAW_SEED = _raw_value(*_SEED)

def _value(c0, c4, c8):
    return _raw_value(c0, c4, c8) / _RAW_SEED

# ---- Build the two quartic-form tensors (4th-order) ----
_basis = [np.array([1.0, 0.0, 0.0]),
          np.array([0.0, 1.0, 0.0]),
          np.array([0.0, 0.0, 1.0])]

def _build_T(fn):
    T = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    T[i,j,k,l] = fn(_basis[i]+_basis[j]+_basis[k]+_basis[l])
    return T

def _g_raw(c):
    b = _b_coeffs(c[0], c[1], c[2]); return _poly4_integral(b)
def _gf_raw(c):
    b = _b_coeffs(c[0], c[1], c[2]); return _poly4_integral(_R_coeffs_u(b))

T_g  = _build_T(_g_raw)
T_gf = _build_T(_gf_raw)

def _quartic(c, T):
    return np.einsum('ijkl, i, j, k, l->', T, c, c, c, c)

def _raw4(c):
    return _quartic(c, T_g) * _quartic(c, T_gf)

def _objective(x):
    return _raw4(x) / _RAW_SEED

def _objective_signed(x):
    # change of variables: c = [x0, x1, x2]; sign handled by c0>=0 constraint
    return _raw4(x) / _RAW_SEED

# ---- Quartic-form eigenproblem (cubic eigenvalue) ----
def _quartic_grad(c, T):
    return np.einsum('ijkl, j, k, l->i', T, c, c, c) * 4.0

def _deflation(T, c, lam):
    # returns matrix M such that eig(M, c) == lam
    c = np.asarray(c, float)
    nrm = np.linalg.norm(c)
    if nrm < 1e-15:
        return None
    u = c / nrm
    M = np.einsum('ijkl, l->ijk', T, u) - lam * np.outer(u, np.eye(3))
    return M

def _rayleigh_quartic(T, c0, iters=2000, tol=1e-15):
    c = np.asarray(c0, float).copy()
    nrm = np.linalg.norm(c)
    if nrm < 1e-15:
        return None
    c = c / nrm
    prev = None
    for _ in range(iters):
        g = _quartic_grad(c, T) / 4.0  # gradient of (1/4)quartic
        # cubic eigenvalue: solve eig(M, c) with M = T(c) - lam I ; use deflation trick
        # lam = <c, grad> (quartic value)
        lam = _quartic(c, T)
        M = _deflation(T, c, lam)
        if M is None:
            return None
        # Rayleigh step: minimize quadratic c^T M c
        w, V = np.linalg.eigh(M)
        idx = int(np.argmin(w))
        c_new = V[:, idx]
        # keep same orientation (dot>0) to avoid flip
        if np.dot(c_new, c) < 0:
            c_new = -c_new
        step = np.linalg.norm(c_new - c)
        c = c_new
        if step < tol:
            break
    return c / np.linalg.norm(c)

def _refine_quartic(c0):
    # polish a direction to a cubic-eigenvalue local min of the quartic
    c = np.asarray(c0, float)
    nrm = np.linalg.norm(c)
    if nrm < 1e-15:
        return None
    c = c / nrm
    for _ in range(30):
        M = _deflation(T_g, c, _quartic(c, T_g))
        if M is None:
            return None
        w, V = np.linalg.eigh(M)
        idx = int(np.argmin(w))
        c_new = V[:, idx]
        if np.dot(c_new, c) < 0:
            c_new = -c_new
        if np.linalg.norm(c_new - c) < 1e-15:
            break
        c = c_new
    return c / np.linalg.norm(c)

def construct():
    global _START, _DEADLINE
    best_x = _SEED.copy()
    best_v = _objective(*_SEED)  # = 1.0

    def consider(x):
        nonlocal best_x, best_v
        v = _objective(x[0], x[1], x[2])
        if v < best_v - 1e-15:
            best_v = v; best_x = np.array([x[0], x[1], x[2]])

    def polish(x0):
        nonlocal best_x, best_v
        x0 = np.asarray(x0, float)
        for method, opts in [
            ('Nelder-Mead', {'maxiter':20000, 'xatol':1e-13, 'fatol':1e-15, 'adaptive':True}),
            ('L-BFGS-B',    {'maxiter':20000, 'gtol':1e-15}),
        ]:
            if _time_left() < 8:
                break
            try:
                r = minimize(_objective, x0, method=method, options=opts)
                if r.fun < best_v:
                    best_v = r.fun; best_x = np.array([r.x[0], r.x[1], r.x[2]])
                    x0 = r.x
            except Exception:
                pass

    try:
        # ---- Global search: differential evolution (sign-symmetric, bounded) ----
        if _time_left() > 40:
            bounds = [(-12,12), (-12,12), (-6,6)]
            for seed in [42, 1, 2, 3, 7, 11]:
                if _time_left() < 30:
                    break
                try:
                    res = differential_evolution(
                        _objective, bounds, seed=seed, maxiter=250, popsize=28,
                        tol=1e-12, polish=False, init='sobol')
                    if res.x is not None and res.fun < best_v:
                        best_v = res.fun; best_x = np.array([res.x[0], res.x[1], res.x[2]])
                except Exception:
                    pass

        # ---- Cubic-eigenvalue (quartic-form) global solver ----
        # The full objective is a product of two quartics; minimize each quartic's
        # direction via the cubic eigenproblem, then combine.
        if _time_left() > 15:
            # random directions to seed the eigenvalue solver
            rng = np.random.default_rng(0)
            cand_dirs = []
            for _ in range(60):
                v = rng.standard_normal(3)
                cand_dirs.append(v/np.linalg.norm(v))
            # also add the seed direction
            cand_dirs.append(_SEED/np.linalg.norm(_SEED))
            for d in cand_dirs:
                if _time_left() < 12:
                    break
                for T in (T_g, T_gf):
                    try:
                        d2 = _rayleigh_quartic(T, d, iters=400, tol=1e-15)
                        if d2 is None:
                            continue
                        # try d2 and -d2 as a full-objective direction
                        for sgn in (1.0, -1.0):
                            # scale to a representative magnitude
                            scale = 1.0
                            x0 = sgn * d2 * scale
                            polish(x0)
                    except Exception:
                        pass

        # ---- Final high-precision polish from best and seed ----
        polish(best_x)
        polish(_SEED)
    except Exception:
        pass

    return np.array([best_x[0], best_x[1], best_x[2]])
