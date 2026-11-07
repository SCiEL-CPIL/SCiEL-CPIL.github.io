"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 315.0
_START = time.time()

# ---- fixed hardware constants ----
_N = 80
_DT = 0.5
_T = _N * _DT
_ANH = -2.0 * np.pi * 0.25
_AMP = 0.4
_SLEW = 0.12
_T1 = 25000.0
_TPHI = 40000.0

# detuning / scale grid for scoring
_DET_GRID = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])

# ---- fixed operators (time independent) ----
_A = np.diag([1.0, np.sqrt(2.0), 1.0])          # a
_ADAG = _A.conj().T                              # a^dagger
_NOP = _ADAG @ _A                                # n
_A_ADAG = _A @ _ADAG                             # a + a^dagger (real sym)
_A_ADAG_I = 1j * (_A - _ADAG)                    # i(a - a^dagger)

# Lindblad terms
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T

# Precompute constant (non-drive) part of Liouvillian superoperator.
# L(rho) = det*n*rho - rho*det*n + 0.5*anh*n(n-1)*rho - rho*0.5*anh*n(n-1)
#         + 0.5*(I AADAG + iQ AADAG_I) rho - rho*0.5*(I AADAG + iQ AADAG_I)
#         -0.5 L1ADAG L1 rho - rho 0.5 L1ADAG L1 + L1 rho L1ADAG
#         -0.5 L2ADAG L2 rho - rho 0.5 L2ADAG L2 + L2 rho L2ADAG
# We build the superoperator as a 9x9 matrix acting on vec(rho) (column-stacked).

def _kron_super(L):
    # superoperator for commutator term:  L rho - rho L = kron(I, L) - kron(L^T, I)
    dim = 3
    return np.kron(np.eye(dim), L) - np.kron(L.T, np.eye(dim))

def _super_diss(L):
    dim = 3
    return (np.kron(L.conj(), L) - 0.5*np.kron(L.conj().T @ L, np.eye(dim))
            - 0.5*np.kron(np.eye(dim), L.conj().T @ L))

# constant superoperator pieces (independent of I, Q)
_H_DET = _NOP                       # coefficient = detuning
_H_ANH = 0.5 * _ANH * (_NOP @ (_NOP - np.eye(3)))   # coefficient = 1
_H_I   = 0.5 * _A_ADAG              # coefficient = I
_H_Q   = 0.5 * _A_ADAG_I            # coefficient = Q

_S_DET = _kron_super(_H_DET)
_S_ANH = _kron_super(_H_ANH)
_S_I   = _kron_super(_H_I)
_S_Q   = _kron_super(_H_Q)
_S_DISS = _super_diss(_L1) + _super_diss(_L2)

# basis states
_B0 = np.zeros(9); _B0[0] = 1.0
_B1 = np.zeros(9); _B1[3] = 1.0
_B2 = np.zeros(9); _B2[6] = 1.0

def _vec_of_state(s):
    v = np.outer(s, s.conj())
    return v.ravel()

def _fidelity_vec(rv, target):
    # tr(rho * P_target)
    P = np.outer(target, target.conj()).ravel()
    return float(np.real(np.dot(P, rv)))

# target vectors for X gate: X|psi>
def _x_state(psi):
    # X = [[0,1],[1,0]] on subspace {|0>,|1>}, |2> fixed
    out = np.zeros(3, dtype=complex)
    out[0] = psi[1]
    out[1] = psi[0]
    out[2] = psi[2]
    return out

_PSI0 = np.array([1,0,0], dtype=complex)
_PSI1 = np.array([0,1,0], dtype=complex)
_PSIp = np.array([1,1,0], dtype=complex)/np.sqrt(2)
_PSMi = np.array([1,-1,0], dtype=complex)/np.sqrt(2)
_PSIpi = np.array([1,1j,0], dtype=complex)/np.sqrt(2)
_PSMi_i = np.array([1,-1j,0], dtype=complex)/np.sqrt(2)

_TARGS = [_x_state(p) for p in (_PSI0,_PSI1,_PSIp,_PSMi,_PSIpi,_PSMi_i)]

def _build_pulse(amp, drag, center, sigma):
    t = np.arange(_N) * _DT
    g = amp * np.exp(-0.5*((t-center)/sigma)**2)
    # derivative for DRAG
    dg = g * (-(t-center)/sigma**2)
    I = g
    Q = drag * dg
    return np.stack([I, Q], axis=1)

def _check_valid(p):
    I, Q = p[:,0], p[:,1]
    amp = np.sqrt(I*I + Q*Q)
    if np.max(amp) > _AMP + 1e-9:
        return False
    # slew including 0->p0 and p_last->0
    d0 = np.hypot(I[0], Q[0]) / _DT
    dlast = np.hypot(I[-1], Q[-1]) / _DT
    d = np.hypot(np.diff(I), np.diff(Q)) / _DT
    if d0 > _SLEW + 1e-9 or dlast > _SLEW + 1e-9 or np.max(d) > _SLEW + 1e-9:
        return False
    return True

def _simulate(p, det, scale):
    # build per-step superoperator and apply to rho (start |0><0|)
    rv = _B0.copy()
    S_const = _S_ANH + _S_DISS + det * _S_DET
    I = p[:,0]; Q = p[:,1]
    for k in range(_N):
        S = S_const + scale * (I[k]*_S_I + Q[k]*_S_Q)
        rv = np.exp(_DT * S) @ rv
    return rv

def _score_case(p, det, scale):
    rv = _simulate(p, det, scale)
    fids = [_fidelity_vec(rv, t) for t in _TARGS]
    inf = 1.0 - np.mean(fids)
    return inf

def _total_score(p):
    infs = [_score_case(p, d, s) for d in _DET_GRID for s in _SCALE_GRID]
    infs = np.array(infs)
    return float(np.mean(infs) + np.max(infs))

def _seed_pulse():
    # Gaussian, sigma = 40/6, offset to 0 at edges, area pi
    t = np.arange(_N) * _DT
    sigma = _T / 6.0
    center = _T / 2.0
    g = np.exp(-0.5*((t-center)/sigma)**2)
    # area pi: integral ~ amp * sigma * sqrt(2*pi) * (erf correction) ~ set amp
    # approximate continuous area = amp * sigma * sqrt(2*pi)
    amp = np.pi / (sigma * np.sqrt(2*np.pi))
    # offset to 0 at edges: subtract value at edges? edges already ~0. Just scale.
    I = amp * g
    Q = np.zeros_like(I)
    p = np.stack([I, Q], axis=1)
    return p

def _project_slew(p, max_iter=200):
    # simple iterative projection to satisfy slew & amplitude bounds
    p = p.copy()
    for _ in range(max_iter):
        # amplitude projection
        amp = np.hypot(p[:,0], p[:,1])
        over = amp > _AMP
        if np.any(over):
            scale = np.where(over, _AMP/np.maximum(amp,1e-12), 1.0)
            p[:,0] *= scale
            p[:,1] *= scale
        # slew projection: limit step differences
        # from 0 to p0
        d0 = np.hypot(p[0,0], p[0,1])
        if d0 > _SLEW*_DT:
            p[0] *= _SLEW*_DT/d0
        # from plast to 0
        dl = np.hypot(p[-1,0], p[-1,1])
        if dl > _SLEW*_DT:
            p[-1] *= _SLEW*_DT/dl
        # interior steps
        for i in range(_N-1):
            dI = p[i+1,0]-p[i,0]
            dQ = p[i+1,1]-p[i,1]
            d = np.hypot(dI, dQ)
            lim = _SLEW*_DT
            if d > lim:
                f = lim/d
                # move midpoint: adjust both by half
                midI = (p[i,0]+p[i+1,0])/2
                midQ = (p[i,1]+p[i+1,1])/2
                p[i,0] = midI - 0.5*f*dI
                p[i,1] = midQ - 0.5*f*dQ
                p[i+1,0] = midI + 0.5*f*dI
                p[i+1,1] = midQ + 0.5*f*dQ
        if _check_valid(p):
            break
    return p

def _local_search(p0):
    # coordinate-wise Nelder-Mead-like perturbations on 4 params
    # params: amp, drag, center, sigma
    # recover from p0: fit amp (max), sigma (from width), center (peak), drag (from Q/I slope)
    t = np.arange(_N) * _DT
    I = p0[:,0]; Q = p0[:,1]
    center = float(t[np.argmax(I)])
    # sigma from half width
    half = 0.5*np.max(I)
    idx = np.where(I > half)[0]
    if len(idx) >= 2:
        width = (t[idx[-1]] - t[idx[0]])
        sigma = width / (2*np.sqrt(2*np.log(2)))
    else:
        sigma = _T/6.0
    amp = float(np.max(I))
    # drag from Q ~ drag * dI/dt at peak region: use correlation
    dI = np.gradient(I, _DT)
    drag = 0.0
    if np.std(dI) > 1e-12:
        drag = float(np.dot(Q, dI)/np.dot(dI, dI))
    params = np.array([amp, drag, center, sigma])
    best = p0
    best_score = _total_score(p0)
    step_sizes = [0.05, 0.05, 1.0, 1.0]
    improved = True
    iters = 0
    while improved and (time.time() - _START) < _DEADLINE - 30:
        improved = False
        iters += 1
        for j in range(4):
            for sign in (+1, -1):
                p_new = params.copy()
                p_new[j] += sign * step_sizes[j]
                p = _build_pulse(p_new[0], p_new[1], p_new[2], p_new[3])
                p = _project_slew(p)
                if not _check_valid(p):
                    continue
                s = _total_score(p)
                if s < best_score:
                    best_score = s
                    best = p
                    params = p_new
                    improved = True
        # shrink steps a bit
        step_sizes = [s*0.8 for s in step_sizes]
    return best, best_score

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        # fallback to seed
        p = _seed_pulse()
        return p
    # start from seed
    p_seed = _seed_pulse()
    # ensure valid
    if not _check_valid(p_seed):
        p_seed = _project_slew(p_seed)
    best = p_seed
    # try a few DRAG-informed starts
    candidates = [p_seed]
    # DRAG: Q = -lambda * dI/dt ; try a few lambdas
    t = np.arange(_N)*_DT
    I = p_seed[:,0]
    dI = np.gradient(I, _DT)
    for lam in [0.0, 0.2, 0.4, 0.6, -0.2]:
        p = np.stack([I, lam*dI], axis=1)
        p = _project_slew(p)
        if _check_valid(p):
            candidates.append(p)
    # pick best candidate by score, then local search
    best = candidates[0]
    best_score = _total_score(best)
    for c in candidates[1:]:
        s = _total_score(c)
        if s < best_score:
            best_score = s
            best = c
    if (time.time() - _START) < _DEADLINE - 40:
        best, best_score = _local_search(best)
    # final validity guard
    if not _check_valid(best):
        best = _project_slew(best)
    return best
# EVOLVE-BLOCK-END
