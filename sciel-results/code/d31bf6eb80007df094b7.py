"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

_DEADLINE = 325.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

# ---- fixed hardware constants ----
_N = 80
_DT = 0.5
_T = _N * _DT  # 40 ns
_ANH = -2.0 * np.pi * 0.25   # anharmonicity
_AMP = 0.4
_SLEW = 0.12
_DET_GRID = np.array([-2*np.pi*0.002, 0.0, 2*np.pi*0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])
_T1 = 25000.0
_TPHI = 40000.0

# a = diag(sqrt([1,2]),1); a_dag = a^H
_A = np.diag([1.0, np.sqrt(2.0), 1.0])
_ADAG = _A.conj().T
_NOP = _ADAG @ _A  # diag(0,1,2)

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _NOP * np.sqrt(2.0 / _TPHI)
_L1ADAG = _L1.conj().T
_L2ADAG = _L2.conj().T

_I3 = np.eye(3, dtype=complex)

# Dissipative part (constant)
_L_DISS = (np.kron(_L1, _L1.conj()) - 0.5*(np.kron(_I3, _L1ADAG @ _L1) + np.kron(_L1ADAG @ _L1, _I3))
          + np.kron(_L2, _L2.conj()) - 0.5*(np.kron(_I3, _L2ADAG @ _L2) + np.kron(_L2ADAG @ _L2, _I3)))

# Precompute static Liouvillian for each detuning
_STATIC_LIOUV = {}
for det in _DET_GRID:
    H_static = det * _NOP + 0.5 * _ANH * (_NOP @ (_NOP - _I3))
    L_static = -1j * (np.kron(H_static, _I3) - np.kron(_I3, H_static.T))
    _STATIC_LIOUV[det] = L_static + _L_DISS

# Drive kron products
_A_PLUS_ADAG = _A + _ADAG
_A_MINUS_ADAG = _A - _ADAG

_KRON_AP_I = np.kron(_A_PLUS_ADAG, _I3)
_KRON_I_APT = np.kron(_I3, _A_PLUS_ADAG.T)
_KRON_AM_I = np.kron(_A_MINUS_ADAG, _I3)
_KRON_I_AMT = np.kron(_I3, _A_MINUS_ADAG.T)

# Precompute propagators for zero-detuning, scale=1 case for speed
# Actually we need per-step propagators that depend on I,Q

def _liouvillian_step_fast(dt, I, Q, detuning, scale):
    L_total = _STATIC_LIOUV[detuning]
    if abs(I) > 1e-15 or abs(Q) > 1e-15:
        coeff_I = 0.5 * scale * I
        coeff_Q = 0.5 * scale * Q
        L_total = L_total + (-1j * coeff_I * _KRON_AP_I + 1j * coeff_I * _KRON_I_APT 
                            + coeff_Q * _KRON_AM_I - coeff_Q * _KRON_I_AMT)
    return expm(L_total * dt)

# ---- Fast 2-level model for optimization ----
# Effective 2-level: |0>,|1> with leakage to |2> approximated
# H_2level = detuning*|1><1| + 0.5*anh*|1><1|*|1><1| (approx: anharmonic shift of |1> relative to |0> is ~0 for 2-level, but |2> shift matters)
# Actually for 2-level: H = detuning*n + 0.5*drive*(I*(a+a†) + iQ*(a-a†)) projected to 2-level
# The anharmonicity effect on 2-level dynamics is small; main effect is leakage to |2>
# For optimization, use 2-level with effective Rabi frequency including anharmonic correction

_I2 = np.eye(2, dtype=complex)
_A2 = np.array([[0,1],[1,0]], dtype=complex)  # sigma_x in 2-level
_A2DAG = _A2.conj().T
_N2 = np.diag([0,1])
_L1_2 = np.array([[0,0],[1,0]], dtype=complex) / np.sqrt(_T1)  # |1>->|0> decay
_L2_2 = np.array([[0,0],[0,1]], dtype=complex) * np.sqrt(2.0/_TPHI)  # dephasing
_L1_2DAG = _L1_2.conj().T
_L2_2DAG = _L2_2.conj().T

_L_DISS_2 = (np.kron(_L1_2, _L1_2.conj()) - 0.5*(np.kron(_I2, _L1_2DAG @ _L1_2) + np.kron(_L1_2DAG @ _L1_2, _I2))
            + np.kron(_L2_2, _L2_2.conj()) - 0.5*(np.kron(_I2, _L2_2DAG @ _L2_2) + np.kron(_L2_2DAG @ _L2_2, _I2)))

# Precompute 2-level static Liouvillian for each detuning
_STATIC_LIOUV_2 = {}
for det in _DET_GRID:
    # In 2-level, n=|1><1|, so anharmonic term 0.5*anh*n*(n-1) = 0 (since n^2=n in 2-level)
    # But the actual 3-level has |2> which shifts |1> by anharmonicity... 
    # For 2-level approximation, include detuning only
    H_static_2 = det * _N2
    L_static_2 = -1j * (np.kron(H_static_2, _I2) - np.kron(_I2, H_static_2.T))
    _STATIC_LIOUV_2[det] = L_static_2 + _L_DISS_2

_KRON_SX_I2 = np.kron(_A2, _I2)
_KRON_I2_SXT = np.kron(_I2, _A2.T)
_KRON_SY_I2 = np.kron(np.array([[0,1j],[-1j,0]], dtype=complex), _I2)
_KRON_I2_SYT = np.kron(_I2, np.array([[0,-1j],[1j,0]], dtype=complex))

def _liouvillian_step_2level(dt, I, Q, detuning, scale):
    L_total = _STATIC_LIOUV_2[detuning]
    if abs(I) > 1e-15 or abs(Q) > 1e-15:
        coeff_I = 0.5 * scale * I
        coeff_Q = 0.5 * scale * Q
        L_total = L_total + (-1j * coeff_I * _KRON_SX_I2 + 1j * coeff_I * _KRON_I2_SXT 
                            + coeff_Q * _KRON_SY_I2 - coeff_Q * _KRON_I2_SYT)
    return expm(L_total * dt)

def _simulate_2level(I_arr, Q_arr, detuning, scale):
    """Fast 2-level simulation returning final density matrices."""
    N = len(I_arr)
    states = [
        np.array([1, 0], dtype=complex),
        np.array([0, 1], dtype=complex),
        np.array([1, 1], dtype=complex) / np.sqrt(2),
        np.array([1, -1], dtype=complex) / np.sqrt(2),
        np.array([1, 1j], dtype=complex) / np.sqrt(2),
        np.array([1, -1j], dtype=complex) / np.sqrt(2),
    ]
    results = []
    for state in states:
        rho = np.outer(state, state.conj())
        vec_rho = rho.flatten()
        for i in range(N):
            prop = _liouvillian_step_2level(_DT, I_arr[i], Q_arr[i], detuning, scale)
            vec_rho = prop @ vec_rho
        results.append(vec_rho.reshape(2,2))
    return results

def _score_2level(I_arr, Q_arr):
    """Fast 2-level score."""
    target_0 = np.array([0, 1], dtype=complex)
    target_1 = np.array([1, 0], dtype=complex)
    all_fids = []
    for det in _DET_GRID:
        for scale in _SCALE_GRID:
            results = _simulate_2level(I_arr, Q_arr, det, scale)
            targets = [target_1, target_0,
                       np.array([1,1],dtype=complex)/np.sqrt(2),
                       np.array([-1,1],dtype=complex)/np.sqrt(2),
                       np.array([1j,1],dtype=complex)/np.sqrt(2),
                       np.array([-1j,1],dtype=complex)/np.sqrt(2)]
            for i, tgt in enumerate(targets):
                trho = np.outer(tgt, tgt.conj())
                all_fids.append(np.real(np.trace(trho @ results[i])))
    mean_fid = np.mean(all_fids)
    max_fid = np.max(all_fids)
    return 1.0 - mean_fid + (1.0 - max_fid)

def _simulate_3level(I_arr, Q_arr, detuning, scale):
    N = len(I_arr)
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    results = []
    for state in states:
        rho = np.outer(state, state.conj())
        vec_rho = rho.flatten()
        for i in range(N):
            prop = _liouvillian_step_fast(_DT, I_arr[i], Q_arr[i], detuning, scale)
            vec_rho = prop @ vec_rho
        results.append(vec_rho.reshape(3,3))
    return results

def _score_3level(I_arr, Q_arr):
    """Full 3-level score."""
    target_0 = np.array([0, 1, 0], dtype=complex)
    target_1 = np.array([1, 0, 0], dtype=complex)
    all_fids = []
    for det in _DET_GRID:
        for scale in _SCALE_GRID:
            results = _simulate_3level(I_arr, Q_arr, det, scale)
            targets = [target_1, target_0,
                       np.array([1,1,0],dtype=complex)/np.sqrt(2),
                       np.array([-1,1,0],dtype=complex)/np.sqrt(2),
                       np.array([1j,1,0],dtype=complex)/np.sqrt(2),
                       np.array([-1j,1,0],dtype=complex)/np.sqrt(2)]
            for i, tgt in enumerate(targets):
                trho = np.outer(tgt, tgt.conj())
                all_fids.append(np.real(np.trace(trho @ results[i])))
    mean_fid = np.mean(all_fids)
    max_fid = np.max(all_fids)
    return 1.0 - mean_fid + (1.0 - max_fid)

def _enforce_constraints(I_arr, Q_arr, amp_bound, slew_bound, dt):
    """Enforce amplitude and slew rate constraints."""
    I_arr = I_arr.copy()
    Q_arr = Q_arr.copy()
    N = len(I_arr)
    for _ in range(300):
        valid = True
        # Amplitude
        for i in range(N):
            amp = np.hypot(I_arr[i], Q_arr[i])
            if amp > amp_bound:
                s = amp_bound / amp
                I_arr[i] *= s
                Q_arr[i] *= s
                valid = False
        # Slew: zero to first
        a0 = np.hypot(I_arr[0], Q_arr[0])
        if a0 / dt > slew_bound:
            s = slew_bound * dt / a0
            I_arr[0] *= s
            Q_arr[0] *= s
            valid = False
        # Slew: between steps
        for i in range(N-1):
            dI = I_arr[i+1]-I_arr[i]
            dQ = Q_arr[i+1]-Q_arr[i]
            amp = np.hypot(dI, dQ)
            if amp / dt > slew_bound:
                s = slew_bound * dt / amp
                I_arr[i+1] = I_arr[i] + dI*s
                Q_arr[i+1] = Q_arr[i] + dQ*s
                valid = False
        # Slew: last to zero
        aL = np.hypot(I_arr[-1], Q_arr[-1])
        if aL / dt > slew_bound:
            s = slew_bound * dt / aL
            I_arr[-1] *= s
            Q_arr[-1] *= s
            valid = False
        if valid:
            break
    return I_arr, Q_arr

def _make_drag_pulse(beta, sigma, area, N=_N, dt=_DT):
    t = np.arange(N) * dt
    center = N * dt / 2.0
    env = np.exp(-0.5 * ((t - center) / sigma)**2)
    edge_val = np.exp(-0.5 * ((t[0] - center) / sigma)**2)
    env = env - edge_val
    cur_area = np.sum(env) * dt
    if cur_area > 1e-12:
        env = env * (area / cur_area)
    I = env.copy()
    denv = np.gradient(env, dt)
    Q = beta * denv
    I, Q = _enforce_constraints(I, Q, _AMP, _SLEW, dt)
    return I, Q

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        raise ValueError(f"Unknown target gate: {target_gate}")

    # Start with DRAG seed
    I, Q = _make_drag_pulse(0.0, 40.0/6.0, np.pi)
    best_I, best_Q = I.copy(), Q.copy()
    best_score = _score_2level(best_I, best_Q)

    # Nelder-Mead on (beta, sigma, area)
    from scipy.optimize import minimize

    def obj(params):
        beta, sigma, area = params
        if sigma < 1.0 or area < 0.5 or area > 5.0:
            return 10.0
        I, Q = _make_drag_pulse(beta, sigma, area)
        return _score_2level(I, Q)

    x0 = np.array([0.0, 40.0/6.0, np.pi])
    
    # Multiple restarts with different initial guesses
    best_2level_score = best_score
    best_2level_I = best_I.copy()
    best_2level_Q = best_Q.copy()

    initial_guesses = [
        [0.0, 40.0/6.0, np.pi],
        [0.1, 40.0/6.0, np.pi],
        [0.2, 40.0/6.0, np.pi],
        [0.0, 30.0, np.pi],
        [0.0, 50.0, np.pi],
        [0.15, 35.0, np.pi],
        [0.25, 45.0, np.pi],
    ]

    for x0 in initial_guesses:
        if _time_remaining() < 30:
            break
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.fun < best_2level_score:
                best_2level_score = res.fun
                best_2level_I, best_2level_Q = _make_drag_pulse(res.x[0], res.x[1], res.x[2])
        except Exception:
            pass

    best_I, best_Q = best_2level_I, best_2level_Q

    # Now verify and polish with 3-level model
    best_3level_score = _score_3level(best_I, best_Q)
    best_3level_I, best_3level_Q = best_I.copy(), best_Q.copy()

    # Fine-tune with 3-level using small perturbations
    if _time_remaining() > 60:
        # Try local refinements
        for trial in range(5):
            if _time_remaining() < 30:
                break
            beta, sigma, area = 0.1, 40.0/6.0, np.pi
            # Perturb around best
            I_try = best_3level_I + np.random.randn(_N) * 0.01
            Q_try = best_3level_Q + np.random.randn(_N) * 0.01
            I_try, Q_try = _enforce_constraints(I_try, Q_try, _AMP, _SLEW, _DT)
            s = _score_3level(I_try, Q_try)
            if s < best_3level_score:
                best_3level_score = s
                best_3level_I = I_try.copy()
                best_3level_Q = Q_try.copy()

    return np.column_stack([best_3level_I, best_3level_Q])

# EVOLVE-BLOCK-END
