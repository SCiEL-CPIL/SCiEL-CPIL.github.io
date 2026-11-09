"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg as sla
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# --- Constants ---
DT = 0.5
N_STEPS = 80
GATE_TIME = 40.0
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_VALUES = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALE_VALUES = [0.97, 1.0, 1.03]
DEADLINE = 300.0

# Precompute matrices
_a = np.array([[0, 1, 0],
               [0, 0, np.sqrt(2)],
               [0, 0, 0]], dtype=complex)
_adag = _a.conj().T
_n = _adag @ _a
_eye3 = np.eye(3, dtype=complex)

# Lindblad operators
_L1 = _a / np.sqrt(T1)
_L2 = _n * np.sqrt(2.0 / Tphi)

# Precompute Liouvillian building blocks
def _precompute_liouvillian_bases():
    I3 = _eye3
    L_det_base = -1j * (np.kron(_n, I3) - np.kron(I3, _n.T))
    H_I_base = 0.5 * (_a + _adag)
    L_I_base = -1j * (np.kron(H_I_base, I3) - np.kron(I3, H_I_base.T))
    H_Q_base = 0.5j * (_a - _adag)
    L_Q_base = -1j * (np.kron(H_Q_base, I3) - np.kron(I3, H_Q_base.T))
    H_anh = 0.5 * ANHARMONICITY * _n @ (_n - _eye3)
    L_anh_base = -1j * (np.kron(H_anh, I3) - np.kron(I3, H_anh.T))
    L_diss = np.zeros((9, 9), dtype=complex)
    for Lk in [_L1, _L2]:
        Lkd = Lk.conj().T
        LdL = Lkd @ Lk
        term1 = np.kron(Lkd.T, Lk)
        term2 = -0.5 * (np.kron(I3, LdL.T) + np.kron(LdL, I3))
        L_diss += term1 + term2
    return L_det_base, L_I_base, L_Q_base, L_anh_base, L_diss

_L_DET, _L_I, _L_Q, _L_ANH, _L_DISS = _precompute_liouvillian_bases()

# Initial states
_psi0 = np.array([1, 0, 0], dtype=complex)
_psi1 = np.array([0, 1, 0], dtype=complex)
_psi_plus = (_psi0 + _psi1) / np.sqrt(2)
_psi_minus = (_psi0 - _psi1) / np.sqrt(2)
_psi_plus_i = (_psi0 + 1j*_psi1) / np.sqrt(2)
_psi_minus_i = (_psi0 - 1j*_psi1) / np.sqrt(2)

_psi_outs = [
    np.array([0, 1, 0], dtype=complex),
    np.array([1, 0, 0], dtype=complex),
    (_psi0 + _psi1) / np.sqrt(2),
    (_psi0 - _psi1) / np.sqrt(2),
    (_psi1 + 1j*_psi0) / np.sqrt(2),
    (_psi1 - 1j*_psi0) / np.sqrt(2),
]

def _evolve_all_states(pulse_I, pulse_Q, detuning, scale):
    rhos = [np.outer(p, p.conj()) for p in [_psi0, _psi1, _psi_plus, _psi_minus, _psi_plus_i, _psi_minus_i]]
    rhos_vec = [r.flatten() for r in rhos]
    L_const = detuning * _L_DET + _L_ANH + _L_DISS
    for i in range(N_STEPS):
        L = L_const + scale * pulse_I[i] * _L_I + scale * pulse_Q[i] * _L_Q
        expL = sla.expm(L * DT)
        for j in range(6):
            rhos_vec[j] = expL @ rhos_vec[j]
    return [v.reshape(3, 3) for v in rhos_vec]

def _fidelity_from_rhos(rhos):
    fidelities = []
    for rho, psi_out in zip(rhos, _psi_outs):
        F = np.real(np.vdot(psi_out, rho @ psi_out))
        F = max(0.0, min(1.0, F))
        fidelities.append(F)
    return np.mean(fidelities)

def _compute_score(pulse_I, pulse_Q):
    infids = []
    for det in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            rhos = _evolve_all_states(pulse_I, pulse_Q, det, scale)
            F = _fidelity_from_rhos(rhos)
            infids.append(1.0 - F)
    return np.mean(infids) + np.max(infids)

def _is_valid(pulse_I, pulse_Q):
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    if np.any(amp > AMP_BOUND + 1e-12):
        return False
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-12:
            return False
    slew0 = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT
    if slew0 > SLEW_BOUND + 1e-12:
        return False
    slewl = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT
    if slewl > SLEW_BOUND + 1e-12:
        return False
    return True

def _enforce_constraints(pulse_I, pulse_Q):
    pulse_I = pulse_I.copy()
    pulse_Q = pulse_Q.copy()
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    mask = amp > AMP_BOUND
    if np.any(mask):
        pulse_I[mask] *= AMP_BOUND / amp[mask]
        pulse_Q[mask] *= AMP_BOUND / amp[mask]
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            scale = SLEW_BOUND * DT / slew
            pulse_I[i+1] = pulse_I[i] + dI * scale
            pulse_Q[i+1] = pulse_Q[i] + dQ * scale
    slew0 = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT
    if slew0 > SLEW_BOUND:
        scale = SLEW_BOUND * DT / slew0
        pulse_I[0] *= scale
        pulse_Q[0] *= scale
    slewl = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT
    if slewl > SLEW_BOUND:
        scale = SLEW_BOUND * DT / slewl
        pulse_I[-1] *= scale
        pulse_Q[-1] *= scale
    return pulse_I, pulse_Q

def _seed_pulse():
    t = np.arange(N_STEPS) * DT
    sigma = GATE_TIME / 6.0
    center = GATE_TIME / 2.0
    g = np.exp(-0.5 * (t - center)**2 / sigma**2)
    g[0] = 0.0
    g[-1] = 0.0
    area = np.sum(g) * DT
    if area > 0:
        g *= np.pi / area
    beta = 0.08
    dI_dt = np.gradient(g, DT)
    pulse_I = g.copy()
    pulse_Q = beta * dI_dt
    pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)
    return pulse_I, pulse_Q

def _pulse_from_params(params):
    sigma = max(3.0, min(25.0, params[0]))
    beta = max(-0.5, min(0.5, params[1]))
    amp_scale = max(0.3, min(2.0, params[2]))
    phase_shift = max(-8.0, min(8.0, params[3]))
    drag_shift = max(-5.0, min(5.0, params[4]))
    t = np.arange(N_STEPS) * DT
    center = GATE_TIME / 2.0 + phase_shift
    g = np.exp(-0.5 * (t - center)**2 / sigma**2)
    g[0] = 0.0
    g[-1] = 0.0
    area = np.sum(g) * DT
    if area > 0:
        g *= np.pi * amp_scale / area
    dI_dt = np.gradient(g, DT)
    pulse_I = g.copy()
    pulse_Q = beta * dI_dt
    if abs(drag_shift) > 1e-6:
        t_shifted = t + drag_shift
        valid = (t_shifted >= 0) & (t_shifted < GATE_TIME)
        if np.any(valid):
            pulse_Q[valid] = np.interp(t_shifted[valid], t, beta * dI_dt)
    pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)
    return pulse_I, pulse_Q

def _score_from_params(params):
    pulse_I, pulse_Q = _pulse_from_params(params)
    if not _is_valid(pulse_I, pulse_Q):
        return 10.0
    return _compute_score(pulse_I, pulse_Q)

def _optimize_param_space(time_limit=120):
    """Optimize in 5-parameter Gaussian+DRAG space with multiple restarts."""
    start_time = time.time()
    
    # Multiple starting points
    starts = [
        [GATE_TIME/6.0, 0.08, 1.0, 0.0, 0.0],
        [GATE_TIME/5.0, 0.12, 1.0, 0.0, 0.0],
        [GATE_TIME/7.0, 0.06, 1.0, 0.0, 0.0],
        [GATE_TIME/6.0, -0.1, 1.0, 0.0, 0.0],
        [GATE_TIME/6.0, 0.15, 1.05, 0.5, 0.0],
        [GATE_TIME/6.0, 0.08, 0.95, -0.5, 0.0],
        [GATE_TIME/6.0, 0.1, 1.0, 0.0, 1.0],
        [GATE_TIME/6.0, 0.1, 1.0, 0.0, -1.0],
    ]
    
    bounds = [
        (4.0, 22.0),
        (-0.3, 0.3),
        (0.7, 1.3),
        (-4.0, 4.0),
        (-4.0, 4.0),
    ]
    
    best_I, best_Q, best_score = _seed_pulse()
    best_score = _compute_score(best_I, best_Q)
    
    for x0 in starts:
        if time.time() - start_time > time_limit:
            break
        try:
            result = minimize(
                _score_from_params, x0, method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-6,
                         'adaptive': True}
            )
            if result.fun < best_score:
                pI, pQ = _pulse_from_params(result.x)
                if _is_valid(pI, pQ):
                    best_I = pI.copy()
                    best_Q = pQ.copy()
                    best_score = result.fun
        except Exception:
            pass
    
    return best_I, best_Q, best_score

def _direct_pulse_optimize(pulse_I, pulse_Q, time_limit=100):
    """Direct optimization on all 160 pulse parameters using L-BFGS-B with finite differences."""
    start_time = time.time()
    
    x0 = np.concatenate([pulse_I, pulse_Q])
    
    def objective(x):
        if time.time() - start_time > time_limit:
            return 100.0
        pI = x[:N_STEPS]
        pQ = x[N_STEPS:]
        pI, pQ = _enforce_constraints(pI, pQ)
        if not _is_valid(pI, pQ):
            return 10.0
        return _compute_score(pI, pQ)
    
    # Bounds: each pulse element in [-AMP_BOUND, AMP_BOUND]
    bounds = [(-AMP_BOUND, AMP_BOUND)] * (2 * N_STEPS)
    
    try:
        result = minimize(
            objective, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6,
                     'maxls': 50, 'maxfun': 200}
        )
        pI = result.x[:N_STEPS]
        pQ = result.x[N_STEPS:]
        pI, pQ = _enforce_constraints(pI, pQ)
        if _is_valid(pI, pQ):
            score = _compute_score(pI, pQ)
            return pI, pQ, score
    except Exception:
        pass
    
    return pulse_I, pulse_Q, _compute_score(pulse_I, pulse_Q)

def _coordinate_refine(pulse_I, pulse_Q, time_limit=60):
    """Coordinate-wise refinement with adaptive step size."""
    start_time = time.time()
    best_I = pulse_I.copy()
    best_Q = pulse_Q.copy()
    best_score = _compute_score(best_I, best_Q)
    
    np.random.seed(42)
    step_size = 0.02
    no_improve = 0
    
    while time.time() - start_time < start_time + time_limit and step_size > 0.002:
        idx = np.random.randint(N_STEPS)
        dI = np.zeros(N_STEPS)
        dQ = np.zeros(N_STEPS)
        dI[idx] = np.random.choice([-1, 1]) * step_size
        dQ[idx] = np.random.choice([-1, 1]) * step_size * 0.5
        
        new_I = best_I + dI
        new_Q = best_Q + dQ
        new_I, new_Q = _enforce_constraints(new_I, new_Q)
        
        if _is_valid(new_I, new_Q):
            new_score = _compute_score(new_I, new_Q)
            if new_score < best_score:
                best_I = new_I.copy()
                best_Q = new_Q.copy()
                best_score = new_score
                no_improve = 0
            else:
                no_improve += 1
        
        if no_improve > 6:
            step_size *= 0.7
            no_improve = 0
    
    return best_I, best_Q, best_score

def _dop853_recheck(pulse_I, pulse_Q):
    det = 0.0
    scale = 1.0
    rhos_ours = _evolve_all_states(pulse_I, pulse_Q, det, scale)
    
    def liouvillian_vec(t, rho_vec):
        idx = int(t / DT)
        if idx >= N_STEPS:
            idx = N_STEPS - 1
        I_val = pulse_I[idx]
        Q_val = pulse_Q[idx]
        L = (det * _L_DET + _L_ANH + _L_DISS 
             + scale * I_val * _L_I + scale * Q_val * _L_Q)
        return L @ rho_vec
    
    max_diff = 0.0
    for idx, psi in enumerate([_psi0, _psi1, _psi_plus, _psi_minus, _psi_plus_i, _psi_minus_i]):
        rho_init = np.outer(psi, psi.conj()).flatten()
        sol = solve_ivp(liouvillian_vec, [0, GATE_TIME], rho_init, 
                       method='DOP853', rtol=1e-12, atol=1e-14, max_step=0.1)
        rho_dop = sol.y[:, -1].reshape(3, 3)
        diff = np.max(np.abs(rho_dop - rhos_ours[idx]))
        max_diff = max(max_diff, diff)
    
    return max_diff

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    
    # Phase 1: Parameter-space optimization (Gaussian + DRAG)
    pulse_I, pulse_Q, score1 = _optimize_param_space(time_limit=120)
    
    # Phase 2: Direct pulse optimization with L-BFGS-B
    remaining = DEADLINE - (time.time() - start_time) - 30
    if remaining > 30:
        pulse_I, pulse_Q, score2 = _direct_pulse_optimize(pulse_I, pulse_Q, time_limit=min(100, remaining))
    
    # Phase 3: Coordinate-wise refinement
    remaining = DEADLINE - (time.time() - start_time) - 15
    if remaining > 10:
        pulse_I, pulse_Q, score3 = _coordinate_refine(pulse_I, pulse_Q, time_limit=min(50, remaining))
    
    # DOP853 recheck
    if time.time() - start_time < DEADLINE - 10:
        diff = _dop853_recheck(pulse_I, pulse_Q)
        if diff > 2e-7:
            remaining = DEADLINE - (time.time() - start_time) - 5
            if remaining > 10:
                pulse_I, pulse_Q, _ = _coordinate_refine(pulse_I, pulse_Q, time_limit=remaining)
    
    # Final validity check
    if not _is_valid(pulse_I, pulse_Q):
        pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)
    
    return np.column_stack((pulse_I, pulse_Q))
# EVOLVE-BLOCK-END
