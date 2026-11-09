"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg as sla
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution

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
DEADLINE = 305.0  # safety margin before 330s kill

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
    """Precompute the matrix coefficients for the Liouvillian."""
    I3 = _eye3
    
    # Detuning term
    L_det_base = -1j * (np.kron(_n, I3) - np.kron(I3, _n.T))
    
    # Drive I term: 0.5*(a+a†)
    H_I_base = 0.5 * (_a + _adag)
    L_I_base = -1j * (np.kron(H_I_base, I3) - np.kron(I3, H_I_base.T))
    
    # Drive Q term: 0.5*i*(a-a†)
    H_Q_base = 0.5j * (_a - _adag)
    L_Q_base = -1j * (np.kron(H_Q_base, I3) - np.kron(I3, H_Q_base.T))
    
    # Anharmonicity term (constant part): 0.5*alpha*n*(n-1)
    H_anh = 0.5 * ANHARMONICITY * _n @ (_n - _eye3)
    L_anh_base = -1j * (np.kron(H_anh, I3) - np.kron(I3, H_anh.T))
    
    # Dissipative part (constant)
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

# Ideal outputs (X applied to each)
_psi_outs = [
    np.array([0, 1, 0], dtype=complex),
    np.array([1, 0, 0], dtype=complex),
    (_psi0 + _psi1) / np.sqrt(2),
    (_psi0 - _psi1) / np.sqrt(2),
    (_psi1 + 1j*_psi0) / np.sqrt(2),
    (_psi1 - 1j*_psi0) / np.sqrt(2),
]

def _evolve_all_states(pulse_I, pulse_Q, detuning, scale):
    """Evolve all 6 input states through the pulse, return final density matrices."""
    rhos = [np.outer(p, p.conj()) for p in [_psi0, _psi1, _psi_plus, _psi_minus, _psi_plus_i, _psi_minus_i]]
    rhos_vec = [r.flatten() for r in rhos]
    
    # Constant part of Liouvillian
    L_const = detuning * _L_DET + _L_ANH + _L_DISS
    
    for i in range(N_STEPS):
        I_val = pulse_I[i]
        Q_val = pulse_Q[i]
        
        L = L_const + scale * I_val * _L_I + scale * Q_val * _L_Q
        expL = sla.expm(L * DT)
        
        for j in range(6):
            rhos_vec[j] = expL @ rhos_vec[j]
    
    return [v.reshape(3, 3) for v in rhos_vec]

def _fidelity_from_rhos(rhos):
    """Compute mean fidelity to X gate from 6 final density matrices."""
    fidelities = []
    for rho, psi_out in zip(rhos, _psi_outs):
        F = np.real(np.vdot(psi_out, rho @ psi_out))
        F = max(0.0, min(1.0, F))
        fidelities.append(F)
    
    return np.mean(fidelities)

def _compute_score(pulse_I, pulse_Q):
    """Compute score = mean + max of 9 infidelities."""
    infids = []
    for det in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            rhos = _evolve_all_states(pulse_I, pulse_Q, det, scale)
            F = _fidelity_from_rhos(rhos)
            infids.append(1.0 - F)
    
    return np.mean(infids) + np.max(infids)

def _is_valid(pulse_I, pulse_Q):
    """Check amplitude and slew constraints."""
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
    """Project pulse onto feasible set (amplitude + slew)."""
    pulse_I = pulse_I.copy()
    pulse_Q = pulse_Q.copy()
    
    # Amplitude
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    mask = amp > AMP_BOUND
    if np.any(mask):
        pulse_I[mask] *= AMP_BOUND / amp[mask]
        pulse_Q[mask] *= AMP_BOUND / amp[mask]
    
    # Slew: forward pass
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            scale = SLEW_BOUND * DT / slew
            pulse_I[i+1] = pulse_I[i] + dI * scale
            pulse_Q[i+1] = pulse_Q[i] + dQ * scale
    
    # Edge slews
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
    """Generate DRAG-style Gaussian seed with beta compensation."""
    t = np.arange(N_STEPS) * DT
    sigma = GATE_TIME / 6.0
    center = GATE_TIME / 2.0
    
    # Gaussian envelope
    g = np.exp(-0.5 * (t - center)**2 / sigma**2)
    g[0] = 0.0
    g[-1] = 0.0
    
    # Scale to area pi
    area = np.sum(g) * DT
    if area > 0:
        g *= np.pi / area
    
    # DRAG beta
    beta = 0.15
    
    # Derivative of I envelope
    dI_dt = np.gradient(g, DT)
    
    pulse_I = g.copy()
    pulse_Q = beta * dI_dt
    
    # Enforce constraints
    pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)
    
    return pulse_I, pulse_Q

def _pulse_from_params(params):
    """
    Generate pulse from low-dimensional parameterization.
    params: [sigma, beta, amp_scale, center_shift, drag_shift]
    """
    sigma = params[0]
    beta = params[1]
    amp_scale = params[2]
    center_shift = params[3]
    drag_shift = params[4]
    
    t = np.arange(N_STEPS) * DT
    center = GATE_TIME / 2.0 + center_shift
    
    # Gaussian envelope
    g = np.exp(-0.5 * (t - center)**2 / sigma**2)
    g[0] = 0.0
    g[-1] = 0.0
    
    # Scale to area pi * amp_scale
    area = np.sum(g) * DT
    if area > 0:
        g *= np.pi * amp_scale / area
    
    # Derivative of I envelope
    dI_dt = np.gradient(g, DT)
    
    pulse_I = g.copy()
    pulse_Q = beta * dI_dt
    
    # Apply drag shift
    if abs(drag_shift) > 1e-6:
        t_shifted = t + drag_shift
        valid = (t_shifted >= 0) & (t_shifted < GATE_TIME)
        if np.any(valid):
            pulse_Q[valid] = np.interp(t_shifted[valid], t, beta * dI_dt)
        pulse_Q[~valid] = 0.0
    
    # Enforce constraints
    pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)
    
    return pulse_I, pulse_Q

def _score_from_params(params):
    """Compute score from parameter vector."""
    pulse_I, pulse_Q = _pulse_from_params(params)
    if not _is_valid(pulse_I, pulse_Q):
        return 10.0
    return _compute_score(pulse_I, pulse_Q)

def _gradient_estimate(pulse_I, pulse_Q, score, step=0.01):
    """Estimate gradient using central differences on a subset of steps."""
    grad_I = np.zeros(N_STEPS)
    grad_Q = np.zeros(N_STEPS)
    
    # Use a subset of steps for efficiency
    subset = np.random.choice(N_STEPS, size=min(20, N_STEPS), replace=False)
    
    for idx in subset:
        # I gradient
        pert_I = pulse_I.copy()
        pert_I[idx] += step
        pert_I, pert_Q = _enforce_constraints(pert_I, pulse_Q)
        if _is_valid(pert_I, pert_Q):
            s_plus = _compute_score(pert_I, pert_Q)
            pert_I2 = pulse_I.copy()
            pert_I2[idx] -= step
            pert_I2, pert_Q2 = _enforce_constraints(pert_I2, pulse_Q)
            if _is_valid(pert_I2, pert_Q2):
                s_minus = _compute_score(pert_I2, pert_Q2)
                grad_I[idx] = (s_plus - s_minus) / (2 * step)
        
        # Q gradient
        pert_Q = pulse_Q.copy()
        pert_Q[idx] += step
        pert_I, pert_Q = _enforce_constraints(pulse_I, pert_Q)
        if _is_valid(pert_I, pert_Q):
            s_plus = _compute_score(pert_I, pert_Q)
            pert_Q2 = pulse_Q.copy()
            pert_Q2[idx] -= step
            pert_I2, pert_Q2 = _enforce_constraints(pulse_I, pert_Q2)
            if _is_valid(pert_I2, pert_Q2):
                s_minus = _compute_score(pert_I2, pert_Q2)
                grad_Q[idx] = (s_plus - s_minus) / (2 * step)
    
    return grad_I, grad_Q

def _local_refine(pulse_I, pulse_Q, time_limit=80):
    """Local refinement using gradient-based optimization."""
    start_time = time.time()
    best_I = pulse_I.copy()
    best_Q = pulse_Q.copy()
    best_score = _compute_score(best_I, best_Q)
    
    np.random.seed(42)
    step_size = 0.02
    no_improve = 0
    lr = 0.01
    
    while time.time() - start_time < time_limit and step_size > 0.001:
        # Estimate gradient
        grad_I, grad_Q = _gradient_estimate(best_I, best_Q, best_score, step=step_size)
        
        # Gradient descent step
        new_I = best_I - lr * grad_I
        new_Q = best_Q - lr * grad_Q
        new_I, new_Q = _enforce_constraints(new_I, new_Q)
        
        if _is_valid(new_I, new_Q):
            new_score = _compute_score(new_I, new_Q)
            if new_score < best_score:
                best_I = new_I.copy()
                best_Q = new_Q.copy()
                best_score = new_score
                no_improve = 0
                lr *= 1.1
            else:
                no_improve += 1
                lr *= 0.8
        else:
            no_improve += 1
            lr *= 0.8
        
        if no_improve > 8:
            step_size *= 0.7
            no_improve = 0
            lr = 0.01
    
    return best_I, best_Q, best_score

def _optimize_with_de(pulse_I, pulse_Q, time_limit=150):
    """Optimize using Differential Evolution on low-dimensional parameter space."""
    start_time = time.time()
    
    # Initial best from seed
    best_I = pulse_I.copy()
    best_Q = pulse_Q.copy()
    best_score = _compute_score(best_I, best_Q)
    
    # Parameter bounds for DE
    bounds = [
        (4.0, 22.0),    # sigma
        (-0.6, 0.6),    # beta
        (0.6, 1.4),     # amp_scale
        (-4.0, 4.0),    # center_shift
        (-2.5, 2.5),    # drag_shift
    ]
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 100.0
        return _score_from_params(params)
    
    try:
        result = differential_evolution(
            objective, bounds,
            seed=42, maxiter=40, popsize=12,
            tol=1e-4, mutation=(0.3, 1.0), recombination=0.8,
            polish=True, init='latinhypercube'
        )
        
        if result.fun < best_score:
            pulse_I, pulse_Q = _pulse_from_params(result.x)
            if _is_valid(pulse_I, pulse_Q):
                best_I = pulse_I.copy()
                best_Q = pulse_Q.copy()
                best_score = result.fun
    except Exception:
        pass
    
    return best_I, best_Q, best_score

def _dop853_recheck(pulse_I, pulse_Q):
    """Recheck nominal case with DOP853, return max difference."""
    det = 0.0
    scale = 1.0
    
    # Our method
    rhos_ours = _evolve_all_states(pulse_I, pulse_Q, det, scale)
    
    # DOP853 method
    def liouvillian_vec(t, rho_vec):
        rho = rho_vec.reshape(3, 3)
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
    """Return an (80, 2) array of I and Q amplitudes for an X-pi gate."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    
    # Generate seed pulse with DRAG
    pulse_I, pulse_Q = _seed_pulse()
    seed_score = _compute_score(pulse_I, pulse_Q)
    
    # Phase 1: Global optimization with DE on low-dimensional parameters
    elapsed = time.time() - start_time
    de_time_limit = min(120, DEADLINE - elapsed - 60)
    if de_time_limit > 20:
        pulse_I, pulse_Q, de_score = _optimize_with_de(pulse_I, pulse_Q, time_limit=de_time_limit)
    else:
        de_score = seed_score
    
    # Phase 2: Local refinement using gradient-based approach
    elapsed = time.time() - start_time
    local_time_limit = min(80, DEADLINE - elapsed - 25)
    if local_time_limit > 10:
        pulse_I, pulse_Q, local_score = _local_refine(pulse_I, pulse_Q, time_limit=local_time_limit)
    else:
        local_score = de_score
    
    # DOP853 recheck
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 15.0:
        diff = _dop853_recheck(pulse_I, pulse_Q)
        if diff > 2e-7:
            # If recheck fails, try one more local refinement
            remaining = DEADLINE - (time.time() - start_time) - 5.0
            if remaining > 15:
                pulse_I, pulse_Q, _ = _local_refine(pulse_I, pulse_Q, time_limit=remaining)
    
    result = np.column_stack((pulse_I, pulse_Q))
    
    return result
# EVOLVE-BLOCK-END
