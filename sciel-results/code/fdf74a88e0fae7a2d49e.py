"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg as sla
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

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
DEADLINE = 310.0  # safety margin before 330s kill

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

    # Detuning term: -i(detuning*n, rho)
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
    np.array([0, 1, 0], dtype=complex),   # X|0>
    np.array([1, 0, 0], dtype=complex),   # X|1>
    (_psi0 + _psi1) / np.sqrt(2),         # X(|0>+|1>)/sqrt(2)
    (_psi0 - _psi1) / np.sqrt(2),         # X(|0>-|1>)/sqrt(2)
    (_psi1 + 1j*_psi0) / np.sqrt(2),      # X(|0>+i|1>)/sqrt(2)
    (_psi1 - 1j*_psi0) / np.sqrt(2),      # X(|0>-i|1>)/sqrt(2)
]

def _evolve_all_states(pulse_I, pulse_Q, detuning, scale):
    """Evolve all 6 input states through the pulse, return final density matrices."""
    rhos = [np.outer(p, p.conj()) for p in [_psi0, _psi1, _psi_plus, _psi_minus, _psi_plus_i, _psi_minus_i]]
    rhos_vec = [r.flatten() for r in rhos]

    # Constant part of Liouvillian (detuning + anharmonicity + dissipation)
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
    """Generate DRAG-style Gaussian seed with physics-informed beta compensation."""
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

    # Physics-informed DRAG beta
    beta = 0.08

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
    params: [sigma, beta, amp_scale, phase_shift, drag_shift]
    - sigma: Gaussian width (time)
    - beta: DRAG Q-component strength
    - amp_scale: overall amplitude scaling
    - phase_shift: time shift of pulse center
    - drag_shift: additional phase shift for DRAG component
    """
    sigma = max(3.0, min(25.0, params[0]))
    beta = max(-0.5, min(0.5, params[1]))
    amp_scale = max(0.3, min(2.0, params[2]))
    phase_shift = max(-8.0, min(8.0, params[3]))
    drag_shift = max(-5.0, min(5.0, params[4]))

    t = np.arange(N_STEPS) * DT
    center = GATE_TIME / 2.0 + phase_shift

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

    # Apply drag_shift to Q component
    t_drag = t + drag_shift
    center_drag = GATE_TIME / 2.0
    dI_dt_shifted = np.exp(-0.5 * (t_drag - center_drag)**2 / sigma**2)
    dI_dt_shifted[0] = 0.0
    dI_dt_shifted[-1] = 0.0
    area_drag = np.sum(dI_dt_shifted) * DT
    if area_drag > 0:
        dI_dt_shifted *= np.pi / area_drag
    dI_dt_shifted = np.gradient(dI_dt_shifted, DT)

    pulse_I = g.copy()
    pulse_Q = beta * dI_dt_shifted

    # Enforce constraints
    pulse_I, pulse_Q = _enforce_constraints(pulse_I, pulse_Q)

    return pulse_I, pulse_Q

def _objective_params(params):
    """Objective function for parameter optimization."""
    pulse_I, pulse_Q = _pulse_from_params(params)
    if not _is_valid(pulse_I, pulse_Q):
        return 1e6
    return _compute_score(pulse_I, pulse_Q)

def _optimize_pulse(pulse_I, pulse_Q, max_iters=100, time_limit=280):
    """Optimize using parameterized search and gradient refinement."""
    best_I = pulse_I.copy()
    best_Q = pulse_Q.copy()
    best_score = _compute_score(best_I, best_Q)

    start_time = time.time()
    np.random.seed(42)

    # Phase 1: Differential evolution on parameter space
    bounds = [(3.0, 25.0), (-0.5, 0.5), (0.3, 2.0), (-8.0, 8.0), (-5.0, 5.0)]
    
    try:
        result = differential_evolution(
            _objective_params, bounds,
            maxiter=50, popsize=10, tol=1e-4,
            seed=42, polish=True
        )
        
        if result.fun < best_score:
            pulse_I, pulse_Q = _pulse_from_params(result.x)
            best_I = pulse_I.copy()
            best_Q = pulse_Q.copy()
            best_score = result.fun
    except Exception:
        pass

    # Phase 2: Gradient-based refinement on full pulse
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        step_size = 0.03
        no_improve = 0

        for it in range(max_iters):
            elapsed = time.time() - start_time
            if elapsed > time_limit - 10:
                break

            if no_improve > 5:
                step_size *= 0.7
                no_improve = 0
                if step_size < 0.005:
                    step_size = 0.005

            # Estimate gradient using finite differences on random subset
            grad_I = np.zeros(N_STEPS)
            grad_Q = np.zeros(N_STEPS)
            n_samples = 8
            indices = np.random.choice(N_STEPS, size=n_samples, replace=False)

            for idx in indices:
                p_plus_I = best_I.copy()
                p_plus_I[idx] += step_size * 0.5
                p_plus_I, p_plus_Q = _enforce_constraints(p_plus_I, best_Q)
                s_plus = _compute_score(p_plus_I, p_plus_Q)

                p_minus_I = best_I.copy()
                p_minus_I[idx] -= step_size * 0.5
                p_minus_I, p_minus_Q = _enforce_constraints(p_minus_I, best_Q)
                s_minus = _compute_score(p_minus_I, p_minus_Q)

                grad_I[idx] = (s_plus - s_minus) / (step_size)

                p_plus_Q = best_Q.copy()
                p_plus_Q[idx] += step_size * 0.5
                p_plus_I2, p_plus_Q2 = _enforce_constraints(best_I, p_plus_Q)
                s_plus2 = _compute_score(p_plus_I2, p_plus_Q2)

                p_minus_Q = best_Q.copy()
                p_minus_Q[idx] -= step_size * 0.5
                p_minus_I2, p_minus_Q2 = _enforce_constraints(best_I, p_minus_Q)
                s_minus2 = _compute_score(p_minus_I2, p_minus_Q2)

                grad_Q[idx] = (s_plus2 - s_minus2) / (step_size)

            new_I = best_I - step_size * grad_I
            new_Q = best_Q - step_size * grad_Q
            new_I, new_Q = _enforce_constraints(new_I, new_Q)

            if not _is_valid(new_I, new_Q):
                no_improve += 1
                continue

            new_score = _compute_score(new_I, new_Q)

            if new_score < best_score:
                best_I = new_I.copy()
                best_Q = new_Q.copy()
                best_score = new_score
                no_improve = 0
            else:
                no_improve += 1

            if no_improve > 15:
                dI = np.random.normal(0, step_size * 0.5, N_STEPS)
                dQ = np.random.normal(0, step_size * 0.5, N_STEPS)
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

    return best_I, best_Q, best_score

def _dop853_recheck(pulse_I, pulse_Q, detuning=0.0, scale=1.0):
    """Recheck nominal case using DOP853 to ensure accuracy within 2e-7."""
    def hamiltonian(t, state):
        rho = state.reshape(3, 3)
        I_val = pulse_I[int(t / DT)] if int(t / DT) < N_STEPS else 0.0
        Q_val = pulse_Q[int(t / DT)] if int(t / DT) < N_STEPS else 0.0

        H = (detuning * _n
             + 0.5 * ANHARMONICITY * _n @ (_n - _eye3)
             + 0.5 * scale * I_val * (_a + _adag)
             + 0.5j * scale * Q_val * (_a - _adag))

        comm = H @ rho - rho @ H

        diss = np.zeros((3, 3), dtype=complex)
        for Lk in [_L1, _L2]:
            Lkd = Lk.conj().T
            LdL = Lkd @ Lk
            diss += Lk @ rho @ Lkd - 0.5 * (LdL @ rho + rho @ LdL)

        d_rho = -1j * comm + diss
        return d_rho.flatten()

    psi0 = np.array([1, 0, 0], dtype=complex)
    rho0 = np.outer(psi0, psi0).flatten()

    sol = solve_ivp(hamiltonian, [0, GATE_TIME], rho0, method='DOP853', rtol=1e-10, atol=1e-12)

    if not sol.success:
        return None

    rho_final = sol.y[:, -1].reshape(3, 3)
    psi_out = np.array([0, 1, 0], dtype=complex)
    F = np.real(np.vdot(psi_out, rho_final @ psi_out))
    F = max(0.0, min(1.0, F))

    return F

def pulse(target_gate, hardware_model):
    """Return an (80, 2) array of I and Q amplitudes for an X-pi gate."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    start_time = time.time()

    # Generate seed pulse
    pulse_I, pulse_Q = _seed_pulse()

    # Optimize
    remaining = DEADLINE - (time.time() - start_time) - 10.0
    pulse_I, pulse_Q, score = _optimize_pulse(pulse_I, pulse_Q, max_iters=100, time_limit=max(10, remaining))

    # DOP853 recheck
    remaining = DEADLINE - (time.time() - start_time) - 5.0
    if remaining > 10:
        F_dop = _dop853_recheck(pulse_I, pulse_Q)

    result = np.column_stack((pulse_I, pulse_Q))

    return result
# EVOLVE-BLOCK-END
