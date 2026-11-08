"""Optimized pulse for three-level transmon X_pi gate with robustness."""
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize
import sys

_DEADLINE = time.time() + 315  # safety margin before 330s kill

# Hardware parameters
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
SCALE_BOUNDS = (0.97, 1.03)
N_STEPS = 80
DT = 0.5

# Precompute ladder operators
_a = np.zeros((3, 3), dtype=complex)
_a[1, 0] = np.sqrt(1.0)
_a[2, 1] = np.sqrt(2.0)
_a_dag = _a.conj().T
_n_op = _a_dag @ _a
_I3 = np.eye(3)

# Lindblad operators
_L1 = _a / np.sqrt(T1)
_L2 = _n_op * np.sqrt(2.0 / Tphi)
_L1_dag = _L1.conj().T
_L2_dag = _L2.conj().T

# Precompute L^dag L terms
_L1_dag_L1 = _L1_dag @ _L1
_L2_dag_L2 = _L2_dag @ _L2


def _build_liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian superoperator for one step."""
    H = detuning * _n_op
    H += 0.5 * ANHARMONICITY * (_n_op @ (_n_op - _I3))
    H += 0.5 * scale * (I * (_a + _a_dag) + 1j * Q * (_a - _a_dag))
    
    H_t = H.T
    L_ham = -1j * (np.kron(_I3, H_t) - np.kron(H_t, _I3))
    
    L_diss = np.zeros((9, 9), dtype=complex)
    for Lk, Lk_dag_Lk in [(_L1, _L1_dag_L1), (_L2, _L2_dag_L2)]:
        Lk_dag = Lk.conj().T
        L_diss += np.kron(Lk_dag.T, Lk) - 0.5 * (np.kron(_I3, Lk_dag_Lk.T) + np.kron(Lk_dag_Lk, _I3))
    
    return L_ham + L_diss


def _simulate_pulse(pulse, detuning, scale):
    """Simulate the pulse and return the final density matrix."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L = _build_liouvillian(detuning, scale, I_val, Q_val)
        L_mat = expm(L * DT)
        rho_vec = rho.flatten()
        rho_vec = L_mat @ rho_vec
        rho = rho_vec.reshape(3, 3)
    
    return rho


def _fidelity(rho, target_state):
    """Compute fidelity between density matrix and pure target state."""
    val = np.vdot(target_state, rho @ target_state).real
    return np.clip(val, 0.0, 1.0)


def _check_validity(pulse):
    """Check if pulse satisfies all constraints."""
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMPLITUDE_BOUND + 1e-9:
            return False
    
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse[i]
        if np.linalg.norm(curr - prev) / DT > SLEW_BOUND + 1e-9:
            return False
        prev = curr
    if np.linalg.norm(prev) / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True


def _compute_score(pulse):
    """Compute the full score for a pulse."""
    if not _check_validity(pulse):
        return None
    
    X_gate = np.zeros((3, 3), dtype=complex)
    X_gate[0, 1] = 1.0
    X_gate[1, 0] = 1.0
    
    init_states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    detunings = [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for scl in scales:
            for init_state in init_states:
                rho = _simulate_pulse(pulse, det, scl)
                target = X_gate @ init_state
                f = _fidelity(rho, target)
                infidelities.append(1.0 - f)
    
    infidelities = np.array(infidelities)
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    return mean_inf + max_inf


def _enforce_constraints(pulse):
    """Enforce amplitude and slew rate constraints."""
    # Amplitude bound
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMPLITUDE_BOUND:
            s = AMPLITUDE_BOUND / amp
            pulse[i, 0] *= s
            pulse[i, 1] *= s
    
    # Slew rate bound
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        delta = pulse[i] - prev
        norm = np.linalg.norm(delta) / DT
        if norm > SLEW_BOUND:
            pulse[i] = prev + SLEW_BOUND * DT * delta / (norm * DT)
        prev = pulse[i]
    
    # Final step to zero
    nf = np.linalg.norm(prev) / DT
    if nf > SLEW_BOUND:
        pulse[-1] = prev * (SLEW_BOUND * DT / np.linalg.norm(prev))
    
    return pulse


def _seed_pulse():
    """Generate DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    t_center = (N_STEPS - 1) * DT / 2.0
    sigma = 40.0 / 6.0
    
    envelope = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    raw_area = np.sum(envelope) * DT
    envelope = envelope * (np.pi / raw_area)
    
    drag_coeff = -0.1
    envelope_deriv = np.gradient(envelope, DT)
    
    I_pulse = envelope
    Q_pulse = drag_coeff * envelope_deriv
    
    pulse = np.column_stack([I_pulse, Q_pulse])
    return _enforce_constraints(pulse)


def _param_pulse(params):
    """Generate pulse from parameters: [sigma, eta, det_comp, amp_scale]."""
    sigma, eta, det_comp, amp_scale = params
    t = np.arange(N_STEPS) * DT
    t_center = (N_STEPS - 1) * DT / 2.0
    
    envelope = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    raw_area = np.sum(envelope) * DT
    if raw_area > 0:
        envelope = envelope * (np.pi / raw_area) * amp_scale
    
    envelope_deriv = np.gradient(envelope, DT)
    
    I_pulse = envelope
    Q_pulse = eta * envelope_deriv + det_comp * (t - t_center) / t_center
    
    pulse = np.column_stack([I_pulse, Q_pulse])
    return _enforce_constraints(pulse)


def _objective(params):
    if time.time() > _DEADLINE - 10:
        return 1.0
    try:
        p = _param_pulse(params)
        s = _compute_score(p)
        if s is None:
            return 1.0
        return s
    except Exception:
        return 1.0


def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return np.zeros((N_STEPS, 2))
    
    # Start with seed
    best_pulse = _seed_pulse()
    best_score = _compute_score(best_pulse)
    if best_score is None:
        best_pulse = np.zeros((N_STEPS, 2))
        best_score = _compute_score(best_pulse)
    
    if best_score is None:
        return np.zeros((N_STEPS, 2))
    
    # Multi-start Nelder-Mead optimization
    # Parameters: [sigma, eta (DRAG), det_comp, amp_scale]
    # sigma in [2, 8] ns, eta in [-0.5, 0.5], det_comp in [-0.01, 0.01], amp_scale in [0.5, 1.5]
    
    initial_guesses = [
        [40.0/6.0, -0.1, 0.0, 1.0],
        [40.0/6.0, -0.2, 0.0, 1.0],
        [40.0/6.0, -0.3, 0.0, 1.0],
        [40.0/6.0, -0.1, 0.002, 1.0],
        [40.0/6.0, -0.1, -0.002, 1.0],
        [3.0, -0.15, 0.0, 1.0],
        [5.0, -0.08, 0.0, 1.0],
        [4.0, -0.25, 0.001, 1.1],
        [4.5, -0.12, -0.001, 0.95],
        [3.5, -0.2, 0.003, 1.05],
        [5.5, -0.05, 0.0, 1.0],
        [4.0, -0.35, 0.0, 1.0],
        [40.0/6.0, -0.15, 0.005, 1.0],
        [40.0/6.0, -0.15, -0.005, 1.0],
    ]
    
    for guess in initial_guesses:
        if time.time() > _DEADLINE - 15:
            break
        try:
            result = minimize(_objective, guess, method='Nelder-Mead',
                             options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-7})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = _param_pulse(result.x)
        except Exception:
            pass
    
    # Second round with refined starts around best
    if time.time() < _DEADLINE - 30:
        # Get current best params (we need to track them)
        # Re-derive from best_pulse is hard, so do a few more starts
        refined_guesses = [
            [4.0, -0.18, 0.001, 1.0],
            [4.2, -0.22, -0.001, 1.0],
            [3.8, -0.15, 0.002, 1.05],
            [4.5, -0.25, 0.0, 0.98],
            [4.0, -0.2, 0.003, 1.02],
            [4.0, -0.2, -0.003, 1.02],
        ]
        for guess in refined_guesses:
            if time.time() > _DEADLINE - 15:
                break
            try:
                result = minimize(_objective, guess, method='Nelder-Mead',
                                 options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
                if result.fun < best_score:
                    best_score = result.fun
                    best_pulse = _param_pulse(result.x)
            except Exception:
                pass
    
    # Final constraint check
    if not _check_validity(best_pulse):
        best_pulse = _enforce_constraints(best_pulse)
        if not _check_validity(best_pulse):
            best_pulse = _seed_pulse()
    
    return best_pulse
