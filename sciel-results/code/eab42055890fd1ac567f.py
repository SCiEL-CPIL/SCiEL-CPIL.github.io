import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import eval_legendre

# Constants
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5
N_LEGENDRE = 8

def _build_matrices():
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

def _precompute_liouvillian_parts(a, a_dag, n):
    """Precompute parts of the Liouvillian that don't change with I, Q, detuning, scale."""
    I3 = np.eye(3, dtype=complex)
    
    # n and n(n-1) for the anharmonicity term
    n_np1 = n @ (n - I3)
    
    # Lindblad parts
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    
    L1_dag = L1.conj().T
    L2_dag = L2.conj().T
    
    # Precompute kron terms for Lindblad
    kron_L1L1d = np.kron(L1, L1.conj())
    kron_L1dL1_I = np.kron(L1_dag @ L1, I3)
    kron_I_L1dL1 = np.kron(I3, (L1_dag @ L1).T)
    
    kron_L2L2d = np.kron(L2, L2.conj())
    kron_L2dL2_I = np.kron(L2_dag @ L2, I3)
    kron_I_L2dL2 = np.kron(I3, (L2_dag @ L2).T)
    
    # Precompute kron terms for drive
    kron_a_plus = np.kron(a + a_dag, I3)
    kron_a_minus = np.kron(a - a_dag, I3)
    kron_I_a_plus_T = np.kron(I3, (a + a_dag).T)
    kron_I_a_minus_T = np.kron(I3, (a - a_dag).T)
    
    # Precompute kron for detuning and anharmonicity
    kron_n_I = np.kron(n, I3)
    kron_I_n_T = np.kron(I3, n.T)
    kron_nn1_I = np.kron(n_np1, I3)
    kron_I_nn1_T = np.kron(I3, n_np1.T)
    
    return {
        'I3': I3,
        'n_np1': n_np1,
        'kron_L1L1d': kron_L1L1d,
        'kron_L1dL1_I': kron_L1dL1_I,
        'kron_I_L1dL1': kron_I_L1dL1,
        'kron_L2L2d': kron_L2L2d,
        'kron_L2dL2_I': kron_L2dL2_I,
        'kron_I_L2dL2': kron_I_L2dL2,
        'kron_a_plus': kron_a_plus,
        'kron_a_minus': kron_a_minus,
        'kron_I_a_plus_T': kron_I_a_plus_T,
        'kron_I_a_minus_T': kron_I_a_minus_T,
        'kron_n_I': kron_n_I,
        'kron_I_n_T': kron_I_n_T,
        'kron_nn1_I': kron_nn1_I,
        'kron_I_nn1_T': kron_I_nn1_T,
    }

def _liouvillian(I_val, Q_val, detuning, scale, parts):
    """Build Liouvillian using precomputed parts."""
    I3 = parts['I3']
    
    L_ham = -1j * (
        detuning * (parts['kron_n_I'] - parts['kron_I_n_T']) +
        0.5 * ANHARMONICITY * (parts['kron_nn1_I'] - parts['kron_I_nn1_T']) +
        0.5 * scale * I_val * (parts['kron_a_plus'] - parts['kron_I_a_plus_T']) +
        0.5 * scale * Q_val * (1j * parts['kron_a_minus'] - 1j * parts['kron_I_a_minus_T'])
    )
    
    L_diss = (
        parts['kron_L1L1d'] - 0.5 * (parts['kron_L1dL1_I'] + parts['kron_I_L1dL1']) +
        parts['kron_L2L2d'] - 0.5 * (parts['kron_L2dL2_I'] + parts['kron_I_L2dL2'])
    )
    
    return L_ham + L_diss

def _simulate_pulse(pulse, detuning, scale, parts, a, a_dag, n):
    """Simulate pulse and return mean fidelity over 6 input states."""
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_2 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_3 = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_4 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_5 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    inputs = [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]
    
    initial_states = []
    for psi in inputs:
        rho = np.outer(psi, psi.conj())
        initial_states.append(rho.flatten(order='F'))
    
    states = np.column_stack(initial_states)
    
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L = _liouvillian(I_val, Q_val, detuning, scale, parts)
        U = expm(L * DT)
        states = U @ states
    
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    fidelities = []
    for j in range(6):
        psi_in = inputs[j]
        psi_target = X_mat @ psi_in
        rho_vec = states[:, j]
        rho_out = rho_vec.reshape((3, 3), order='F')
        fidelity = np.real(np.conj(psi_target) @ rho_out @ psi_target)
        fidelity = max(0.0, min(1.0, fidelity))
        fidelities.append(fidelity)
    
    return np.mean(fidelities)

def _check_validity(pulse):
    """Check amplitude and slew constraints."""
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-10):
        return False
    
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND + 1e-10:
        return False
    
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        if diff > SLEW_BOUND + 1e-10:
            return False
    
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND + 1e-10:
        return False
    
    return True

def _compute_score(pulse, parts, a, a_dag, n):
    """Compute score: mean + max infidelity over 9 cases."""
    infidelities = []
    for detuning in [DE_TUNING_BOUNDS[0], 0.0, DE_TUNING_BOUNDS[1]]:
        for scale in [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]:
            fid = _simulate_pulse(pulse, detuning, scale, parts, a, a_dag, n)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    return np.mean(infidelities) + np.max(infidelities)

def _enforce_constraints(pulse):
    """Project pulse onto feasible set: amplitude and slew bounds."""
    pulse = pulse.copy()
    
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / amp
            pulse[i, 0] *= scale
            pulse[i, 1] *= scale
    
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND:
        scale = SLEW_BOUND * DT / np.linalg.norm(pulse[0])
        pulse[0] *= scale
    
    for i in range(N_STEPS - 1):
        diff = pulse[i+1] - pulse[i]
        norm_diff = np.linalg.norm(diff) / DT
        if norm_diff > SLEW_BOUND:
            max_step = SLEW_BOUND * DT
            direction = diff / np.linalg.norm(diff)
            pulse[i+1] = pulse[i] + direction * max_step
    
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND:
        scale = SLEW_BOUND * DT / np.linalg.norm(pulse[-1])
        pulse[-1] *= scale
    
    return pulse

def _legendre_to_pulse(coeffs_I, coeffs_Q):
    """Convert Legendre coefficients to pulse array."""
    t = np.linspace(0, 1, N_STEPS)
    I = np.zeros(N_STEPS)
    Q = np.zeros(N_STEPS)
    for k in range(N_LEGENDRE):
        Pk = eval_legendre(k, 2*t - 1)
        I += coeffs_I[k] * Pk
        Q += coeffs_Q[k] * Pk
    pulse = np.column_stack([I, Q])
    return pulse

def _pulse_to_legendre(pulse):
    """Convert pulse to Legendre coefficients via least squares."""
    t = np.linspace(0, 1, N_STEPS)
    x = 2*t - 1
    P = np.column_stack([eval_legendre(k, x) for k in range(N_LEGENDRE)])
    
    # Least squares fit
    coeffs_I, _, _, _ = np.linalg.lstsq(P, pulse[:, 0], rcond=None)
    coeffs_Q, _, _, _ = np.linalg.lstsq(P, pulse[:, 1], rcond=None)
    
    return coeffs_I, coeffs_Q

def _coeffs_to_valid_pulse(coeffs_I, coeffs_Q):
    """Convert coefficients to a valid pulse."""
    pulse = _legendre_to_pulse(coeffs_I, coeffs_Q)
    pulse = _enforce_constraints(pulse)
    return pulse

def _generate_initial_coeffs():
    """Generate initial Legendre coefficient sets."""
    initial_sets = []
    
    # Simple Gaussian-like: mostly first few Legendre polynomials
    # P0 = 1, P1 = x, P2 = (3x^2-1)/2
    # For a bell-shaped pulse, we want positive center, zero-ish edges
    
    # Set 1: Pure P0 (flat)
    c1_I = np.zeros(N_LEGENDRE)
    c1_I[0] = 0.3
    c1_Q = np.zeros(N_LEGENDRE)
    initial_sets.append((c1_I, c1_Q))
    
    # Set 2: P0 + P2 (bell shape)
    c2_I = np.zeros(N_LEGENDRE)
    c2_I[0] = 0.25
    c2_I[2] = -0.15
    c2_Q = np.zeros(N_LEGENDRE)
    initial_sets.append((c2_I, c2_Q))
    
    # Set 3: With some DRAG-like Q component
    c3_I = np.zeros(N_LEGENDRE)
    c3_I[0] = 0.25
    c3_I[2] = -0.15
    c3_Q = np.zeros(N_LEGENDRE)
    c3_Q[1] = 0.05
    initial_sets.append((c3_I, c3_Q))
    
    # Set 4: Different amplitude
    c4_I = np.zeros(N_LEGENDRE)
    c4_I[0] = 0.35
    c4_I[2] = -0.2
    c4_Q = np.zeros(N_LEGENDRE)
    c4_Q[1] = -0.05
    initial_sets.append((c4_I, c4_Q))
    
    # Set 5: More complex shape
    c5_I = np.zeros(N_LEGENDRE)
    c5_I[0] = 0.2
    c5_I[2] = -0.1
    c5_I[4] = 0.05
    c5_Q = np.zeros(N_LEGENDRE)
    c5_Q[1] = 0.03
    c5_Q[3] = -0.02
    initial_sets.append((c5_I, c5_Q))
    
    return initial_sets

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 280.0
    
    a, a_dag, n = _build_matrices()
    parts = _precompute_liouvillian_parts(a, a_dag, n)
    
    # Generate initial coefficient sets
    initial_sets = _generate_initial_coeffs()
    
    # Evaluate initial pulses
    best_pulse = None
    best_score = float('inf')
    best_coeffs = None
    
    for cI, cQ in initial_sets:
        if time.time() > deadline - 120.0:
            break
        p = _coeffs_to_valid_pulse(cI, cQ)
        if not _check_validity(p):
            continue
        score = _compute_score(p, parts, a, a_dag, n)
        if score < best_score:
            best_score = score
            best_pulse = p.copy()
            best_coeffs = (cI.copy(), cQ.copy())
    
    # If no valid pulse found, use a fallback
    if best_pulse is None:
        cI = np.zeros(N_LEGENDRE)
        cI[0] = 0.3
        cQ = np.zeros(N_LEGENDRE)
        best_pulse = _coeffs_to_valid_pulse(cI, cQ)
        best_score = _compute_score(best_pulse, parts, a, a_dag, n)
        best_coeffs = (cI, cQ)
    
    # Optimize in Legendre coefficient space
    def objective(coeffs):
        cI = coeffs[:N_LEGENDRE]
        cQ = coeffs[N_LEGENDRE:]
        p = _coeffs_to_valid_pulse(cI, cQ)
        if not _check_validity(p):
            return 1.0
        return _compute_score(p, parts, a, a_dag, n)
    
    # Bounds for coefficients: keep them reasonable
    # The pulse values should stay within [-0.4, 0.4]
    # Legendre polynomials are bounded by 1 in [-1,1], so coefficients roughly bounded by 0.4
    coeff_bounds = [(-0.5, 0.5)] * (2 * N_LEGENDRE)
    
    # L-BFGS-B optimization in coefficient space
    if time.time() < deadline - 60.0:
        try:
            x0 = np.concatenate([best_coeffs[0], best_coeffs[1]])
            res = minimize(objective, x0, method='L-BFGS-B', bounds=coeff_bounds,
                          options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7, 'maxls': 30})
            if res.fun < best_score:
                cI_opt = res.x[:N_LEGENDRE]
                cQ_opt = res.x[N_LEGENDRE:]
                best_pulse = _coeffs_to_valid_pulse(cI_opt, cQ_opt)
                best_score = res.fun
                best_coeffs = (cI_opt, cQ_opt)
        except Exception:
            pass
    
    # Try a few more local optimizations from different starts if time permits
    for cI, cQ in initial_sets:
        if time.time() > deadline - 30.0:
            break
        try:
            x0 = np.concatenate([cI, cQ])
            res = minimize(objective, x0, method='L-BFGS-B', bounds=coeff_bounds,
                          options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 20})
            if res.fun < best_score:
                cI_opt = res.x[:N_LEGENDRE]
                cQ_opt = res.x[N_LEGENDRE:]
                best_pulse = _coeffs_to_valid_pulse(cI_opt, cQ_opt)
                best_score = res.fun
                best_coeffs = (cI_opt, cQ_opt)
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 10.0:
        try:
            x0 = np.concatenate([best_coeffs[0], best_coeffs[1]])
            res = minimize(objective, x0, method='L-BFGS-B', bounds=coeff_bounds,
                          options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20})
            if res.fun < best_score:
                cI_opt = res.x[:N_LEGENDRE]
                cQ_opt = res.x[N_LEGENDRE:]
                best_pulse = _coeffs_to_valid_pulse(cI_opt, cQ_opt)
                best_score = res.fun
                best_coeffs = (cI_opt, cQ_opt)
        except Exception:
            pass
    
    # Final validity check
    if not _check_validity(best_pulse):
        cI = np.zeros(N_LEGENDRE)
        cI[0] = 0.3
        cQ = np.zeros(N_LEGENDRE)
        best_pulse = _coeffs_to_valid_pulse(cI, cQ)
    
    return best_pulse
