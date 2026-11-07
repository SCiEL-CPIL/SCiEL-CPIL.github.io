"""Robust DRAG pulse for 3-level transmon X-pi gate with analytic second-order correction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm
import warnings
warnings.filterwarnings('ignore')

# --- Constants from the task ---
DT = 0.5          # ns
N_STEPS = 80
T_TOTAL = N_STEPS * DT  # 40 ns
AMPL_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUND = 2 * np.pi * 0.002
SCALE_LO, SCALE_HI = 0.97, 1.03
T1 = 25000.0
T2 = 40000.0  # Tphi
ANHARMONICITY = -2 * np.pi * 0.25

# 3-level system matrices
sqrt1 = np.sqrt(1.0)
sqrt2 = np.sqrt(2.0)
a_mat = np.array([[0, sqrt1, 0],
                  [0, 0, sqrt2],
                  [0, 0, 0]], dtype=complex)
ad_mat = a_mat.conj().T
n_mat = ad_mat @ a_mat  # diag(0, 1, 2)

# Lindblad operators
L1 = a_mat / np.sqrt(T1)
L2 = n_mat * np.sqrt(2.0 / T2)

def _build_matrices():
    """Precompute matrix constants."""
    I3 = np.eye(3, dtype=complex)
    return I3, a_mat, ad_mat, n_mat, L1, L2

def _liouvillian_superoperator(detuning, scale, I_val, Q_val):
    """Build the Liouvillian superoperator for one step."""
    I3, a, ad, n, L1, L2 = _build_matrices()
    
    # Hamiltonian
    H = (detuning * n 
         + 0.5 * ANHARMONICITY * (n @ (n - I3))
         + 0.5 * scale * (I_val * (a + ad) + 1j * Q_val * (a - ad)))
    
    # Coherent part: -i[H, rho] in vectorized form
    coherent = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
    
    # Dissipative part
    dissipative = np.zeros((9, 9), dtype=complex)
    for L in [L1, L2]:
        Ldag = L.conj().T
        LdagL = Ldag @ L
        dissipative += (np.kron(Ldag.T, L) 
                       - 0.5 * (np.kron(I3, LdagL) + np.kron(LdagL.T, I3)))
    
    return coherent + dissipative

def _simulate_pulse_fast(pulse_IQ, detuning, scale):
    """Evolve the density matrix through all steps. Returns final rho (3x3)."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    
    I3, a, ad, n, L1, L2 = _build_matrices()
    
    for i in range(N_STEPS):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        L = _liouvillian_superoperator(detuning, scale, I_val, Q_val)
        U = expm(L * DT)
        rho_vec = U @ rho.flatten()
        rho = rho_vec.reshape(3, 3)
    
    return rho

def _check_validity(pulse_IQ):
    """Check amplitude and slew constraints."""
    amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    if np.any(amps > AMPL_BOUND + 1e-12):
        return False
    
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse_IQ[i]
        if np.linalg.norm(curr - prev) / DT > SLEW_BOUND + 1e-12:
            return False
        prev = curr
    if np.linalg.norm(prev) / DT > SLEW_BOUND + 1e-12:
        return False
    
    return True

def _compute_score(pulse_IQ):
    """Compute the score: mean + max of the 9 cases' infidelities."""
    detunings = [-DETUNING_BOUND, 0.0, DETUNING_BOUND]
    scales = [SCALE_LO, 1.0, SCALE_HI]
    
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = (psi0 + psi1) / np.sqrt(2)
    psi_minus = (psi0 - psi1) / np.sqrt(2)
    psi_i_plus = (psi0 + 1j * psi1) / np.sqrt(2)
    psi_i_minus = (psi0 - 1j * psi1) / np.sqrt(2)
    initial_states = [psi0, psi1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]
    
    case_infidelities = []
    for det in detunings:
        for sc in scales:
            case_inf = 0.0
            for psi_init in initial_states:
                psi_target = np.array([psi_init[1], psi_init[0], psi_init[2]], dtype=complex)
                rho_final = _simulate_pulse_fast(pulse_IQ, det, sc)
                F = np.real(np.conj(psi_target.flatten()) @ rho_final @ psi_target.flatten())
                case_inf += (1.0 - F)
            case_inf /= len(initial_states)
            case_infidelities.append(case_inf)
    
    mean_inf = np.mean(case_infidelities)
    max_inf = np.max(case_infidelities)
    return mean_inf + max_inf

def _generate_pulse_drag(beta=-0.2, sigma_factor=40.0/6.0):
    """Generate a DRAG-style Gaussian pulse."""
    t = np.arange(N_STEPS) * DT
    t_center = (N_STEPS - 1) * DT / 2.0
    sigma = sigma_factor
    A = np.sqrt(np.pi / 2.0) / sigma
    envelope = A * np.exp(-(t - t_center)**2 / (2 * sigma**2))
    
    edge_val = A * np.exp(-(t_center)**2 / (2 * sigma**2))
    envelope = envelope - edge_val
    
    d_envelope = np.gradient(envelope, DT)
    
    I = envelope
    Q = beta * d_envelope
    
    return np.column_stack([I, Q])

def _generate_pulse_second_order_drag():
    """Generate pulse with analytic second-order DRAG correction.
    
    The optimal beta for DRAG in a transmon is:
    beta = -g * anharmonicity / (2 * Omega^2) * (dOmega/dt) / Omega
    where g is the coupling strength. For a pi pulse with Gaussian envelope,
    we can compute this analytically.
    """
    t = np.arange(N_STEPS) * DT
    t_center = (N_STEPS - 1) * DT / 2.0
    sigma = 40.0 / 6.0
    A = np.sqrt(np.pi / 2.0) / sigma
    envelope = A * np.exp(-(t - t_center)**2 / (2 * sigma**2))
    
    edge_val = A * np.exp(-(t_center)**2 / (2 * sigma**2))
    envelope = envelope - edge_val
    
    # Compute dOmega/dt
    d_envelope = np.gradient(envelope, DT)
    
    # Analytic beta: beta = -anharmonicity * (dOmega/dt) / (2 * Omega^2)
    # But we need to be careful with division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        beta_local = np.where(np.abs(envelope) > 1e-6, 
                             -ANHARMONICITY * d_envelope / (2 * envelope**2),
                             0.0)
    
    # Average the local beta to get a single effective beta
    # Weight by envelope^2 to emphasize the peak
    weights = envelope**2
    total_weight = np.sum(weights)
    if total_weight > 1e-10:
        beta_opt = np.sum(beta_local * weights) / total_weight
    else:
        beta_opt = -0.2
    
    # Clamp beta to reasonable range
    beta_opt = np.clip(beta_opt, -1.0, 1.0)
    
    I = envelope
    Q = beta_opt * d_envelope
    
    return np.column_stack([I, Q]), beta_opt

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    DEADLINE = 300.0  # seconds
    SAFETY_MARGIN = 10.0  # seconds
    
    # Generate initial pulse with second-order DRAG correction
    pulse_IQ, beta_opt = _generate_pulse_second_order_drag()
    
    # Check validity, if not valid, reduce amplitude
    while not _check_validity(pulse_IQ):
        pulse_IQ *= 0.95
        if np.all(np.abs(pulse_IQ) < 1e-10):
            break
    
    best_score = _compute_score(pulse_IQ)
    best_pulse = pulse_IQ.copy()
    
    # Try local optimization to fine-tune
    if time.time() - start_time < DEADLINE - SAFETY_MARGIN - 60:
        # Parameterize: [beta, amplitude_scale, sigma_scale]
        def objective(params):
            beta, amp_scale, sigma_scale = params
            t = np.arange(N_STEPS) * DT
            t_center = (N_STEPS - 1) * DT / 2.0
            sigma = (40.0 / 6.0) * sigma_scale
            A = np.sqrt(np.pi / 2.0) / sigma
            envelope = A * np.exp(-(t - t_center)**2 / (2 * sigma**2))
            edge_val = A * np.exp(-(t_center)**2 / (2 * sigma**2))
            envelope = envelope - edge_val
            d_envelope = np.gradient(envelope, DT)
            I = amp_scale * envelope
            Q = beta * d_envelope
            p = np.column_stack([I, Q])
            if not _check_validity(p):
                return 1e6
            return _compute_score(p)
        
        initial_params = np.array([beta_opt, 1.0, 1.0])
        bounds = [(-1.0, 1.0), (0.5, 1.5), (0.5, 1.5)]
        
        try:
            result = minimize(objective, initial_params, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6,
                                      'adaptive': True})
            if result.fun < best_score:
                beta, amp_scale, sigma_scale = result.x
                t = np.arange(N_STEPS) * DT
                t_center = (N_STEPS - 1) * DT / 2.0
                sigma = (40.0 / 6.0) * sigma_scale
                A = np.sqrt(np.pi / 2.0) / sigma
                envelope = A * np.exp(-(t - t_center)**2 / (2 * sigma**2))
                edge_val = A * np.exp(-(t_center)**2 / (2 * sigma**2))
                envelope = envelope - edge_val
                d_envelope = np.gradient(envelope, DT)
                I = amp_scale * envelope
                Q = beta * d_envelope
                candidate = np.column_stack([I, Q])
                if _check_validity(candidate):
                    best_pulse = candidate
                    best_score = result.fun
        except Exception:
            pass
    
    # Ensure final pulse is valid
    if not _check_validity(best_pulse):
        best_pulse = _generate_pulse_drag(beta=-0.1)
        while not _check_validity(best_pulse):
            best_pulse *= 0.95
    
    return best_pulse
# EVOLVE-BLOCK-END
