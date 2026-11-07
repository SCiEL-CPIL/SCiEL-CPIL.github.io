import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

# Constants from the task contract
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5


def _build_matrices():
    """Build 3x3 matrices for a, a_dag, n."""
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n


def _liouvillian_superoperator(I, Q, detuning, scale, a, a_dag, n):
    """Build the 9x9 Liouvillian superoperator."""
    H = detuning * n + 0.5 * ANHARMONICITY * (n @ (n - np.eye(3))) + \
        0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))

    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)

    I3 = np.eye(3, dtype=complex)

    L_ham = -1j * (np.kron(H, I3) - np.kron(I3, H.T))

    L_total = L_ham.copy()
    for L in [L1, L2]:
        L_dag = L.conj().T
        L_total += np.kron(L, L.conj()) - 0.5 * (np.kron(L_dag @ L, I3) + np.kron(I3, (L_dag @ L).T))

    return L_total


def _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n):
    """
    Simulate the pulse for 6 input states simultaneously.
    Returns mean fidelity.
    """
    # Input states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_2 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_3 = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_4 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_5 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)

    # Initial density matrices vectorized
    initial_states = []
    for psi in [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]:
        rho = np.outer(psi, psi.conj())
        initial_states.append(rho.flatten(order='F'))

    # Stack into a 9x6 matrix
    states = np.column_stack(initial_states)

    # Precompute propagators for each step
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L_super = _liouvillian_superoperator(I_val, Q_val, detuning, scale, a, a_dag, n)
        U_super = expm(L_super * DT)
        states = U_super @ states

    # Compute fidelities
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    inputs = [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]

    fidelities = []
    for j in range(6):
        psi_in = inputs[j]
        psi_target = X_mat @ psi_in

        # Extract rho_out
        rho_vec = states[:, j]
        rho_out = rho_vec.reshape((3, 3), order='F')

        fidelity = np.real(np.conj(psi_target) @ rho_out @ psi_target)
        fidelity = max(0.0, min(1.0, fidelity))
        fidelities.append(fidelity)

    return np.mean(fidelities)


def _check_validity(pulse):
    """Check amplitude and slew constraints."""
    amps = np.sqrt(pulse[:, 0] ** 2 + pulse[:, 1] ** 2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-12):
        return False

    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND + 1e-12:
        return False

    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i + 1] - pulse[i]) / DT
        if diff > SLEW_BOUND + 1e-12:
            return False

    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND + 1e-12:
        return False

    return True


def _compute_score_nominal(pulse, a, a_dag, n):
    """Compute score for nominal case only (fast)."""
    fid = _simulate_pulse_fast(pulse, 0.0, 1.0, a, a_dag, n)
    return 1.0 - fid


def _compute_score_full(pulse, a, a_dag, n):
    """Compute the score: mean + max infidelity over 9 cases."""
    infidelities = []
    for detuning in [DE_TUNING_BOUNDS[0], 0.0, DE_TUNING_BOUNDS[1]]:
        for scale in [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]:
            fid = _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n)
            infid = 1.0 - fid
            infidelities.append(infid)

    return np.mean(infidelities) + np.max(infidelities)


def _generate_drag_pulse(A, sigma, beta, t_center=20.0):
    """Generate a DRAG-style pulse."""
    t = np.arange(N_STEPS) * DT
    
    # Gaussian envelope
    gauss = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # Offset to ensure zero at edges
    edge_val = np.exp(-0.5 * ((0 - t_center) / sigma) ** 2)
    gauss = gauss - edge_val
    
    # Normalize area to pi
    area = np.sum(gauss) * DT
    if area > 0:
        gauss_norm = gauss / area
    else:
        gauss_norm = gauss
    
    # Compute derivative using central differences
    dgauss = np.zeros(N_STEPS)
    if N_STEPS > 2:
        dgauss[1:-1] = (gauss_norm[2:] - gauss_norm[:-2]) / (2 * DT)
        dgauss[0] = (gauss_norm[1] - gauss_norm[0]) / DT
        dgauss[-1] = (gauss_norm[-1] - gauss_norm[-2]) / DT
    else:
        dgauss[:] = 0
    
    I = A * gauss_norm
    Q = beta * A * dgauss
    
    pulse = np.column_stack([I, Q])
    
    # Enforce amplitude bound
    amps = np.sqrt(pulse[:, 0] ** 2 + pulse[:, 1] ** 2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
    
    return pulse


def _enforce_constraints(pulse):
    """Project pulse to satisfy constraints."""
    # Enforce amplitude bound
    amps = np.sqrt(pulse[:, 0] ** 2 + pulse[:, 1] ** 2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse = pulse * scale_factor
    
    # Enforce slew rate constraint
    # Iterate a few times to converge
    for _ in range(10):
        # Check and fix slew from 0 to step 0
        slew0 = np.linalg.norm(pulse[0]) / DT
        if slew0 > SLEW_BOUND:
            pulse[0] = pulse[0] * (SLEW_BOUND / slew0)
        
        # Check and fix slew between steps
        for i in range(N_STEPS - 1):
            diff = pulse[i + 1] - pulse[i]
            norm_diff = np.linalg.norm(diff)
            if norm_diff / DT > SLEW_BOUND:
                # Scale the difference
                scale = (SLEW_BOUND * DT) / norm_diff
                pulse[i + 1] = pulse[i] + diff * scale
        
        # Check and fix slew from last step to 0
        slew_last = np.linalg.norm(pulse[-1]) / DT
        if slew_last > SLEW_BOUND:
            pulse[-1] = pulse[-1] * (SLEW_BOUND / slew_last)
        
        # Re-enforce amplitude after slew adjustments
        amps = np.sqrt(pulse[:, 0] ** 2 + pulse[:, 1] ** 2)
        max_amp = np.max(amps)
        if max_amp > AMPLITUDE_BOUND:
            scale_factor = AMPLITUDE_BOUND / max_amp
            pulse = pulse * scale_factor
    
    return pulse


def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    a, a_dag, n = _build_matrices()

    # Start with a good DRAG pulse
    best_pulse = _generate_drag_pulse(0.3, 5.0, 0.2)
    best_pulse = _enforce_constraints(best_pulse)
    best_score = _compute_score_full(best_pulse, a, a_dag, n)

    # Phase 1: Optimize on nominal case only (fast)
    # Use DRAG parameters: A, sigma, beta
    def objective_nominal(params):
        A, sigma, beta = params
        p = _generate_drag_pulse(A, sigma, beta)
        p = _enforce_constraints(p)
        if not _check_validity(p):
            return 1.0
        return _compute_score_nominal(p, a, a_dag, n)

    # Multiple starts
    np.random.seed(42)
    nominal_starts = [
        np.array([0.3, 5.0, 0.2]),
        np.array([0.35, 4.5, 0.15]),
        np.array([0.25, 5.5, 0.25]),
        np.array([0.3, 4.0, 0.1]),
        np.array([0.35, 5.0, 0.3]),
        np.array([0.28, 4.8, 0.18]),
        np.array([0.32, 5.2, 0.22]),
    ]

    for x_start in nominal_starts:
        if time.time() > deadline - 60.0:
            break

        try:
            res = minimize(objective_nominal, x_start, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.fun < best_score:
                p = _generate_drag_pulse(*res.x)
                p = _enforce_constraints(p)
                if _check_validity(p):
                    # Verify with full score
                    full_score = _compute_score_full(p, a, a_dag, n)
                    if full_score < best_score:
                        best_score = full_score
                        best_pulse = p
        except Exception:
            pass

    # Phase 2: Fine-tune with full score evaluation
    # Use the best DRAG parameters as starting point
    def objective_full(params):
        A, sigma, beta = params
        p = _generate_drag_pulse(A, sigma, beta)
        p = _enforce_constraints(p)
        if not _check_validity(p):
            return 1.0
        return _compute_score_full(p, a, a_dag, n)

    # Get current best parameters from best_pulse by fitting
    # For simplicity, try a few refinements around the best found
    if time.time() < deadline - 30.0:
        # Try small perturbations
        for perturbation in [0.01, -0.01, 0.02, -0.02]:
            if time.time() > deadline - 30.0:
                break
            try:
                # Perturb the best pulse slightly
                p_pert = best_pulse.copy()
                # Add small noise to I channel
                noise = np.random.randn(N_STEPS) * perturbation * 0.01
                p_pert[:, 0] += noise
                p_pert = _enforce_constraints(p_pert)
                
                if _check_validity(p_pert):
                    score = _compute_score_full(p_pert, a, a_dag, n)
                    if score < best_score:
                        best_score = score
                        best_pulse = p_pert
            except Exception:
                pass

    # Final validity check
    if not _check_validity(best_pulse):
        # Fallback to a safe pulse
        safe_pulse = _generate_drag_pulse(0.3, 5.0, 0.0)
        safe_pulse = _enforce_constraints(safe_pulse)
        best_pulse = safe_pulse

    return best_pulse
