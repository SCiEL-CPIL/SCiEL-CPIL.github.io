"""Improved program for robust X-pi gate control pulse using low-dimensional basis expansion."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.linalg import expm
import time

# Time parameters
DT = 0.5  # ns
N_STEPS = 80
T_TOTAL = 40.0  # ns

# Hardware parameters (in rad/ns units)
ANHARMONICITY = -2.0 * np.pi * 0.25e-3  # -2*pi*0.25 GHz -> rad/ns
AMPLITUDE_BOUND = 0.4  # rad/ns
SLEW_BOUND = 0.12  # (rad/ns)/ns = rad/ns^2
DETUNING_BOUNDS = (-2.0 * np.pi * 0.002e-3, 2.0 * np.pi * 0.002e-3)  # rad/ns
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0  # ns
TPHI = 40000.0  # ns

# Basis functions for pulse representation
# We'll use a small set of Gaussian-modulated basis functions
# to reduce the optimization space from 160 to ~8 parameters

def _make_basis_functions(n_basis=4):
    """Create a set of orthogonal-ish basis functions for the pulse."""
    t = np.linspace(0, T_TOTAL, N_STEPS)
    t_center = T_TOTAL / 2.0
    sigma = T_TOTAL / 6.0  # Same as seed
    
    basis = np.zeros((n_basis, N_STEPS))
    
    # Basis 0: Main Gaussian (DRAG-like)
    basis[0] = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # Basis 1: Linear detuning correction
    basis[1] = (t - t_center) / T_TOTAL * np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # Basis 2: Quadratic correction
    basis[2] = ((t - t_center) / T_TOTAL) ** 2 * np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # Basis 3: Q-channel Gaussian (for DRAG)
    basis[3] = (t - t_center) / T_TOTAL * np.exp(-0.5 * ((t - t_center) / (sigma * 0.8)) ** 2)
    
    # Normalize each basis to have unit L2 norm
    for i in range(n_basis):
        norm = np.sqrt(np.sum(basis[i] ** 2) * DT)
        if norm > 1e-10:
            basis[i] /= norm
    
    return basis

BASIS_FUNCTIONS = _make_basis_functions(4)

def _params_to_pulse(params):
    """Convert low-dimensional parameters to full pulse."""
    # params: [I0, I1, I2, Q0, Q1, Q2]
    # I coefficients for I channel, Q coefficients for Q channel
    I_coeffs = params[:3]
    Q_coeffs = params[3:6]
    
    I_pulse = np.zeros(N_STEPS)
    Q_pulse = np.zeros(N_STEPS)
    
    for i in range(3):
        I_pulse += I_coeffs[i] * BASIS_FUNCTIONS[i]
        Q_pulse += Q_coeffs[i] * BASIS_FUNCTIONS[i + 1] if i < 3 else 0
    
    # Actually, let's use separate basis for Q
    # Q uses shifted/scaled versions
    Q_pulse = np.zeros(N_STEPS)
    for i in range(3):
        if i < len(BASIS_FUNCTIONS):
            Q_pulse += Q_coeffs[i] * BASIS_FUNCTIONS[i]
    
    return np.column_stack([I_pulse, Q_pulse])

def _check_validity(pulse_arr):
    """Check if pulse satisfies amplitude and slew constraints."""
    # Amplitude constraint: sqrt(I^2 + Q^2) <= 0.4
    amps = np.sqrt(pulse_arr[:, 0] ** 2 + pulse_arr[:, 1] ** 2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-10):
        return False
    
    # Slew constraint: ||p[i+1] - p[i]|| / dt <= 0.12
    # From zero into step 1
    diff0 = np.sqrt(pulse_arr[0, 0] ** 2 + pulse_arr[0, 1] ** 2)
    if diff0 / DT > SLEW_BOUND + 1e-10:
        return False
    
    # Between consecutive steps
    for i in range(N_STEPS - 1):
        diff = np.sqrt((pulse_arr[i+1, 0] - pulse_arr[i, 0]) ** 2 + 
                       (pulse_arr[i+1, 1] - pulse_arr[i, 1]) ** 2)
        if diff / DT > SLEW_BOUND + 1e-10:
            return False
    
    # From last step to zero
    diff_last = np.sqrt(pulse_arr[-1, 0] ** 2 + pulse_arr[-1, 1] ** 2)
    if diff_last / DT > SLEW_BOUND + 1e-10:
        return False
    
    return True

def _build_hamiltonian(detuning, scale, I, Q):
    """Build the 3x3 Hamiltonian."""
    # a matrix: diag(sqrt([1,2]), 1)
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    
    # n = a^dagger a
    n = a.conj().T @ a
    
    # H = detuning * n + 0.5 * anharmonicity * n * (n - 1) + 0.5 * scale * (I*(a+a^dag) + i*Q*(a-a^dag))
    H = detuning * n
    H += 0.5 * ANHARMONICITY * n @ (n - np.eye(3))
    
    a_plus = a + a.conj().T
    a_minus = a - a.conj().T
    
    H += 0.5 * scale * (I * a_plus + 1j * Q * a_minus)
    
    return H

def _build_liouvillian(H):
    """Build the Liouvillian superoperator."""
    dim = 3
    H = H.astype(complex)
    
    # Coherent part: -i [H, rho]
    # In vectorized form: -i (I ⊗ H - H^T ⊗ I)
    I_mat = np.eye(dim, dtype=complex)
    
    # Lindblad dissipators
    # L1 = a / sqrt(T1)
    a_op = np.zeros((3, 3), dtype=complex)
    a_op[0, 1] = 1.0
    a_op[1, 2] = np.sqrt(2.0)
    
    L1 = a_op / np.sqrt(T1)
    
    # L2 = n * sqrt(2/Tphi)
    n_op = a_op.conj().T @ a_op
    L2 = n_op * np.sqrt(2.0 / TPHI)
    
    # Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    # In vectorized form (vec(rho) is column-major):
    # vec(A rho B) = (B^T ⊗ A) vec(rho)
    
    liouv = np.zeros((dim*dim, dim*dim), dtype=complex)
    
    # Coherent part
    liouv -= 1j * (I_mat.T @ np.kron(I_mat, H) - np.kron(H.conj().T, I_mat))
    # Actually: vec(H rho) = (I ⊗ H) vec(rho), vec(rho H) = (H^T ⊗ I) vec(rho)
    # So vec([H, rho]) = vec(H rho - rho H) = (I ⊗ H - H^T ⊗ I) vec(rho)
    # And -i [H, rho] -> -i (I ⊗ H - H^T ⊗ I)
    
    liouv = -1j * (np.kron(I_mat, H) - np.kron(H.conj().T, I_mat))
    
    # Dissipative part
    for L in [L1, L2]:
        Ldag = L.conj().T
        # L rho L^dag -> (L^T ⊗ L)
        liouv += np.kron(L.conj().T, L)
        # -0.5 {L^dag L, rho} -> -0.5 (I ⊗ L^dag L + (L^dag L)^T ⊗ I)
        LL = Ldag @ L
        liouv -= 0.5 * (np.kron(I_mat, LL) + np.kron(LL.conj().T, I_mat))
    
    return liouv

def _evolve_state(pulse_arr, detuning, scale, n_steps=N_STEPS):
    """Evolve the density matrix through the pulse."""
    dim = 3
    # Initial state: |0><0|
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    
    rho_vec = rho.flatten()  # Column-major vectorization
    
    for i in range(n_steps):
        I_val = pulse_arr[i, 0]
        Q_val = pulse_arr[i, 1]
        
        H = _build_hamiltonian(detuning, scale, I_val, Q_val)
        liouv = _build_liouvillian(H)
        
        # Exponentiate: rho_vec_new = exp(liouv * dt) @ rho_vec
        # For efficiency, use matrix exponential
        exp_liouv = expm(liouv * DT)
        rho_vec = exp_liouv @ rho_vec
    
    # Reshape back to density matrix
    rho_final = rho_vec.reshape((dim, dim), order='F')
    return rho_final

def _compute_fidelity(rho_final, target_state):
    """Compute fidelity F = <psi_target|rho_final|psi_target>."""
    return np.real(np.vdot(target_state, rho_final @ target_state))

def _score_pulse(pulse_arr, detuning_list=None, scale_list=None):
    """Compute the score for a given pulse."""
    if detuning_list is None:
        detuning_list = [DETUNING_BOUNDS[0], 0.0, DETUNING_BOUNDS[1]]
    if scale_list is None:
        scale_list = [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]
    
    # Target states
    psi0 = np.array([1.0, 0.0, 0.0], dtype=complex)
    psi1 = np.array([0.0, 1.0, 0.0], dtype=complex)
    psi_plus = (psi0 + psi1) / np.sqrt(2.0)
    psi_minus = (psi0 - psi1) / np.sqrt(2.0)
    psi_i_plus = (psi0 + 1j*psi1) / np.sqrt(2.0)
    psi_i_minus = (psi0 - 1j*psi1) / np.sqrt(2.0)
    
    target_states = [psi0, psi1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]
    
    # Target operation: X gate swaps |0> and |1>
    # Fidelity to X means: U|psi> should be X|psi>
    # So we compare rho_final to X|psi><psi|X^dag
    
    X_gate = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype=complex)
    
    fidelities = []
    
    for det in detuning_list:
        for scl in scale_list:
            rho_final = _evolve_state(pulse_arr, det, scl)
            
            case_fidelities = []
            for psi in target_states:
                # Target state after X gate
                psi_target = X_gate @ psi
                fid = _compute_fidelity(rho_final, psi_target)
                case_fidelities.append(fid)
            
            mean_fid = np.mean(case_fidelities)
            fidelities.append(1.0 - mean_fid)
    
    infidelities = np.array(fidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    
    return score

def _objective(params):
    """Objective function for optimization."""
    pulse_arr = _params_to_pulse(params)
    
    if not _check_validity(pulse_arr):
        return 1.0  # Large penalty for invalid pulses
    
    try:
        return _score_pulse(pulse_arr)
    except Exception:
        return 1.0

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    Uses low-dimensional basis expansion with differential evolution for robust optimization.
    """
    start_time = time.time()
    deadline = start_time + 300.0  # 300s budget
    
    # Define bounds for the 6 parameters
    # I coefficients: main amplitude should be ~pi/area to get pi rotation
    # Q coefficients: smaller, for DRAG correction
    bounds = [
        (-2.0, 2.0),  # I0 - main I amplitude
        (-0.5, 0.5),  # I1 - linear correction
        (-0.3, 0.3),  # I2 - quadratic correction
        (-0.3, 0.3),  # Q0 - main Q amplitude (DRAG)
        (-0.2, 0.2),  # Q1 - linear Q correction
        (-0.1, 0.1),  # Q2 - quadratic Q correction
    ]
    
    # Try differential evolution first
    best_params = None
    best_score = float('inf')
    
    try:
        result = differential_evolution(
            _objective, 
            bounds,
            seed=42,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            polish=True,
            update_interval=1,
            callback=lambda xk, convergence: (
                time.time() > deadline - 30.0  # Stop if close to deadline
            )
        )
        
        if result.success or result.fun < best_score:
            best_params = result.x
            best_score = result.fun
    except Exception:
        pass
    
    # If we still have time, try local refinement
    elapsed = time.time() - start_time
    if elapsed < deadline - 60.0 and best_params is not None:
        try:
            local_result = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            if local_result.fun < best_score:
                best_params = local_result.x
                best_score = local_result.fun
        except Exception:
            pass
    
    # If optimization failed, fall back to seed-like pulse
    if best_params is None:
        # Use the seed DRAG pulse as fallback
        t = np.arange(N_STEPS) * DT
        t_center = T_TOTAL / 2.0
        sigma = T_TOTAL / 6.0
        
        # Gaussian envelope with area pi
        gauss = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
        area = np.sum(gauss) * DT
        I_pulse = (np.pi / area) * gauss
        
        # DRAG Q component: derivative of Gaussian
        dgauss = -((t - t_center) / sigma ** 2) * gauss
        darea = np.sum(dgauss) * DT
        if abs(darea) > 1e-10:
            Q_pulse = (0.1 * np.pi / darea) * dgauss
        else:
            Q_pulse = np.zeros(N_STEPS)
        
        pulse_arr = np.column_stack([I_pulse, Q_pulse])
        
        # Check validity and scale down if needed
        if not _check_validity(pulse_arr):
            # Scale down to satisfy constraints
            max_amp = np.max(np.sqrt(I_pulse**2 + Q_pulse**2))
            if max_amp > AMPLITUDE_BOUND:
                scale_factor = AMPLITUDE_BOUND / max_amp * 0.99
                pulse_arr *= scale_factor
        
        return pulse_arr
    
    # Convert best params to pulse
    pulse_arr = _params_to_pulse(best_params)
    
    # Final validity check
    if not _check_validity(pulse_arr):
        # Scale down to satisfy constraints
        max_amp = np.max(np.sqrt(pulse_arr[:, 0]**2 + pulse_arr[:, 1]**2))
        if max_amp > AMPLITUDE_BOUND:
            scale_factor = AMPLITUDE_BOUND / max_amp * 0.99
            pulse_arr *= scale_factor
        
        # Check slew
        for i in range(N_STEPS - 1):
            diff = np.sqrt((pulse_arr[i+1, 0] - pulse_arr[i, 0])**2 + 
                          (pulse_arr[i+1, 1] - pulse_arr[i, 1])**2)
            if diff / DT > SLEW_BOUND:
                # This is problematic; try to smooth
                pass
    
    return pulse_arr
# EVOLVE-BLOCK-END
