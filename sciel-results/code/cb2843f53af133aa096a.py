"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# Constants
DT = 0.5  # ns
N_STEPS = 80
TOTAL_TIME = 40.0  # ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUND = 2 * np.pi * 0.002
SCALE_LOW, SCALE_HIGH = 0.97, 1.03
ANHARMONICITY = -2 * np.pi * 0.25
T1 = 25000.0
Tphi = 40000.0

# 3-level system matrices
# a = [[0, sqrt(1), 0], [0, 0, sqrt(2)], [0, 0, 0]]
a_mat = np.array([
    [0, 1, 0],
    [0, 0, np.sqrt(2)],
    [0, 0, 0]
], dtype=complex)
a_dag = a_mat.conj().T
n_mat = a_dag @ a_mat  # diag(0, 1, 2)

# Lindblad operators
L1 = a_mat / np.sqrt(T1)
L2 = n_mat * np.sqrt(2.0 / Tphi)
L1_dag = L1.conj().T
L2_dag = L2.conj().T

# Precompute Liouvillian basis terms
# H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
# We need to exponentiate the Liouvillian for each step.
# For a 3-level system, the Liouvillian acts on the density matrix (reshaped to 9-vector).
# L(rho) = -i [H, rho] + sum_k (L_k rho L_k† - 0.5 {L_k† L_k, rho})

# Precompute the matrix representation of the Lindblad part (time-independent)
# and the Hamiltonian part (time-dependent).

def _liouvillian_matrix(detuning, scale, I, Q):
    """Build the 9x9 Liouvillian matrix for given parameters."""
    # H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
    n_n1 = n_mat @ (n_mat - np.eye(3))
    drive = I * (a_mat + a_dag) + 1j * Q * (a_mat - a_dag)
    H = detuning * n_mat + 0.5 * ANHARMONICITY * n_n1 + 0.5 * scale * drive
    
    # Liouvillian: -i [H, rho] + dissipators
    # Represent rho as 9-vector (row-major)
    # -i [H, rho] -> -i (H ⊗ I - I ⊗ H^T) in vec representation
    # Actually, vec(H rho H†) = (H† ⊗ H) vec(rho)
    # vec([H, rho]) = vec(H rho - rho H) = (I ⊗ H - H^T ⊗ I) vec(rho)
    # So -i [H, rho] -> -i (I ⊗ H - H^T ⊗ I) vec(rho)
    
    H_T = H.T
    H_comm = np.kron(np.eye(9), H) - np.kron(H_T, np.eye(9))
    
    # Dissipator: L rho L† - 0.5 {L† L, rho}
    # vec(L rho L†) = (L† ⊗ L) vec(rho)
    # vec({L† L, rho}) = vec(L† L rho + rho L† L) = (I ⊗ L† L + (L† L)^T ⊗ I) vec(rho)
    
    L1_L1dag = L1 @ L1_dag
    L2_L2dag = L2 @ L2_dag
    
    D1 = np.kron(L1_dag, L1) - 0.5 * (np.kron(np.eye(9), L1_L1dag) + np.kron(L1_L1dag.T, np.eye(9)))
    D2 = np.kron(L2_dag, L2) - 0.5 * (np.kron(np.eye(9), L2_L2dag) + np.kron(L2_L2dag.T, np.eye(9)))
    
    return -1j * H_comm + D1 + D2


def _evolve_state(rho0, detuning, scale, I, Q):
    """Evolve density matrix through one step of duration DT."""
    L_mat = _liouvillian_matrix(detuning, scale, I, Q)
    # Exponentiate: rho1 = exp(L_mat * DT) @ rho0
    from scipy.linalg import expm
    U = expm(L_mat * DT)
    return U @ rho0


def _fidelity(rho, target):
    """Compute fidelity between density matrix rho and pure state target."""
    # Fidelity = <target|rho|target> for pure target
    return np.real(np.vdot(target, rho @ target))


def _score_pulse(I_arr, Q_arr):
    """Compute the score for a given pulse."""
    # Test cases: 3 detunings x 3 scales
    detunings = [-DETUNING_BOUND, 0.0, DETUNING_BOUND]
    scales = [SCALE_LOW, 1.0, SCALE_HIGH]
    
    # Target states
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_i = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_minus_i = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    targets = [psi0, psi1, psi_plus, psi_minus, psi_i, psi_minus_i]
    
    X_mat = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ], dtype=complex)
    
    infidelities = []
    
    for det in detunings:
        for sc in scales:
            case_infs = []
            for target in targets:
                # Initial state: |target><target|
                rho = np.outer(target, target.conj())
                
                # Evolve through all 80 steps
                for i in range(N_STEPS):
                    rho = _evolve_state(rho, det, sc, I_arr[i], Q_arr[i])
                
                # After evolution, compute fidelity to X|target>
                target_final = X_mat @ target
                fid = _fidelity(rho, target_final)
                case_infs.append(1.0 - fid)
            
            mean_inf = np.mean(case_infs)
            infidelities.append(mean_inf)
    
    mean_score = np.mean(infidelities)
    max_score = np.max(infidelities)
    return mean_score + max_score


def _check_constraints(I_arr, Q_arr):
    """Check amplitude and slew constraints."""
    # Amplitude: sqrt(I^2 + Q^2) <= 0.4
    amps = np.sqrt(I_arr**2 + Q_arr**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-9):
        return False
    
    # Slew: ||p[i+1] - p[i]|| / dt <= 0.12
    # From zero to step 1
    p0 = np.array([0.0, 0.0])
    p1 = np.array([I_arr[0], Q_arr[0]])
    if np.linalg.norm(p1 - p0) / DT > SLEW_BOUND + 1e-9:
        return False
    
    # Between consecutive steps
    for i in range(N_STEPS - 1):
        pi = np.array([I_arr[i], Q_arr[i]])
        pi1 = np.array([I_arr[i+1], Q_arr[i+1]])
        if np.linalg.norm(pi1 - pi) / DT > SLEW_BOUND + 1e-9:
            return False
    
    # From last step to zero
    plast = np.array([I_arr[-1], Q_arr[-1]])
    if np.linalg.norm(p0 - plast) / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True


def _penalty(I_arr, Q_arr):
    """Penalty function for constraint violations."""
    penalty = 0.0
    # Amplitude penalty
    amps = np.sqrt(I_arr**2 + Q_arr**2)
    viol_amp = np.maximum(0, amps - AMPLITUDE_BOUND)
    penalty += 100.0 * np.sum(viol_amp**2)
    
    # Slew penalty
    # From zero to step 1
    p1 = np.array([I_arr[0], Q_arr[0]])
    slew1 = np.linalg.norm(p1) / DT
    if slew1 > SLEW_BOUND:
        penalty += 100.0 * (slew1 - SLEW_BOUND)**2
    
    # Between steps
    for i in range(N_STEPS - 1):
        diff = np.array([I_arr[i+1] - I_arr[i], Q_arr[i+1] - Q_arr[i]])
        slew = np.linalg.norm(diff) / DT
        if slew > SLEW_BOUND:
            penalty += 100.0 * (slew - SLEW_BOUND)**2
    
    # From last step to zero
    plast = np.array([I_arr[-1], Q_arr[-1]])
    slew_last = np.linalg.norm(plast) / DT
    if slew_last > SLEW_BOUND:
        penalty += 100.0 * (slew_last - SLEW_BOUND)**2
    
    return penalty


def _objective(params):
    """Objective function for optimization."""
    I_arr = params[:N_STEPS]
    Q_arr = params[N_STEPS:2*N_STEPS]
    
    score = _score_pulse(I_arr, Q_arr)
    pen = _penalty(I_arr, Q_arr)
    return score + pen


def _seed_pulse():
    """Generate DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    sigma = 40.0 / 6.0
    # Gaussian centered at 20 ns
    g = np.exp(-0.5 * ((t - 20.0) / sigma)**2)
    # Normalize area to pi
    area = np.sum(g) * DT
    scale_factor = np.pi / area
    I_arr = g * scale_factor
    # DRAG term: proportional to derivative
    Q_arr = np.zeros(N_STEPS)
    for i in range(N_STEPS):
        if i == 0:
            Q_arr[i] = (I_arr[1] - I_arr[0]) / DT * 0.1
        elif i == N_STEPS - 1:
            Q_arr[i] = (I_arr[i] - I_arr[i-1]) / DT * 0.1
        else:
            Q_arr[i] = (I_arr[i+1] - I_arr[i-1]) / (2 * DT) * 0.1
    return I_arr, Q_arr


def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I and Q amplitudes."""
    start_time = time.time()
    deadline = start_time + 300.0
    
    # Generate seed
    I_seed, Q_seed = _seed_pulse()
    best_I = I_seed.copy()
    best_Q = Q_seed.copy()
    best_score = _score_pulse(best_I, best_Q)
    best_params = np.concatenate([best_I, best_Q])
    
    # Try optimization if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        try:
            x0 = np.concatenate([I_seed, Q_seed])
            result = minimize(_objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
            elapsed = time.time() - start_time
            if elapsed < deadline - 10.0:
                new_params = result.x
                new_I = new_params[:N_STEPS]
                new_Q = new_params[N_STEPS:2*N_STEPS]
                if _check_constraints(new_I, new_Q):
                    new_score = _score_pulse(new_I, new_Q)
                    if new_score < best_score:
                        best_I = new_I
                        best_Q = new_Q
                        best_score = new_score
                        best_params = new_params
        except Exception:
            pass
    
    # Final constraint check
    if not _check_constraints(best_I, best_Q):
        # Fall back to seed
        best_I, best_Q = _seed_pulse()
    
    return np.column_stack([best_I, best_Q])


# EVOLVE-BLOCK-END
