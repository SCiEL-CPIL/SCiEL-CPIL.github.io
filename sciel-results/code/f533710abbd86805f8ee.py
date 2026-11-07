"""Robust X_pi control pulse for a three-level transmon using gradient optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import time

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
T1 = 25000.0
Tphi = 40000.0
DT = 0.5
N_STEPS = 80

# Precompute 3x3 matrices
a_mat = np.zeros((3, 3), dtype=complex)
a_mat[0, 1] = 1.0
a_mat[1, 2] = np.sqrt(2.0)
a_dag = a_mat.conj().T
n_mat = a_dag @ a_mat
n_nm1 = n_mat @ (n_mat - np.eye(3))
I3 = np.eye(3, dtype=complex)

# Lindblad operators
L1 = a_mat / np.sqrt(T1)
L2 = n_mat * np.sqrt(2.0 / Tphi)

# Precompute the fixed part of Liouvillian (dissipation only)
def build_dissipation_superoperator():
    """Build the dissipation part of the Liouvillian (independent of drive)."""
    M = np.zeros((9, 9), dtype=complex)
    for L in [L1, L2]:
        Ldag = L.conj().T
        LdagL = Ldag @ L
        M += np.kron(L.conj(), L)
        M -= 0.5 * np.kron(I3, LdagL)
        M -= 0.5 * np.kron(LdagL.T, I3)
    return M

DISS_M = build_dissipation_superoperator()

def build_coherent_superoperator(detuning, scale, I, Q):
    """Build the coherent part of the Liouvillian superoperator."""
    H = detuning * n_mat + 0.5 * ANHARMONICITY * n_nm1
    drive = 0.5 * scale * (I * (a_mat + a_dag) + 1j * Q * (a_mat - a_dag))
    H += drive
    return -1j * (np.kron(I3, H) - np.kron(H.conj().T, I3))

def simulate_all_cases(pulse_IQ):
    """
    Simulate all 9 detuning/scale cases from all 5 input states.
    Returns array of fidelities (45 values).
    """
    detunings = [-2*np.pi*0.002, 0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_i = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    
    input_states = [psi0, psi1, psi_plus, psi_minus, psi_i]
    target_states = [psi1, psi0, psi_plus, psi_minus, np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)]
    
    fidelities = []
    
    for det in detunings:
        for scale in scales:
            # Precompute all 80 step propagators for this (det, scale)
            step_propagators = []
            for i in range(N_STEPS):
                I_val = pulse_IQ[i, 0]
                Q_val = pulse_IQ[i, 1]
                coh_M = build_coherent_superoperator(det, scale, I_val, Q_val)
                total_M = coh_M + DISS_M
                U = expm(total_M * DT)
                step_propagators.append(U)
            
            # Simulate from each input state
            for psi_in, psi_target in zip(input_states, target_states):
                rho_vec = np.outer(psi_in, psi_in.conj()).reshape(-1)
                for U in step_propagators:
                    rho_vec = U @ rho_vec
                rho_final = rho_vec.reshape(3, 3)
                fid = np.real(np.vdot(psi_target, rho_final @ psi_target))
                fidelities.append(np.clip(fid, 0, 1))
    
    return np.array(fidelities)

def check_constraints(pulse_IQ):
    """Check amplitude and slew constraints."""
    amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-6):
        return False
    if np.sqrt(pulse_IQ[0, 0]**2 + pulse_IQ[0, 1]**2) / DT > SLEW_BOUND + 1e-6:
        return False
    for i in range(N_STEPS - 1):
        diff = np.sqrt((pulse_IQ[i+1, 0] - pulse_IQ[i, 0])**2 + (pulse_IQ[i+1, 1] - pulse_IQ[i, 1])**2)
        if diff / DT > SLEW_BOUND + 1e-6:
            return False
    if np.sqrt(pulse_IQ[-1, 0]**2 + pulse_IQ[-1, 1]**2) / DT > SLEW_BOUND + 1e-6:
        return False
    return True

def scoring_function(pulse_IQ):
    """Compute score: mean + max infidelity."""
    if not check_constraints(pulse_IQ):
        return 1e6
    fids = simulate_all_cases(pulse_IQ)
    infids = 1 - fids
    return np.mean(infids) + np.max(infids)

def make_drag_pulse():
    """Create initial DRAG-style Gaussian pulse."""
    t = np.arange(N_STEPS) * DT
    center = TOTAL_TIME / 2 if 'TOTAL_TIME' in dir() else N_STEPS * DT / 2
    sigma = 40.0 / 6.0
    gauss = np.exp(-0.5 * ((t - center) / sigma)**2)
    # Offset to 0 at edges - already handled by Gaussian
    integral = np.sum(gauss) * DT
    A = np.pi / integral if integral > 0 else 0
    I = A * gauss
    Q = np.zeros(N_STEPS)
    return np.column_stack([I, Q])

TOTAL_TIME = N_STEPS * DT

def pulse(target_gate, hardware_model):
    """Return an (80, 2) array of I and Q amplitudes."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 320
    
    pulse_IQ = make_drag_pulse()
    best_score = scoring_function(pulse_IQ)
    best_pulse = pulse_IQ.copy()
    
    def objective(x):
        p = x.reshape(N_STEPS, 2)
        if not check_constraints(p):
            return 1e6
        return scoring_function(p)
    
    bounds = [(-AMPLITUDE_BOUND, AMPLITUDE_BOUND)] * (2 * N_STEPS)
    
    # Optimization
    try:
        result = minimize(objective, best_pulse.flatten(), method='SLSQP', bounds=bounds,
                         options={'maxiter': 30, 'ftol': 1e-7})
        if result.fun < best_score:
            best_score = result.fun
            best_pulse = result.x.reshape(N_STEPS, 2)
    except:
        pass
    
    if time.time() > deadline - 15:
        return best_pulse
    
    # Additional local optimization
    try:
        np.random.seed(123)
        for trial in range(2):
            if time.time() > deadline - 20:
                break
            x_init = best_pulse.flatten() + np.random.normal(0, 0.005, size=2*N_STEPS)
            x_init = np.clip(x_init, -AMPLITUDE_BOUND, AMPLITUDE_BOUND)
            result2 = minimize(objective, x_init, method='SLSQP', bounds=bounds,
                              options={'maxiter': 20, 'ftol': 1e-7})
            if result2.fun < best_score:
                best_score = result2.fun
                best_pulse = result2.x.reshape(N_STEPS, 2)
    except:
        pass
    
    return best_pulse
# EVOLVE-BLOCK-END
