"""Improved program for robust X-pi gate control pulse with DRAG and optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
from itertools import product
import warnings
warnings.filterwarnings('ignore')

def _build_hamiltonian(detuning, scale, I, Q, anharmonicity):
    """Build the 3x3 Hamiltonian for the transmon system."""
    # Creation and annihilation operators for 3-level system
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    
    # Number operator
    n = adag @ a
    
    # Hamiltonian components
    H = detuning * n + 0.5 * anharmonicity * n @ (n - np.eye(3))
    H += 0.5 * scale * (I * (a + adag) + 1j * Q * (a - adag))
    
    return H

def _build_liouvillian(H, T1, Tphi):
    """Build the Liouvillian superoperator for the density matrix evolution."""
    # Lindblad operators
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n = adag @ a
    
    L1 = a / np.sqrt(T1)  # Relaxation
    L2 = n * np.sqrt(2.0 / Tphi)  # Dephasing
    
    # Build superoperator
    dim = 3
    L = np.zeros((dim*dim, dim*dim), dtype=complex)
    
    # Coherent part: -i * [H, rho]
    # In vectorized form: -i * (I ⊗ H - H^T ⊗ I)
    I3 = np.eye(dim)
    H_vec = H.reshape(dim*dim)
    L_coherent = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
    
    # Dissipative part
    for L_op in [L1, L2]:
        L_op_vec = L_op.reshape(dim*dim)
        L_diss = np.kron(L_op, L_op.conj()) - 0.5 * np.kron(L_op.conj().T @ L_op, I3) - 0.5 * np.kron(I3, (L_op.conj().T @ L_op).T)
        L += L_diss
    
    L = L_coherent + L
    return L

def _evolve_step(rho, H, dt, T1, Tphi):
    """Evolve density matrix by one time step using matrix exponential."""
    L = _build_liouvillian(H, T1, Tphi)
    # Vectorize rho
    rho_vec = rho.reshape(9, 1)
    # Exponentiate
    from scipy.linalg import expm
    U = expm(L * dt)
    rho_new_vec = U @ rho_vec
    rho_new = rho_new_vec.reshape(3, 3)
    # Ensure trace is 1
    rho_new = rho_new / np.trace(rho_new)
    return rho_new

def _fidelity(rho, state):
    """Compute fidelity between density matrix and pure state."""
    # Fidelity = <state|rho|state>
    return np.real(np.vdot(state, rho @ state))

def _compute_score(pulse_array, detuning, scale, T1, Tphi, anharmonicity):
    """Compute the infidelity for a given pulse and hardware parameters."""
    n_steps = pulse_array.shape[0]
    dt = 0.5
    
    # Initial states
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
    ]
    
    fidelities = []
    
    for state in states:
        # Initial density matrix
        rho = np.outer(state, state.conj())
        
        # Evolve through all steps
        for i in range(n_steps):
            I_val = pulse_array[i, 0]
            Q_val = pulse_array[i, 1]
            H = _build_hamiltonian(detuning, scale, I_val, Q_val, anharmonicity)
            rho = _evolve_step(rho, H, dt, T1, Tphi)
        
        # Compute fidelity with target X gate
        # X gate maps |0> -> |1>, |1> -> |0>
        # So we want the final state to be X @ initial_state
        X = np.array([[0, 1, 0],
                      [1, 0, 0],
                      [0, 0, 0]], dtype=complex)
        target_state = X @ state
        fid = _fidelity(rho, target_state)
        fidelities.append(fid)
    
    mean_fid = np.mean(fidelities)
    infidelity = 1.0 - mean_fid
    return infidelity

def _score_all_cases(pulse_array, T1, Tphi, anharmonicity):
    """Compute score over all 9 detuning/scale cases."""
    detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            inf = _compute_score(pulse_array, det, sc, T1, Tphi, anharmonicity)
            infidelities.append(inf)
    
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    score = mean_inf + max_inf
    return score

def _check_constraints(pulse_array, dt=0.5, amp_bound=0.4, slew_bound=0.12):
    """Check if pulse satisfies amplitude and slew rate constraints."""
    # Amplitude constraint
    amplitudes = np.sqrt(pulse_array[:, 0]**2 + pulse_array[:, 1]**2)
    if np.any(amplitudes > amp_bound + 1e-10):
        return False
    
    # Slew rate constraint: including from zero to first step and last step to zero
    # From zero to step 0
    p0 = np.array([0.0, 0.0])
    p1 = pulse_array[0]
    if np.linalg.norm(p1 - p0) / dt > slew_bound + 1e-10:
        return False
    
    # Between consecutive steps
    for i in range(len(pulse_array) - 1):
        diff = pulse_array[i+1] - pulse_array[i]
        if np.linalg.norm(diff) / dt > slew_bound + 1e-10:
            return False
    
    # From last step to zero
    plast = pulse_array[-1]
    pzero = np.array([0.0, 0.0])
    if np.linalg.norm(pzero - plast) / dt > slew_bound + 1e-10:
        return False
    
    return True

def _generate_drag_pulse(params, n_steps=80, dt=0.5):
    """Generate a DRAG pulse from parameters."""
    # params: [amp, sigma, drag_beta, phase, offset]
    amp, sigma, drag_beta, phase, offset = params
    
    t = np.arange(n_steps) * dt + dt/2  # Center of each step
    t_center = (n_steps - 1) * dt / 2 + dt/2
    
    # Gaussian envelope
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    
    # Offset to make edges zero (approximately)
    gauss = gauss - gauss[0]
    gauss = gauss - gauss[-1]
    if np.max(gauss) > 0:
        gauss = gauss / np.max(gauss)
    
    # DRAG: add derivative term
    # d/dt of Gaussian
    dgauss = -((t - t_center) / sigma**2) * gauss
    
    # I and Q components
    I = amp * gauss * np.cos(phase) - amp * drag_beta * dgauss * np.sin(phase)
    Q = amp * gauss * np.sin(phase) + amp * drag_beta * dgauss * np.cos(phase)
    
    pulse = np.column_stack([I, Q])
    
    # Normalize area to pi
    area = np.sum(np.sqrt(I**2 + Q**2)) * dt
    if area > 1e-10:
        pulse *= np.pi / area
    
    return pulse

def _objective(params, T1, Tphi, anharmonicity):
    """Objective function for optimization."""
    pulse_array = _generate_drag_pulse(params)
    
    if not _check_constraints(pulse_array):
        return 1e10
    
    score = _score_all_cases(pulse_array, T1, Tphi, anharmonicity)
    return score

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25e-3  # rad/ns
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    n_steps = 80
    
    start_time = time.time()
    deadline = start_time + 300  # 300 seconds
    
    # Initial DRAG pulse parameters
    # [amp, sigma, drag_beta, phase, offset]
    initial_params = [0.3, 40.0/6.0, 0.08, 0.0, 0.0]
    
    # Try a few different initial guesses
    best_score = 1e10
    best_pulse = None
    
    # Generate initial pulse
    pulse_array = _generate_drag_pulse(initial_params)
    if _check_constraints(pulse_array):
        best_pulse = pulse_array.copy()
        best_score = _score_all_cases(pulse_array, T1, Tphi, anharmonicity)
    
    # Optimization
    bounds = [(0.1, 0.35), (3.0, 8.0), (0.0, 0.3), (0.0, 2*np.pi), (-0.1, 0.1)]
    
    # Nelder-Mead local optimization
    if time.time() < deadline - 60:
        try:
            result = minimize(_objective, initial_params, args=(T1, Tphi, anharmonicity),
                             method='Nelder-Mead', bounds=bounds,
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.fun < best_score:
                pulse_array = _generate_drag_pulse(result.x)
                if _check_constraints(pulse_array):
                    best_pulse = pulse_array.copy()
                    best_score = result.fun
        except Exception:
            pass
    
    # Differential evolution for global search
    if time.time() < deadline - 30:
        try:
            de_result = differential_evolution(_objective, bounds, args=(T1, Tphi, anharmonicity),
                                              maxiter=50, popsize=10, tol=1e-6,
                                              seed=42, polish=True)
            
            if de_result.fun < best_score:
                pulse_array = _generate_drag_pulse(de_result.x)
                if _check_constraints(pulse_array):
                    best_pulse = pulse_array.copy()
                    best_score = de_result.fun
        except Exception:
            pass
    
    # Final check
    if best_pulse is None:
        # Fallback to seed
        best_pulse = _generate_drag_pulse(initial_params)
    
    # Ensure constraints are satisfied
    if not _check_constraints(best_pulse):
        # Project onto feasible set by scaling
        amplitudes = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
        if np.max(amplitudes) > amp_bound:
            best_pulse *= amp_bound / np.max(amplitudes)
    
    return best_pulse
# EVOLVE-BLOCK-END
