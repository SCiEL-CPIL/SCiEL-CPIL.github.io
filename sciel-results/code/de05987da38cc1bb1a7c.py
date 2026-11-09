import numpy as np
from scipy.linalg import expm
import time

_PRECOMPUTED = None

def _get_precomputed():
    global _PRECOMPUTED
    if _PRECOMPUTED is None:
        a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
        a_dag = a.conj().T
        n = a_dag @ a
        n_nm1 = n @ (n - np.eye(3))
        a_plus_adag = a + a_dag
        a_minus_adag = a - a_dag
        _PRECOMPUTED = {
            'n': n,
            'n_nm1': n_nm1,
            'a_plus_adag': a_plus_adag,
            'a_minus_adag': a_minus_adag
        }
    return _PRECOMPUTED

def _simulate_unitary(pulse_IQ, detuning, scale, anharmonicity, dt, n_steps):
    """Simulate the quantum dynamics and return the final unitary matrix."""
    ops = _get_precomputed()
    n = ops['n']
    n_nm1 = ops['n_nm1']
    a_plus_adag = ops['a_plus_adag']
    a_minus_adag = ops['a_minus_adag']
    
    I3 = np.eye(3, dtype=complex)
    U_total = I3.copy()
    
    H_static_base = detuning * n + 0.5 * anharmonicity * n_nm1
    
    for i in range(n_steps):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        
        # H = H_static + 0.5 * scale * (I * (a+a†) + iQ * (a-a†))
        H = H_static_base + 0.5 * scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
        
        # Evolve
        U_step = expm(-1j * H * dt)
        U_total = U_step @ U_total
        
    return U_total

def _compute_score_from_unitary(U):
    """Compute the score (mean + max infidelity) from the final unitary."""
    # X gate
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    # Initial states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_01 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_01n = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_0i1 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_0mi1 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    initial_states = [psi_0, psi_1, psi_01, psi_01n, psi_0i1, psi_0mi1]
    
    infidelities = []
    for psi in initial_states:
        phi = X @ psi
        # Final state vector
        psi_final = U @ psi
        # Fidelity
        F = np.abs(np.vdot(phi, psi_final))**2
        infid = 1.0 - F
        infidelities.append(infid)
        
    mean_infid = np.mean(infidelities)
    max_infid = np.max(infidelities)
    return mean_infid + max_infid

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    dt = 0.5
    n_steps = 80
    T_total = 40.0
    
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002
    
    t_samples = (np.arange(n_steps) + 0.5) * dt
    t_center = T_total / 2.0
    
    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]
    
    def check_validity(pulse_IQ):
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amplitudes > amp_bound + 1e-10):
            return False
        
        p = pulse_IQ
        prev = np.array([0.0, 0.0])
        for i in range(n_steps):
            diff = np.linalg.norm(p[i] - prev)
            if diff / dt > slew_bound + 1e-10:
                return False
            prev = p[i]
        diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
        if diff / dt > slew_bound + 1e-10:
            return False
            
        return True

    def evaluate_pulse(pulse_IQ):
        if not check_validity(pulse_IQ):
            return float('inf')
            
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary(pulse_IQ, det, sc, anharmonicity, dt, n_steps)
                score = _compute_score_from_unitary(U)
                scores.append(score)
                
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        return mean_score + max_score

    best_pulse = None
    best_score = float('inf')
    best_sigma = T_total / 6.0
    best_beta = 0.0
    
    # Seed pulse
    sigma_seed = T_total / 6.0
    A_seed = np.sqrt(np.pi / 2) / sigma_seed
    I_seed = A_seed * np.exp(-(t_samples - t_center)**2 / (2 * sigma_seed**2))
    Q_seed = np.zeros(n_steps)
    pulse_seed = np.column_stack([I_seed, Q_seed])
    
    score_seed = evaluate_pulse(pulse_seed)
    best_pulse = pulse_seed.copy()
    best_score = score_seed
    
    # Search parameters: sigma, beta
    sigma_range = np.linspace(T_total / 8.0, T_total / 4.0, 15)
    beta_range = np.linspace(-0.1, 0.1, 15)
    
    # Coarse grid search
    for sigma in sigma_range:
        if time.time() > deadline - 60.0:
            break
        A = np.sqrt(np.pi / 2) / sigma
        I_base = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        dI_dt = -A * (t_samples - t_center) / sigma**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        
        for beta in beta_range:
            if time.time() > deadline - 60.0:
                break
            
            Q_drag = -beta * dI_dt
            pulse_IQ = np.column_stack([I_base, Q_drag])
            
            score = evaluate_pulse(pulse_IQ)
            if score < best_score:
                best_score = score
                best_pulse = pulse_IQ.copy()
                best_sigma = sigma
                best_beta = beta
    
    # Fine grid search around best found
    if time.time() < deadline - 10.0:
        sigma_fine_range = np.linspace(best_sigma - 0.5, best_sigma + 0.5, 10)
        beta_fine_range = np.linspace(best_beta - 0.02, best_beta + 0.02, 10)
        
        for sigma in sigma_fine_range:
            if time.time() > deadline - 10.0:
                break
            A = np.sqrt(np.pi / 2) / sigma
            I_base = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
            dI_dt = -A * (t_samples - t_center) / sigma**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
            
            for beta in beta_fine_range:
                if time.time() > deadline - 10.0:
                    break
                
                Q_drag = -beta * dI_dt
                pulse_IQ = np.column_stack([I_base, Q_drag])
                
                score = evaluate_pulse(pulse_IQ)
                if score < best_score:
                    best_score = score
                    best_pulse = pulse_IQ.copy()

    return best_pulse
