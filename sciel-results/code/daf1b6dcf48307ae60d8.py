import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import differential_evolution, minimize

def _precompute_operators():
    a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)
    n_nm1 = n @ (n - I3)
    a_plus_adag = a + a_dag
    a_minus_adag = a - a_dag
    return a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag

def _simulate_unitary(pulse_IQ, detuning, scale, anharmonicity, dt, n_steps,
                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag):
    U_total = I3.copy()
    H_static = detuning * n + 0.5 * anharmonicity * n_nm1
    for i in range(n_steps):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        H = H_static + 0.5 * scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
        U_step = expm(-1j * H * dt)
        U_total = U_step @ U_total
    return U_total

def _compute_score_from_unitary(U, X):
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
        psi_final = U @ psi
        F = np.abs(np.vdot(phi, psi_final))**2
        infid = 1.0 - F
        infidelities.append(infid)
    mean_infid = np.mean(infidelities)
    max_infid = np.max(infidelities)
    return mean_infid + max_infid

def _check_validity(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
    amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    if np.any(amplitudes > amp_bound + 1e-10):
        return False
    prev = np.array([0.0, 0.0])
    for i in range(n_steps):
        diff = np.linalg.norm(pulse_IQ[i] - prev)
        if diff / dt > slew_bound + 1e-10:
            return False
        prev = pulse_IQ[i]
    diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
    if diff / dt > slew_bound + 1e-10:
        return False
    return True

def _enforce_constraints(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
    p = pulse_IQ.copy()
    
    # Enforce amplitude bound
    for _ in range(20):
        amps = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        mask = amps > amp_bound
        if not np.any(mask):
            break
        p[mask, 0] *= (amp_bound / amps[mask])
        p[mask, 1] *= (amp_bound / amps[mask])
    
    # Enforce slew bound iteratively
    for iteration in range(100):
        prev = np.array([0.0, 0.0])
        violated = False
        for i in range(n_steps):
            diff = p[i] - prev
            dist = np.linalg.norm(diff)
            max_dist = slew_bound * dt
            if dist > max_dist:
                p[i] = prev + diff * (max_dist / dist)
                violated = True
            prev = p[i]
        
        # Check final step to zero
        diff = np.array([0.0, 0.0]) - prev
        dist = np.linalg.norm(diff)
        max_dist = slew_bound * dt
        if dist > max_dist:
            violated = True
        
        if not violated:
            break
    
    # Final amplitude check
    amps = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    mask = amps > amp_bound
    if np.any(mask):
        p[mask, 0] *= (amp_bound / amps[mask])
        p[mask, 1] *= (amp_bound / amps[mask])
    
    return p

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 300.0

    dt = 0.5
    n_steps = 80
    T_total = 40.0
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002

    a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag = _precompute_operators()
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    t_samples = (np.arange(n_steps) + 0.5) * dt
    t_center = T_total / 2.0
    t = t_samples - t_center

    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]

    def _generate_pulse_from_params(params):
        """Generate a DRAG pulse from 5 parameters:
        A: amplitude, sigma: width, beta: DRAG coefficient, 
        gamma: shape exponent (Gaussian-like), phi: phase shift
        """
        A, sigma, beta, gamma, phi = params
        # Clamp to valid ranges
        A = np.clip(A, 0.0, amp_bound)
        sigma = np.clip(sigma, 1.0, 10.0)
        beta = np.clip(beta, -0.5, 0.5)
        gamma = np.clip(gamma, 1.0, 3.0)
        phi = np.clip(phi, -np.pi, np.pi)
        
        # Generalized Gaussian shape
        envelope = np.exp(-np.abs(t / sigma)**gamma)
        I_base = A * envelope * np.cos(phi)
        
        # DRAG: Q proportional to derivative of I
        dI_dt = np.zeros(n_steps)
        for i in range(1, n_steps):
            dI_dt[i] = (I_base[i] - I_base[i-1]) / dt
        if n_steps > 1:
            dI_dt[0] = (I_base[1] - I_base[0]) / dt
            dI_dt[-1] = (I_base[-1] - I_base[-2]) / dt
        
        Q_drag = -beta * dI_dt * sigma  # Scale by sigma to normalize
        
        pulse = np.column_stack([I_base, Q_drag])
        return pulse

    def evaluate_pulse(pulse_IQ):
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary(pulse_IQ, det, sc, anharmonicity, dt, n_steps,
                                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        raw_score = np.mean(scores) + np.max(scores)
        return raw_score

    def objective_param(params):
        if time.time() > deadline - 5:
            return 1e6
        pulse_IQ = _generate_pulse_from_params(params)
        pulse_IQ = _enforce_constraints(pulse_IQ, amp_bound, slew_bound, dt, n_steps)
        score = evaluate_pulse(pulse_IQ)
        return score

    # Phase 1: Differential evolution on 5-parameter space
    bounds_de = [(0.1, 0.4), (2.0, 8.0), (-0.3, 0.3), (1.0, 2.5), (-0.5, 0.5)]
    
    best_params = None
    best_score = 1e6
    
    if time.time() < deadline - 120:
        try:
            de_result = differential_evolution(
                objective_param, bounds_de,
                maxiter=30, popsize=15, tol=1e-4,
                seed=42, polish=False,
                init='latinhypercube'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x
        except Exception:
            pass
    
    # Phase 2: Local refinement with L-BFGS-B on full 160-dim pulse
    if best_params is not None:
        pulse_best = _generate_pulse_from_params(best_params)
        pulse_best = _enforce_constraints(pulse_best, amp_bound, slew_bound, dt, n_steps)
    else:
        # Fallback seed
        A_seed = 0.35
        sigma_seed = T_total / 6.0
        beta_seed = 0.08
        I_base = A_seed * np.exp(-t**2 / (2 * sigma_seed**2))
        dI_dt = -A_seed * t / sigma_seed**2 * np.exp(-t**2 / (2 * sigma_seed**2))
        Q_drag = -beta_seed * dI_dt
        pulse_best = np.column_stack([I_base, Q_drag])
        pulse_best = _enforce_constraints(pulse_best, amp_bound, slew_bound, dt, n_steps)
    
    # Gradient-based refinement on full pulse
    if time.time() < deadline - 30:
        def objective_direct(p_flat):
            if time.time() > deadline - 5:
                return 1e6
            p = p_flat.reshape(80, 2)
            # Amplitude penalty
            amps = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
            amp_pen = np.sum(np.maximum(0, amps - amp_bound)**2)
            
            # Slew penalty
            slew_pen = 0.0
            prev = np.array([0.0, 0.0])
            for i in range(80):
                diff = np.linalg.norm(p[i] - prev)
                if diff > slew_bound * dt:
                    slew_pen += (diff - slew_bound * dt)**2
                prev = p[i]
            diff = np.linalg.norm(np.array([0.0, 0.0]) - prev)
            if diff > slew_bound * dt:
                slew_pen += (diff - slew_bound * dt)**2
            
            score = evaluate_pulse(p)
            return score + 500.0 * (amp_pen + slew_pen)
        
        try:
            res = minimize(objective_direct, pulse_best.flatten(), method='L-BFGS-B',
                           bounds=[(-0.4, 0.4)] * 160,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                pulse_best = res.x.reshape(80, 2)
        except Exception:
            pass
    
    # Final validity check and enforcement
    if not _check_validity(pulse_best, amp_bound, slew_bound, dt, n_steps):
        pulse_best = _enforce_constraints(pulse_best, amp_bound, slew_bound, dt, n_steps)
    
    return pulse_best
