import numpy as np
from scipy.linalg import expm
import time
from scipy.optimize import differential_evolution

def _precompute_operators():
    """Precompute all constant operators."""
    a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)
    
    n_nm1 = n @ (n - I3)
    a_plus_adag = a + a_dag
    a_minus_adag = a - a_dag
    
    return a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag

def _simulate_unitary_fast(pulse_IQ, detuning, scale, anharmonicity, dt, n_steps,
                           a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag):
    """Simulate the quantum dynamics using matrix exponentiation."""
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
    """Compute the score (mean + max infidelity) from the final unitary."""
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
    """Check if pulse satisfies amplitude and slew rate constraints."""
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

def _gaussian_basis(t, t_center, sigma):
    """Generate Gaussian basis functions."""
    return np.exp(-(t - t_center)**2 / (2 * sigma**2))

def _make_pulse_from_basis_coeffs(t_samples, coeffs, n_steps):
    """
    Build a pulse from basis coefficients.
    coeffs: [A0, A1, A2, B0, B1, B2] where
      I(t) = A0*g0 + A1*g1 + A2*g2
      Q(t) = B0*g0 + B1*g1 + B2*g2
    g0, g1, g2 are Gaussian basis functions with different widths.
    """
    T_total = 40.0
    t_center = T_total / 2.0
    
    # Three Gaussian widths
    sigmas = [T_total/6.0, T_total/8.0, T_total/4.0]
    
    I_vals = np.zeros(n_steps)
    Q_vals = np.zeros(n_steps)
    
    for j, sigma in enumerate(sigmas):
        g = _gaussian_basis(t_samples, t_center, sigma)
        I_vals += coeffs[j] * g
        Q_vals += coeffs[3+j] * g
    
    return np.column_stack([I_vals, Q_vals])

def _normalize_area(pulse_IQ, t_samples, dt, n_steps):
    """Scale pulse so that the area under I is approximately pi."""
    area = np.sum(pulse_IQ[:, 0]) * dt
    if abs(area) > 1e-10:
        scale_factor = np.pi / area
        pulse_IQ = pulse_IQ * scale_factor
    return pulse_IQ

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 320.0  # 320s safety margin (330s limit, 10s buffer)
    
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
    
    detunings = [-detuning_bound, 0.0, detuning_bound]
    scales = [0.97, 1.0, 1.03]
    
    def evaluate_pulse(pulse_IQ):
        """Return the score (lower is better)."""
        if not _check_validity(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
            return float('inf')
            
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary_fast(pulse_IQ, det, sc, anharmonicity, dt, n_steps,
                                           a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
                
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        return mean_score + max_score
    
    # Start with seed pulse
    sigma_seed = T_total / 6.0
    A_seed = np.sqrt(np.pi / 2) / sigma_seed
    I_seed = A_seed * np.exp(-(t_samples - t_center)**2 / (2 * sigma_seed**2))
    Q_seed = np.zeros(n_steps)
    pulse_seed = np.column_stack([I_seed, Q_seed])
    
    best_pulse = pulse_seed.copy()
    best_score = evaluate_pulse(pulse_seed)
    
    # Coarse grid search over DRAG parameters
    sigma_range = np.linspace(T_total / 8.0, T_total / 4.0, 8)
    beta_range = np.linspace(-0.2, 0.2, 8)
    
    def make_drag_pulse(sigma, beta):
        A = np.sqrt(np.pi / 2) / sigma
        I_base = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        dI_dt = -A * (t_samples - t_center) / sigma**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma**2))
        Q_drag = -beta * dI_dt
        return np.column_stack([I_base, Q_drag])
    
    for sigma in sigma_range:
        if time.time() > deadline - 120.0:
            break
        for beta in beta_range:
            if time.time() > deadline - 120.0:
                break
            
            pulse_IQ = make_drag_pulse(sigma, beta)
            score = evaluate_pulse(pulse_IQ)
            if score < best_score:
                best_score = score
                best_pulse = pulse_IQ.copy()
    
    # Differential Evolution over basis coefficients
    if time.time() < deadline - 40.0:
        # Define bounds for the 6 basis coefficients
        # I coefficients: allow negative to positive
        # Q coefficients: allow negative to positive
        bounds = [
            (-1.0, 1.0),  # A0
            (-1.0, 1.0),  # A1
            (-1.0, 1.0),  # A2
            (-0.5, 0.5),  # B0
            (-0.5, 0.5),  # B1
            (-0.5, 0.5),  # B2
        ]
        
        def objective(coeffs):
            pulse_IQ = _make_pulse_from_basis_coeffs(t_samples, coeffs, n_steps)
            pulse_IQ = _normalize_area(pulse_IQ, t_samples, dt, n_steps)
            return evaluate_pulse(pulse_IQ)
        
        try:
            result = differential_evolution(
                objective, bounds,
                maxiter=50, popsize=15,
                tol=1e-4, mut=(0.3, 0.9),
                recomb=0.7, seed=42,
                polish=False
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = _make_pulse_from_basis_coeffs(t_samples, result.x, n_steps)
                best_pulse = _normalize_area(best_pulse, t_samples, dt, n_steps)
        except Exception:
            pass
    
    # Final local refinement using Nelder-Mead on the best pulse found
    if time.time() < deadline - 10.0:
        # Convert best pulse to a parameterized form for local optimization
        # Use a simple parametrization: scale and phase of the best pulse
        def local_objective(params):
            scale = params[0]
            phase = params[1]
            pulse_IQ = best_pulse * scale
            # Apply a small Q adjustment
            pulse_IQ[:, 1] += phase * np.sum(best_pulse[:, 0]) * dt / n_steps
            return evaluate_pulse(pulse_IQ)
        
        from scipy.optimize import minimize
        try:
            result = minimize(local_objective, [1.0, 0.0], method='Nelder-Mead',
                             options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
            
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = best_pulse * result.x[0]
                best_pulse[:, 1] += result.x[1] * np.sum(best_pulse[:, 0]) * dt / n_steps
        except Exception:
            pass
    
    # Ensure final pulse is valid
    if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
        # Fall back to seed
        best_pulse = pulse_seed.copy()
    
    return best_pulse
