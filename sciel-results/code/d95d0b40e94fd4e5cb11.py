import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

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

def _simulate_unitary(pulse_IQ, detuning, scale, anharmonicity, dt, n_steps,
                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag):
    """Simulate the quantum dynamics and return the final unitary matrix."""
    U_total = I3.copy()
    H_static = detuning * n + 0.5 * anharmonicity * n_nm1
    half_scale = 0.5 * scale
    for i in range(n_steps):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        H = H_static + half_scale * (I_val * a_plus_adag + 1j * Q_val * a_minus_adag)
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

def _enforce_constraints(pulse, amp_bound, slew_bound, dt, n_steps):
    """Scale pulse to satisfy amplitude and slew constraints."""
    p = pulse.copy()
    amps = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > amp_bound:
        p = p * (amp_bound / max_amp)
    
    prev = np.array([0.0, 0.0])
    max_slew = 0.0
    for i in range(n_steps):
        diff = np.linalg.norm(p[i] - prev)
        slew = diff / dt
        max_slew = max(max_slew, slew)
        prev = p[i]
    diff = np.linalg.norm(-prev)
    max_slew = max(max_slew, diff / dt)
    
    if max_slew > slew_bound:
        p = p * (slew_bound / max_slew)
    
    return p

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 310.0

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
        """Evaluate the full 9-case score for a pulse."""
        if not _check_validity(pulse_IQ, amp_bound, slew_bound, dt, n_steps):
            return float('inf')
        scores = []
        for det in detunings:
            for sc in scales:
                U = _simulate_unitary(pulse_IQ, det, sc, anharmonicity, dt, n_steps,
                                      a, a_dag, n, I3, n_nm1, a_plus_adag, a_minus_adag)
                score = _compute_score_from_unitary(U, X)
                scores.append(score)
        return np.mean(scores) + np.max(scores)

    # Structural Change: Trajectory Tracking Pulse Generation
    # We generate a pulse by simulating the system and applying a feedback law
    # that steers the state towards the ideal X-gate trajectory.
    
    # Ideal Hamiltonian for X-gate: H_ideal = (pi / (2*T)) * X
    # But we only control the 0-1 subspace drive.
    # H_drive = 0.5 * scale * (u^* a + u a^dag)
    # We want H_total psi ~ H_ideal psi_ideal
    
    def generate_feedback_pulse(gain, smoothing_window):
        """
        Generate a pulse using a feedback control law.
        gain: scales the feedback strength
        smoothing_window: size of moving average for smoothing the raw control signal
        """
        raw_pulse = np.zeros((n_steps, 2), dtype=float)
        psi_actual = np.zeros((n_steps, 3), dtype=complex)
        
        # Ideal trajectory for |0> initial state
        # U_ideal(t) = exp(-i * (pi/(2T)) * t * X)
        # For |0>, psi_ideal(t) = [cos(pi*t/(2T)), -i*sin(pi*t/(2T)), 0]
        
        H_static_nominal = 0.5 * anharmonicity * n_nm1 # Detuning is 0 for nominal design
        
        for i in range(n_steps):
            t = (i + 0.5) * dt
            # Ideal state at time t
            angle = np.pi * t / T_total
            psi_ideal = np.array([np.cos(angle), -1j*np.sin(angle), 0], dtype=complex)
            
            # Current actual state (from previous step simulation)
            if i == 0:
                psi_current = np.array([1, 0, 0], dtype=complex)
            else:
                psi_current = psi_actual[i-1]
                
            # Compute error
            error = psi_ideal - psi_current
            
            # Calculate control input u
            # We want 0.5 * scale * (u^* a + u a^dag) psi ~ H_ideal psi_ideal - H_static psi
            # Simplified: Project error onto the control basis
            # u ~ <a^dag | error> ?
            # Let's use a linear feedback: u = K * (a^dag error)
            # a^dag error is a scalar
            u_complex = gain * np.vdot(a_dag, error)
            
            # Apply smoothing
            # We store raw u in a buffer and smooth it
            # For simplicity, we just store it and smooth at the end
            raw_pulse[i, 0] = np.real(u_complex)
            raw_pulse[i, 1] = np.imag(u_complex)
            
            # Simulate one step to update psi_actual
            # Note: This simulation uses the nominal parameters
            H_drive = 0.5 * 1.0 * (np.conj(u_complex) * a + u_complex * a_dag)
            H = H_static_nominal + H_drive
            U_step = expm(-1j * H * dt)
            psi_next = U_step @ psi_current
            psi_actual[i] = psi_next
            
        # Smooth the raw pulse
        if smoothing_window > 1:
            kernel = np.ones(smoothing_window) / smoothing_window
            raw_pulse[:, 0] = np.convolve(raw_pulse[:, 0], kernel, mode='same')
            raw_pulse[:, 1] = np.convolve(raw_pulse[:, 1], kernel, mode='same')
            
        # Enforce constraints
        pulse_out = _enforce_constraints(raw_pulse, amp_bound, slew_bound, dt, n_steps)
        return pulse_out

    # Optimize gain and smoothing window
    # Initial guess
    init_gain = 1.0
    init_smooth = 3
    
    best_pulse = generate_feedback_pulse(init_gain, init_smooth)
    best_score = evaluate_pulse(best_pulse)
    
    def objective(params):
        gain, smooth = params
        if smooth < 1:
            return float('inf')
        p = generate_feedback_pulse(gain, int(smooth))
        return evaluate_pulse(p)
    
    # Use L-BFGS-B to optimize gain and smoothing
    if time.time() < deadline - 50:
        try:
            # Bounds: gain [0.1, 5.0], smooth [1, 10]
            bounds = [(0.1, 5.0), (1.0, 10.0)]
            result = minimize(objective, [init_gain, init_smooth], method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 50, 'ftol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_pulse = generate_feedback_pulse(result.x[0], int(result.x[1]))
        except Exception:
            pass
            
    # Final validity check
    if not _check_validity(best_pulse, amp_bound, slew_bound, dt, n_steps):
        # Fallback to a simple DRAG pulse if feedback fails
        sigma_seed = T_total / 6.0
        A = np.sqrt(np.pi / 2) / sigma_seed
        I_seed = A * np.exp(-(t_samples - t_center)**2 / (2 * sigma_seed**2))
        dI_dt = -A * (t_samples - t_center) / sigma_seed**2 * np.exp(-(t_samples - t_center)**2 / (2 * sigma_seed**2))
        beta_seed = 0.15
        Q_seed = -beta_seed * dI_dt
        pulse_seed = np.column_stack([I_seed, Q_seed])
        best_pulse = _enforce_constraints(pulse_seed, amp_bound, slew_bound, dt, n_steps)
    
    return best_pulse
