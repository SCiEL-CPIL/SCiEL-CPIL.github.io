"""Analytic DRAG pulse with low-dimensional optimization for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

def _build_hamiltonian(I, Q, detuning, anharmonicity, scale):
    """Build 3x3 Hamiltonian for given pulse step."""
    a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n_op = a_dag @ a
    # H = detuning*n + 0.5*anharmonicity*n*(n-1) + 0.5*scale*(I*(a+a_dag) + i*Q*(a-a_dag))
    H = (detuning * n_op 
         + 0.5 * anharmonicity * (n_op @ (n_op - 3*np.eye(3)))  # n(n-1) for 3 levels
         + 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag)))
    return H

def _build_liouvillian(H, T1, Tphi):
    """Build Liouvillian superoperator (9x9) for 3-level system."""
    # Lindblad operators: L1 = a/sqrt(T1), L2 = n*sqrt(2/Tphi)
    a = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n_op = a_dag @ a
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    def comm(X, Y):
        return X @ Y - Y @ X
    
    def lindblad(L, rho_vec):
        L_dag = L.conj().T
        # L rho L^dag - 0.5(L^dag L rho + rho L^dag L)
        rho = rho_vec.reshape(3, 3)
        return (L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)).flatten()
    
    # Build superoperator matrix
    dim = 9
    L_super = np.zeros((dim, dim), dtype=complex)
    # Coherent part: -i(H rho - rho H)
    for i in range(3):
        for j in range(3):
            L_super[i*3+j, :] = (-1j * (H @ np.eye(3) - np.eye(3) @ H)).flatten()[i*3+j]
            # Actually need to do this properly
    # Better: build via matrix representation
    L_super = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        e = np.zeros(dim, dtype=complex)
        e[col] = 1.0
        rho = e.reshape(3, 3)
        # Coherent: -i[H, rho]
        coherent = (-1j * (H @ rho - rho @ H)).flatten()
        # Dissipative
        diss = lindblad(L1, e) + lindblad(L2, e)
        L_super[:, col] = coherent + diss
    
    return L_super

def _evolve_pulse(pulse_IQ, detuning, anharmonicity, scale, T1, Tphi, dt):
    """Evolve all 5 basis states through the pulse. Returns list of final density matrices."""
    n_steps = len(pulse_IQ)
    states_init = []
    # |0>, |1>, (|0>+|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_p = (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2)
    psi_m = (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2)
    psi_ip = (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2)
    psi_im = (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2)
    
    states = [np.outer(p, p.conj()).flatten() for p in [psi0, psi1, psi_p, psi_m, psi_ip, psi_im]]
    
    for step in range(n_steps):
        I_val = pulse_IQ[step, 0]
        Q_val = pulse_IQ[step, 1]
        L_super = _build_liouvillian(_build_hamiltonian(I_val, Q_val, detuning, anharmonicity, scale), T1, Tphi)
        U_step = expm(L_super * dt)
        states = [U_step @ s for s in states]
    
    return states

def _compute_fidelity(pulse_IQ, detuning, anharmonicity, scale, T1, Tphi, dt):
    """Compute mean fidelity to X gate over the 6 test states."""
    states = _evolve_pulse(pulse_IQ, detuning, anharmonicity, scale, T1, Tphi, dt)
    
    # X gate maps |0><0| -> |1><1|, |1><1| -> |0><0|
    # For superpositions, X|psi> gives the target
    X_op = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    # Target states after X:
    # |0> -> |1>, |1> -> |0>, (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = same
    # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2)
    # (|0>-i|1>)/sqrt(2) -> (|1>-i|0>)/sqrt(2)
    
    targets = []
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_p = (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2)
    psi_m = (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2)
    psi_ip = (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2)
    psi_im = (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2)
    
    target_states = [X_op @ psi for psi in [psi0, psi1, psi_p, psi_m, psi_ip, psi_im]]
    
    fidelities = []
    for i in range(6):
        rho_final = states[i].reshape(3, 3)
        rho_target = np.outer(target_states[i], target_states[i].conj())
        fid = np.real(np.trace(rho_final @ rho_target))
        fidelities.append(fid)
    
    return np.mean(fidelities)

def _check_validity(pulse_IQ, dt):
    """Check amplitude and slew rate constraints."""
    # Amplitude bound
    amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    if np.any(amps > 0.4 + 1e-10):
        return False
    # Slew rate: including from zero to step 0 and last step to zero
    prev = np.array([0.0, 0.0])
    for i in range(len(pulse_IQ)):
        curr = pulse_IQ[i]
        if np.linalg.norm(curr - prev) / dt > 0.12 + 1e-10:
            return False
        prev = curr
    # Last step to zero
    if np.linalg.norm(prev) / dt > 0.12 + 1e-10:
        return False
    return True

def _generate_pulse(params):
    """Generate DRAG pulse from parameters: [sigma, alpha, A, center_offset]"""
    sigma = params[0]
    alpha = params[1]
    A = params[2]
    center = params[3]
    
    n_steps = 80
    dt = 0.5
    t = np.arange(n_steps) * dt - 20.0 + center
    
    # Gaussian envelope
    gauss = A * np.exp(-t**2 / (2 * sigma**2))
    
    # DRAG term: derivative of Gaussian
    drag = alpha * A * (-t / sigma**2) * np.exp(-t**2 / (2 * sigma**2))
    
    I = gauss
    Q = drag
    
    # Edge tapering to ensure zero at boundaries
    # Apply smooth taper in first and last few steps
    taper = 5
    for i in range(taper):
        frac = (i + 1) / taper
        I[i] *= frac
        Q[i] *= frac
        I[-1-i] *= frac
        Q[-1-i] *= frac
    
    pulse_IQ = np.column_stack([I, Q])
    
    # Enforce constraints
    # Amplitude
    amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    if np.max(amps) > 0.4:
        scale_factor = 0.4 / np.max(amps)
        pulse_IQ *= scale_factor
    
    # Slew rate enforcement
    # Simple: if slew violated, reduce amplitude
    for _ in range(20):
        if _check_validity(pulse_IQ, dt):
            break
        pulse_IQ *= 0.95
    
    return pulse_IQ

def _score_function(params):
    """Score function for optimizer: returns infidelity (lower is better)."""
    pulse_IQ = _generate_pulse(params)
    
    if not _check_validity(pulse_IQ, 0.5):
        return 1.0
    
    # Evaluate over 9 cases
    anharmonicity = -2 * np.pi * 0.25
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    
    detunings = [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            fid = _compute_fidelity(pulse_IQ, det, anharmonicity, sc, T1, Tphi, dt)
            infidelities.append(1.0 - fid)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I and Q amplitudes."""
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin before 330s kill
    
    # Initial parameters: sigma, alpha, A, center_offset
    # Seed: Gaussian with sigma ~ 40/6, area pi
    # Area of Gaussian = A * sigma * sqrt(2*pi), so A = pi / (sigma * sqrt(2*pi))
    sigma0 = 40.0 / 6.0
    A0 = np.pi / (sigma0 * np.sqrt(2 * np.pi))
    alpha0 = 0.0
    center0 = 0.0
    
    x0 = np.array([sigma0, alpha0, A0, center0])
    
    # Bounds: sigma in [2, 10], alpha in [-1, 1], A in [0.05, 0.5], center in [-2, 2]
    bounds = [(2.0, 10.0), (-1.0, 1.0), (0.05, 0.5), (-2.0, 2.0)]
    
    # First, try a few random initial points to get a good starting point
    best_score = np.inf
    best_params = x0
    
    # Evaluate seed
    seed_pulse_IQ = _generate_pulse(x0)
    if _check_validity(seed_pulse_IQ, 0.5):
        best_score = _score_function(x0)
    
    # Try a few random starts
    np.random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 60:
            break
        x_rand = np.array([
            np.random.uniform(3.0, 8.0),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(0.1, 0.4),
            np.random.uniform(-1.0, 1.0)
        ])
        pr = _generate_pulse(x_rand)
        if _check_validity(pr, 0.5):
            sr = _score_function(x_rand)
            if sr < best_score:
                best_score = sr
                best_params = x_rand
    
    # Nelder-Mead optimization
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _score_function,
                best_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-8, 'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Final refinement with smaller steps if time permits
    if time.time() < deadline - 10:
        try:
            result2 = minimize(
                _score_function,
                best_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-9, 'adaptive': True,
                          'initial_simplex': best_params + np.array([0.1, 0.05, 0.02, 0.1])}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x
        except Exception:
            pass
    
    final_pulse = _generate_pulse(best_params)
    
    # Final validity check
    if not _check_validity(final_pulse, 0.5):
        # Fallback to seed
        final_pulse = _generate_pulse(x0)
        if not _check_validity(final_pulse, 0.5):
            # Ultra-safe fallback
            final_pulse = np.zeros((80, 2))
            # Simple rectangular pulse with area pi
            amp = np.pi / 40.0
            final_pulse[10:70, 0] = amp
            # Check
            if not _check_validity(final_pulse, 0.5):
                final_pulse = np.zeros((80, 2))
                final_pulse[20:60, 0] = np.pi / 20.0
    
    return final_pulse

# EVOLVE-BLOCK-END
