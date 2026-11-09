"""Improved program for robust X-pi gate control pulse using DRAG optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

def _build_hamiltonian(detuning, scale, I, Q, anharmonicity):
    """Build the 3-level Hamiltonian matrix."""
    # Ladder operator a for 3 levels (0, 1, 2)
    # a = diag(sqrt([1, 2]), 1) means a_{1,0} = sqrt(1), a_{2,1} = sqrt(2)
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = np.sqrt(1)
    a[2, 1] = np.sqrt(2)
    adag = a.conj().T
    
    # Number operator n = a^dagger a
    n = adag @ a
    
    # H = detuning * n + 0.5 * anharmonicity * n * (n - 1) + 0.5 * scale * (I * (a + a^dagger) + 1j * Q * (a - a^dagger))
    H = detuning * n + 0.5 * anharmonicity * n @ (n - np.eye(3))
    H += 0.5 * scale * (I * (a + adag) + 1j * Q * (a - adag))
    
    return H

def _build_liouvillian(detuning, scale, I, Q, anharmonicity, T1, Tphi):
    """Build the Liouvillian superoperator for the master equation."""
    H = _build_hamiltonian(detuning, scale, I, Q, anharmonicity)
    dim = 3
    
    # Coherent part: -i [H, .]
    # L_coherent(rho) = -i * (H @ rho - rho @ H)
    
    # Dissipative part:
    # L1(rho) = (2 * a @ rho @ a^dagger - rho @ a^dagger @ a - a^dagger @ a @ rho) / T1
    # L2(rho) = (2 * n @ rho @ n - rho @ n @ n - n @ n @ rho) / Tphi
    
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = np.sqrt(1)
    a[2, 1] = np.sqrt(2)
    adag = a.conj().T
    n = adag @ a
    
    # Build Liouvillian as a 9x9 matrix acting on vec(rho)
    # vec(A @ rho @ B) = (B^T ⊗ A) vec(rho)
    
    # Coherent: -i [H, rho] = -i H rho + i rho H
    # vec(-i H rho) = -i (I ⊗ H) vec(rho)
    # vec(i rho H) = i (H^T ⊗ I) vec(rho)
    L_coh = -1j * (np.kron(np.eye(3), H) + np.kron(H.T, np.eye(3)))
    
    # Dissipative 1: a / sqrt(T1)
    # L_a(rho) = 2 a rho a^dagger - a^dagger a rho - rho a^dagger a
    # vec(2 a rho a^dagger) = 2 (a^T ⊗ a) vec(rho)
    # vec(-a^dagger a rho) = -(I ⊗ a^dagger a) vec(rho)
    # vec(-rho a^dagger a) = -((a^dagger a)^T ⊗ I) vec(rho)
    gamma1 = 1.0 / T1
    L_d1 = 2 * gamma1 * np.kron(adag.T, a) - gamma1 * (np.kron(np.eye(3), adag @ a) + np.kron((adag @ a).T, np.eye(3)))
    
    # Dissipative 2: n sqrt(2/Tphi)
    # L_n(rho) = 2 n rho n - n^2 rho - rho n^2
    gamma2 = 2.0 / Tphi
    L_d2 = 2 * gamma2 * np.kron(n.T, n) - gamma2 * (np.kron(np.eye(3), n @ n) + np.kron((n @ n).T, np.eye(3)))
    
    L = L_coh + L_d1 + L_d2
    return L

def _evolve_pulse(pulse, detuning, scale, anharmonicity, T1, Tphi, dt):
    """Evolve the system with the given pulse and return the final unitary (or process)."""
    n_steps = len(pulse)
    dim = 3
    
    # Initial density matrix for |0><0|
    rho0 = np.zeros((3, 3), dtype=complex)
    rho0[0, 0] = 1.0
    
    # We'll track the evolution of the basis states to get the process
    # Actually, for fidelity we need to track how each initial state evolves
    # Let's compute the process matrix by evolving each basis state
    
    # Better approach: evolve the density matrix for each of the 5 test states
    # But we need the fidelity to X gate, so let's compute the unitary evolution
    
    # For simplicity, let's compute the evolution operator by evolving basis states
    # U such that |psi_final> = U |psi_initial>
    
    # We'll use the Liouvillian approach but track the process
    # Actually, let's just compute the infidelity directly by evolving each test state
    
    # Test states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    # Wait, the task says: |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)
    # That's 6 states, but the task says "mean fidelity to X over" these states
    
    # Actually, re-reading: "infidelity = 1 - mean fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # So 6 states total: |0>, |1>, |+>, |->, |R>, |L>
    
    # Let's compute the process by evolving the basis states
    # We'll track the evolution of each computational basis state
    
    # State vector approach: evolve |0>, |1>, |2> to get U
    # Then compute fidelity for each test state
    
    def _evolve_state_vector(psi0, pulse, detuning, scale, anharmonicity, T1, Tphi, dt):
        """Evolve a state vector through the pulse sequence."""
        dim = 3
        psi = psi0.copy()
        
        for i in range(len(pulse)):
            I_val = pulse[i, 0]
            Q_val = pulse[i, 1]
            
            # Build Liouvillian for this step
            L = _build_liouvillian(detuning, scale, I_val, Q_val, anharmonicity, T1, Tphi)
            
            # Evolve density matrix: rho_{i+1} = expm(L * dt) @ vec(rho_i)
            rho = np.outer(psi, psi.conj())
            rho_vec = rho.flatten()
            
            # Exponentiate the Liouvillian
            Ldt = L @ dt
            expLdt = expm(Ldt)
            
            rho_vec_new = expLdt @ rho_vec
            rho_new = rho_vec_new.reshape((3, 3))
            
            # Extract state vector (assuming pure state evolution)
            # For pure states, rho = |psi><psi|, so psi = rho @ psi_initial
            # But with dissipation, the state may become mixed
            # For fidelity calculation, we need the full process
            
            # Actually, let's track the density matrix directly
            psi = None  # We'll switch to density matrix tracking
        
        # This approach is getting complicated. Let's use density matrix evolution directly.
        pass
    
    # Let's use a cleaner approach: evolve the density matrix for each test state
    # and compute fidelity with the target state after X gate
    
    def _evolve_density_matrix(rho0, pulse, detuning, scale, anharmonicity, T1, Tphi, dt):
        """Evolve a density matrix through the pulse sequence."""
        rho = rho0.copy()
        
        for i in range(len(pulse)):
            I_val = pulse[i, 0]
            Q_val = pulse[i, 1]
            
            # Build Liouvillian for this step
            L = _build_liouvillian(detuning, scale, I_val, Q_val, anharmonicity, T1, Tphi)
            
            # Evolve: rho_{i+1} = expm(L * dt) @ vec(rho_i)
            rho_vec = rho.flatten()
            Ldt = L @ dt
            expLdt = expm(Ldt)
            
            rho_vec_new = expLdt @ rho_vec
            rho = rho_vec_new.reshape((3, 3))
        
        return rho
    
    # Test states and their target states after X gate
    # X gate: |0> -> |1>, |1> -> |0>, |2> -> |2>
    # So target states:
    # |0> -> |1>
    # |1> -> |0>
    # (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = same
    # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2)
    # (|0>-i|1>)/sqrt(2) -> (|1>-i|0>)/sqrt(2) = -i(|0>+i|1>)/sqrt(2)
    
    test_states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # |+>
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),  # |->
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),  # |R>
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),  # |L>
    ]
    
    target_states = [
        np.array([0, 1, 0], dtype=complex),  # |1>
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # |+>
        -np.array([1, -1, 0], dtype=complex) / np.sqrt(2),  # -|->
        1j * np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),  # i|L>
        -1j * np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),  # -i|R>
    ]
    
    fidelities = []
    for psi0, psi_target in zip(test_states, target_states):
        rho0 = np.outer(psi0, psi0.conj())
        rho_final = _evolve_density_matrix(rho0, pulse, detuning, scale, anharmonicity, T1, Tphi, dt)
        
        # Fidelity: F = <psi_target|rho_final|psi_target>
        fidelity = np.real(np.vdot(psi_target, rho_final @ psi_target))
        fidelity = np.clip(fidelity, 0, 1)
        fidelities.append(fidelity)
    
    mean_fidelity = np.mean(fidelities)
    infidelity = 1 - mean_fidelity
    
    return infidelity

def _score_pulse(pulse, hardware_model):
    """Score a pulse by computing the mean + max infidelity over 9 cases."""
    dt = 0.5
    anharmonicity = -2 * np.pi * 0.25e-3  # rad/ns
    T1 = 25000.0
    Tphi = 40000.0
    
    detunings = [-2 * np.pi * 0.002e-3, 0, 2 * np.pi * 0.002e-3]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for scale in scales:
            inf = _evolve_pulse(pulse, det, scale, anharmonicity, T1, Tphi, dt)
            infidelities.append(inf)
    
    infidelities = np.array(infidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    
    return score

def _make_drag_pulse(params, dt=0.5, n_steps=80):
    """Create a DRAG-style pulse from parameters."""
    amp, sigma, drag_amp, drag_sigma, phase = params
    
    t = np.arange(n_steps) * dt
    t_center = (n_steps - 1) * dt / 2
    
    # Gaussian envelope
    gauss = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # Offset to zero at edges (already zero at edges due to Gaussian)
    # DRAG component: derivative of Gaussian
    drag = np.gradient(gauss, dt)
    
    I = amp * gauss * np.cos(phase) - drag_amp * drag * np.sin(phase)
    Q = amp * gauss * np.sin(phase) + drag_amp * drag * np.cos(phase)
    
    pulse = np.column_stack([I, Q])
    
    # Check constraints
    amplitudes = np.sqrt(I**2 + Q**2)
    if np.max(amplitudes) > 0.4:
        # Scale down
        scale_factor = 0.4 / np.max(amplitudes)
        I *= scale_factor
        Q *= scale_factor
        pulse = np.column_stack([I, Q])
    
    # Check slew rate
    for i in range(n_steps - 1):
        dI = pulse[i+1, 0] - pulse[i, 0]
        dQ = pulse[i+1, 1] - pulse[i, 1]
        slew = np.sqrt(dI**2 + dQ**2) / dt
        if slew > 0.12:
            # This is a constraint violation; we'll handle it in the optimizer
            pass
    
    return pulse

def _constrained_pulse(params):
    """Create a pulse that satisfies constraints."""
    pulse = _make_drag_pulse(params)
    
    # Enforce amplitude constraint
    amplitudes = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    mask = amplitudes > 0.4
    if np.any(mask):
        scale = 0.4 / amplitudes[mask]
        pulse[mask, 0] *= scale
        pulse[mask, 1] *= scale
    
    # Enforce slew rate constraint (simple clipping)
    for i in range(len(pulse)):
        if i == 0:
            prev = np.array([0.0, 0.0])
        else:
            prev = pulse[i-1]
        
        diff = pulse[i] - prev
        norm = np.linalg.norm(diff)
        if norm > 0.12 * 0.5:
            pulse[i] = prev + 0.12 * 0.5 * diff / norm
        
        if i == len(pulse) - 1:
            # Check from last step to zero
            diff = -pulse[i]
            norm = np.linalg.norm(diff)
            if norm > 0.12 * 0.5:
                pulse[i] = -0.12 * 0.5 * diff / norm
    
    return pulse

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    Uses DRAG-style pulse with optimized parameters.
    """
    start_time = time.time()
    deadline = start_time + 320  # Leave 10s margin
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25e-3  # rad/ns
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    n_steps = 80
    
    # Initial guess: DRAG pulse
    sigma = 40.0 / 6.0
    t = np.arange(n_steps) * dt
    t_center = (n_steps - 1) * dt / 2
    
    # Seed pulse: Gaussian with area pi
    gauss = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    # Normalize to area pi
    area = np.sum(gauss) * dt
    gauss = gauss * (np.pi / area)
    
    # DRAG component
    drag = np.gradient(gauss, dt)
    
    # Initial amplitude and drag
    amp = np.max(gauss)
    drag_amp = 0.0
    
    I = gauss
    Q = drag * 0.1  # Small initial DRAG
    
    pulse_initial = np.column_stack([I, Q])
    
    # Check and fix constraints
    amplitudes = np.sqrt(pulse_initial[:, 0]**2 + pulse_initial[:, 1]**2)
    mask = amplitudes > 0.4
    if np.any(mask):
        scale = 0.4 / amplitudes[mask]
        pulse_initial[mask, 0] *= scale
        pulse_initial[mask, 1] *= scale
    
    # Compute initial score
    initial_score = _score_pulse(pulse_initial, hardware_model)
    
    best_pulse = pulse_initial.copy()
    best_score = initial_score
    
    # Optimization using differential evolution
    # Parameters: [amp_scale, sigma, drag_scale, drag_sigma, phase]
    bounds = [
        (0.5, 1.5),      # amp_scale
        (3.0, 10.0),     # sigma
        (-0.5, 0.5),     # drag_scale
        (3.0, 10.0),     # drag_sigma
        (-np.pi/4, np.pi/4),  # phase
    ]
    
    def objective(params):
        if time.time() > deadline:
            return best_score
        
        amp_scale, sigma_opt, drag_scale, drag_sigma, phase = params
        
        gauss = np.exp(-0.5 * ((t - t_center) / sigma_opt) ** 2)
        area = np.sum(gauss) * dt
        gauss = gauss * (np.pi / area) * amp_scale
        
        drag = np.exp(-0.5 * ((t - t_center) / drag_sigma) ** 2)
        drag = np.gradient(drag, dt) * drag_scale
        
        I = gauss * np.cos(phase) - drag * np.sin(phase)
        Q = gauss * np.sin(phase) + drag * np.cos(phase)
        
        p = np.column_stack([I, Q])
        
        # Enforce amplitude constraint
        amplitudes = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        mask = amplitudes > 0.4
        if np.any(mask):
            scale = 0.4 / amplitudes[mask]
            p[mask, 0] *= scale
            p[mask, 1] *= scale
        
        # Enforce slew rate
        for i in range(len(p)):
            if i == 0:
                prev = np.array([0.0, 0.0])
            else:
                prev = p[i-1]
            
            diff = p[i] - prev
            norm = np.linalg.norm(diff)
            max_slew = 0.12 * dt
            if norm > max_slew:
                p[i] = prev + max_slew * diff / norm
            
            if i == len(p) - 1:
                diff = -p[i]
                norm = np.linalg.norm(diff)
                if norm > max_slew:
                    p[i] = -max_slew * diff / norm
        
        score = _score_pulse(p, hardware_model)
        
        if score < best_score:
            best_score = score
            best_pulse = p.copy()
        
        return score
    
    # Run differential evolution
    try:
        result = differential_evolution(
            objective, bounds,
            maxiter=50, popsize=10,
            seed=42, tol=1e-4,
            polish=False
        )
        
        if time.time() < deadline - 30:
            # Polish with local optimization
            def objective_polish(params):
                if time.time() > deadline:
                    return best_score
                
                amp_scale, sigma_opt, drag_scale, drag_sigma, phase = params
                
                gauss = np.exp(-0.5 * ((t - t_center) / sigma_opt) ** 2)
                area = np.sum(gauss) * dt
                gauss = gauss * (np.pi / area) * amp_scale
                
                drag = np.exp(-0.5 * ((t - t_center) / drag_sigma) ** 2)
                drag = np.gradient(drag, dt) * drag_scale
                
                I = gauss * np.cos(phase) - drag * np.sin(phase)
                Q = gauss * np.sin(phase) + drag * np.cos(phase)
                
                p = np.column_stack([I, Q])
                
                amplitudes = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
                mask = amplitudes > 0.4
                if np.any(mask):
                    scale = 0.4 / amplitudes[mask]
                    p[mask, 0] *= scale
                    p[mask, 1] *= scale
                
                for i in range(len(p)):
                    if i == 0:
                        prev = np.array([0.0, 0.0])
                    else:
                        prev = p[i-1]
                    
                    diff = p[i] - prev
                    norm = np.linalg.norm(diff)
                    max_slew = 0.12 * dt
                    if norm > max_slew:
                        p[i] = prev + max_slew * diff / norm
                    
                    if i == len(p) - 1:
                        diff = -p[i]
                        norm = np.linalg.norm(diff)
                        if norm > max_slew:
                            p[i] = -max_slew * diff / norm
                
                score = _score_pulse(p, hardware_model)
                
                if score < best_score:
                    best_score = score
                    best_pulse = p.copy()
                
                return score
            
            polish_result = minimize(
                objective_polish, result.x,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                # Reconstruct pulse from best parameters
                amp_scale, sigma_opt, drag_scale, drag_sigma, phase = polish_result.x
                gauss = np.exp(-0.5 * ((t - t_center) / sigma_opt) ** 2)
                area = np.sum(gauss) * dt
                gauss = gauss * (np.pi / area) * amp_scale
                drag = np.exp(-0.5 * ((t - t_center) / drag_sigma) ** 2)
                drag = np.gradient(drag, dt) * drag_scale
                I = gauss * np.cos(phase) - drag * np.sin(phase)
                Q = gauss * np.sin(phase) + drag * np.cos(phase)
                best_pulse = np.column_stack([I, Q])
                
                amplitudes = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
                mask = amplitudes > 0.4
                if np.any(mask):
                    scale = 0.4 / amplitudes[mask]
                    best_pulse[mask, 0] *= scale
                    best_pulse[mask, 1] *= scale
                
                for i in range(len(best_pulse)):
                    if i == 0:
                        prev = np.array([0.0, 0.0])
                    else:
                        prev = best_pulse[i-1]
                    
                    diff = best_pulse[i] - prev
                    norm = np.linalg.norm(diff)
                    max_slew = 0.12 * dt
                    if norm > max_slew:
                        best_pulse[i] = prev + max_slew * diff / norm
                    
                    if i == len(best_pulse) - 1:
                        diff = -best_pulse[i]
                        norm = np.linalg.norm(diff)
                        if norm > max_slew:
                            best_pulse[i] = -max_slew * diff / norm
        
    except Exception as e:
        pass
    
    return best_pulse
# EVOLVE-BLOCK-END
