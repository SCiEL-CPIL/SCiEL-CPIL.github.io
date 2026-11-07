"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    """
    Generate a robust pulse for X_pi gate using model-based optimization.
    
    Strategy:
    1. Parametric DRAG-style pulse with Gaussian envelope.
    2. Full Lindblad simulation for accurate infidelity calculation.
    3. Optimization of sigma, alpha (DRAG), and area_scale.
    4. Deadline guard to ensure completion within time limit.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    start_time = time.time()
    deadline = 320.0  # seconds, with safety margin from 330s
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5  # ns
    n_steps = 80
    total_time = n_steps * dt  # 40 ns
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns
    
    # Time points
    t = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2
    t_centered = t - total_time / 2.0
    
    # Pre-compute matrices
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n_op = adag @ a
    eye3 = np.eye(3, dtype=complex)
    
    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    # Anharmonicity part of Hamiltonian
    H_anh = 0.5 * anharmonicity * n_op @ (n_op - eye3)
    
    # Test states (as density matrices)
    psi_states = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0] / np.sqrt(2),
        [1, -1, 0] / np.sqrt(2),
        [1, 1j, 0] / np.sqrt(2),
        [1, -1j, 0] / np.sqrt(2)
    ], dtype=complex)
    
    rho_initial = np.array([np.outer(p, p.conj()) for p in psi_states])
    
    # X gate
    X = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 0]], dtype=complex)
    
    # Target density matrices
    psi_target = np.array([X @ p for p in psi_states], dtype=complex)
    rho_target = np.array([np.outer(p, p.conj()) for p in psi_target])
    
    def get_lindbladian_superoperator(H, L1, L2):
        """
        Construct the 9x9 superoperator for the Lindblad equation.
        d(rho)/dt = -i[H, rho] + L1 rho L1^dag - 0.5{L1^dag L1, rho} + ...
        Vectorization: vec(rho) = (rho_00, rho_01, ..., rho_22)
        Superoperator M: vec(d(rho)/dt) = M vec(rho)
        """
        dim = 3
        # Commutator part: -i (H \otimes I - I \otimes H^T)
        # Note: vec(AB C) = (A \otimes C^T) vec(B)
        # vec(H rho) = (I \otimes H) vec(rho)
        # vec(rho H) = (H^T \otimes I) vec(rho)
        # vec([H, rho]) = (I \otimes H - H^T \otimes I) vec(rho)
        # So -i[H, rho] -> -i (I \otimes H - H^T \otimes I)
        
        term_commutator = -1j * (np.kron(eye3, H) - np.kron(H.T, eye3))
        
        # Dissipation part
        def lindblad_term(L):
            # L rho L^dag -> (L^dag^T \otimes L) vec(rho)
            # -0.5 {L^dag L, rho} -> -0.5 (I \otimes L^dag L - (L^dag L)^T \otimes I) vec(rho)
            LdagL = L.conj().T @ L
            term = np.kron(L.conj().T.T, L) - 0.5 * (np.kron(eye3, LdagL) - np.kron(LdagL.T, eye3))
            return term
        
        term_diss = lindblad_term(L1) + lindblad_term(L2)
        
        return term_commutator + term_diss
    
    def simulate_full(pulse_data, delta, scale):
        """
        Full quantum simulation including dissipation using Lindblad superoperator.
        Returns the final density matrices for the 6 test states.
        """
        # Pre-compute superoperators for each step
        # Since H depends on I, Q, delta, scale, we compute them on the fly or pre-compute if pulse is fixed.
        # For optimization, pulse changes, so we compute inside the loop.
        
        # We evolve all 6 states simultaneously by stacking their density matrices?
        # No, the superoperator is linear, so we can apply it to a batch.
        # Let's keep rho as a list of 6 matrices, or a (6, 3, 3) array.
        
        rho_current = rho_initial.copy()
        
        for i in range(n_steps):
            I_i = pulse_data[i, 0]
            Q_i = pulse_data[i, 1]
            
            H_int = 0.5 * scale * (I_i * (a + adag) + 1j * Q_i * (a - adag))
            H_t = delta * n_op + H_anh + H_int
            
            # Compute superoperator
            # This is the expensive part. 9x9 matrix construction.
            # We can optimize this by noting that L1, L2 are constant.
            # The commutator part changes with H_t.
            
            # Optimization: Pre-compute constant parts of Lindblad terms
            # L1, L2 are constant.
            # term_diss is constant!
            # term_commutator depends on H_t.
            
            # Let's compute term_diss once outside the loop in the closure?
            # It depends on L1, L2 which are constant.
            
            L1 = a / np.sqrt(T1)
            L2 = n_op * np.sqrt(2.0 / Tphi)
            
            def lindblad_term_const(L):
                LdagL = L.conj().T @ L
                return np.kron(L.conj().T.T, L) - 0.5 * (np.kron(eye3, LdagL) - np.kron(LdagL.T, eye3))
            
            # This is still inside the loop. Let's move it out.
            
            # Actually, let's just compute the full superoperator.
            # 9x9 is small.
            
            S = get_lindbladian_superoperator(H_t, L1, L2)
            
            # Exponentiate S * dt
            S_step = expm(S * dt)
            
            # Apply to all 6 states
            # rho_vec shape: (6, 9)
            rho_vecs = np.array([rho.flatten() for rho in rho_current])
            rho_vecs_new = S_step @ rho_vecs.T
            rho_current = [rho_vecs_new[:, i].reshape(3, 3) for i in range(6)]
            
        return rho_current
    
    # Pre-compute constant Lindblad terms to speed up get_lindbladian_superoperator
    L1_const = a / np.sqrt(T1)
    L2_const = n_op * np.sqrt(2.0 / Tphi)
    
    def lindblad_term_precomp(L):
        LdagL = L.conj().T @ L
        return np.kron(L.conj().T.T, L) - 0.5 * (np.kron(eye3, LdagL) - np.kron(LdagL.T, eye3))
    
    term_diss_precomp = lindblad_term_precomp(L1_const) + lindblad_term_precomp(L2_const)
    
    def get_superoperator_fast(H_t):
        term_commutator = -1j * (np.kron(eye3, H_t) - np.kron(H_t.T, eye3))
        return term_commutator + term_diss_precomp
    
    def simulate_full_fast(pulse_data, delta, scale):
        rho_current = rho_initial.copy()
        
        for i in range(n_steps):
            I_i = pulse_data[i, 0]
            Q_i = pulse_data[i, 1]
            
            H_int = 0.5 * scale * (I_i * (a + adag) + 1j * Q_i * (a - adag))
            H_t = delta * n_op + H_anh + H_int
            
            S = get_superoperator_fast(H_t)
            S_step = expm(S * dt)
            
            rho_vecs = np.array([rho.flatten() for rho in rho_current])
            rho_vecs_new = S_step @ rho_vecs.T
            rho_current = [rho_vecs_new[:, i].reshape(3, 3) for i in range(6)]
            
        return rho_current
    
    def compute_fidelity(rho_outs):
        """
        Compute mean fidelity over the 6 test states.
        Fidelity F = <psi_target| rho_out |psi_target>
        """
        fidelities = []
        for i in range(6):
            rho_out = rho_outs[i]
            psi_t = psi_target[i]
            # F = psi_t^dag rho_out psi_t
            fid = np.vdot(psi_t, rho_out @ psi_t).real
            fidelities.append(fid)
        
        return np.mean(fidelities)
    
    def enforce_constraints(p):
        """Enforce amplitude and slew rate constraints."""
        p = p.copy()
        
        # Amplitude
        amp = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            p *= amp_bound / max_amp
        
        # Slew rate
        max_slew = 0
        for i in range(n_steps):
            if i == 0:
                dp = np.linalg.norm(p[0]) / dt
            else:
                dp = np.linalg.norm(p[i] - p[i-1]) / dt
            if dp > max_slew:
                max_slew = dp
        
        dp_last = np.linalg.norm(p[-1]) / dt
        if dp_last > max_slew:
            max_slew = dp_last
            
        if max_slew > slew_bound:
            p *= slew_bound / max_slew
            
        return p
    
    def parametric_pulse(params):
        """
        Generate pulse from parameters: [sigma, alpha, area_scale]
        """
        sigma, alpha, area_scale = params
        
        # Gaussian envelope
        I = np.exp(-t_centered**2 / (2 * sigma**2))
        area = np.sum(I) * dt
        I = I * (np.pi * area_scale / area)
        
        # DRAG: Q proportional to dI/dt
        dI = np.gradient(I, dt)
        Q = alpha * dI
        
        p = np.column_stack([I, Q])
        return enforce_constraints(p)
    
    def compute_score(pulse_data):
        """
        Compute the scoring metric: mean + max infidelity over 9 cases.
        """
        deltas = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        infidelities = []
        for d in deltas:
            for s in scales:
                rho_outs = simulate_full_fast(pulse_data, d, s)
                fid = compute_fidelity(rho_outs)
                infidelities.append(1 - fid)
        
        infidelities = np.array(infidelities)
        return np.mean(infidelities) + np.max(infidelities)
    
    # Optimization using Nelder-Mead
    def objective(params):
        p = parametric_pulse(params)
        return compute_score(p)
    
    # Initial guess: sigma=6, alpha=0.5, area_scale=1.0
    x0 = np.array([6.0, 0.5, 1.0])
    
    best_params = x0.copy()
    best_score = objective(x0)
    
    # Nelder-Mead optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7,
                              'adaptive': True})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        best_pulse = parametric_pulse(best_params)
        return best_pulse
    
    # Try a few more local searches around the best point
    best_params_copy = best_params.copy()
    
    for restart in range(20):
        if time.time() - start_time > deadline - 20:
            break
        
        # Perturb parameters
        perturbation = np.random.normal(0, 0.3, size=3)
        new_params = best_params_copy + perturbation
        
        # Ensure sigma is positive
        new_params[0] = max(new_params[0], 2.0)
        
        new_score = objective(new_params)
        
        if new_score < best_score:
            best_score = new_score
            best_params_copy = new_params
    
    best_pulse = parametric_pulse(best_params_copy)
    
    # Final validity check
    amp = np.sqrt(best_pulse[:, 0]**2 + best_pulse[:, 1]**2)
    if np.max(amp) > amp_bound:
        best_pulse *= amp_bound / np.max(amp)
    
    max_slew = 0
    for i in range(n_steps):
        if i == 0:
            dp = np.linalg.norm(best_pulse[0]) / dt
        else:
            dp = np.linalg.norm(best_pulse[i] - best_pulse[i-1]) / dt
        if dp > max_slew:
            max_slew = dp
    dp_last = np.linalg.norm(best_pulse[-1]) / dt
    if dp_last > max_slew:
        max_slew = dp_last
        
    if max_slew > slew_bound:
        best_pulse *= slew_bound / max_slew
        
    return best_pulse
# EVOLVE-BLOCK-END
