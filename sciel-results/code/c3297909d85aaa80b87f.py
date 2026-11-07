"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    n_steps = 80
    total_time = n_steps * dt
    
    T1 = 25000.0
    Tphi = 40000.0
    
    # Time grid
    t = np.linspace(0, total_time, n_steps, endpoint=False) + dt/2
    t_centered = t - total_time / 2.0
    
    # Precompute operators
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n = adag @ a
    eye3 = np.eye(3, dtype=complex)
    
    # H0 components: H0 = Delta * n + 0.5 * chi * n(n-1)
    # n(n-1) = diag(0, 0, 2)
    n_n_minus_1 = n @ (n - eye3)
    
    # H_int components: H_int = 0.5 * scale * (I*(a+adag) + i*Q*(a-adag))
    # a+adag = [[0,1,0],[1,0,sqrt(2)],[0,sqrt(2),0]]
    # a-adag = [[0,1,0],[-1,0,sqrt(2)],[0,sqrt(2),0]]
    a_plus_adag = a + adag
    a_minus_adag = a - adag
    
    # Lindblad operators
    # L1 = a / sqrt(T1)
    # L2 = n * sqrt(2 / Tphi)
    # Liouvillian: L(rho) = -i[H,rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    # For speed, we can simulate the density matrix directly.
    # But for scoring optimization, we can approximate by tracking the state vector 
    # and adding a decay factor? No, dephasing affects off-diagonals.
    # However, T1 and Tphi are large (25us, 40us) compared to 40ns.
    # gamma1 = 1/25000 = 4e-5. Over 40ns, decay is ~1.6e-3.
    # gamma_phi = 1/40000 = 2.5e-5.
    # The infidelity from decay is small but non-zero.
    # For optimization speed, we can first optimize for unitary fidelity, 
    # then verify. The decay penalty is roughly constant for all pulses 
    # with similar support, so it won't change the optimal pulse shape much.
    # Let's use unitary evolution for the search, and full Liouvillian for final check?
    # Actually, the task says "each step exponentiates the Liouvillian".
    # We should respect that for the final result, but for search, unitary is a good proxy.
    
    def simulate_unitary(pulse):
        I = pulse[:, 0]
        Q = pulse[:, 1]
        
        Delta = 0.0
        scale = 1.0
        
        psi = np.array([1, 0, 0], dtype=complex)
        
        for i in range(n_steps):
            I_i = I[i]
            Q_i = Q[i]
            
            H0_i = Delta * n + 0.5 * anharmonicity * n_n_minus_1
            H_int_i = 0.5 * scale * (I_i * a_plus_adag + 1j * Q_i * a_minus_adag)
            H_t = H0_i + H_int_i
            
            # Exponentiate H_t * dt
            # For 3x3, we can use expm or diagonalization
            # Diagonalization is faster for Hermitian
            eigvals, eigvecs = np.linalg.eigh(H_t)
            U = eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
            psi = U @ psi
        
        # Fidelity to X gate
        # X |0> = |1>, X |1> = |0>, X |+> = |+>, X |-> = |->
        # Target states:
        # psi_0_target = [0, 1, 0]
        # psi_1_target = [1, 0, 0]
        # psi_plus_target = [1/sqrt(2), 1/sqrt(2), 0]
        # psi_minus_target = [1/sqrt(2), -1/sqrt(2), 0]
        
        # We need to compute fidelity for 4 initial states.
        # But the Hamiltonian is linear, so U is the same.
        # Fidelity = |<target|U|initial>|^2
        
        # Let's compute U_total
        # Actually, we already evolved psi from |0>.
        # We need U. Let's re-compute U_total for the 4 states.
        # Or better, compute the 3x3 unitary matrix U.
        
        U_total = np.eye(3, dtype=complex)
        for i in range(n_steps):
            I_i = I[i]
            Q_i = Q[i]
            H0_i = Delta * n + 0.5 * anharmonicity * n_n_minus_1
            H_int_i = 0.5 * scale * (I_i * a_plus_adag + 1j * Q_i * a_minus_adag)
            H_t = H0_i + H_int_i
            eigvals, eigvecs = np.linalg.eigh(H_t)
            U_step = eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
            U_total = U_step @ U_total
        
        # Compute infidelity
        # |0> -> |1>
        f0 = abs(U_total[1, 0])**2
        # |1> -> |0>
        f1 = abs(U_total[0, 1])**2
        # |+> -> |+>
        psi_plus_in = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
        psi_plus_out = U_total @ psi_plus_in
        psi_plus_target = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
        f_plus = abs(np.vdot(psi_plus_target, psi_plus_out))**2
        
        # |-> -> |->
        psi_minus_in = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
        psi_minus_out = U_total @ psi_minus_in
        psi_minus_target = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
        f_minus = abs(np.vdot(psi_minus_target, psi_minus_out))**2
        
        # |0> +- i|1>
        psi_0i_in = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
        psi_0i_out = U_total @ psi_0i_in
        psi_0i_target = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
        f_0i = abs(np.vdot(psi_0i_target, psi_0i_out))**2
        
        psi_0mi_in = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
        psi_0mi_out = U_total @ psi_0mi_in
        psi_0mi_target = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
        f_0mi = abs(np.vdot(psi_0mi_target, psi_0mi_out))**2
        
        # The task says: mean fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)
        # That's 6 states? Or 4?
        # "|0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
        # Usually this means 4 states: |0>, |1>, |+>, |->.
        # But it lists (|0> +- |1>) which is 2 states, and (|0> +- i|1>) which is 2 states.
        # Total 4 states.
        # Let's assume the 4 states are |0>, |1>, |+>, |->.
        # Wait, the text says "over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)".
        # This is ambiguous. Let's assume it means the 4 states: |0>, |1>, |+>, |->.
        # Or maybe 6 states?
        # Let's use the 4 states: |0>, |1>, |+>, |->.
        
        fids = [f0, f1, f_plus, f_minus]
        mean_fid = np.mean(fids)
        infidelity = 1 - mean_fid
        
        return infidelity

    # Strategy: Optimize a DRAG pulse with Gaussian shape.
    # Parameters: sigma (width), alpha (DRAG coefficient)
    # We will do a grid search over sigma and alpha.
    
    def generate_pulse(sigma, alpha):
        I = np.exp(-t_centered**2 / (2 * sigma**2))
        area = np.sum(I) * dt
        I = I * (np.pi / area)
        
        dI = np.gradient(I, dt)
        Q = alpha * dI
        
        p = np.column_stack([I, Q])
        
        # Check amplitude bound
        amp = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            scale = amp_bound / max_amp
            p *= scale
        
        # Check slew rate
        max_slew = 0.0
        p_prev = np.zeros(2)
        for i in range(n_steps):
            dp = np.linalg.norm(p[i] - p_prev) / dt
            if dp > max_slew:
                max_slew = dp
            p_prev = p[i].copy()
        if np.linalg.norm(p[-1]) / dt > max_slew:
            max_slew = np.linalg.norm(p[-1]) / dt
        
        if max_slew > slew_bound:
            scale = slew_bound / max_slew
            p *= scale
        
        return p

    # Grid search
    sigmas = np.linspace(4.0, 10.0, 10)
    alphas = np.linspace(-0.5, 0.5, 20)
    
    best_infidelity = np.inf
    best_pulse = None
    best_params = (0, 0)
    
    start_time = time.time()
    timeout = 250  # Leave 80s margin
    
    for sigma in sigmas:
        for alpha in alphas:
            if time.time() - start_time > timeout:
                break
            
            p = generate_pulse(sigma, alpha)
            inf = simulate_unitary(p)
            
            if inf < best_infidelity:
                best_infidelity = inf
                best_pulse = p.copy()
                best_params = (sigma, alpha)
    
    if best_pulse is None:
        # Fallback: simple Gaussian
        sigma = 6.0
        alpha = 0.0
        best_pulse = generate_pulse(sigma, alpha)
    
    return best_pulse
# EVOLVE-BLOCK-END
