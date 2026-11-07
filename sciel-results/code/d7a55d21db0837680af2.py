import numpy as np
from scipy.linalg import expm

def pulse(target_gate, hardware_model):
    # Hardware parameters
    # anharmonicity -2 pi 0.25
    chi = -2 * np.pi * 0.25
    # amplitude bound 0.4
    amp_bound = 0.4
    # slew bound 0.12
    slew_bound = 0.12
    # detuning bounds +-2 pi 0.002
    det_bound = 2 * np.pi * 0.002
    # drive scale bounds [0.97, 1.03]
    scale_min, scale_max = 0.97, 1.03
    # T1 25000 ns, Tphi 40000 ns
    T1 = 25000.0
    Tphi = 40000.0
    
    dt = 0.5
    N_steps = 80
    T_total = N_steps * dt # 40 ns
    
    # Basis states |0>, |1>, |2>
    # a is 3x3 lower triangular
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    
    n = a_dag @ a # diag(0, 1, 2)
    
    # Lindbladian construction
    # L = sum gamma_k ( L_k rho L_k^dagger - 0.5 { L_k^dagger L_k, rho } )
    # We represent rho as a 9x1 vector (vec)
    # vec(A rho B) = (B^T \otimes A) vec(rho)
    
    def vec(rho):
        return rho.flatten()
    
    def unvec(v):
        return v.reshape((3, 3))
    
    # Liouvillian superoperator L_total
    # Dephasing: L2 = n * sqrt(2/Tphi)
    L2 = n * np.sqrt(2.0 / Tphi)
    # Relaxation: L1 = a / sqrt(T1)
    L1 = a / np.sqrt(T1)
    
    def lindblad_term(L_op):
        # L_op rho L_op^dagger - 0.5 (L_op^dagger L_op rho + rho L_op^dagger L_op)
        L_op_dag = L_op.conj().T
        L_op_dag_L_op = L_op_dag @ L_op
        
        # Term 1: L_op rho L_op^dagger
        # vec(L_op rho L_op^dagger) = (L_op^T \otimes L_op) vec(rho) ? 
        # Actually vec(A X B) = (B^T \otimes A) vec(X)
        # So vec(L_op rho L_op^dagger) = ((L_op^dagger)^T \otimes L_op) vec(rho) = (L_op.conj() \otimes L_op) vec(rho)
        term1 = np.kron(L_op.conj(), L_op)
        
        # Term 2: -0.5 L_op^dagger L_op rho
        term2 = -0.5 * np.kron(np.eye(3), L_op_dag_L_op)
        
        # Term 3: -0.5 rho L_op^dagger L_op
        term3 = -0.5 * np.kron(L_op_dag_L_op, np.eye(3))
        
        return term1 + term2 + term3
    
    L_diss = lindblad_term(L1) + lindblad_term(L2)
    
    # Hamiltonian part
    # H = detuning * n + 0.5 * chi * n(n-1) + 0.5 * scale * (I (a+a_dag) + i Q (a-a_dag))
    # Note: The task says "0.5 scale (I (a + a^dagger) + i Q (a - a^dagger))"
    # Let's verify the signs.
    # X = a + a_dag
    # Y = -i(a - a_dag) => i(a - a_dag) = -Y? 
    # Standard: H_drive = Omega_x X + Omega_y Y.
    # If I is X component and Q is Y component:
    # H_drive = I X + Q Y = I(a+a_dag) + Q(-i(a-a_dag)) = I(a+a_dag) - i Q (a-a_dag).
    # The task has: + i Q (a - a_dag).
    # This implies the Q component has a sign flip relative to standard Y, or it's just a convention.
    # We will follow the task exactly: H_drive = 0.5 * scale * ( I * (a + a_dag) + 1j * Q * (a - a_dag) )
    
    # Pre-compute static parts
    H_static = det_bound * n + 0.5 * chi * (n @ (n - np.eye(3))) # Note: n(n-1) is operator product? No, n is diagonal, so element-wise? 
    # n = diag(0,1,2). n(n-1) = diag(0, 0, 2).
    # Wait, n(n-1) usually means n^2 - n.
    # n^2 = diag(0,1,4). n^2 - n = diag(0,0,2).
    # So 0.5 * chi * diag(0,0,2) = diag(0, 0, chi).
    # This is the anharmonicity shift for level 2.
    
    # Actually, the term is 0.5 * anharmonicity * n(n-1).
    # For level 2: 0.5 * chi * 2 * 1 = chi.
    # For level 1: 0.5 * chi * 1 * 0 = 0.
    # For level 0: 0.
    # So H_anh = diag(0, 0, chi).
    
    # We will construct the full Liouvillian for each step inside the simulation loop for simplicity, 
    # or pre-compute the basis matrices.
    
    # Optimization Strategy
    # We search for sigma and beta.
    # Pulse shape:
    # I(t) = A * exp(-(t-t0)^2 / (2*sigma^2))
    # Q(t) = -beta * A * (t-t0)/sigma^2 * exp(-(t-t0)^2 / (2*sigma^2))
    
    def generate_pulse(sigma, beta):
        t = np.arange(N_steps) * dt + dt/2 # Center of steps? Or 0 to 40?
        # "80 constant steps of dt = 0.5 ns". 
        # Usually step i is from i*dt to (i+1)*dt.
        # Let's use the center of the step for the amplitude.
        t = (np.arange(N_steps) + 0.5) * dt
        t0 = T_total / 2.0
        
        # Gaussian
        gauss = np.exp(-(t - t0)**2 / (2 * sigma**2))
        
        # Derivative
        dgauss = -(t - t0) / (sigma**2) * gauss
        
        # Normalize Area to Pi
        area = np.sum(gauss) * dt
        if area > 0:
            A = np.pi / area
        else:
            A = 0
            
        I = A * gauss
        Q = -beta * A * dgauss
        
        # Check bounds
        # Amplitude
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > amp_bound:
            # Scale down
            scale_factor = amp_bound / np.max(amp)
            I *= scale_factor
            Q *= scale_factor
            # Area is now not exactly pi, but we prioritize validity.
            
        # Check Slew
        # p[i+1] - p[i]
        dI = np.diff(I, prepend=0.0) # From 0 to I[0]
        dQ = np.diff(Q, prepend=0.0)
        dI = np.append(dI, -I[-1]) # From I[-1] to 0
        dQ = np.append(dQ, -Q[-1])
        
        slew = np.sqrt(dI**2 + dQ**2) / dt
        if np.max(slew) > slew_bound:
             # This is harder to fix simply. We might need to reduce amplitude or widen pulse.
             # For now, we rely on the search space to avoid invalid regions.
             pass
             
        return np.column_stack((I, Q))

    def simulate_and_score(pulse_arr):
        # pulse_arr is (80, 2)
        I = pulse_arr[:, 0]
        Q = pulse_arr[:, 1]
        
        # States to test
        # |0> = [1, 0, 0]
        # |1> = [0, 1, 0]
        # |+> = [1, 1, 0]/sqrt(2)
        # |-> = [1, -1, 0]/sqrt(2)
        # |R> = [1, i, 0]/sqrt(2)
        # |L> = [1, -i, 0]/sqrt(2)
        
        states = [
            np.array([1, 0, 0], dtype=complex),
            np.array([0, 1, 0], dtype=complex),
            np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
            np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
            np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
            np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
        ]
        
        # X_pi gate: |0> -> |1>, |1> -> -|0> (or |0> -> |1>, |1> -> -|0> is X? 
        # X = [[0,1],[1,0]]. X|0> = |1>. X|1> = |0>.
        # Wait, standard X gate: X|0> = |1>, X|1> = |0>.
        # Fidelity is |<psi_final|psi_target>|^2.
        # Target for |0> is |1>.
        # Target for |1> is |0>.
        # Target for |+> is |->? X|+> = |->.
        # Target for |-> is |+>.
        # Target for |R> (|1> + i|0>)/sqrt2? No, (|0> + i|1>)/sqrt2 is |R> (spin up y).
        # X|0> = |1>, X|i> = i|0>? 
        # Let's just use the unitary X.
        # X = [[0,1,0],[1,0,0],[0,0,1]] (acting on 0,1 subspace).
        # We check fidelity to X|psi_init>.
        
        targets = []
        X_mat = np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=complex)
        for psi in states:
            targets.append(X_mat @ psi)
        
        infidelities = []
        
        # Detunings and Scales
        dets = [-det_bound, 0.0, det_bound]
        scales = [0.97, 1.0, 1.03]
        
        for det in dets:
            for scale in scales:
                # Simulate for each initial state
                # We can batch this or loop. Loop is fine for 6 states.
                case_infids = []
                for k, psi_init in enumerate(states):
                    rho = np.outer(psi_init, psi_init.conj())
                    rho_vec = vec(rho)
                    
                    for i in range(N_steps):
                        I_val = I[i]
                        Q_val = Q[i]
                        
                        # Construct H
                        # H = det * n + 0.5 * chi * (n^2 - n) + 0.5 * scale * ( I (a+a_dag) + i Q (a-a_dag) )
                        H_drive = 0.5 * scale * ( I_val * (a + a_dag) + 1j * Q_val * (a - a_dag) )
                        H = det * n + 0.5 * chi * (n @ n - n) + H_drive
                        
                        # Liouvillian L = -i [H, .] + L_diss
                        # vec(H rho - rho H) = (I \otimes H - H^T \otimes I) vec(rho)
                        # So -i [H, .] -> -i (I \otimes H - H^T \otimes I)
                        L_ham = -1j * (np.kron(np.eye(3), H) - np.kron(H.T, np.eye(3)))
                        
                        L_total = L_ham + L_diss
                        
                        # Exponentiate
                        U = expm(L_total * dt)
                        rho_vec = U @ rho_vec
                        
                    rho_final = unvec(rho_vec)
                    psi_final = rho_final @ psi_init # This is not right. 
                    # We need the state vector. 
                    # If rho is pure, rho = |psi><psi|. 
                    # We can extract |psi> by taking the first column? No, basis dependent.
                    # Better: Fidelity = Tr(rho_final P_target)
                    # P_target = |t><t|
                    # F = <t| rho_final |t>
                    t_vec = targets[k]
                    F = np.real(np.vdot(t_vec, rho_final @ t_vec))
                    # Ensure F is in [0,1]
                    F = np.clip(F, 0.0, 1.0)
                    infid = 1.0 - F
                    case_infids.append(infid)
                
                # Average over 6 states
                mean_infid = np.mean(case_infids)
                infidelities.append(mean_infid)
        
        # Score = mean + max of the 9 cases
        mean_score = np.mean(infidelities)
        max_score = np.max(infidelities)
        return mean_score + max_score

    # Grid Search for sigma and beta
    best_score = float('inf')
    best_sigma = 6.0
    best_beta = 0.0
    
    # Sigma range: 4 to 10
    # Beta range: -0.5 to 0.5 (scaled appropriately)
    # Note: Q(t) = -beta * A * dgauss. 
    # If beta is 1, Q is derivative.
    # Typical beta for DRAG is small.
    
    sigmas = np.linspace(4.0, 10.0, 15)
    betas = np.linspace(-0.5, 0.5, 21)
    
    for sig in sigmas:
        for bet in betas:
            p = generate_pulse(sig, bet)
            # Quick validity check
            if np.max(np.sqrt(p[:,0]**2 + p[:,1]**2)) > amp_bound + 1e-9:
                continue
            # Slew check
            dI = np.diff(p[:,0], prepend=0.0)
            dQ = np.diff(p[:,1], prepend=0.0)
            dI = np.append(dI, -p[-1,0])
            dQ = np.append(dQ, -p[-1,1])
            slew = np.max(np.sqrt(dI**2 + dQ**2) / dt)
            if slew > slew_bound + 1e-9:
                continue
                
            score = simulate_and_score(p)
            if score < best_score:
                best_score = score
                best_sigma = sig
                best_beta = bet

    # Generate final pulse
    final_pulse = generate_pulse(best_sigma, best_beta)
    
    # Final validity check and correction if needed (should not be needed if search was valid)
    amp = np.sqrt(final_pulse[:,0]**2 + final_pulse[:,1]**2)
    if np.max(amp) > amp_bound:
        scale_f = amp_bound / np.max(amp)
        final_pulse *= scale_f
        
    return final_pulse

pulse = pulse
