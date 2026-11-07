import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    # Hardware parameters
    chi = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    det_bound = 2 * np.pi * 0.002
    scale_min, scale_max = 0.97, 1.03
    T1 = 25000.0
    Tphi = 40000.0
    
    dt = 0.5
    N_steps = 80
    T_total = N_steps * dt
    
    # Basis states |0>, |1>, |2>
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    
    # Lindbladian construction
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    def lindblad_term(L_op):
        L_op_dag = L_op.conj().T
        L_op_dag_L_op = L_op_dag @ L_op
        term1 = np.kron(L_op.conj(), L_op)
        term2 = -0.5 * np.kron(np.eye(3), L_op_dag_L_op)
        term3 = -0.5 * np.kron(L_op_dag_L_op, np.eye(3))
        return term1 + term2 + term3
    
    L_diss = lindblad_term(L1) + lindblad_term(L2)
    
    # Pre-compute Hamiltonian parts for Liouvillian construction
    # H = det * n + 0.5 * chi * (n^2 - n) + 0.5 * scale * ( I (a+a_dag) + i Q (a-a_dag) )
    # Let H = H_det + H_anh + H_drive
    # H_drive = 0.5 * scale * I * (a+a_dag) + 0.5 * scale * i * Q * (a-a_dag)
    
    # Superoperator for commutator: -i [H, .]
    # vec(-i (H rho - rho H)) = -i (I \otimes H - H^T \otimes I) vec(rho)
    
    def ham_liouvillian(H):
        return -1j * (np.kron(np.eye(3), H) - np.kron(H.T, np.eye(3)))
    
    # Pre-compute static parts for each detuning
    H_anh = 0.5 * chi * (n @ n - n)
    
    # States and Targets
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    ]
    
    X_mat = np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=complex)
    targets = [X_mat @ psi for psi in states]
    # Projectors for fidelity
    P_targets = [np.outer(t, t.conj()) for t in targets]
    # vec(P_target) for efficient trace calculation?
    # F = Tr(rho_final P_target) = sum(rho_final * P_target^T) = dot(rho_final_vec, P_target_vec.conj())?
    # vec(A) dot vec(B^T) = Tr(A^T B).
    # Tr(rho P) = sum(rho_ij P_ji).
    # Let's just use np.trace(rho_final @ P_target).

    def generate_pulse(sigma, beta):
        t = (np.arange(N_steps) + 0.5) * dt
        t0 = T_total / 2.0
        
        gauss = np.exp(-(t - t0)**2 / (2 * sigma**2))
        dgauss = -(t - t0) / (sigma**2) * gauss
        
        area = np.sum(gauss) * dt
        if area > 0:
            A = np.pi / area
        else:
            A = 0
            
        I = A * gauss
        Q = -beta * A * dgauss
        
        # Validity Check and Correction
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > amp_bound:
            scale_factor = amp_bound / np.max(amp)
            I *= scale_factor
            Q *= scale_factor
            
        dI = np.diff(I, prepend=0.0)
        dQ = np.diff(Q, prepend=0.0)
        dI = np.append(dI, -I[-1])
        dQ = np.append(dQ, -Q[-1])
        slew = np.sqrt(dI**2 + dQ**2) / dt
        if np.max(slew) > slew_bound:
             # Return a very large penalty or invalid pulse?
             # For optimization, we can return a penalty.
             # But generate_pulse should return a valid pulse if possible.
             # If slew is violated, we can't easily fix it by scaling I/Q down uniformly?
             # Scaling down reduces slew.
             slew_scale = slew_bound / np.max(slew)
             I *= slew_scale
             Q *= slew_scale
             
        return np.column_stack((I, Q))

    def simulate_and_score(pulse_arr):
        I = pulse_arr[:, 0]
        Q = pulse_arr[:, 1]
        
        # Check validity again to be safe
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > amp_bound + 1e-9:
            return 10.0 # Penalty
            
        dI = np.diff(I, prepend=0.0)
        dQ = np.diff(Q, prepend=0.0)
        dI = np.append(dI, -I[-1])
        dQ = np.append(dQ, -Q[-1])
        slew = np.max(np.sqrt(dI**2 + dQ**2) / dt)
        if slew > slew_bound + 1e-9:
            return 10.0 # Penalty

        infidelities = []
        
        dets = [-det_bound, 0.0, det_bound]
        scales = [0.97, 1.0, 1.03]
        
        for det in dets:
            for scale in scales:
                H_static = det * n + H_anh
                # Coefficients for I and Q in H
                # H_drive = 0.5 * scale * I * (a+a_dag) + 0.5 * scale * i * Q * (a-a_dag)
                C_I = 0.5 * scale * (a + a_dag)
                C_Q = 0.5 * scale * 1j * (a - a_dag)
                
                # Pre-compute Liouvillian coefficients
                L_static = L_diss + ham_liouvillian(H_static)
                L_I_coeff = ham_liouvillian(C_I)
                L_Q_coeff = ham_liouvillian(C_Q)
                
                # Initialize density matrices for all 6 states
                rho_vecs = np.array([np.outer(psi, psi.conj()).flatten() for psi in states])
                
                for i in range(N_steps):
                    I_val = I[i]
                    Q_val = Q[i]
                    
                    L_total = L_static + I_val * L_I_coeff + Q_val * L_Q_coeff
                    U_step = expm(L_total * dt)
                    
                    # Apply to all states
                    # rho_vecs is (6, 9)
                    # U_step is (9, 9)
                    # New rho_vecs[j] = U_step @ rho_vecs[j]
                    rho_vecs = (U_step @ rho_vecs.T).T
                    
                # Calculate infidelities
                case_infids = []
                for k in range(6):
                    rho_final = rho_vecs[k].reshape((3, 3))
                    P_t = P_targets[k]
                    F = np.real(np.trace(rho_final @ P_t))
                    F = np.clip(F, 0.0, 1.0)
                    case_infids.append(1.0 - F)
                
                mean_infid = np.mean(case_infids)
                infidelities.append(mean_infid)
        
        mean_score = np.mean(infidelities)
        max_score = np.max(infidelities)
        return mean_score + max_score

    def objective(params):
        sigma, beta = params
        # Ensure sigma is positive
        if sigma <= 0:
            return 10.0
        p = generate_pulse(sigma, beta)
        score = simulate_and_score(p)
        return score

    # Optimization
    # Start near seed values
    # Seed: Gaussian sigma 40/6 = 6.67? "sigma 40/6 ns" in task description.
    # Task says: "The seed is a DRAG-style Gaussian (sigma 40/6 ns, offset to 0 at the edges, area pi)."
    # 40/6 = 6.66...
    # DRAG beta is usually small.
    
    x0 = [6.66, 0.2]
    bounds = [(2.0, 12.0), (-1.0, 1.0)]
    
    # Use Nelder-Mead for speed and robustness
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 100, 'maxfev': 200})
    
    best_params = res.x
    best_sigma, best_beta = best_params
    
    # Generate final pulse
    final_pulse = generate_pulse(best_sigma, best_beta)
    
    # Final validity check
    amp = np.sqrt(final_pulse[:,0]**2 + final_pulse[:,1]**2)
    if np.max(amp) > amp_bound:
        scale_f = amp_bound / np.max(amp)
        final_pulse *= scale_f
        
    dI = np.diff(final_pulse[:,0], prepend=0.0)
    dQ = np.diff(final_pulse[:,1], prepend=0.0)
    dI = np.append(dI, -final_pulse[-1,0])
    dQ = np.append(dQ, -final_pulse[-1,1])
    slew = np.max(np.sqrt(dI**2 + dQ**2) / dt)
    if slew > slew_bound:
        scale_f = slew_bound / slew
        final_pulse *= scale_f

    return final_pulse

pulse = pulse
