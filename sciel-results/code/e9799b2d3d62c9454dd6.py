import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    # Hardware parameters
    chi = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    det_bound = 2 * np.pi * 0.002
    T1 = 25000.0
    Tphi = 40000.0
    
    dt = 0.5
    N_steps = 80
    T_total = N_steps * dt
    
    # Basis operators
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    
    # Pre-compute Lindblad dissipator
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
    
    # Pre-compute Hamiltonian components
    H_anh = 0.5 * chi * (n @ n - n)
    X_op = a + a_dag
    Y_op = 1j * (a - a_dag)
    I3 = np.eye(3)

    def simulate_and_score(I, Q):
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
        
        dets = [-det_bound, 0.0, det_bound]
        scales = [0.97, 1.0, 1.03]
        
        infidelities = []
        
        for det in dets:
            for scale in scales:
                case_infids = []
                for k, psi_init in enumerate(states):
                    rho = np.outer(psi_init, psi_init.conj())
                    rho_vec = rho.flatten()
                    
                    for i in range(N_steps):
                        I_val = I[i]
                        Q_val = Q[i]
                        
                        H_drive = 0.5 * scale * (I_val * X_op + Q_val * Y_op)
                        H = det * n + H_anh + H_drive
                        
                        L_ham = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
                        L_total = L_ham + L_diss
                        
                        U = expm(L_total * dt)
                        rho_vec = U @ rho_vec
                    
                    rho_final = rho_vec.reshape((3, 3))
                    t_vec = targets[k]
                    F = np.real(np.vdot(t_vec, rho_final @ t_vec))
                    F = np.clip(F, 0.0, 1.0)
                    case_infids.append(1.0 - F)
                
                infidelities.append(np.mean(case_infids))
        
        return np.mean(infidelities) + np.max(infidelities)

    def objective(params):
        I = params[:N_steps]
        Q = params[N_steps:]
        
        # Penalty for amplitude violation
        amp = np.sqrt(I**2 + Q**2)
        amp_viol = np.sum(np.maximum(0, amp - amp_bound)**2)
        
        # Penalty for slew rate violation
        dI = np.diff(I, prepend=0.0)
        dQ = np.diff(Q, prepend=0.0)
        dI = np.append(dI, -I[-1])
        dQ = np.append(dQ, -Q[-1])
        slew = np.sqrt(dI**2 + dQ**2) / dt
        slew_viol = np.sum(np.maximum(0, slew - slew_bound)**2)
        
        penalty = 100.0 * (amp_viol + slew_viol)
        
        if penalty > 1e-3:
            return penalty
        
        score = simulate_and_score(I, Q)
        return score + penalty

    # Generate initial seed pulse (Gaussian)
    t = (np.arange(N_steps) + 0.5) * dt
    t0 = T_total / 2.0
    sigma = 6.0
    gauss = np.exp(-(t - t0)**2 / (2 * sigma**2))
    area = np.sum(gauss) * dt
    A = np.pi / area
    I_seed = A * gauss
    Q_seed = np.zeros(N_steps)
    
    # Scale to satisfy bounds
    amp = np.sqrt(I_seed**2 + Q_seed**2)
    if np.max(amp) > amp_bound:
        scale = amp_bound / np.max(amp)
        I_seed *= scale
        Q_seed *= scale
    
    dI = np.diff(I_seed, prepend=0.0)
    dQ = np.diff(Q_seed, prepend=0.0)
    dI = np.append(dI, -I_seed[-1])
    dQ = np.append(dQ, -Q_seed[-1])
    slew = np.max(np.sqrt(dI**2 + dQ**2) / dt)
    if slew > slew_bound:
        scale = slew_bound / slew
        I_seed *= scale
        Q_seed *= scale
        
    x0 = np.concatenate([I_seed, Q_seed])
    
    # Optimize
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
        if res.success or res.fun < 1.0:
            I_opt = res.x[:N_steps]
            Q_opt = res.x[N_steps:]
        else:
            I_opt = I_seed
            Q_opt = Q_seed
    except Exception:
        I_opt = I_seed
        Q_opt = Q_seed

    # Final safety check and clipping
    final_I = I_opt
    final_Q = Q_opt
    
    amp = np.sqrt(final_I**2 + final_Q**2)
    if np.max(amp) > amp_bound:
        scale = amp_bound / np.max(amp)
        final_I *= scale
        final_Q *= scale
    
    dI = np.diff(final_I, prepend=0.0)
    dQ = np.diff(final_Q, prepend=0.0)
    dI = np.append(dI, -final_I[-1])
    dQ = np.append(dQ, -final_Q[-1])
    slew = np.max(np.sqrt(dI**2 + dQ**2) / dt)
    if slew > slew_bound:
        scale = slew_bound / slew
        final_I *= scale
        final_Q *= scale

    return np.column_stack((final_I, final_Q))

pulse = pulse
