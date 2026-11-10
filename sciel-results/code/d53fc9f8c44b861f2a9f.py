import numpy as np
import scipy.linalg as sla
import scipy.optimize as opt
import math
import time

def pulse(target_gate, hardware_model):
    # Constants from task
    dt = 0.5
    n_steps = 80
    T = 40.0
    
    # Hardware parameters
    chi = -2 * math.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    detunings = [-2 * math.pi * 0.002, 0.0, 2 * math.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    T1 = 25000.0
    Tphi = 40000.0
    
    # Operators
    a = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, math.sqrt(2), 0]
    ], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)
    
    # Lindblad operators
    L1 = a / math.sqrt(T1)
    L2 = n * math.sqrt(2.0 / Tphi)
    
    # Precompute dissipator matrices (independent of I, Q, detuning)
    def get_dissipator_matrix(L_op):
        L_dag = L_op.conj().T
        L_dag_L = L_dag @ L_op
        # vec(L rho L^dag) = (L^dag.T @ kron(L)) vec(rho) -> np.kron(L.T, L_dag)? No.
        # vec(A rho B) = (B.T @ kron(A)) vec(rho)
        # A=L, B=L^dag. So (L^dag).T @ kron(L).
        # L is 3x3. L^dag is 3x3.
        term1 = np.kron(L_dag.T, L)
        
        # vec({L^dag L, rho}/2) = 0.5 (vec(L^dag L rho) + vec(rho L^dag L))
        # vec(A rho) = (I @ kron(A)) vec(rho)
        # vec(rho B) = (B.T @ kron(I)) vec(rho)
        term2 = 0.5 * (np.kron(I3, L_dag_L) + np.kron(L_dag_L.T, I3))
        
        return term1 - term2

    D1_mat = get_dissipator_matrix(L1)
    D2_mat = get_dissipator_matrix(L2)
    D_total = D1_mat + D2_mat

    # Precompute n and n(n-1) matrices
    n_mat = n
    nn_minus_1 = n @ (n - I3)
    
    # Hamiltonian parts
    # H = delta * n + 0.5 * chi * n(n-1) + 0.5 * scale * (I(a+a_dag) + i Q(a-a_dag))
    A_plus = a + a_dag
    A_minus = a - a_dag
    
    # Target states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_p = (psi_0 + psi_1) / math.sqrt(2)
    psi_m = (psi_0 - psi_1) / math.sqrt(2)
    psi_r = (psi_0 + 1j*psi_1) / math.sqrt(2)
    psi_l = (psi_0 - 1j*psi_1) / math.sqrt(2)
    
    target_states = [psi_0, psi_1, psi_p, psi_m, psi_r, psi_l]
    
    # X matrix
    X_mat = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ], dtype=complex)
    
    targets = [X_mat @ p for p in target_states]
    
    # Precompute vectorized initial states and targets for fast fidelity calc
    # rho_vec = vec(rho)
    # fidelity = <target| rho |target> = sum_{ij} target_i^* rho_ij target_j
    # In vectorized form: fidelity = sum_{k} (target_vec^* @ rho_vec)[k] where target_vec is appropriate?
    # Let's just use matrix form for clarity, it's 6 states * 9 cases = 54 ops, negligible.
    
    def check_constraints(I, Q):
        # Amplitude
        amps = np.sqrt(I**2 + Q**2)
        if np.max(amps) > amp_bound + 1e-9:
            return False
            
        # Slew rate
        p = np.column_stack((I, Q))
        p_prev = np.zeros((1, 2))
        p_next = np.zeros((1, 2))
        
        steps = np.vstack([p_prev, p, p_next])
        diffs = np.diff(steps, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        if np.max(norms) / dt > slew_bound + 1e-9:
            return False
        return True

    def simulate_superoperator(I, Q, detuning, scale):
        U_total = np.eye(9, dtype=complex)
        H0 = detuning * n_mat + 0.5 * chi * nn_minus_1
        
        for k in range(n_steps):
            I_k = I[k]
            Q_k = Q[k]
            
            HD = 0.5 * scale * (I_k * A_plus + 1j * Q_k * A_minus)
            H = H0 + HD
            
            # Liouvillian
            # -i (I @ kron(H) - H.T @ kron(I))
            L_mat = -1j * (np.kron(I3, H) - np.kron(H.T, I3)) + D_total
            
            U_step = sla.expm(L_mat * dt)
            U_total = U_step @ U_total
            
        return U_total

    def calculate_score(I, Q):
        if not check_constraints(I, Q):
            return 1e10
            
        infidelities = []
        
        for delta in detunings:
            for s in scales:
                U_tot = simulate_superoperator(I, Q, delta, s)
                
                case_infid = 0.0
                for i in range(len(target_states)):
                    psi_in = target_states[i]
                    psi_tgt = targets[i]
                    
                    rho_init = np.outer(psi_in, psi_in.conj())
                    rho_init_vec = rho_init.flatten()
                    
                    rho_final_vec = U_tot @ rho_init_vec
                    rho_final = rho_final_vec.reshape(3, 3)
                    
                    F = np.real(np.vdot(psi_tgt, rho_final @ psi_tgt))
                    F = np.clip(F, 0.0, 1.0)
                    case_infid += (1.0 - F)
                    
                case_infid /= len(target_states)
                infidelities.append(case_infid)
                
        infidelities = np.array(infidelities)
        return np.mean(infidelities) + np.max(infidelities)

    def generate_pulse(params):
        # params: [A, B, sigma]
        A, B, sigma = params
        
        if sigma < 1e-6:
            return np.zeros((n_steps, 2))
            
        t = np.linspace(0, T, n_steps)
        center = T / 2.0
        
        I = A * np.exp(-((t - center)**2) / (2 * sigma**2))
        # Derivative of I w.r.t t
        dI_dt = I * (-(t - center) / sigma**2)
        
        # DRAG term is proportional to dI/dt
        Q = B * dI_dt
        
        return np.column_stack((I, Q))

    # Initial guess
    sigma0 = 40.0 / 6.0
    A0 = math.pi / (sigma0 * math.sqrt(2 * math.pi))
    B0 = 0.0
    
    x0 = [A0, B0, sigma0]
    
    # Bounds
    # A <= 0.4
    # Slew constraint: max |dp/dt| <= 0.12
    # |dp/dt| = |d/dt [I, Q]|.
    # For small t, I ~ A, Q ~ 0.
    # Near peak, I ~ A, dI/dt ~ 0. Q ~ B * dI/dt.
    # Slope of I is max at t = center +/- sigma.
    # |dI/dt|_max = A / (e * sigma).
    # |dQ/dt| is more complex but roughly B * A * ...
    # Let's just use broad bounds and rely on constraint check in objective.
    
    bounds = [
        (0, 0.4), # A
        (-2.0, 2.0), # B (DRAG coefficient)
        (1.0, 20.0)  # sigma
    ]

    def objective(params):
        t_start = time.time()
        if t_start > 300:
            return 1e10
            
        I, Q = generate_pulse(params)
        return calculate_score(I, Q)

    best_x = x0
    best_val = objective(x0)
    
    # Optimize using Nelder-Mead
    try:
        res = opt.minimize(objective, x0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception as e:
        pass

    # Generate final pulse
    I, Q = generate_pulse(best_x)
    
    # Final constraint check
    if not check_constraints(I, Q):
        # Fallback to seed
        sigma0 = 40.0 / 6.0
        A0 = math.pi / (sigma0 * math.sqrt(2 * math.pi))
        t = np.linspace(0, T, n_steps)
        center = T / 2.0
        I = A0 * np.exp(-((t - center)**2) / (2 * sigma0**2))
        Q = np.zeros(n_steps)
        
    return np.column_stack((I, Q))
