import numpy as np
from scipy.linalg import expm
import time
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 325.0  # safety margin before 330s kill
    
    # Hardware parameters
    alpha = -2 * np.pi * 0.25  # anharmonicity
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    T1 = 25000.0
    Tphi = 40000.0
    
    # Detuning and scale cases
    deltas = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # The 6 states as specified
    sqrt2 = np.sqrt(2)
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / sqrt2,
        np.array([1, -1, 0], dtype=complex) / sqrt2,
        np.array([1, 1j, 0], dtype=complex) / sqrt2,
        np.array([1, -1j, 0], dtype=complex) / sqrt2
    ]
    
    # Target states for X gate
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / sqrt2,
        np.array([-1, 1, 0], dtype=complex) / sqrt2,
        np.array([1j, 1, 0], dtype=complex) / sqrt2,
        np.array([-1j, 1, 0], dtype=complex) / sqrt2
    ]
    
    # Operators
    a_op = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    
    # Precompute constant parts
    H_anharm = 0.5 * alpha * (n_op @ (n_op - np.eye(3)))
    
    # Dissipator superoperator (constant)
    dim = 3
    d2 = dim * dim
    L_diss_super = np.zeros((d2, d2), dtype=complex)
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    for L_op in [L1, L2]:
        L_dag = L_op.conj().T
        L1_term = np.kron(L_op, L_op.conj())
        L2_term = 0.5 * (np.kron(L_dag @ L_op, np.eye(dim)) + np.kron(np.eye(dim), (L_dag @ L_op).T))
        L_diss_super += L1_term - L2_term
    
    # Precompute identity for kron
    I3 = np.eye(3)
    
    # Precompute H terms for all 9 cases
    # H = delta * n_op + H_anharm + drive
    # drive = 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
    # H_term = -1j * (kron(H, I) - kron(I, H.T))
    # Since H is linear in I and Q (for fixed delta, scale), we can precompute:
    # H = (delta * n_op + H_anharm) + 0.5 * scale * I * (a_op + a_dag) + 0.5j * scale * Q * (a_op - a_dag)
    # Let H0 = delta * n_op + H_anharm
    # Let HI = 0.5 * scale * (a_op + a_dag)
    # Let HQ = 0.5j * scale * (a_op - a_dag)
    # Then H = H0 + I*HI + Q*HQ
    # H_term = -1j * (kron(H0, I) - kron(I, H0.T)) -1j * (kron(I*HI, I) - kron(I, (I*HI).T)) -1j * (kron(Q*HQ, I) - kron(I, (Q*HQ).T))
    # = H_term_0 + I * H_term_I + Q * H_term_Q
    
    # Precompute for each (delta, scale) pair
    liouvillian_parts = {}
    for delta in deltas:
        for scale in scales:
            H0 = delta * n_op + H_anharm
            HI = 0.5 * scale * (a_op + a_dag)
            HQ = 0.5j * scale * (a_op - a_dag)
            
            H_term_0 = -1j * (np.kron(H0, I3) - np.kron(I3, H0.T))
            H_term_I = -1j * (np.kron(HI, I3) - np.kron(I3, HI.T))
            H_term_Q = -1j * (np.kron(HQ, I3) - np.kron(I3, HQ.T))
            
            L_base_0 = H_term_0 + L_diss_super
            liouvillian_parts[(delta, scale)] = (L_base_0, H_term_I, H_term_Q)
    
    def simulate_pulse(I_Q, delta, scale):
        """
        Simulates the pulse and returns mean infidelity over 6 states.
        Vectorized over states for speed.
        """
        L_base_0, H_term_I, H_term_Q = liouvillian_parts[(delta, scale)]
        
        # Precompute all propagators
        L_steps = []
        for i in range(N):
            I = I_Q[i, 0]
            Q = I_Q[i, 1]
            L_total = L_base_0 + I * H_term_I + Q * H_term_Q
            L_steps.append(expm(L_total * dt))
        
        # Vectorize: stack all 6 initial density matrices
        rho_vecs = np.array([np.outer(s, s.conj()).flatten() for s in states])  # (6, 9)
        
        # Apply all steps
        for i in range(N):
            rho_vecs = (L_steps[i] @ rho_vecs.T).T
        
        # Compute fidelities
        total_infidelity = 0.0
        for idx in range(6):
            rho = rho_vecs[idx].reshape(3, 3)
            target = targets[idx]
            proj = np.outer(target, target.conj())
            fid = np.real(np.trace(rho @ proj))
            fid = np.clip(fid, 0, 1)
            total_infidelity += (1.0 - fid)
        
        return total_infidelity / 6.0
    
    def check_constraints(I_Q):
        # Amplitude
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        if np.max(amps) > amp_bound + 1e-12:
            return False
        # Slew
        if np.sqrt(I_Q[0,0]**2 + I_Q[0,1]**2) / dt > slew_bound + 1e-12:
            return False
        if np.sqrt(I_Q[-1,0]**2 + I_Q[-1,1]**2) / dt > slew_bound + 1e-12:
            return False
        diffs = np.diff(I_Q, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / dt > slew_bound + 1e-12:
            return False
        return True
    
    def calculate_score(I_Q):
        """
        Calculates the score: mean + max of the 9 cases.
        """
        case_scores = []
        for delta in deltas:
            for scale in scales:
                mean_inf = simulate_pulse(I_Q, delta, scale)
                # We need max infidelity too, but simulate_pulse returns mean.
                # Let's modify to return both or recompute.
                # For speed, let's just use mean as proxy first, or modify simulate_pulse.
                # Actually, the task says "infidelity = 1 - mean fidelity... Score = mean + max of the 9."
                # This implies for each case, we compute infidelity (which is mean fidelity deficit?).
                # "infidelity = 1 - mean fidelity to X over [states]"
                # So infidelity_case = 1 - (1/6) * sum(fid_i)
                # Then Score = mean(infidelity_case) + max(infidelity_case)
                # My simulate_pulse returns mean_inf which is infidelity_case.
                case_scores.append(mean_inf)
        
        return np.mean(case_scores) + np.max(case_scores)
    
    def generate_pulse(sigma, eta, center=20.0):
        t = np.linspace(0, 40, N)
        # Area of I should be pi for a pi-pulse
        # Integral of A*exp(-(t-c)^2/(2*sigma^2)) dt = A * sigma * sqrt(2*pi)
        # So A = pi / (sigma * sqrt(2*pi)) = sqrt(pi/2) / sigma
        A = np.sqrt(np.pi / 2) / sigma
        I = A * np.exp(-(t - center)**2 / (2 * sigma**2))
        # Derivative of I with respect to t
        dI_dt = -(t - center) / (sigma**2) * I
        # DRAG: Q = -eta * dI/dt (negative sign for proper DRAG with negative anharmonicity)
        Q = -eta * dI_dt
        I_Q = np.column_stack((I, Q))
        return I_Q
    
    # Seed pulse
    sigma_seed = 40.0 / 6.0
    I_Q_seed = generate_pulse(sigma_seed, 0.0)
    
    # Check seed
    if not check_constraints(I_Q_seed):
        # Adjust seed if necessary
        pass
    
    best_I_Q = I_Q_seed.copy()
    best_score = calculate_score(best_I_Q)
    
    # Also try a few DRAG seeds
    for sigma_try in [6.0, 8.0, 10.0, 12.0]:
        for eta_try in [0.5, 1.0, 1.5, 2.0, 3.0]:
            I_Q_try = generate_pulse(sigma_try, eta_try)
            if check_constraints(I_Q_try):
                s_try = calculate_score(I_Q_try)
                if s_try < best_score:
                    best_score = s_try
                    best_I_Q = I_Q_try.copy()
    
    # Objective function for optimization
    def objective(params):
        if time.time() > deadline:
            return 1e10
        sigma, eta = params
        if sigma <= 0.5 or sigma > 50:
            return 1e10
        I_Q = generate_pulse(sigma, eta)
        if not check_constraints(I_Q):
            return 1e10
        return calculate_score(I_Q)
    
    # Phase 1: Coarse Grid Search
    # Time budget: 50s
    grid_deadline = start_time + 50.0
    
    sigmas = [5, 6, 7, 8, 9, 10, 11, 12, 14, 16]
    etas = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    best_grid_params = [sigma_seed, 0.0]
    
    for s in sigmas:
        for e in etas:
            if time.time() > grid_deadline:
                break
            score = objective([s, e])
            if score < best_score:
                best_score = score
                best_I_Q = generate_pulse(s, e)
                best_grid_params = [s, e]
        if time.time() > grid_deadline:
            break
    
    # Phase 2: Local Optimization (Nelder-Mead)
    # Time budget: 100s
    local_deadline = start_time + 150.0
    
    # Start from best grid point and a few diverse points
    nm_starts = [
        best_grid_params,
        [8.0, 1.0],
        [10.0, 2.0],
        [6.0, 0.5],
        [12.0, 1.5],
        [9.0, 1.0],
        [sigma_seed, 0.0]
    ]
    
    # Remove duplicates
    unique_starts = []
    for sp in nm_starts:
        if sp not in unique_starts:
            unique_starts.append(sp)
    
    for sp in unique_starts:
        if time.time() > local_deadline:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                           options={'maxiter': 400, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_I_Q = generate_pulse(res.x[0], res.x[1])
                best_grid_params = [res.x[0], res.x[1]]
        except Exception as e:
            pass
    
    # Phase 3: Final Polish with tighter tolerances
    if time.time() < deadline - 5:
        final_starts = [
            best_grid_params,
            [8.0, 1.0],
            [9.0, 1.5],
            [10.0, 2.0],
            [7.0, 0.8],
            [11.0, 1.2]
        ]
        
        unique_final = []
        for sp in final_starts:
            if sp not in unique_final:
                unique_final.append(sp)
        
        for sp in unique_final:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(objective, sp, method='Nelder-Mead', 
                               options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_I_Q = generate_pulse(res.x[0], res.x[1])
            except Exception as e:
                pass
    
    # Final Constraint Check
    if not check_constraints(best_I_Q):
        best_I_Q = I_Q_seed.copy()
    
    return best_I_Q
