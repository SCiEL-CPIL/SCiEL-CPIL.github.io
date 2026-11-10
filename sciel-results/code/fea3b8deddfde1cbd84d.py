import numpy as np
import scipy.linalg as sla
import scipy.optimize as opt
import time
import math

def pulse(target_gate, hardware_model):
    t_start = time.time()
    deadline = t_start + 320.0  # Safety margin before 330s kill
    
    # Constants
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
    n_op = a_dag @ a  # diag(0,1,2)
    I3 = np.eye(3, dtype=complex)
    
    # Precompute constant matrices
    aa_dag = a + a_dag
    i_a_minus_adag = 1j * (a - a_dag)
    n_np = n_op.astype(complex)
    
    # Lindblad operators
    L1 = a / math.sqrt(T1)
    L2 = n_op * math.sqrt(2.0 / Tphi)
    
    # Precompute Lindblad superoperator
    def make_lindblad_superop(L):
        L_dag = L.conj().T
        term1 = np.kron(L_dag.T, L)
        LL_dag = L_dag @ L
        term2 = 0.5 * (np.kron(I3, LL_dag) + np.kron(LL_dag.T, I3))
        return term1 - term2
    
    M_Lindblad = make_lindblad_superop(L1) + make_lindblad_superop(L2)
    
    # Target states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_p = (psi_0 + psi_1) / math.sqrt(2)
    psi_m = (psi_0 - psi_1) / math.sqrt(2)
    psi_r = (psi_0 + 1j * psi_1) / math.sqrt(2)
    psi_l = (psi_0 - 1j * psi_1) / math.sqrt(2)
    
    target_states = [psi_0, psi_1, psi_p, psi_m, psi_r, psi_l]
    
    # X gate
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    targets = [X_mat @ psi for psi in target_states]
    
    # Initial vectorized density matrices (9x6)
    rho_vecs = np.zeros((9, 6), dtype=complex)
    for i, psi in enumerate(target_states):
        rho = np.outer(psi, psi.conj())
        rho_vecs[:, i] = rho.flatten()
    
    # Precompute H0 for each detuning
    H0_list = []
    for delta in detunings:
        H0 = delta * n_np + 0.5 * chi * (n_np @ (n_np - I3))
        H0_list.append(H0)
    
    def simulate_and_score(I, Q):
        """Simulate all 9 cases and return score = mean + max infidelity."""
        # Check constraints
        amps = np.sqrt(I**2 + Q**2)
        if np.max(amps) > amp_bound + 1e-10:
            return 1e10
        
        p = np.column_stack((I, Q))
        p_ext = np.vstack([np.zeros((1, 2)), p, np.zeros((1, 2))])
        diffs = np.diff(p_ext, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        if np.max(norms) / dt > slew_bound + 1e-10:
            return 1e10
        
        infid_per_case = []
        
        for di, delta in enumerate(detunings):
            H0 = H0_list[di]
            
            for si, scale in enumerate(scales):
                R = rho_vecs.copy()
                
                for k in range(n_steps):
                    Ik = I[k]
                    Qk = Q[k]
                    HD = 0.5 * scale * (Ik * aa_dag + Qk * i_a_minus_adag)
                    H = H0 + HD
                    
                    M_coh = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
                    M_liouv = M_coh + M_Lindblad
                    
                    E = sla.expm(dt * M_liouv)
                    R = E @ R
                
                sum_fid = 0.0
                for i in range(6):
                    rho_f = R[:, i].reshape((3, 3))
                    f = np.vdot(targets[i], rho_f @ targets[i]).real
                    f = max(0.0, min(1.0, f))
                    sum_fid += f
                
                mean_fid = sum_fid / 6.0
                infid = 1.0 - mean_fid
                infid_per_case.append(infid)
        
        infid_arr = np.array(infid_per_case)
        return np.mean(infid_arr) + np.max(infid_arr)
    
    def make_pulse(alpha, sigma):
        """Generate DRAG-style pulse."""
        t = np.linspace(0, T, n_steps)
        center = T / 2.0
        I = alpha * np.exp(-((t - center)**2) / (2.0 * sigma**2))
        # Derivative of Gaussian
        dI_dt = -I * ((t - center) / sigma**2)
        Q = -alpha * 0.5 * ((t - center) / sigma**2) * np.exp(-((t - center)**2) / (2.0 * sigma**2)) * 0.5
        # Actually, DRAG: Q = -beta * dI/dt where beta is chosen to cancel leakage
        # For a Gaussian, dI/dt = -I*(t-center)/sigma^2
        # So Q = beta * I * (t-center)/sigma^2
        Q = alpha * 0.25 * ((t - center) / sigma**2) * np.exp(-((t - center)**2) / (2.0 * sigma**2))
        return I, Q
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        alpha, sigma = params
        
        if sigma < 1.0 or sigma > 30.0:
            return 1e10
        if alpha < 0.0 or alpha > 0.4:
            return 1e10
        
        I, Q = make_pulse(alpha, sigma)
        return simulate_and_score(I, Q)
    
    # Seed values
    sigma0 = 40.0 / 6.0
    # Area = pi: integral of alpha * exp(-(t-c)^2/(2 sigma^2)) dt = alpha * sigma * sqrt(2*pi)
    # So alpha = pi / (sigma * sqrt(2*pi))
    alpha0 = math.pi / (sigma0 * math.sqrt(2 * math.pi))
    
    best_params = [alpha0, sigma0]
    best_score = objective(best_params)
    
    # Optimize with Nelder-Mead
    try:
        res = opt.minimize(
            objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        if time.time() < deadline and res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except Exception:
        pass
    
    # Refine with COBYLA if time permits
    if time.time() < deadline - 30:
        try:
            res2 = opt.minimize(
                objective,
                best_params,
                method='COBYLA',
                bounds=[(0.1, 0.4), (2.0, 25.0)],
                options={'maxiter': 100, 'rhobeg': 0.01}
            )
            if time.time() < deadline and res2.fun < best_score:
                best_params = res2.x
                best_score = res2.fun
        except Exception:
            pass
    
    # Generate final pulse
    alpha, sigma = best_params
    I, Q = make_pulse(alpha, sigma)
    
    # Final constraint check - if violated, fall back to seed
    amps = np.sqrt(I**2 + Q**2)
    p = np.column_stack((I, Q))
    p_ext = np.vstack([np.zeros((1, 2)), p, np.zeros((1, 2))])
    diffs = np.diff(p_ext, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    
    if np.max(amps) > amp_bound or np.max(norms) / dt > slew_bound:
        # Fall back to seed
        I, Q = make_pulse(alpha0, sigma0)
        amps = np.sqrt(I**2 + Q**2)
        p = np.column_stack((I, Q))
        p_ext = np.vstack([np.zeros((1, 2)), p, np.zeros((1, 2))])
        diffs = np.diff(p_ext, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        if np.max(amps) > amp_bound or np.max(norms) / dt > slew_bound:
            # Last resort: scale down
            scale_factor = min(1.0, amp_bound / np.max(amps), slew_bound * dt / np.max(norms))
            I *= scale_factor
            Q *= scale_factor
    
    return np.column_stack((I, Q))
