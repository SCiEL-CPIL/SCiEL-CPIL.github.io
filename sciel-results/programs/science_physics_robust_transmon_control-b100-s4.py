"""Pulse optimization for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time
import math
from scipy.optimize import minimize

def pulse(target_gate, hardware_model):
    """
    Returns an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    80 constant steps of dt = 0.5 ns.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin before 330s kill
    
    dt = 0.5
    n_steps = 80
    T = 40.0
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002
    scale_min, scale_max = 0.97, 1.03
    T1 = 25000.0
    Tphi = 40000.0
    
    # Build 3-level operators
    a_op = np.zeros((3, 3), dtype=complex)
    a_op[0, 1] = np.sqrt(1.0)
    a_op[1, 2] = np.sqrt(2.0)
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    
    # Lindblad operators
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    # Precompute constant parts of Lindblad superoperator
    dim = 3
    I3 = np.eye(dim)
    
    def lindblad_term(L):
        L_dag = L.conj().T
        LdL = L_dag @ L
        term = np.kron(L.conj(), L) + np.kron(L, L_dag) - 0.5 * (np.kron(LdL, I3) + np.kron(I3, LdL))
        return term
    
    L_diss = lindblad_term(L1) + lindblad_term(L2)
    
    def build_hamiltonian(detuning, scale, I, Q):
        H = detuning * n_op + 0.5 * anharmonicity * (n_op @ n_op - n_op)
        H += 0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag))
        return H
    
    def lindblad_hamiltonian(H):
        H_comm = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
        return H_comm

    def simulate_corner(I_arr, Q_arr, detuning, scale):
        """Simulate one corner case and return mean fidelity."""
        # Pre-compute evolution operators for each step
        exp_ops = []
        for i in range(n_steps):
            H = build_hamiltonian(detuning, scale, I_arr[i], Q_arr[i])
            L_total = lindblad_hamiltonian(H) + L_diss
            exp_ops.append(sla.expm(L_total * dt))
        
        # Initial states
        initial_states = [
            np.array([1.0, 0, 0], dtype=complex),
            np.array([0, 1.0, 0], dtype=complex),
            np.array([1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex),
            np.array([1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex),
            np.array([1/np.sqrt(2), 1j/np.sqrt(2), 0], dtype=complex),
            np.array([1/np.sqrt(2), -1j/np.sqrt(2), 0], dtype=complex),
        ]
        
        X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
        
        fids = []
        for psi in initial_states:
            rho_v = np.outer(psi, psi.conj()).flatten()
            for op in exp_ops:
                rho_v = op @ rho_v
            rho_final = rho_v.reshape(dim, dim)
            psi_t = X_mat @ psi
            fid = np.real(np.vdot(psi_t, rho_final @ psi_t))
            fids.append(fid)
        
        return np.mean(fids)
    
    def compute_score(I_arr, Q_arr):
        """Compute mean + max infidelity over 9 corners."""
        detunings = [-detuning_bound, 0.0, detuning_bound]
        scales = [scale_min, 1.0, scale_max]
        
        infs = []
        for det in detunings:
            for sc in scales:
                mean_fid = simulate_corner(I_arr, Q_arr, det, sc)
                infs.append(1.0 - mean_fid)
        
        infs = np.array(infs)
        return np.mean(infs) + np.max(infs)
    
    def is_valid(I_arr, Q_arr):
        amps = np.sqrt(I_arr**2 + Q_arr**2)
        if np.any(amps > amp_bound + 1e-12):
            return False
        # Slew from zero to step 1
        if np.linalg.norm([I_arr[0], Q_arr[0]]) / dt > slew_bound + 1e-12:
            return False
        # Between steps
        for i in range(n_steps - 1):
            diff = np.array([I_arr[i+1] - I_arr[i], Q_arr[i+1] - Q_arr[i]])
            if np.linalg.norm(diff) / dt > slew_bound + 1e-12:
                return False
        # From last step to zero
        if np.linalg.norm([I_arr[-1], Q_arr[-1]]) / dt > slew_bound + 1e-12:
            return False
        return True
    
    def make_pulse(params):
        """
        params: [A, sigma, alpha, t_center, phase]
        Returns I, Q arrays.
        """
        A, sigma, alpha, t_center, phase = params
        t = np.arange(n_steps) * dt
        t_centered = t - t_center
        gauss = np.exp(-t_centered**2 / (2 * sigma**2))
        
        # Normalize area to pi
        area = np.sum(gauss) * dt
        if area < 1e-12:
            return np.zeros(n_steps), np.zeros(n_steps)
        
        I = A * gauss * (np.pi / area) * np.cos(phase)
        
        # DRAG Q component
        dI = np.gradient(I, dt)
        Q = -alpha * dI
        
        # Check constraints and scale down if necessary
        amps = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amps)
        if max_amp > amp_bound:
            scale = amp_bound / max_amp
            I *= scale
            Q *= scale
        
        if not is_valid(I, Q):
            # Iterative scaling for slew
            for _ in range(20):
                I *= 0.9
                Q *= 0.9
                if is_valid(I, Q):
                    break
        
        return I, Q

    # Initial guess
    sigma_init = 40.0 / 6.0
    alpha_init = 1.0 / (4.0 * abs(anharmonicity))
    A_init = 0.3
    t_center_init = T / 2
    phase_init = 0.0
    
    x0 = np.array([A_init, sigma_init, alpha_init, t_center_init, phase_init])
    
    def objective(params):
        I, Q = make_pulse(params)
        if not is_valid(I, Q):
            return 1e6
        return compute_score(I, Q)
    
    # Optimize using Nelder-Mead
    best_x = x0.copy()
    best_score = objective(best_x)
    
    if time.time() < deadline - 30:
        try:
            res = minimize(objective, best_x, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score:
                best_x = res.x
                best_score = res.fun
        except Exception:
            pass
    
    I_best, Q_best = make_pulse(best_x)
    
    # If we have time, try to polish with random perturbations on the raw array
    if time.time() < deadline - 10:
        I_curr = I_best.copy()
        Q_curr = Q_best.copy()
        score_curr = best_score
        
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            # Small perturbation
            I_pert = I_curr + 0.01 * np.random.randn(n_steps)
            Q_pert = Q_curr + 0.01 * np.random.randn(n_steps)
            
            # Project to constraints
            amps = np.sqrt(I_pert**2 + Q_pert**2)
            max_amp = np.max(amps)
            if max_amp > amp_bound:
                scale = amp_bound / max_amp
                I_pert *= scale
                Q_pert *= scale
            
            if not is_valid(I_pert, Q_pert):
                continue
            
            score_pert = compute_score(I_pert, Q_pert)
            if score_pert < score_curr:
                score_curr = score_pert
                I_curr = I_pert.copy()
                Q_curr = Q_pert.copy()
        
        I_best = I_curr
        Q_best = Q_curr
    
    return np.column_stack([I_best, Q_best])
# EVOLVE-BLOCK-END
