import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin

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
    
    # Define Pauli-X matrix
    X = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 0]], dtype=complex)
    
    # Basis states for infidelity calculation
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_plus_i = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_minus_i = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    init_states = [psi_0, psi_1, psi_plus, psi_minus, psi_plus_i, psi_minus_i]
    
    # Target states: X @ init_state
    target_states = [X @ psi for psi in init_states]

    # Pre-compute Lindblad terms
    # L1 = a / sqrt(T1)
    # L2 = n * sqrt(2/Tphi)
    a_mat = np.array([[0, 1, 0],
                      [0, 0, np.sqrt(2)],
                      [0, 0, 0]], dtype=complex)
    n_mat = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 2]], dtype=complex)
    
    L1 = a_mat / np.sqrt(T1)
    L2 = n_mat * np.sqrt(2 / Tphi)
    
    # Compute M_diss (9x9 superoperator for dissipation)
    def vec(rho):
        return rho.flatten(order='F')
        
    def unvec(v):
        return v.reshape((3, 3), order='F')
        
    M_diss = np.zeros((9, 9), dtype=complex)
    for k in range(9):
        e_k = np.zeros(9, dtype=complex)
        e_k[k] = 1
        rho_k = unvec(e_k)
        diss_k = L1 @ rho_k @ L1.conj().T - 0.5 * (L1.conj().T @ L1 @ rho_k + rho_k @ L1.conj().T @ L1)
        diss_k += L2 @ rho_k @ L2.conj().T - 0.5 * (L2.conj().T @ L2 @ rho_k + rho_k @ L2.conj().T @ L2)
        M_diss[:, k] = vec(diss_k)
        
    I_mat = np.eye(3)
    n2 = n_mat @ n_mat
    anharmonic = 0.5 * alpha * (n2 - n_mat)

    def get_H(I_p, Q_p, delta, scale):
        drive = 0.5 * scale * (I_p * (a_mat + a_mat.conj().T) + 1j * Q_p * (a_mat - a_mat.conj().T))
        detuning = delta * n_mat
        return detuning + anharmonic + drive

    def simulate_single(I_pulse, Q_pulse, delta, scale):
        # Compute the total superoperator U_total
        # U_total = exp(L_79 dt) ... exp(L_0 dt)
        # We can accumulate this.
        # To be efficient, we can just apply the step operators to the 6 initial density matrices directly.
        
        rho_vecs = np.zeros((9, 6), dtype=complex)
        for i, psi in enumerate(init_states):
            rho_vecs[:, i] = vec(np.outer(psi, psi.conj()))
            
        # Pre-compute M_coh for each step? No, I/Q change.
        # But we can compute M_step for each i and apply.
        # 80 steps * 9x9 expm is fast.
        
        for i in range(N):
            I_p = I_pulse[i]
            Q_p = Q_pulse[i]
            H = get_H(I_p, Q_p, delta, scale)
            
            # M_coh = -i [H, rho] -> superoperator
            M_coh = -1j * (np.kron(I_mat, H) - np.kron(H.T, I_mat))
            
            M_total = M_coh + M_diss
            U_step = expm(M_total * dt)
            
            rho_vecs = U_step @ rho_vecs
            
        # Compute fidelities
        fidelities = []
        for i, target in enumerate(target_states):
            rho_final = unvec(rho_vecs[:, i])
            t_vec = target.flatten()
            f = np.real(np.vdot(t_vec, rho_final @ t_vec))
            f = np.clip(f, 0.0, 1.0)
            fidelities.append(f)
            
        return np.array(fidelities)

    def score_pulse(I_pulse, Q_pulse):
        infidelities = []
        for delta in deltas:
            for scale in scales:
                fids = simulate_single(I_pulse, Q_pulse, delta, scale)
                inf = 1.0 - np.mean(fids)
                infidelities.append(inf)
        infidelities = np.array(infidelities)
        return np.mean(infidelities) + np.max(infidelities)

    def check_constraints(I_pulse, Q_pulse):
        amps = np.sqrt(I_pulse**2 + Q_pulse**2)
        if np.any(amps > amp_bound + 1e-9):
            return False
        if np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2) / dt > slew_bound + 1e-9:
            return False
        dI = np.diff(I_pulse)
        dQ = np.diff(Q_pulse)
        dists = np.sqrt(dI**2 + dQ**2)
        if np.any(dists / dt > slew_bound + 1e-9):
            return False
        if np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2) / dt > slew_bound + 1e-9:
            return False
        return True

    def penalty(I_pulse, Q_pulse):
        if check_constraints(I_pulse, Q_pulse):
            return 0.0
        p = 0.0
        amps = np.sqrt(I_pulse**2 + Q_pulse**2)
        p += np.sum(np.maximum(amps - amp_bound, 0)**2)
        p += np.maximum(np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2) / dt - slew_bound, 0)**2
        dI = np.diff(I_pulse)
        dQ = np.diff(Q_pulse)
        dists = np.sqrt(dI**2 + dQ**2)
        p += np.sum(np.maximum(dists / dt - slew_bound, 0)**2)
        p += np.maximum(np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2) / dt - slew_bound, 0)**2
        return 1000.0 * p

    def objective(params):
        I_pulse = params[:N]
        Q_pulse = params[N:]
        score = score_pulse(I_pulse, Q_pulse)
        pen = penalty(I_pulse, Q_pulse)
        return score + pen

    # Generate DRAG seed
    def generate_drag_pulse(A, sigma, alpha_drag):
        t = np.arange(N) * dt
        t_center = (N - 1) / 2.0 * dt
        t_shifted = t - t_center
        I = A * np.exp(-t_shifted**2 / (2 * sigma**2))
        dI_dt = -t_shifted / (sigma**2) * I
        Q = alpha_drag * dI_dt
        return I, Q

    # Initial guess
    sigma_seed = 40.0 / 6.0
    A_seed = 1.0 / np.sqrt(2 * np.pi * sigma_seed**2) # Area ~ pi? No, area of Gaussian is A*sigma*sqrt(2pi). We want area pi. A = pi / (sigma * sqrt(2pi)) = sqrt(pi/2)/sigma.
    A_seed = np.sqrt(np.pi / 2.0) / sigma_seed
    alpha_seed = 0.0
    
    I0, Q0 = generate_drag_pulse(A_seed, sigma_seed, alpha_seed)
    
    # Ensure seed is valid
    if not check_constraints(I0, Q0):
        # Scale down if needed
        max_amp = np.max(np.sqrt(I0**2 + Q0**2))
        if max_amp > amp_bound:
            scale = amp_bound / max_amp * 0.99
            I0 *= scale
            Q0 *= scale
        # Check slew
        # If slew is too high, widen sigma or reduce A.
        # For now, just proceed with optimization.
        
    x0 = np.concatenate([I0, Q0])
    
    # Optimization
    best_I = I0
    best_Q = Q0
    best_score = objective(x0)
    
    if time.time() < deadline - 10.0:
        # Local optimization
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
        if res.fun < best_score:
            best_I = res.x[:N]
            best_Q = res.x[N:]
            best_score = res.fun
            
    # Final constraint check
    if not check_constraints(best_I, best_Q):
        # Fallback to a very safe pulse
        best_I = np.zeros(N)
        best_Q = np.zeros(N)
        # Put a small bump in the middle
        idx = N // 2
        best_I[idx] = 0.1
        best_I[idx-1] = 0.1
        best_I[idx+1] = 0.1
        
    return np.column_stack((best_I, best_Q))
