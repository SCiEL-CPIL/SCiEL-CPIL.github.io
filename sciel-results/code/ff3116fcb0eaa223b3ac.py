# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amplitude_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0

    dt = 0.5
    N = 80
    total_time = N * dt

    # 3-level operators
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    a_dag = a.conj().T
    n_op = a_dag @ a
    I3 = np.eye(3)

    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)

    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2

    L1_term = np.kron(L1, L1.T)
    L2_term = np.kron(L2, L2.T)
    L1_diss_left = np.kron(I3, L1_dag_L1)
    L1_diss_right = np.kron(L1_dag_L1.T, I3)
    L2_diss_left = np.kron(I3, L2_dag_L2)
    L2_diss_right = np.kron(L2_dag_L2.T, I3)

    static_Liouv = L1_term - 0.5 * (L1_diss_left + L1_diss_right) \
                  + L2_term - 0.5 * (L2_diss_left + L2_diss_right)

    an_ham = 0.5 * anharmonicity * n_op @ (n_op - I3)

    detunings = np.array([-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002])
    scales = np.array([0.97, 1.0, 1.03])

    case_static_L = np.zeros((9, 9, 9), dtype=complex)
    case_scale = np.zeros(9)
    case_idx = 0
    for det in detunings:
        for sc in scales:
            H_static = det * n_op + an_ham
            H_left = np.kron(I3, H_static)
            H_right = np.kron(H_static.T, I3)
            case_static_L[case_idx] = -1j * (H_left - H_right) + static_Liouv
            case_scale[case_idx] = sc
            case_idx += 1

    # Precompute drive derivative Liouvillian components (constant part)
    dHdI_base = 0.5 * (a + a_dag)
    dHdQ_base = 0.5j * (a - a_dag)
    dL_dI_base = -1j * (np.kron(I3, dHdI_base) - np.kron(dHdI_base.T, I3))
    dL_dQ_base = -1j * (np.kron(I3, dHdQ_base) - np.kron(dHdQ_base.T, I3))

    # Test states
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]

    # X_pi target: -i * swap |0> and |1>
    target_states = []
    for s in states:
        t = np.zeros(3, dtype=complex)
        t[0] = -1j * s[1]
        t[1] = -1j * s[0]
        t[2] = s[2]
        target_states.append(t)

    # Precompute fidelity gradient matrices
    fid_grad_mats = np.zeros((6, 9), dtype=float)
    for si in range(6):
        psi_t = target_states[si]
        fid_grad_mats[si] = np.outer(np.conj(psi_t), psi_t).real.flatten()

    # Precompute rho_init
    rho_init = np.zeros((6, 9), dtype=complex)
    for si in range(6):
        psi = states[si]
        rho_mat = np.outer(psi, psi.conj())
        rho_init[si] = rho_mat.flatten()

    def check_validity(pulse_IQ):
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amplitudes > amplitude_bound + 1e-12):
            return False
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            if np.linalg.norm(curr - prev) / dt > slew_bound + 1e-12:
                return False
            prev = curr
        if np.linalg.norm(prev) / dt > slew_bound + 1e-12:
            return False
        return True

    def make_drag_pulse(alpha=0.0, sigma=40.0/6.0):
        t = np.arange(N) * dt
        center = total_time / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        edge_val = gaussian[0]
        gaussian -= edge_val
        area = np.sum(gaussian) * dt
        if abs(area) > 1e-15:
            gaussian *= np.pi / area
        I = gaussian.copy()
        if alpha != 0:
            dI_dt = np.gradient(I, dt)
            Q = alpha * dI_dt
        else:
            Q = np.zeros(N)
        pulse_IQ = np.column_stack([I, Q])
        if not check_validity(pulse_IQ):
            max_amp = np.max(np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2))
            if max_amp > amplitude_bound:
                pulse_IQ *= amplitude_bound / max_amp
        return pulse_IQ

    def simulate_with_steps(pulse_IQ):
        """
        Returns U_all (9, N, 9, 9) and rho_steps (9, N+1, 6, 9).
        """
        U_all = np.zeros((9, N, 9, 9), dtype=complex)
        for ci in range(9):
            sc = case_scale[ci]
            for step in range(N):
                I_val = pulse_IQ[step, 0]
                Q_val = pulse_IQ[step, 1]
                H_drive = 0.5 * sc * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag))
                H_left = np.kron(I3, H_drive)
                H_right = np.kron(H_drive.T, I3)
                L = case_static_L[ci] - 1j * (H_left - H_right)
                U_all[ci, step] = expm(dt * L)

        rho_steps = np.zeros((9, N + 1, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                vec = rho_init[si].copy()
                rho_steps[ci, 0, si] = vec
                for step in range(N):
                    vec = U_all[ci, step] @ vec
                    rho_steps[ci, step + 1, si] = vec

        return U_all, rho_steps

    def compute_score_and_grad(params, beta=50.0):
        I = params[:N]
        Q = params[N:2*N]
        p = np.column_stack([I, Q])
        
        # Validity check
        if not check_validity(p):
            # Return a high penalty and zero gradient
            return 1e10, np.zeros(2*N)

        U_all, rho_steps = simulate_with_steps(p)
        rho_final = rho_steps[:, N, :, :]
        
        # Compute fidelities
        fids = np.zeros((9, 6))
        for ci in range(9):
            for si in range(6):
                rho_mat = rho_final[ci, si].reshape(3, 3)
                psi_t = target_states[si]
                fids[ci, si] = np.real(np.conj(psi_t) @ rho_mat @ psi_t)
        
        infids = 1.0 - fids
        m = np.max(infids)
        exp_vals = np.exp(beta * (infids - m))
        sum_exp = np.sum(exp_vals)
        score = np.mean(infids) + (m + np.log(sum_exp) / beta)
        
        # Gradient calculation
        d_s_d_infid = np.ones(54) / 54.0 + exp_vals.flatten() / (sum_exp * beta)
        d_s_d_fid = -d_s_d_infid.reshape(9, 6)
        
        grad_params = np.zeros(2*N)
        
        for ci in range(9):
            sc = case_scale[ci]
            # Adjoint variables lambda, shape (6, 9)
            # lambda corresponds to dS / d rho
            lam = np.zeros((6, 9), dtype=float)
            for si in range(6):
                lam[si] = d_s_d_fid[ci, si] * fid_grad_mats[si]
            
            # Backward pass
            for step in range(N - 1, -1, -1):
                # dS / d U_step
                # U_step maps rho_step to rho_step+1
                # dS = sum_lam_{step+1} dot rho_{step+1}
                # rho_{step+1} = U_step rho_step
                # dS = sum_lam_{step+1} dot (U_step rho_step)
                # dS/dU_step = sum_lam_{step+1} rho_step^T
                # Note: lam is dS/d rho_{step+1}.
                # We need dS/d I_step and dS/d Q_step.
                
                # Accumulate gradient for this step
                # dU_dI = dt * dL_dI * sc (since H_drive has sc)
                # Actually, L = Static - 1j (H_drive_left - H_drive_right)
                # dL_dI = -1j (dH_drive_dI_left - dH_drive_dI_right)
                # dH_drive_dI = 0.5 * sc * (a + a_dag)
                # So dL_dI = sc * dL_dI_base
                dL_dI_step = sc * dL_dI_base
                dL_dQ_step = sc * dL_dQ_base
                
                # First order approx: dU_dI = dt * dL_dI
                dU_dI = dt * dL_dI_step
                dU_dQ = dt * dL_dQ_step
                
                # dS_dU = sum_si outer(lam[si], rho_step[si])
                # rho_step is rho_steps[ci, step, si]
                dS_dU = np.zeros((9, 9), dtype=float)
                for si in range(6):
                    dS_dU += np.outer(lam[si], rho_steps[ci, step, si].real)
                
                grad_params[step] += np.real(np.sum(dS_dU * dU_dI.T))
                grad_params[N + step] += np.real(np.sum(dS_dU * dU_dQ.T))
                
                # Update lam for previous step
                # lam_step = U_step^T lam_{step+1}
                U_step = U_all[ci, step]
                for si in range(6):
                    lam[si] = U_step.T @ lam[si]
                    
        return score, grad_params

    # Time management
    start_time = time.time()
    deadline = start_time + 290.0

    def time_remaining():
        return deadline - time.time()

    # Phase 1: DRAG alpha sweep
    best_pulse = make_drag_pulse(0.0)
    best_score, _ = compute_score_and_grad(best_pulse.flatten(), beta=50.0)
    
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3]
    for alpha in alphas:
        if time_remaining() < 30:
            break
        p_test = make_drag_pulse(alpha)
        if check_validity(p_test):
            s, _ = compute_score_and_grad(p_test.flatten(), beta=50.0)
            if s < best_score:
                best_score = s
                best_pulse = p_test.copy()

    # Phase 2: L-BFGS-B Optimization
    if time_remaining() > 30:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2 * N)
        
        def objective(params):
            s, g = compute_score_and_grad(params, beta=50.0)
            return s, g
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                              bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfev': 500})
            if time_remaining() > 10:
                p_opt = result.x.reshape(N, 2)
                if check_validity(p_opt):
                    s_opt, _ = compute_score_and_grad(p_opt.flatten(), beta=50.0)
                    if s_opt < best_score:
                        best_score = s_opt
                        best_pulse = p_opt.copy()
        except Exception:
            pass

    # Final validity check
    if not check_validity(best_pulse):
        best_pulse = make_drag_pulse(0.0)

    return best_pulse
# EVOLVE-BLOCK-END
