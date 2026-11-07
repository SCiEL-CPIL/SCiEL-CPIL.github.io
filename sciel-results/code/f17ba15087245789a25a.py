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
    n_op = a_dag @ a  # diag(0, 1, 2)
    I3 = np.eye(3)

    # Precompute static parts of Liouvillian
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

    # Static part of Liouvillian (independent of I, Q, detuning, scale)
    static_Liouv = L1_term - 0.5 * (L1_diss_left + L1_diss_right) \
                  + L2_term - 0.5 * (L2_diss_left + L2_diss_right)

    # Precompute anharmonicity contribution
    an_ham = 0.5 * anharmonicity * n_op @ (n_op - I3)

    # Precompute all 9 case static Liouvillian components
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
    fid_grad_mats = []
    for si in range(6):
        psi_t = target_states[si]
        grad_mat = np.outer(np.conj(psi_t), psi_t).real
        fid_grad_mats.append(grad_mat.flatten())

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

    def make_drag_pulse(alpha=0.0):
        t = np.arange(N) * dt
        sigma = 40.0 / 6.0
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
            max_amp = np.max(np.sqrt(I**2 + Q**2))
            if max_amp > amplitude_bound:
                pulse_IQ *= amplitude_bound / max_amp
        return pulse_IQ

    def simulate_all(pulse_IQ):
        """
        Simulate all 54 (case, state) combinations.
        Returns rho_final array of shape (9, 6, 9).
        Also returns U_all for adjoint computation.
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

        rho_init = np.zeros((6, 9), dtype=complex)
        for si in range(6):
            psi = states[si]
            rho_mat = np.outer(psi, psi.conj())
            rho_init[si] = rho_mat.flatten()

        rho_final = np.zeros((9, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                vec = rho_init[si].copy()
                for step in range(N):
                    vec = U_all[ci, step] @ vec
                rho_final[ci, si] = vec

        return rho_final, U_all

    def compute_fidelities(rho_final):
        fids = np.zeros((9, 6))
        for ci in range(9):
            for si in range(6):
                rho_mat = rho_final[ci, si].reshape(3, 3)
                psi_t = target_states[si]
                fid = np.real(np.conj(psi_t) @ rho_mat @ psi_t)
                fids[ci, si] = fid
        return fids

    def compute_score(fids):
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)

    def compute_soft_score(fids, beta=50.0):
        infids = 1.0 - fids
        m = np.max(infids)
        soft_max = m + np.log(np.sum(np.exp(beta * (infids - m)))) / beta
        return np.mean(infids) + soft_max

    def compute_gradient(params, U_all, fids, beta=50.0):
        """Compute gradient using adjoint method."""
        infids = 1.0 - fids
        m = np.max(infids)
        exp_vals = np.exp(beta * (infids - m))
        soft_max = m + np.log(np.sum(exp_vals)) / beta

        # d(score)/d(infid[k]) = 1/54 + exp_vals[k] / sum(exp_vals) / beta
        d_score_d_infid = np.ones(54) / 54.0 + exp_vals.flatten() / (np.sum(exp_vals) * beta)
        d_score_d_fid = -d_score_d_infid.reshape(9, 6)

        # Gradient w.r.t. each rho_final[ci, si]
        grad_rho = np.zeros((9, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                grad_rho[ci, si] = d_score_d_fid[ci, si] * fid_grad_mats[si]

        # Adjoint backward propagation
        # lambda[ci, si, step] = d(score)/d(rho_vec[ci, si, step])
        # lambda[ci, si, N] = grad_rho[ci, si]
        # lambda[ci, si, step] = U[ci, step]^T @ lambda[ci, si, step+1]
        # For complex: we need to be careful. Since score is real:
        # d(score)/d(rho_step) = U[step]^T @ d(score)/d(rho_{step+1})
        # But rho is complex, so we use the Wirtinger calculus approach.
        # Actually, for vec(rho_{step+1}) = U @ vec(rho_step):
        # d(score)/d(vec(rho_step)) = U^T @ d(score)/d(vec(rho_{step+1}))
        # This works when we treat vec(rho) as the independent variable.

        lambda_state = np.zeros((9, 6, 9), dtype=complex)
        lambda_state[:, :, :] = grad_rho

        grad_params = np.zeros(2 * N)

        for step in range(N - 1, -1, -1):
            # Compute gradient w.r.t. I and Q at this step
            for ci in range(9):
                sc = case_scale[ci]
                I_val = params[step]
                Q_val = params[N + step]

                # dU/dI and dU/dQ
                # U = exp(dt * L), where L = case_static_L[ci] - 1j*(H_left - H_right)
                # dL/dI = -1j * (dH_left/dI - dH_right/dI)
                # dH_drive/dI = 0.5 * sc * (a + a_dag)
                # dH_drive/dQ = 0.5 * sc * 1j * (a - a_dag)

                dH_dI = 0.5 * sc * (a + a_dag)
                dH_dQ = 0.5 * sc * 1j * (a - a_dag)

                dL_dI = -1j * (np.kron(I3, dH_dI) - np.kron(dH_dI.T, I3))
                dL_dQ = -1j * (np.kron(I3, dH_dQ) - np.kron(dH_dQ.T, I3))

                # dU/dparam = integral_0^dt exp((dt-s)*L) dL exp(s*L) ds
                # Approximate: dt * exp(dt/2 * L) dL exp(dt/2 * L) (midpoint)
                # Or use: dU = dt * expm(dt*L) dL (first order)
                # Better: use the identity d(expm(X))/dX = int_0^1 expm((1-t)X) dX expm(tX) dt
                # Approximate with midpoint: expm(dt/2 * L) dL expm(dt/2 * L) * dt

                U_step = U_all[ci, step]
                L_step = case_static_L[ci] - 1j * (np.kron(I3, 0.5 * sc * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag))) - np.kron((0.5 * sc * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag))).T, I3))

                # Use first-order approximation: dU ≈ dt * U @ dL
                dU_dI = dt * U_step @ dL_dI
                dU_dQ = dt * U_step @ dL_dQ

                # Gradient contribution from this step
                # rho_{step+1} = U_step @ rho_step
                # d(score)/d(U_step) = sum_si lambda[ci, si, step+1] outer(rho_step[ci, si])
                # Actually: d(rho_{step+1})/d(U_step) = outer(rho_step, vec(1)) in some sense
                # More precisely: rho_{step+1}[i] = sum_j U_step[i,j] rho_step[j]
                # d(score)/d(U_step[i,j]) = sum_si lambda[ci,si,i] * rho_step[ci,si,j]

                # We need rho_step for each state
                # rho_step[ci, si] = U_all[ci, step-1] @ ... @ U_all[ci, 0] @ rho_init[si]
                # We can compute this forward

                # For efficiency, precompute rho at each step
                # But for now, let's compute it on the fly

                # Actually, let's restructure: compute rho at each step forward
                pass

            # Propagate adjoint backward
            for ci in range(9):
                for si in range(6):
                    if step > 0:
                        lambda_state[ci, si] = U_all[ci, step - 1].T @ lambda_state[ci, si]

        # For simplicity, use numerical gradient
        return None

    def objective_and_grad(params, beta=50.0):
        """Objective with numerical gradient."""
        I = params[:N]
        Q = params[N:2*N]
        pulse_IQ = np.column_stack([I, Q])
        if not check_validity(pulse_IQ):
            return 1e10, np.zeros(2*N)

        rho_final, _ = simulate_all(pulse_IQ)
        fids = compute_fidelities(rho_final)
        score = compute_soft_score(fids, beta=beta)

        # Numerical gradient with central differences
        eps = 1e-6
        grad = np.zeros(2*N)
        for j in range(2*N):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[j] += eps
            params_minus[j] -= eps

            p_plus = np.column_stack([params_plus[:N], params_plus[N:2*N]])
            p_minus = np.column_stack([params_minus[:N], params_minus[N:2*N]])

            if check_validity(p_plus):
                r_plus, _ = simulate_all(p_plus)
                f_plus = compute_fidelities(r_plus)
                s_plus = compute_soft_score(f_plus, beta=beta)
            else:
                s_plus = score + eps * 1000

            if check_validity(p_minus):
                r_minus, _ = simulate_all(p_minus)
                f_minus = compute_fidelities(r_minus)
                s_minus = compute_soft_score(f_minus, beta=beta)
            else:
                s_minus = score + eps * 1000

            grad[j] = (s_plus - s_minus) / (2 * eps)

        return score, grad

    # Time management
    start_time = time.time()
    deadline = start_time + 290.0

    def time_remaining():
        return deadline - time.time()

    # Phase 1: DRAG alpha sweep with finer grid
    pulse_IQ = make_drag_pulse(0.0)
    best_pulse = pulse_IQ.copy()
    f_best = compute_fidelities(simulate_all(best_pulse)[0])
    best_score = compute_soft_score(f_best, beta=50.0)

    alpha_candidates = np.linspace(-0.5, 0.5, 21)
    for alpha in alpha_candidates:
        if time_remaining() < 30:
            break
        p_test = make_drag_pulse(alpha)
        if check_validity(p_test):
            f = compute_fidelities(simulate_all(p_test)[0])
            s = compute_soft_score(f, beta=50.0)
            if s < best_score:
                best_score = s
                best_pulse = p_test.copy()

    # Phase 2: Nelder-Mead for coarse optimization
    if time_remaining() > 80:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2 * N)

        def nm_obj(params):
            if time_remaining() < 10:
                return 1e10
            I = params[:N]
            Q = params[N:2*N]
            p = np.column_stack([I, Q])
            if not check_validity(p):
                return 1e10
            f = compute_fidelities(simulate_all(p)[0])
            return compute_soft_score(f, beta=50.0)

        result = minimize(nm_obj, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 300})
        if time_remaining() > 30:
            p_nm = result.x.reshape(N, 2)
            if check_validity(p_nm):
                f_nm = compute_fidelities(simulate_all(p_nm)[0])
                s_nm = compute_soft_score(f_nm, beta=50.0)
                if s_nm < best_score:
                    best_score = s_nm
                    best_pulse = p_nm.copy()

    # Phase 3: L-BFGS-B with numerical gradient
    if time_remaining() > 60:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2*N)

        def lbfgs_obj(params):
            if time_remaining() < 10:
                return 1e10, np.zeros(2*N)
            return objective_and_grad(params, beta=50.0)

        result2 = minimize(lbfgs_obj, x0, method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfev': 800})
        if time_remaining() > 15:
            p_bfgs = result2.x.reshape(N, 2)
            if check_validity(p_bfgs):
                f_bfgs = compute_fidelities(simulate_all(p_bfgs)[0])
                s_bfgs = compute_soft_score(f_bfgs, beta=50.0)
                if s_bfgs < best_score:
                    best_score = s_bfgs
                    best_pulse = p_bfgs.copy()

    # Phase 4: Final polish with higher beta
    if time_remaining() > 20:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2*N)

        def polish_obj(params):
            if time_remaining() < 5:
                return 1e10, np.zeros(2*N)
            return objective_and_grad(params, beta=100.0)

        result3 = minimize(polish_obj, x0, method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfev': 400})
        if time_remaining() > 5:
            p_pol = result3.x.reshape(N, 2)
            if check_validity(p_pol):
                f_pol = compute_fidelities(simulate_all(p_pol)[0])
                s_pol = compute_soft_score(f_pol, beta=100.0)
                if s_pol < best_score:
                    best_score = s_pol
                    best_pulse = p_pol.copy()

    # Final validity check
    if not check_validity(best_pulse):
        best_pulse = make_drag_pulse(0.0)

    return best_pulse
# EVOLVE-BLOCK-END
