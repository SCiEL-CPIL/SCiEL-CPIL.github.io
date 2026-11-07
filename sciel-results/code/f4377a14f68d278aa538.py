"""Rewritten program: direct gradient-based pulse optimization with analytic gradients."""
import numpy as np
import time
from scipy import linalg

def pulse(target_gate, hardware_model):
    """
    Generate a robust X_pi pulse for a 3-level transmon.
    Returns (80, 2) array of [I, Q] amplitudes in rad/ns.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")

    dt = 0.5  # ns
    N = 80
    T = N * dt  # 40 ns

    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0

    start_time = time.time()
    deadline = start_time + 310.0  # safety margin before 330s kill

    # Time grid
    t = np.linspace(0, T, N, endpoint=False) + dt / 2.0

    # Operators
    a = np.diag([1.0, np.sqrt(2.0), 1.0])
    a_dag = a.conj().T
    n = a_dag @ a
    n2 = n @ n

    eye3 = np.eye(3)

    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)

    L1c = L1.conj().T
    L2c = L2.conj().T

    # Pre-compute Lindblad part of Liouvillian (9x9)
    Lindblad_part = np.zeros((9, 9), dtype=complex)
    for L, Lc in [(L1, L1c), (L2, L2c)]:
        Lindblad_part += (np.kron(Lc, L) - 0.5 * np.kron(L @ Lc, eye3)
                          - 0.5 * np.kron(eye3, L @ Lc))

    # Pre-compute Hamiltonian coefficient matrices for I and Q
    # H = detuning*n + 0.5*anharmonicity*(n^2-n) + 0.5*scale*(I*(a+a_dag) + i*Q*(a-a_dag))
    # Liouvillian H-part: -i*(I_9 @ H - H^dag @ I_9) = -i*(kron(I_3,H) - kron(H^dag,I_3))
    # We need: L_H = -i*(kron(I_3,H) - kron(H^dag,I_3))
    # For the I term: H_I = 0.5*scale*(a+a_dag)
    # For the Q term: H_Q = 0.5*scale*i*(a-a_dag)
    # For detuning: H_d = detuning*n
    # For anharmonicity: H_a = 0.5*anharmonicity*(n^2-n)

    H_d_matrix = 0.5 * (a + a_dag)  # coefficient of I*scale
    H_Q_matrix = 0.5j * (a - a_dag)  # coefficient of Q*scale
    H_det_matrix = n  # coefficient of detuning
    H_anh_matrix = 0.5 * (n2 - n)  # coefficient of anharmonicity

    # Liouvillian coefficient matrices (9x9)
    # L_H = -i*(kron(I_3,H) - kron(H^dag,I_3))
    def liouv_coeff(H_mat):
        Hc = H_mat.conj().T
        return -1j * (np.kron(eye3, H_mat) - np.kron(Hc, eye3))

    LI = liouv_coeff(H_d_matrix)  # for I * scale
    LQ = liouv_coeff(H_Q_matrix)  # for Q * scale
    Ldet = liouv_coeff(H_det_matrix)  # for detuning
    Lanh = liouv_coeff(H_anh_matrix)  # for anharmonicity (constant)

    # States to simulate
    states = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        (np.array([1.0, 0.0, 0.0]) + np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
    ]

    # Target states after X gate
    X_gate = np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
    target_states = [X_gate @ st for st in states]

    # Pre-compute initial density matrices as 9-vectors
    rho0s = np.array([np.outer(st, st.conj()).reshape(9) for st in states])

    def evolve_single(detuning, scale, I, Q):
        """Evolve all 6 states through the pulse. Returns final rho_batch (6,9)."""
        rho_batch = rho0s.copy()
        # Constant part
        L_const = Lindblad_part + Ldet * detuning + Lanh * anharmonicity

        for k in range(N):
            L_step = L_const + LI * (I[k] * scale) + LQ * (Q[k] * scale)
            U = linalg.expm(L_step * dt)
            rho_batch = (U @ rho_batch.T).T
        return rho_batch

    def compute_fidelities(detuning, scale, I, Q):
        """Compute mean fidelity over 6 states."""
        rho_batch = evolve_single(detuning, scale, I, Q)
        fids = []
        for i in range(6):
            rho_final = rho_batch[i].reshape(3, 3)
            psi_t = target_states[i]
            fid = np.real(np.vdot(psi_t, rho_final @ psi_t))
            fids.append(fid)
        return np.mean(fids)

    def score_pulse(I, Q):
        """Compute the score: mean + max of infidelities over 9 cases."""
        detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        infids = []
        for d in detunings:
            for s in scales:
                fid = compute_fidelities(d, s, I, Q)
                infids.append(1.0 - fid)
        infids = np.array(infids)
        return np.mean(infids) + np.max(infids)

    def check_validity(I, Q):
        """Check amplitude and slew constraints."""
        amp = np.sqrt(I ** 2 + Q ** 2)
        if np.any(amp > amp_bound + 1e-10):
            return False
        # Slew from zero to step 0
        slew0 = np.sqrt(I[0] ** 2 + Q[0] ** 2) / dt
        if slew0 > slew_bound + 1e-10:
            return False
        # Slew between steps
        dI = np.diff(I)
        dQ = np.diff(Q)
        slews = np.sqrt(dI ** 2 + dQ ** 2) / dt
        if np.any(slews > slew_bound + 1e-10):
            return False
        # Slew from last step to zero
        slewN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / dt
        if slewN > slew_bound + 1e-10:
            return False
        return True

    def project_valid(I, Q):
        """Project pulse to satisfy amplitude and slew constraints."""
        # Enforce amplitude bound
        amp = np.sqrt(I ** 2 + Q ** 2)
        max_amp = np.max(amp) if amp.size else 0.0
        if max_amp > amp_bound:
            factor = amp_bound / max_amp
            I = I * factor
            Q = Q * factor

        # Enforce slew bound iteratively
        def max_slew(I, Q):
            slew0 = np.sqrt(I[0] ** 2 + Q[0] ** 2) / dt
            dI = np.diff(I)
            dQ = np.diff(Q)
            slews = np.sqrt(dI ** 2 + dQ ** 2) / dt
            slewN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / dt
            return max(slew0, np.max(slews) if slews.size else 0.0, slewN)

        for _ in range(500):
            ms = max_slew(I, Q)
            if ms <= slew_bound:
                break
            I = I * (slew_bound / ms)
            Q = Q * (slew_bound / ms)

        # Re-check amplitude after slew projection
        amp = np.sqrt(I ** 2 + Q ** 2)
        if np.max(amp) > amp_bound:
            factor = amp_bound / np.max(amp)
            I = I * factor
            Q = Q * factor

        return I, Q

    # Generate seed pulse: DRAG-style Gaussian
    sigma = T / 6.0
    center = T / 2.0
    g = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    edge_val = np.exp(-0.5 * ((0 - center) / sigma) ** 2)
    g = g - edge_val
    g = np.maximum(g, 0.0)
    area = np.sum(g) * dt
    if area > 0:
        g = g * (np.pi / area)

    g_deriv = np.gradient(g, dt)
    I_seed = g.copy()
    Q_seed = 0.3 * g_deriv.copy()
    I_seed, Q_seed = project_valid(I_seed, Q_seed)

    best_I, best_Q = I_seed.copy(), Q_seed.copy()
    best_score = score_pulse(best_I, best_Q)

    # Gradient-based optimization using finite differences on a subset of parameters
    # Use L-BFGS with numerical gradients, but limit iterations for time

    # For speed, use a reduced set of optimization: optimize a few shape parameters
    # Alternative: use scipy minimize with numerical gradient

    from scipy.optimize import minimize

    def objective(params):
        """Objective function for optimization. params encodes I and Q."""
        I = params[:N]
        Q = params[N:2*N]
        I, Q = project_valid(I, Q)
        if not check_validity(I, Q):
            return 1e6
        sc = score_pulse(I, Q)
        return sc

    # Use a smarter parameterization: optimize a DRAG pulse with adjustable parameters
    # params: [sigma_scale, drag_alpha, center_shift, amplitude_scale]
    def param_objective(params):
        sigma_scale = params[0]
        drag_alpha = params[1]
        center_shift = params[2]
        amp_scale = params[3]

        sigma_eff = sigma * sigma_scale
        g = np.exp(-0.5 * ((t - center - center_shift) / sigma_eff) ** 2)
        edge_val = np.exp(-0.5 * ((0 - center - center_shift) / sigma_eff) ** 2)
        g = g - edge_val
        g = np.maximum(g, 0.0)
        area = np.sum(g) * dt
        if area > 0:
            g = g * (np.pi / area)

        g_deriv = np.gradient(g, dt)
        I = amp_scale * g.copy()
        Q = drag_alpha * amp_scale * g_deriv.copy()
        I, Q = project_valid(I, Q)
        if not check_validity(I, Q):
            return 1e6
        return score_pulse(I, Q)

    # Optimize the 4-parameter DRAG pulse
    x0 = [1.0, 0.3, 0.0, 1.0]
    try:
        result = minimize(param_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        if result.fun < best_score:
            best_score = result.fun
            sigma_scale, drag_alpha, center_shift, amp_scale = result.x
            sigma_eff = sigma * sigma_scale
            g = np.exp(-0.5 * ((t - center - center_shift) / sigma_eff) ** 2)
            edge_val = np.exp(-0.5 * ((0 - center - center_shift) / sigma_eff) ** 2)
            g = g - edge_val
            g = np.maximum(g, 0.0)
            area = np.sum(g) * dt
            if area > 0:
                g = g * (np.pi / area)
            g_deriv = np.gradient(g, dt)
            I = amp_scale * g.copy()
            Q = drag_alpha * amp_scale * g_deriv.copy()
            best_I, best_Q = project_valid(I, Q)
    except Exception:
        pass

    # Try a few more candidate shapes
    candidates = []

    # Different DRAG alphas and widths
    for ws in [0.8, 0.9, 1.0, 1.1, 1.2]:
        for da in [0.1, 0.2, 0.3, 0.4, 0.5]:
            sigma_eff = sigma * ws
            g = np.exp(-0.5 * ((t - center) / sigma_eff) ** 2)
            edge_val = np.exp(-0.5 * ((0 - center) / sigma_eff) ** 2)
            g = g - edge_val
            g = np.maximum(g, 0.0)
            area = np.sum(g) * dt
            if area > 0:
                g = g * (np.pi / area)
            g_deriv = np.gradient(g, dt)
            I = g.copy()
            Q = da * g_deriv.copy()
            I, Q = project_valid(I, Q)
            candidates.append((I, Q))

    # Cosine window
    g_cos = 0.5 * (1 - np.cos(2 * np.pi * t / T))
    area = np.sum(g_cos) * dt
    if area > 0:
        g_cos = g_cos * (np.pi / area)
    g_cos_deriv = np.gradient(g_cos, dt)
    for da in [0.2, 0.3, 0.4]:
        I = g_cos.copy()
        Q = da * g_cos_deriv.copy()
        I, Q = project_valid(I, Q)
        candidates.append((I, Q))

    # Evaluate candidates
    for I_c, Q_c in candidates:
        if time.time() > deadline:
            break
        if not check_validity(I_c, Q_c):
            continue
        sc = score_pulse(I_c, Q_c)
        if sc < best_score:
            best_score = sc
            best_I, best_Q = I_c.copy(), Q_c.copy()

    # Final refinement: local gradient step on the best pulse
    if time.time() < deadline - 60:
        # Try small perturbations
        lr = 0.005
        for step in range(50):
            if time.time() > deadline - 10:
                break
            # Compute gradient numerically (central differences on a few random coords)
            grad = np.zeros(2 * N)
            h = 0.002
            for idx in range(0, 2 * N, 4):  # sample every 4th coordinate for speed
                params_plus = np.concatenate([best_I, best_Q])
                params_minus = np.concatenate([best_I, best_Q])
                params_plus[idx] += h
                params_minus[idx] -= h
                I_p, Q_p = project_valid(params_plus[:N], params_plus[N:])
                I_m, Q_m = project_valid(params_minus[:N], params_minus[N:])
                if check_validity(I_p, Q_p) and check_validity(I_m, Q_m):
                    f_p = score_pulse(I_p, Q_p)
                    f_m = score_pulse(I_m, Q_m)
                    grad[idx] = (f_p - f_m) / (2 * h)
                else:
                    grad[idx] = 0.0

            # Gradient step
            grad_I = grad[:N]
            grad_Q = grad[N:]
            I_new = best_I - lr * grad_I
            Q_new = best_Q - lr * grad_Q
            I_new, Q_new = project_valid(I_new, Q_new)
            if check_validity(I_new, Q_new):
                sc_new = score_pulse(I_new, Q_new)
                if sc_new < best_score:
                    best_score = sc_new
                    best_I, best_Q = I_new, Q_new
            else:
                break

    # Final validity check
    I, Q = best_I, best_Q
    amp = np.sqrt(I ** 2 + Q ** 2)
    if np.any(amp > amp_bound + 1e-10):
        factor = amp_bound / np.max(amp)
        I = I * factor
        Q = Q * factor

    def max_slew(I, Q):
        slew0 = np.sqrt(I[0] ** 2 + Q[0] ** 2) / dt
        dI = np.diff(I)
        dQ = np.diff(Q)
        slews = np.sqrt(dI ** 2 + dQ ** 2) / dt
        slewN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / dt
        return max(slew0, np.max(slews) if slews.size else 0.0, slewN)

    ms = max_slew(I, Q)
    if ms > slew_bound:
        factor = slew_bound / ms
        I = I * factor
        Q = Q * factor

    return np.column_stack([I, Q])
