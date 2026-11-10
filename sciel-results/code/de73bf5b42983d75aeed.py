import numpy as np
import scipy.linalg as sla
import scipy.optimize as opt
import time
import math

def pulse(target_gate, hardware_model):
    """
    Returns an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    Strategy: DRAG pulse with optimized sigma, k, and amplitude scale, plus
    fast 3-level Liouvillian simulation for scoring.
    """
    start_time = time.time()
    deadline = start_time + 310.0  # safety margin

    dt = 0.5
    n_steps = 80
    T = 40.0

    # Hardware parameters
    chi = -2 * math.pi * 0.25
    T1 = 25000.0
    Tphi = 40000.0

    # Operators (3 levels)
    a = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, math.sqrt(2), 0]
    ], dtype=complex)
    a_dag = a.conj().T
    n_op = a_dag @ a
    n_nm1 = n_op @ (n_op - np.eye(3))
    eye3 = np.eye(3)

    # Pre-compute dissipator superoperators (constant)
    L1 = a / math.sqrt(T1)
    L1_dag_L1 = L1.conj().T @ L1
    M_L1 = np.kron(L1.conj().T.T, L1) - 0.5 * np.kron(eye3, L1_dag_L1) - 0.5 * np.kron(L1_dag_L1.T, eye3)

    L2 = n_op * math.sqrt(2.0 / Tphi)
    L2_dag_L2 = L2.conj().T @ L2
    M_L2 = np.kron(L2.conj().T.T, L2) - 0.5 * np.kron(eye3, L2_dag_L2) - 0.5 * np.kron(L2_dag_L2.T, eye3)
    M_diss = M_L1 + M_L2

    # Test states and targets
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_p = (psi_0 + psi_1) / math.sqrt(2)
    psi_m = (psi_0 - psi_1) / math.sqrt(2)
    psi_ip = (psi_0 + 1j*psi_1) / math.sqrt(2)
    psi_im = (psi_0 - 1j*psi_1) / math.sqrt(2)

    test_states = [psi_0, psi_1, psi_p, psi_m, psi_ip, psi_im]
    targets = [psi_1, psi_0, (psi_1 + psi_0)/math.sqrt(2), (-psi_1 + psi_0)/math.sqrt(2),
               (1j*psi_1 + psi_0)/math.sqrt(2), (-1j*psi_1 + psi_0)/math.sqrt(2)]

    # Pre-compute target projectors as vectors
    target_projectors = [np.outer(psi_t, psi_t.conj()).flatten() for psi_t in targets]
    initial_states = [np.outer(psi, psi.conj()).flatten() for psi in test_states]

    def get_hamiltonian(detuning, scale, I_t, Q_t):
        H_static = detuning * n_op + 0.5 * chi * n_nm1
        H_drive = 0.5 * scale * (I_t * (a + a_dag) + 1j * Q_t * (a - a_dag))
        return H_static + H_drive

    def get_M_H(H_total):
        return -1j * (np.kron(eye3, H_total) - np.kron(H_total.T, eye3))

    def is_valid(I, Q):
        amps = np.sqrt(I**2 + Q**2)
        if np.max(amps) > 0.4 + 1e-9:
            return False
        if math.sqrt(I[0]**2 + Q[0]**2) / dt > 0.12 + 1e-9:
            return False
        if math.sqrt(I[-1]**2 + Q[-1]**2) / dt > 0.12 + 1e-9:
            return False
        dI = np.diff(I)
        dQ = np.diff(Q)
        slews = np.sqrt(dI**2 + dQ**2) / dt
        if np.max(slews) > 0.12 + 1e-9:
            return False
        return True

    def parametric_pulse(sigma, k, amp_scale):
        t = np.arange(n_steps) * dt
        tc = T / 2.0
        gauss = np.exp(-(t - tc)**2 / (2 * sigma**2))
        # Offset to 0 at edges
        gauss = gauss - gauss[0] - gauss[-1] * (np.arange(n_steps) / (n_steps - 1))
        gauss = np.maximum(gauss, 0)
        area_sum = np.sum(gauss) * dt
        if area_sum < 1e-10:
            return np.zeros(n_steps), np.zeros(n_steps)
        A = math.pi * amp_scale / area_sum
        I = A * gauss
        dI_dt = np.gradient(I, dt)
        Q = -k * dI_dt
        return I, Q

    def simulate_all(I, Q, detunings, scales):
        """Simulate all 9 corners and return mean infidelity and max infidelity."""
        total_infid = 0.0
        max_infid_all = 0.0
        count = 0

        for det in detunings:
            for sc in scales:
                corner_min_fid = 1.0
                for init_vec, target_proj in zip(initial_states, target_projectors):
                    rho_vec = init_vec.copy()
                    for t_idx in range(n_steps):
                        H_total = get_hamiltonian(det, sc, I[t_idx], Q[t_idx])
                        M_H = get_M_H(H_total)
                        M_total = M_H + M_diss
                        M_exp = sla.expm(M_total * dt)
                        rho_vec = M_exp @ rho_vec
                    rho_final = rho_vec.reshape(3, 3)
                    fidelity = np.real(np.trace(target_proj.reshape(3, 3).conj().T @ rho_final))
                    fidelity = max(0.0, min(1.0, fidelity))
                    if fidelity < corner_min_fid:
                        corner_min_fid = fidelity

                infid = 1.0 - corner_min_fid
                total_infid += infid
                if infid > max_infid_all:
                    max_infid_all = infid
                count += 1

        mean_infid = total_infid / count
        return mean_infid, max_infid_all

    def objective(params):
        sigma, k, amp_scale = params
        I, Q = parametric_pulse(sigma, k, amp_scale)

        if not is_valid(I, Q):
            return 1e6

        detunings = [-2 * math.pi * 0.002, 0, 2 * math.pi * 0.002]
        scales = [0.97, 1.0, 1.03]

        mean_infid, max_infid_all = simulate_all(I, Q, detunings, scales)
        return mean_infid + max_infid_all

    # Initial guess
    sigma0 = 40.0 / 6.0
    k0 = 0.1
    amp0 = 1.0

    x0 = np.array([sigma0, k0, amp0])
    bounds = [(3.0, 10.0), (-0.5, 0.5), (0.5, 1.5)]

    best_score = float('inf')
    best_params = x0
    best_I, best_Q = parametric_pulse(sigma0, k0, amp0)

    # Evaluate initial
    s = objective(x0)
    if s < best_score:
        best_score = s
        best_I, best_Q = parametric_pulse(sigma0, k0, amp0)

    # Check deadline
    if time.time() > deadline - 30:
        return np.column_stack((best_I, best_Q))

    # Nelder-Mead optimization
    try:
        result = opt.minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-8,
                                       'adaptive': True})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
            best_I, best_Q = parametric_pulse(result.x[0], result.x[1], result.x[2])
    except Exception:
        pass

    # Check deadline
    if time.time() > deadline - 15:
        return np.column_stack((best_I, best_Q))

    # Try a second round with different initial guesses
    guesses = [
        [4.0, 0.15, 1.0],
        [5.0, 0.2, 1.1],
        [6.0, 0.1, 0.95],
        [4.5, 0.25, 1.0],
        [5.5, 0.15, 1.05],
    ]

    for guess in guesses:
        if time.time() > deadline - 10:
            break
        try:
            result = opt.minimize(objective, guess, method='Nelder-Mead',
                                  options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-7})
            if result.fun < best_score:
                best_score = result.fun
                best_I, best_Q = parametric_pulse(result.x[0], result.x[1], result.x[2])
        except Exception:
            pass

    return np.column_stack((best_I, best_Q))
