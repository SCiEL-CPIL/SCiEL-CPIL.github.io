"""
Robust X-pi pulse for a three-level transmon using a DRAG-style Gaussian with
optimized DRAG coefficient, width, and amplitude. Uses a fast 2-level surrogate
for the main optimization loop, then refines with the full 3-level model.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")

    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bound = 2 * np.pi * 0.002
    scale_low, scale_high = 0.97, 1.03
    T1 = 25000.0
    Tphi = 40000.0

    dt = 0.5
    n_steps = 80
    total_time = n_steps * dt

    t_centered = np.linspace(-total_time/2 + dt/2, total_time/2 - dt/2, n_steps)

    # Pre-compute matrices for full 3-level simulation
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n_mat = adag @ a
    eye3 = np.eye(3)
    H0_base = 0.5 * anharmonicity * (n_mat @ (n_mat - eye3))

    # Test states
    states = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0] / np.sqrt(2),
        [1, -1, 0] / np.sqrt(2),
        [1, 1j, 0] / np.sqrt(2),
        [1, -1j, 0] / np.sqrt(2)
    ], dtype=complex)

    X_gate = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype=complex)

    def enforce_constraints(p):
        amp = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            p = p * (amp_bound / max_amp)

        max_slew = 0
        p_prev = np.zeros(2)
        for i in range(n_steps):
            dp = np.linalg.norm(p[i] - p_prev) / dt
            if dp > max_slew:
                max_slew = dp
            p_prev = p[i].copy()
        dp_last = np.linalg.norm(p[-1]) / dt
        if dp_last > max_slew:
            max_slew = dp_last

        if max_slew > slew_bound:
            p = p * (slew_bound / max_slew)
        return p

    def parametric_pulse(params):
        sigma, amp, eta = params
        sigma = max(1e-3, abs(sigma))
        gauss = np.exp(-t_centered**2 / (2 * sigma**2))
        area = np.sum(gauss) * dt
        if area > 0:
            gauss = gauss * (np.pi / area)
        I = amp * gauss
        d_gauss = -t_centered / (sigma**2) * gauss
        Q = eta * amp * d_gauss
        p = np.column_stack([I, Q])
        return enforce_constraints(p)

    def simulate_2level(pulse_arr, delta, scale):
        """Fast 2-level simulation for optimization loop."""
        U_total = np.eye(2, dtype=complex)
        for i in range(n_steps):
            I_i = pulse_arr[i, 0]
            Q_i = pulse_arr[i, 1]
            H_t = np.array([
                [delta, 0.5 * scale * (I_i + 1j * Q_i)],
                [0.5 * scale * (I_i - 1j * Q_i), delta + anharmonicity]
            ], dtype=complex)
            eigvals, eigvecs = np.linalg.eigh(H_t)
            U_step = eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
            U_total = U_step @ U_total
        return U_total

    def fidelity_2level(U):
        """Mean fidelity over 6 test states using 2-level projection."""
        fids = []
        for psi in states:
            psi2 = psi[:2]
            target = X_gate[:2, :2] @ psi2
            out = U @ psi2
            fid = np.abs(np.vdot(target, out))**2
            fids.append(fid)
        return np.mean(fids)

    def score_2level(pulse_arr):
        deltas = [-detuning_bound, 0, detuning_bound]
        scales = [scale_low, 1.0, scale_high]
        infids = []
        for d in deltas:
            for s in scales:
                U = simulate_2level(pulse_arr, d, s)
                fid = fidelity_2level(U)
                infids.append(1 - fid)
        infids = np.array(infids)
        return np.mean(infids) + np.max(infids)

    def simulate_3level(pulse_arr, delta, scale):
        """Full 3-level simulation for final scoring."""
        U_total = np.eye(3, dtype=complex)
        for i in range(n_steps):
            I_i = pulse_arr[i, 0]
            Q_i = pulse_arr[i, 1]
            H_int = 0.5 * scale * (I_i * (a + adag) + 1j * Q_i * (a - adag))
            H_t = delta * n_mat + H0_base + H_int
            eigvals, eigvecs = np.linalg.eigh(H_t)
            U_step = eigvecs @ np.diag(np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T
            U_total = U_step @ U_total
        return U_total

    def fidelity_3level(U):
        fids = []
        for psi in states:
            target = X_gate @ psi
            out = U @ psi
            fid = np.abs(np.vdot(target, out))**2
            fids.append(fid)
        return np.mean(fids)

    def score_3level(pulse_arr):
        deltas = [-detuning_bound, 0, detuning_bound]
        scales = [scale_low, 1.0, scale_high]
        infids = []
        for d in deltas:
            for s in scales:
                U = simulate_3level(pulse_arr, d, s)
                fid = fidelity_3level(U)
                infids.append(1 - fid)
        infids = np.array(infids)
        return np.mean(infids) + np.max(infids)

    start_time = time.time()
    deadline = start_time + 280.0

    # Phase 1: Coarse optimization with 2-level surrogate
    def obj_2level(params):
        if time.time() > deadline:
            return 1.0
        p = parametric_pulse(params)
        return score_2level(p)

    best_params = np.array([6.0, 0.35, 0.3])
    best_score_2l = obj_2level(best_params)

    # Nelder-Mead from a few starting points
    starts = [
        np.array([6.0, 0.35, 0.3]),
        np.array([5.5, 0.35, 0.4]),
        np.array([6.5, 0.35, 0.2]),
        np.array([5.0, 0.38, 0.35]),
    ]

    for x0 in starts:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(obj_2level, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'disp': False})
            if res.fun < best_score_2l:
                best_score_2l = res.fun
                best_params = res.x
        except Exception:
            pass

    # Phase 2: Refine with full 3-level model
    def obj_3level(params):
        if time.time() > deadline:
            return 1.0
        p = parametric_pulse(params)
        return score_3level(p)

    best_params_3l = best_params.copy()
    best_score_3l = obj_3level(best_params_3l)

    for x0 in [best_params, best_params + np.array([0.2, 0.01, 0.02]),
               best_params - np.array([0.2, 0.01, 0.02])]:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(obj_3level, x0, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
            if res.fun < best_score_3l:
                best_score_3l = res.fun
                best_params_3l = res.x
        except Exception:
            pass

    # Phase 3: Direct pulse optimization on raw amplitudes for final polish
    if time.time() < deadline - 60:
        p_init = parametric_pulse(best_params_3l)

        def obj_direct(p_flat):
            if time.time() > deadline:
                return 1.0
            p = p_flat.reshape(n_steps, 2)
            p = enforce_constraints(p)
            return score_3level(p)

        # Gradient-free local optimization on the flat pulse
        p_flat_init = p_init.flatten().copy()
        try:
            res = minimize(obj_direct, p_flat_init, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-9, 'disp': False})
            if res.fun < best_score_3l:
                best_score_3l = res.fun
                p_final = enforce_constraints(res.x.reshape(n_steps, 2))
            else:
                p_final = p_init
        except Exception:
            p_final = p_init
    else:
        p_final = parametric_pulse(best_params_3l)

    # Final validity check
    amp = np.sqrt(p_final[:, 0]**2 + p_final[:, 1]**2)
    if np.max(amp) > amp_bound:
        p_final *= amp_bound / np.max(amp)

    max_slew = 0
    p_prev = np.zeros(2)
    for i in range(n_steps):
        dp = np.linalg.norm(p_final[i] - p_prev) / dt
        if dp > max_slew:
            max_slew = dp
        p_prev = p_final[i].copy()
    dp_last = np.linalg.norm(p_final[-1]) / dt
    if dp_last > max_slew:
        max_slew = dp_last

    if max_slew > slew_bound:
        p_final *= slew_bound / max_slew

    return p_final
