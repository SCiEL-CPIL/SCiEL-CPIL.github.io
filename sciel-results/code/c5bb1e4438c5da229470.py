"""Rewritten program for robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 300.0  # seconds wall-clock budget


def pulse(target_gate, hardware_model):
    """
    Generate a robust pulse for X_pi gate.
    Uses full 80-step time-domain pulse optimized via projected gradient descent.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")

    dt = 0.5  # ns
    n_steps = 80
    total_time = n_steps * dt  # 40 ns

    # Hardware parameters
    amp_bound = 0.4
    slew_bound = 0.12
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    T1 = 25000.0  # ns
    Tphi = 40000.0  # ns

    # Initial guess: DRAG-style Gaussian
    p0 = _make_initial_pulse(dt, n_steps)

    # Optimize
    p_opt = _optimize_pulse(p0, dt, n_steps, amp_bound, slew_bound,
                            anharmonicity, T1, Tphi)

    # Final constraint enforcement
    p_opt = _enforce_constraints(p_opt, amp_bound, slew_bound, dt, n_steps)

    return p_opt


def _make_initial_pulse(dt, n_steps):
    """Create initial DRAG-style Gaussian pulse."""
    t = np.linspace(0, n_steps * dt, n_steps, endpoint=False) + dt / 2
    t_centered = t - n_steps * dt / 2.0

    sigma = 4.5  # ns
    drag_alpha = 0.35

    I = np.exp(-t_centered**2 / (2 * sigma**2))
    area = np.sum(I) * dt
    I = I * (np.pi / area)

    dI = np.gradient(I, dt)
    Q = drag_alpha * dI

    p = np.column_stack([I, Q])
    return p


def _optimize_pulse(p0, dt, n_steps, amp_bound, slew_bound,
                    anharmonicity, T1, Tphi):
    """Optimize pulse via projected gradient descent."""
    p = p0.copy()
    amp_bound_sq = amp_bound**2
    slew_bound_sq = slew_bound**2

    # Precompute Liouvillian basis operators
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n_op = a_dag @ a
    I3 = np.eye(3, dtype=complex)
    L0 = np.kron(I3, I3)
    L1 = np.kron(I3, a)
    L2 = np.kron(I3, a_dag)
    L3 = np.kron(I3, n_op)
    L4 = np.kron(I3, n_op @ (n_op - I3))
    L5 = np.kron(I3, a + a_dag)
    L6 = np.kron(I3, 1j * (a - a_dag))
    L7 = np.kron(I3, a)
    L8 = np.kron(I3, n_op)

    # Precompute detuning/scale grid
    detunings = [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]

    def compute_score(pulse_arr):
        """Compute mean+max infidelity over 9 cases."""
        infidelities = []
        for det in detunings:
            for scale in scales:
                inf = _compute_infidelity(pulse_arr, det, scale, dt, n_steps,
                                          anharmonicity, T1, Tphi,
                                          L0, L1, L2, L3, L4, L5, L6, L7, L8)
                infidelities.append(inf)
        infidelities = np.array(infidelities)
        return np.mean(infidelities) + np.max(infidelities)

    def _compute_infidelity(pulse_arr, det, scale, dt, n_steps,
                            anharmonicity, T1, Tphi,
                            L0, L1, L2, L3, L4, L5, L6, L7, L8):
        """Compute mean infidelity over 5 test states for given det/scale."""
        # Precompute Liouvillian for this det/scale
        gamma1 = 1.0 / T1
        gphi = 2.0 / Tphi

        # Build static part of Liouvillian
        # H = det*n + 0.5*anh*n*(n-1) + 0.5*scale*(I*(a+a†) + iQ*(a-a†))
        # For each step, the drive part changes with I[i], Q[i]
        # Liouvillian: L(rho) = -i*[H, rho] + dissipation
        # In superoperator form:
        # -i*[H, rho] = -i*(H⊗I - I⊗H^T) rho_vec
        # Dissipation: gamma1*(a rho a† - 0.5{a†a, rho}) + gphi*(n rho n† - 0.5{n, rho})
        #   = gamma1*(a⊗a* - 0.5(a†a⊗I + I⊗(a†a)^T))
        #   + gphi*(n⊗n* - 0.5(n⊗I + I⊗n^T))

        H_static = det * n_op + 0.5 * anharmonicity * (n_op @ (n_op - I3))
        H_static_vec = H_static.ravel()

        # Static Liouvillian (without drive)
        # L_static = -i*(H⊗I - I⊗H^T) + diss
        # Using our precomputed operators:
        # -i*H⊗I part: we need to express H in terms of a, a†, n
        # H_static = det*n + 0.5*anh*(n^2 - n)
        # So -i*H_static⊗I = -i*det*L3 + 0.5i*anh*(L3⊗L3 - L3)
        # Wait, let me just compute directly

        # Direct computation of static Liouvillian
        L_static = np.zeros((9, 9), dtype=complex)
        # -i*[H_static, .]
        L_static -= 1j * (np.kron(H_static, I3) - np.kron(I3, H_static.T))
        # Dissipation
        L_static += gamma1 * (np.kron(a, a.conj()) - 0.5 * (np.kron(n_op, I3) + np.kron(I3, n_op.T)))
        L_static += gphi * (np.kron(n_op, n_op.conj()) - 0.5 * (np.kron(n_op, I3) + np.kron(I3, n_op.T)))

        # Drive Liouvillian coefficients (per step)
        # Drive H = 0.5*scale*(I*(a+a†) + iQ*(a-a†))
        # -i*[H_drive, .] = -0.5i*scale*(I*(a+a†)⊗I - I*(a+a†)⊗I^T + iQ*(a-a†)⊗I - iQ*(a-a†)⊗I^T)
        # = -0.5i*scale*I*(a+a†)⊗I + 0.5i*scale*I*(a+a†)⊗(a+a†)^T
        #   + 0.5*scale*Q*(a-a†)⊗I - 0.5*scale*Q*(a-a†)⊗(a-a†)^T
        # Since (a+a†)^T = a^T + a†^T and (a-a†)^T = a^T - a†^T
        # Let's precompute the drive superoperators
        # D_I = -0.5i*(a+a†)⊗I + 0.5i*(a+a†)⊗(a+a†)^T  (coefficient for I)
        # D_Q = 0.5*(a-a†)⊗I - 0.5*(a-a†)⊗(a-a†)^T  (coefficient for Q)
        # Actually, let me be more careful:
        # H_drive = 0.5*scale*(I_val*(a+a†) + i*Q_val*(a-a†))
        # -i*[H_drive, rho] = -i*H_drive*rho + i*rho*H_drive
        # = -0.5i*scale*I_val*(a+a†)*rho + 0.5i*scale*I_val*rho*(a+a†)
        #   + 0.5*scale*Q_val*(a-a†)*rho - 0.5*scale*Q_val*rho*(a-a†)
        # In vectorized form (column-major):
        # vec(A*rho*B) = (B^T ⊗ A) vec(rho)
        # So:
        # -0.5i*scale*I_val*(a+a†)*rho → -0.5i*scale*I_val * (I ⊗ (a+a†)) vec(rho)
        # 0.5i*scale*I_val*rho*(a+a†) → 0.5i*scale*I_val * ((a+a†)^T ⊗ I) vec(rho)
        # 0.5*scale*Q_val*(a-a†)*rho → 0.5*scale*Q_val * (I ⊗ (a-a†)) vec(rho)
        # -0.5*scale*Q_val*rho*(a-a†) → -0.5*scale*Q_val * ((a-a†)^T ⊗ I) vec(rho)

        D_I = -0.5j * (np.kron(I3, a + a_dag) - np.kron((a + a_dag).T, I3))
        D_Q = 0.5 * (np.kron(I3, a - a_dag) - np.kron((a - a_dag).T, I3))

        # Test states
        psi0 = np.array([1, 0, 0], dtype=complex)
        psi1 = np.array([0, 1, 0], dtype=complex)
        psi_01p = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
        psi_01m = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
        psi_0ip = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
        psi_0im = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)

        test_states = [psi0, psi1, psi_01p, psi_01m, psi_01p, psi_0im]
        # Actually need 5 states: |0>, |1>, (|0>+|1>)/sqrt2, (|0>-|1>)/sqrt2, (|0>+i|1>)/sqrt2
        # But the task says (|0> +- i|1>)/sqrt(2) which is 2 states, plus the others = 5 total?
        # Re-reading: "|0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
        # That's 1 + 1 + 2 + 2 = 6 states? Or does "+-" mean both signs?
        # I'll interpret as 5 states: |0>, |1>, |+>, |->, |+_i>
        # Actually let me re-read: "(|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
        # This likely means 4 states: (|0>+|1>)/√2, (|0>-|1>)/√2, (|0>+i|1>)/√2, (|0>-i|1>)/√2
        # Plus |0> and |1> = 6 total
        test_states = [psi0, psi1, psi_01p, psi_01m, psi_0ip, psi_0im]

        fidelities = []
        for psi in test_states:
            rho = np.outer(psi, psi.conj())
            rho_vec = rho.ravel()

            # Evolve
            for i in range(n_steps):
                L_total = L_static + scale * (pulse_arr[i, 0] * D_I + pulse_arr[i, 1] * D_Q)
                # Matrix exponential
                from scipy.linalg import expm
                U = expm(L_total * dt)
                rho_vec = U @ rho_vec

            # Final fidelity with X gate
            # X gate: |0> -> |1>, |1> -> |0>
            # So target state is X|psi>
            X_mat = np.array([[0, 1, 0],
                              [1, 0, 0],
                              [0, 0, 0]], dtype=complex)
            psi_target = X_mat @ psi
            rho_target = np.outer(psi_target, psi_target.conj())

            # Fidelity for pure target: F = <psi_target|rho_final|psi_target>
            rho_final = rho_vec.reshape(3, 3)
            fid = np.real(np.vdot(psi_target, rho_final @ psi_target))
            fidelities.append(fid)

        mean_fid = np.mean(fidelities)
        return 1.0 - mean_fid

    # Gradient descent with finite differences
    best_score = compute_score(p)
    best_p = p.copy()

    n_params = n_steps * 2
    lr = 0.01
    step_size = 1e-3

    # Deadline guard
    max_iter = 500
    for it in range(max_iter):
        if time.time() - _START_TIME > _DEADLINE - 30:
            break

        # Compute gradient via central differences
        grad = np.zeros(n_params)
        for j in range(n_params):
            p_plus = p.copy()
            p_minus = p.copy()
            idx = j % n_steps
            comp = j // n_steps
            p_plus[idx, comp] += step_size
            p_minus[idx, comp] -= step_size

            # Enforce constraints for gradient evaluation
            p_plus = _enforce_constraints(p_plus, amp_bound, slew_bound, dt, n_steps)
            p_minus = _enforce_constraints(p_minus, amp_bound, slew_bound, dt, n_steps)

            f_plus = compute_score(p_plus)
            f_minus = compute_score(p_minus)
            grad[j] = (f_plus - f_minus) / (2 * step_size)

        # Gradient step
        p_new = p - lr * grad

        # Project onto feasible set
        p_new = _project_to_feasible(p_new, amp_bound, slew_bound, dt, n_steps)

        score_new = compute_score(p_new)

        if score_new < best_score:
            best_score = score_new
            best_p = p_new.copy()
            p = p_new
            # Adaptive learning rate
            if it % 50 == 49 and it > 0:
                lr *= 0.9
        else:
            # Reduce learning rate
            lr *= 0.8
            p = best_p.copy()

        if it % 100 == 99:
            break

    return best_p


def _project_to_feasible(p, amp_bound, slew_bound, dt, n_steps):
    """Project pulse onto feasible set (amplitude and slew constraints)."""
    p = p.copy()

    # Enforce amplitude constraint
    amp = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    max_amp = np.max(amp)
    if max_amp > amp_bound:
        p *= amp_bound / max_amp

    # Enforce slew rate constraint
    max_slew = 0.0
    for i in range(n_steps):
        if i == 0:
            dp = np.linalg.norm(p[0]) / dt
        else:
            dp = np.linalg.norm(p[i] - p[i-1]) / dt
        max_slew = max(max_slew, dp)

    dp_last = np.linalg.norm(p[-1]) / dt
    max_slew = max(max_slew, dp_last)

    if max_slew > slew_bound:
        p *= slew_bound / max_slew

    return p


def _enforce_constraints(p, amp_bound, slew_bound, dt, n_steps):
    """Enforce amplitude and slew rate constraints on the pulse."""
    p = p.copy()

    # Enforce amplitude constraint
    amp = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    max_amp = np.max(amp)
    if max_amp > amp_bound:
        p *= amp_bound / max_amp

    # Enforce slew rate constraint
    max_slew = 0.0
    for i in range(n_steps):
        if i == 0:
            dp = np.linalg.norm(p[0]) / dt
        else:
            dp = np.linalg.norm(p[i] - p[i-1]) / dt
        max_slew = max(max_slew, dp)

    dp_last = np.linalg.norm(p[-1]) / dt
    max_slew = max(max_slew, dp_last)

    if max_slew > slew_bound:
        p *= slew_bound / max_slew

    return p
# EVOLVE-BLOCK-END
