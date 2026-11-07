import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    start = time.time()
    deadline = start + 318.0

    # ---- constants ----
    alpha = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    T1 = 25000.0
    Tphi = 40000.0
    deltas = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]

    # ---- operators ----
    a_op = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    n_nm1 = n_op @ (n_op - np.eye(3))

    H0 = 0.5 * alpha * n_nm1
    driveI = 0.5 * (a_op + a_dag)
    driveQ = 0.5j * (a_op - a_dag)

    # Lindblad operators for the Liouvillian (3x3 system)
    # L0 = a / sqrt(T1), L1 = n * sqrt(2/Tphi)
    L0 = a_op / np.sqrt(T1)
    L1 = n_op * np.sqrt(2.0 / Tphi)

    def get_H(I, Q, delta, scale):
        return H0 + delta * n_op + scale * (I * driveI + Q * driveQ)

    def build_liouvillian(H):
        """Build the Liouvillian superoperator in vectorized form for a 3x3 system."""
        # For a 3-level system, the Liouvillian acting on vec(rho) is:
        # L = -i (H ⊗ I - I ⊗ H^T) + sum_k (Lk ⊗ Lk* - 0.5 {Lk†Lk, .})
        # We use the standard form: d(rho)/dt = -i[H,rho] + sum_k (Lk rho Lk† - 0.5(Lk†Lk rho + rho Lk†Lk))
        # Vectorized: d(vec(rho))/dt = M @ vec(rho)
        I3 = np.eye(3, dtype=complex)
        # Coherent part: -i(H ⊗ I - I ⊗ H^T)
        M = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
        # Dissipative part
        for L in [L0, L1]:
            Ldag = L.conj().T
            LdagL = Ldag @ L
            # L rho L† -> kron(L, L†*)
            M += np.kron(L, L.conj().T)
            # -0.5 L†L rho -> -0.5 kron(I, L†L^T)
            M += -0.5 * np.kron(I3, LdagL.T)
            # -0.5 rho L†L -> -0.5 kron(L†L, I)
            M += -0.5 * np.kron(LdagL, I3)
        return M

    # Precompute the Liouvillian matrix for the nominal case (delta=0, scale=1)
    # But H depends on I,Q so we need to build it per step.
    # For speed, we'll compute expm of the Liouvillian per step.

    # ---- states and targets ----
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1j, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2),
    ]

    # For efficiency, precompute the Liouvillian evolution as a matrix acting on the state vector
    # Actually, since we're tracking pure states, we can use the unitary evolution plus a small correction
    # for dissipation. But the task says "exponentiates the Liouvillian", so we need to be careful.
    # For pure states, the Liouvillian evolution is equivalent to the Schrödinger equation with
    # an effective non-Hermitian Hamiltonian if we ignore the back-action. But to be safe, let's
    # use the full Liouvillian on the density matrix.
    #
    # However, for a pure state |psi>, rho = |psi><psi|. The Liouvillian maps rho to rho'.
    # We can track the state vector if the evolution preserves purity, but with dissipation it doesn't.
    # So we need to track the full density matrix.
    #
    # For efficiency, let's precompute the Liouvillian matrices for each step and then
    # exponentiate them. But this is expensive. Let's use the fact that for pure states,
    # the fidelity can be computed from the evolved state if we use the correct evolution.
    #
    # Actually, the task says "each step exponentiates the Liouvillian". This means we compute
    # U = expm(L * dt) where L is the Liouvillian, and apply it to the density matrix.
    #
    # For speed, let's precompute the Liouvillian for each step and each (delta, scale) combination,
    # then exponentiate. But that's 80 * 9 = 720 exponentiations, which is too slow.
    #
    # Alternative: use the fact that for small dt, we can use a first-order approximation.
    # But the task says "exponentiates", so we need to be accurate.
    #
    # Let's use a faster approach: for each step, compute the Liouvillian, exponentiate it,
    # and apply it to the density matrix. We'll do this for each of the 6 states and 9 (delta, scale)
    # combinations. That's 6 * 9 * 80 = 4320 exponentiations, which is way too slow.
    #
    # We need a faster approach. Let's use the fact that the Liouvillian is linear in I, Q, delta, scale,
    # and precompute the matrix elements. Then we can use a fast exponential of a 9x9 matrix.
    #
    # Actually, for a 3-level system, the Liouvillian is a 9x9 matrix. Exponentiating a 9x9 matrix
    # is fast. Let's benchmark: 4320 exponentiations of 9x9 matrices should take ~4320 * 0.1ms = 0.4s,
    # which is acceptable.

    def check_constraints(I_Q):
        amps = np.sqrt(I_Q[:, 0] ** 2 + I_Q[:, 1] ** 2)
        if np.max(amps) > amp_bound + 1e-9:
            return False
        p = np.vstack([np.zeros(2), I_Q, np.zeros(2)])
        diffs = np.diff(p, axis=0)
        if np.max(np.sqrt(np.sum(diffs ** 2, axis=1))) / dt > slew_bound + 1e-9:
            return False
        return True

    def propagate(I_Q, delta, scale):
        """Propagate all 6 states through the pulse and return final states."""
        # Precompute the Liouvillian evolution matrices for each step
        U_list = []
        for i in range(N):
            H = get_H(I_Q[i, 0], I_Q[i, 1], delta, scale)
            L = build_liouvillian(H)
            U = expm(L * dt)
            U_list.append(U)

        out = np.empty((6, 3), dtype=complex)
        for s in range(6):
            psi = states[s]
            rho = np.outer(psi, psi.conj())
            rho_vec = rho.flatten()
            for U in U_list:
                rho_vec = U @ rho_vec
            rho_final = rho_vec.reshape(3, 3)
            # For fidelity, we need the final state. Since the evolution is not unitary,
            # the final state may be mixed. We compute the fidelity as <target|rho_final|target>.
            # But the task says "fidelity to X over |0>, |1>, ...", which suggests we're
            # computing the fidelity of the final state with the target state.
            # For a mixed state, fidelity = <psi_target|rho_final|psi_target>.
            # But the task likely expects the state to remain close to pure, so we can
            # approximate by extracting the dominant eigenvector or just using the diagonal.
            # Actually, let's compute the fidelity properly.
            out[s] = rho_final
        return out

    def score_case(I_Q, delta, scale):
        rho_f = propagate(I_Q, delta, scale)
        inf = np.array([0.0] * 6)
        for s in range(6):
            rho = rho_f[s]
            target = targets[s]
            # Fidelity = <target|rho|target>
            f = np.vdot(target, rho @ target).real
            inf[s] = 1.0 - f
        return inf.mean() + inf.max()

    def score(I_Q):
        if not check_constraints(I_Q):
            return 1e6
        return float(np.mean([score_case(I_Q, d, sc) for d in deltas for sc in scales]))

    # ---- parametric DRAG family with correct Q sign ----
    def gen_drag(p):
        A, sI, eta, sQ, shQ, c, q0 = p
        t = np.linspace(0, 40, N)
        tc = 20.0
        I = A * np.exp(-(t - tc) ** 2 / (2 * sI ** 2))
        dI = -(t - tc) / sI ** 2 * I
        # Correct DRAG: Q = -eta * dI/dt + ...
        Q = -eta * dI + c * np.exp(-(t - tc - shQ) ** 2 / (2 * sQ ** 2)) + q0
        return np.column_stack((I, Q))

    def seed_pulse():
        s = 40.0 / 6.0
        t = np.linspace(0, 40, N)
        I = (np.sqrt(np.pi / 2) / s) * np.exp(-(t - 20.0) ** 2 / (2 * s ** 2))
        return np.column_stack((I, np.zeros(N)))

    def seed_score():
        return float(np.mean([score_case(seed_pulse(), d, sc) for d in deltas for sc in scales]))

    best = seed_pulse()
    best_s = seed_score()

    # ---- coarse grid over DRAG family with correct sign ----
    for A in [0.2, 0.25, 0.3, 0.35]:
        for sI in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
            for eta in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
                if time.time() > deadline - 150:
                    break
                I_Q = gen_drag([A, sI, eta, 5.0, 0.0, 0.0, 0.0])
                if not check_constraints(I_Q):
                    continue
                s = score(I_Q)
                if s < best_s:
                    best_s, best = s, I_Q

    # ---- refine the parametric family ----
    def opt_drag(p):
        if time.time() > deadline - 120:
            return 1e6
        I_Q = gen_drag(p)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    best_p = None
    for init in [[0.3, 5.0, 2.0, 5.0, 0.0, 0.0, 0.0],
                 [0.3, 6.0, 3.0, 5.0, 0.0, 0.0, 0.0],
                 [0.35, 4.0, 1.0, 6.0, 0.0, 0.0, 0.0],
                 [0.25, 7.0, 4.0, 5.0, 0.0, 0.0, 0.0]]:
        if time.time() > deadline - 120:
            break
        try:
            r = minimize(opt_drag, init, method='Nelder-Mead',
                         options={'maxiter': 150, 'xatol': 0.01, 'fatol': 1e-5})
            if r.fun < best_s:
                best_s, best, best_p = r.fun, gen_drag(r.x), r.x
        except Exception:
            pass

    # ---- direct 80-step refinement (raw amplitudes) ----
    def opt_direct(x):
        if time.time() > deadline - 25:
            return 1e6
        I_Q = x.reshape(N, 2)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    best_x = best.reshape(-1)
    for k in range(3):
        if time.time() > deadline - 25:
            break
        try:
            r = minimize(opt_direct, best_x, method='L-BFGS-B',
                         options={'maxiter': 50, 'ftol': 1e-7, 'gtol': 1e-4,
                                  'maxfun': 500, 'maxls': 20})
            I_Q = r.x.reshape(N, 2)
            if check_constraints(I_Q):
                s = score(I_Q)
                if s < best_s:
                    best_s, best, best_x = s, I_Q, r.x
            else:
                break
        except Exception:
            break

    if not check_constraints(best):
        best = seed_pulse()
    return best
