import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    start = time.time()
    deadline = start + 310.0

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

    a_op = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    n_nm1 = n_op @ (n_op - np.eye(3))

    H0_nom = 0.5 * alpha * n_nm1
    driveI = 0.5 * (a_op + a_dag)
    driveQ = 0.5j * (a_op - a_dag)

    # Lindblad terms
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)

    d = 3
    eye_d = np.eye(d)

    # Precompute incoherent Liouvillian part (constant)
    incoh_L = np.zeros((d*d, d*d), dtype=complex)
    for Lk in [L1, L2]:
        Lkd = Lk.conj().T
        incoh_L += np.kron(Lk.conj(), Lk)
        LkdLk = Lkd @ Lk
        incoh_L += -0.5 * np.kron(eye_d, LkdLk)
        incoh_L += -0.5 * np.kron(LkdLk.T, eye_d)

    def get_H(I, Q, delta, scale):
        return H0_nom + delta * n_op + scale * (I * driveI + Q * driveQ)

    def build_Liouvillian(H):
        """Build the Liouvillian superoperator for a given Hamiltonian."""
        coh_L = -1j * (np.kron(H, eye_d) - np.kron(eye_d, H.T))
        return coh_L + incoh_L

    def check_constraints(I_Q):
        amps = np.sqrt(I_Q[:, 0] ** 2 + I_Q[:, 1] ** 2)
        if np.max(amps) > amp_bound + 1e-9:
            return False
        p = np.vstack([np.zeros(2), I_Q, np.zeros(2)])
        diffs = np.diff(p, axis=0)
        if np.max(np.sqrt(np.sum(diffs ** 2, axis=1))) / dt > slew_bound + 1e-9:
            return False
        return True

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

    # Precompute projector matrices for fidelity
    # fidelity = <psi_t| rho_f |psi_t>
    # In vec form: Tr(P rho) where P = |psi_t><psi_t|
    # Tr(P rho) = vec(P^T)^T vec(rho) for vec column-major
    # Actually: Tr(A B) = vec(A^T)^T vec(B)
    # So fidelity = vec(P^T)^T vec(rho_f)
    # But we need real part, and P is Hermitian so P^T = P.conj()
    # Let's just use: f = Re( sum_{ij} P_{ji} rho_{ij} ) = Re( vec(P^T)^T vec(rho) )
    # For Hermitian P: P^T = P.conj(), so vec(P^T) = vec(P.conj())
    # Simpler: f = Re( np.vdot(psi_t, rho @ psi_t) )
    # In vectorized form for multiple states at once:
    # f_s = Re( psi_s^dagger rho psi_s )
    # We can precompute: for each target state, the linear functional
    # f_s = Re( sum_{i,j} conj(psi_s[i]) rho[i,j] psi_s[j] )
    # = Re( vec(conj(psi_s) psi_s^T)^T vec(rho) )  -- this is getting complex
    # Let's just precompute the 6x9 coefficient matrices

    # For each of 6 initial states and 9 (delta, scale) combos, we propagate
    # and compute fidelity. Let's vectorize over the 6 states.
    # rho_s = |psi_s><psi_s|, vec(rho_s)
    # After propagation: vec(rho_f) = U vec(rho_s)
    # fidelity = Re( <target_s | rho_f | target_s> )
    # = Re( sum_{i,j} conj(target_s[i]) (rho_f)_{ij} target_s[j] )
    # = Re( vec(M_s)^T vec(rho_f) ) where M_s[i,j] = conj(target_s[i]) * target_s[j]
    # M_s = np.outer(target_s.conj(), target_s)
    # vec(M_s) in column-major: M_s.flatten()

    # Precompute M matrices
    M_mats = []
    for s in range(6):
        M = np.outer(targets[s].conj(), targets[s])
        M_mats.append(M.flatten())  # column-major vec
    M_mats = np.array(M_mats)  # shape (6, 9)

    def propagate_all(I_Q, delta, scale):
        """Propagate all 6 states and return fidelities."""
        # Build superoperators for each step
        Us = []
        for i in range(N):
            H = get_H(I_Q[i, 0], I_Q[i, 1], delta, scale)
            L = build_Liouvillian(H)
            U = expm(L * dt)
            Us.append(U)

        # Vectorize: stack all 6 initial density matrices
        # rho_s = outer(states[s], states[s].conj())
        rho_vecs = np.array([np.outer(states[s], states[s].conj()).flatten() for s in range(6)])
        # shape (6, 9)

        # Apply all superoperators
        for U in Us:
            rho_vecs = (U @ rho_vecs.T).T

        # Compute fidelities
        # f_s = Re( M_mats[s] @ rho_vecs[s] )
        fids = np.real(np.sum(M_mats * rho_vecs, axis=1))
        return fids

    def score_case(I_Q, delta, scale):
        res = propagate_all(I_Q, delta, scale)
        infs = 1.0 - res
        return infs.mean() + infs.max()

    def score(I_Q):
        if not check_constraints(I_Q):
            return 1e6
        total = 0.0
        for d in deltas:
            for sc in scales:
                total += score_case(I_Q, d, sc)
        return total / 9.0

    # ---- parametric DRAG family ----
    def gen_drag(p):
        A, sI, eta, sQ, shQ, c, q0 = p
        t = np.linspace(0, 40, N)
        tc = 20.0
        I = A * np.exp(-(t - tc) ** 2 / (2 * sI ** 2))
        dI = -(t - tc) / sI ** 2 * I
        Q = eta * dI + c * np.exp(-(t - tc - shQ) ** 2 / (2 * sQ ** 2)) + q0
        return np.column_stack((I, Q))

    def seed_pulse():
        s = 40.0 / 6.0
        t = np.linspace(0, 40, N)
        I = (np.sqrt(np.pi / 2) / s) * np.exp(-(t - 20.0) ** 2 / (2 * s ** 2))
        return np.column_stack((I, np.zeros(N)))

    best = seed_pulse()
    best_s = score(best)

    # ---- coarse grid over DRAG family ----
    for A in [0.2, 0.25, 0.3, 0.35, 0.38]:
        for sI in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
            for eta in [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]:
                if time.time() > deadline - 100:
                    break
                I_Q = gen_drag([A, sI, eta, 5.0, 0.0, 0.0, 0.0])
                if not check_constraints(I_Q):
                    continue
                s = score(I_Q)
                if s < best_s:
                    best_s, best = s, I_Q

    # ---- refine the parametric family ----
    def opt_drag(p):
        if time.time() > deadline - 80:
            return 1e6
        I_Q = gen_drag(p)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    for init in [[0.3, 5.0, -1.0, 5.0, 0.0, 0.0, 0.0],
                 [0.3, 6.0, -2.0, 5.0, 0.0, 0.0, 0.0],
                 [0.35, 4.0, 1.0, 6.0, 0.0, 0.0, 0.0],
                 [0.38, 5.5, -1.5, 5.0, 0.0, 0.0, 0.0]]:
        if time.time() > deadline - 80:
            break
        try:
            r = minimize(opt_drag, init, method='Nelder-Mead',
                         options={'maxiter': 150, 'xatol': 0.01, 'fatol': 1e-6})
            if r.fun < best_s:
                best_s, best = r.fun, gen_drag(r.x)
        except Exception:
            pass

    # ---- direct 80-step refinement (raw amplitudes) ----
    def opt_direct(x):
        if time.time() > deadline - 20:
            return 1e6
        I_Q = x.reshape(N, 2)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    best_x = best.reshape(-1)
    for k in range(5):
        if time.time() > deadline - 20:
            break
        try:
            r = minimize(opt_direct, best_x, method='L-BFGS-B',
                         options={'maxiter': 60, 'ftol': 1e-8, 'gtol': 1e-5,
                                  'maxfun': 600, 'maxls': 30})
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
