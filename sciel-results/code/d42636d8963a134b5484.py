import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    start = time.time()
    deadline = start + 315.0

    # ---- constants ----
    alpha = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    deltas = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]

    a_op = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    n_nm1 = n_op @ (n_op - np.eye(3))

    H0_nom = 0.5 * alpha * n_nm1
    driveI = 0.5 * (a_op + a_dag)
    driveQ = 0.5j * (a_op - a_dag)

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

    def get_H(I, Q, delta, scale):
        return H0_nom + delta * n_op + scale * (I * driveI + Q * driveQ)

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
        Hs = [get_H(I_Q[i, 0], I_Q[i, 1], delta, scale) for i in range(N)]
        Us = [expm(-1j * H * dt) for H in Hs]
        out = np.empty((6, 3), dtype=complex)
        for s in range(6):
            psi = states[s]
            for U in Us:
                psi = U @ psi
            out[s] = psi
        return out

    def score_case(I_Q, delta, scale):
        psi_f = propagate(I_Q, delta, scale)
        inf = np.array([1.0 - np.abs(np.vdot(targets[s], psi_f[s])) ** 2 for s in range(6)])
        return inf.mean() + inf.max()

    def score(I_Q):
        if not check_constraints(I_Q):
            return 1e6
        return float(np.mean([score_case(I_Q, d, sc) for d in deltas for sc in scales]))

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

    def seed_score():
        return float(np.mean([score_case(seed_pulse(), d, sc) for d in deltas for sc in scales]))

    best = seed_pulse()
    best_s = seed_score()

    # ---- coarse grid over DRAG family ----
    for A in [0.2, 0.25, 0.3, 0.35]:
        for sI in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
            for eta in [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]:
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
    for init in [[0.3, 5.0, -1.0, 5.0, 0.0, 0.0, 0.0],
                 [0.3, 6.0, -2.0, 5.0, 0.0, 0.0, 0.0],
                 [0.35, 4.0, 1.0, 6.0, 0.0, 0.0, 0.0]]:
        if time.time() > deadline - 120:
            break
        try:
            r = minimize(opt_drag, init, method='Nelder-Mead',
                         options={'maxiter': 120, 'xatol': 0.01, 'fatol': 1e-5})
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
                         options={'maxiter': 40, 'ftol': 1e-7, 'gtol': 1e-4,
                                  'maxfun': 400, 'maxls': 20})
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
