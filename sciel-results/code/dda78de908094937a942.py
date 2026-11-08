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
    L1dag = L1.conj().T
    L2dag = L2.conj().T

    d = 3
    eye_d = np.eye(d)

    # Precompute incoherent Liouvillian part
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
        HkronI = np.kron(H, eye_d)
        IkronHt = np.kron(eye_d, H.T)
        return -1j * (HkronI - IkronHt) + incoh_L

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
        Us = []
        for i in range(N):
            H = get_H(I_Q[i, 0], I_Q[i, 1], delta, scale)
            L = build_Liouvillian(H)
            U = expm(L * dt)
            Us.append(U)
        
        out = np.empty(6, dtype=float)
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
        for s in range(6):
            rho = np.outer(states[s], states[s].conj())
            vec_rho = rho.flatten()
            for U in Us:
                vec_rho = U @ vec_rho
            rho_f = vec_rho.reshape(d, d)
            psi_t = targets[s]
            f = np.vdot(psi_t, rho_f @ psi_t).real
            out[s] = f
        return out

    def score_case(I_Q, delta, scale):
        res = propagate(I_Q, delta, scale)
        infs = 1.0 - res
        return infs.mean() + infs.max()

    def score(I_Q):
        if not check_constraints(I_Q):
            return 1e6
        return float(np.mean([score_case(I_Q, d, sc) for d in deltas for sc in scales]))

    # ---- Seed pulse ----
    def seed_pulse():
        s = 40.0 / 6.0
        t = np.linspace(0, 40, N)
        I = (np.sqrt(np.pi / 2) / s) * np.exp(-(t - 20.0) ** 2 / (2 * s ** 2))
        return np.column_stack((I, np.zeros(N)))

    best = seed_pulse()
    best_s = score(best)

    # ---- Linearized DRAG optimization via closed-form least squares ----
    # Basis: [Gaussian, dGaussian/dt, shifted Gaussian, constant] for I and Q
    # Pulse: I = c1*g + c2*dg + c3*gs, Q = c4*gs2 + c5
    # where g = exp(-(t-tc)^2/(2*s^2)), dg = -d/dt g, gs = exp(-(t-tc-sh)^2/(2*sq^2))
    
    t = np.linspace(0, 40, N)
    tc = 20.0
    s_base = 40.0 / 6.0
    
    # Basis functions
    g = np.exp(-(t - tc) ** 2 / (2 * s_base ** 2))
    dg = -(t - tc) / s_base ** 2 * g  # derivative of g
    
    # For Q channel, use a shifted Gaussian and constant
    sh = 0.0
    sq = s_base
    gs_q = np.exp(-(t - tc - sh) ** 2 / (2 * sq ** 2))
    ones = np.ones(N)
    
    # 6 coefficients: c1,c2,c3 for I; c4,c5 for Q (c6 unused, keep simple)
    # I = c1*g + c2*dg + c3*gs_i, Q = c4*gs_q + c5
    gs_i = g  # reuse same shape for I shifted term
    
    # Build basis matrix: 80x6
    B = np.column_stack([g, dg, gs_i, np.zeros(N), gs_q, ones])
    
    # Compute first-order error operator for each basis element and each case
    # Error: deviation from ideal X gate
    # We'll compute the effect of each basis element on the 9 cases
    
    # First, get the nominal propagation for the seed to establish baseline
    seed_IQ = seed_pulse()
    
    # For linearized approach: compute how each basis perturbation affects fidelity
    # Use central difference with small perturbation epsilon
    eps = 0.01
    
    # Error vectors: for each of 6 basis functions, for each of 9 cases, 
    # compute the change in (mean_inf + max_inf)
    
    num_cases = 9
    num_basis = 6
    
    # Precompute all case (delta, scale) pairs
    case_list = [(d, sc) for d in deltas for sc in scales]
    
    # Compute baseline scores for each case
    base_scores = np.zeros(num_cases)
    base_infs = np.zeros((num_cases, 6))
    for ci, (dl, sc) in enumerate(case_list):
        fids = propagate(seed_IQ, dl, sc)
        infs = 1.0 - fids
        base_infs[ci] = infs
        base_scores[ci] = infs.mean() + infs.max()
    
    base_total = np.mean(base_scores)
    
    # For each basis function, perturb the pulse and compute error change
    # J[b, c] = (score_with_perturbation_b - base_total) / eps
    J = np.zeros((num_basis, num_cases))
    
    for b in range(num_basis):
        perturbed = seed_IQ.copy()
        perturbed[:, 0] += eps * B[:, b] if b < 3 else perturbed[:, 0]
        perturbed[:, 1] += eps * B[:, b] if b >= 3 else perturbed[:, 1]
        
        # Actually, let's be more careful:
        # b=0: I += eps*g, b=1: I += eps*dg, b=2: I += eps*gs_i
        # b=3: Q += eps*gs_q, b=4: Q += eps*ones, b=5: unused
        perturbed = seed_IQ.copy()
        if b == 0:
            perturbed[:, 0] += eps * g
        elif b == 1:
            perturbed[:, 0] += eps * dg
        elif b == 2:
            perturbed[:, 0] += eps * gs_i
        elif b == 3:
            perturbed[:, 1] += eps * gs_q
        elif b == 4:
            perturbed[:, 1] += eps * ones
        
        if not check_constraints(perturbed):
            J[b, :] = 1e6
            continue
        
        scores_b = np.zeros(num_cases)
        for ci, (dl, sc) in enumerate(case_list):
            fids = propagate(perturbed, dl, sc)
            infs = 1.0 - fids
            scores_b[ci] = infs.mean() + infs.max()
        
        J[b, :] = (np.mean(scores_b) - base_total) / eps
    
    # Now solve: minimize ||J @ c||^2 subject to constraints
    # Use least squares with regularization
    reg = 1e-4
    A_mat = J.T @ J + reg * np.eye(num_basis)
    b_vec = -J.T @ base_scores  # gradient direction
    
    # Actually, we want to minimize total score, so we solve:
    # (J^T J + reg*I) c = -J^T * (something)
    # Simpler: use scipy lstsq on the full problem
    
    # Let's just do a proper least squares: minimize sum_c (base_score_c + J[:,c] @ c)^2
    # This is: min_c ||A c + b||^2 where A = J, b = base_scores
    # Solution: c = -pinv(J) @ base_scores (for the linear part)
    
    # But we also need to handle the nonlinear constraint. Let's try:
    try:
        c_sol = np.linalg.lstsq(J, -base_scores, rcond=None)[0]
    except:
        c_sol = np.zeros(num_basis)
    
    # Build candidate pulse
    candidate = seed_IQ.copy()
    for b in range(5):  # skip b=5
        if b < 3:
            candidate[:, 0] += c_sol[b] * B[:, b]
        else:
            candidate[:, 1] += c_sol[b] * B[:, b]
    
    if check_constraints(candidate):
        cand_s = score(candidate)
        if cand_s < best_s:
            best = candidate
            best_s = cand_s
    
    # ---- Fallback: if linearized didn't help, do a quick parametric search ----
    if time.time() < deadline - 100:
        def gen_drag(p):
            A, sI, eta, sQ, shQ, c, q0 = p
            t_local = np.linspace(0, 40, N)
            tc_local = 20.0
            I = A * np.exp(-(t_local - tc_local) ** 2 / (2 * sI ** 2))
            dI = -(t_local - tc_local) / sI ** 2 * I
            Q = eta * dI + c * np.exp(-(t_local - tc_local - shQ) ** 2 / (2 * sQ ** 2)) + q0
            return np.column_stack((I, Q))
        
        # Coarse grid
        for A in [0.25, 0.3, 0.35]:
            for sI in [5.0, 6.0, 7.0]:
                for eta in [-2.0, -1.0, 0.0, 1.0]:
                    if time.time() > deadline - 80:
                        break
                    I_Q = gen_drag([A, sI, eta, 5.0, 0.0, 0.0, 0.0])
                    if not check_constraints(I_Q):
                        continue
                    s = score(I_Q)
                    if s < best_s:
                        best_s, best = s, I_Q
        
        # Nelder-Mead refinement
        def opt_drag(p):
            if time.time() > deadline - 50:
                return 1e6
            I_Q = gen_drag(p)
            if not check_constraints(I_Q):
                return 1e6
            return score(I_Q)
        
        for init in [[0.3, 6.0, -1.0, 5.0, 0.0, 0.0, 0.0],
                     [0.35, 5.0, -2.0, 6.0, 0.0, 0.0, 0.0]]:
            if time.time() > deadline - 50:
                break
            try:
                r = minimize(opt_drag, init, method='Nelder-Mead',
                             options={'maxiter': 80, 'xatol': 0.01, 'fatol': 1e-5})
                if r.fun < best_s:
                    best_s, best = r.fun, gen_drag(r.x)
            except Exception:
                pass
    
    # ---- Final constraint check ----
    if not check_constraints(best):
        best = seed_pulse()
    
    return best
