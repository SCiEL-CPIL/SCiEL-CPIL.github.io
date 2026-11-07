<<<<<<< SEARCH
    # ---------------- Simulation + adjoint gradient ----------------
    def simulate_and_score(pulse_IQ, soft_beta=50.0):
        """
        Evolve all (case, state) and return (score, U_all, rho_final).
        U_all[ci, step] is the 9x9 step propagator.
        """
        U_all = np.zeros((9, N, 9, 9), dtype=complex)
        for ci in range(9):
            sc = case_scale[ci]
            for step in range(N):
                Iv = pulse_IQ[step, 0]
                Qv = pulse_IQ[step, 1]
                H_drive = 0.5 * sc * (Iv * (a + a_dag) + 1j * Qv * (a - a_dag))
                H_left = np.kron(I3, H_drive)
                H_right = np.kron(H_drive.T, I3)
                L = case_static_L[ci] - 1j * (H_left - H_right)
                U_all[ci, step] = expm(dt * L)
        rho_final = np.zeros((9, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                vec = rho_init[si].copy()
                for step in range(N):
                    vec = U_all[ci, step] @ vec
                rho_final[ci, si] = vec
        fids = np.zeros((9, 6))
        for ci in range(9):
            for si in range(6):
                fids[ci, si] = np.real(np.vdot(fid_vec[si], rho_final[ci, si]))
        infids = 1.0 - fids
        if soft_beta is None:
            score = np.mean(infids) + np.max(infids)
        else:
            m = np.max(infids)
            soft_max = m + np.log(np.sum(np.exp(soft_beta * (infids - m)))) / soft_beta
            score = np.mean(infids) + soft_max
        return score, U_all, rho_final

    def exact_score(pulse_IQ):
        fids = np.zeros((9, 6))
        for ci in range(9):
            sc = case_scale[ci]
            for si in range(6):
                vec = rho_init[si].copy()
                for step in range(N):
                    Iv = pulse_IQ[step, 0]
                    Qv = pulse_IQ[step, 1]
                    H_drive = 0.5 * sc * (Iv * (a + a_dag) + 1j * Qv * (a - a_dag))
                    H_left = np.kron(I3, H_drive)
                    H_right = np.kron(H_drive.T, I3)
                    L = case_static_L[ci] - 1j * (H_left - H_right)
                    vec = expm(dt * L) @ vec
                fids[ci, si] = np.real(np.vdot(fid_vec[si], vec))
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)

    def adjoint_grad(pulse_IQ, U_all, rho_final, soft_beta=50.0):
        """
        Analytic gradient of soft score w.r.t. pulse_IQ (N,2).
        Uses chain rule through the step propagators.
        """
        # Backward: for each case, accumulate adjoint of rho_final
        # score = sum_ci [ mean_si infid_ci,si + softmax over (ci,si) ]
        # d score / d rho_final[ci,si] = - (1/(54)) * fid_vec[si]  (for the mean part)
        #                              + -softmax_weight[ci,si] * fid_vec[si]
        infids = 1.0 - np.zeros((9, 6))
        fids_mat = np.zeros((9, 6))
        for ci in range(9):
            for si in range(6):
                fids_mat[ci, si] = np.real(np.vdot(fid_vec[si], rho_final[ci, si]))
        infids = 1.0 - fids_mat
        m = np.max(infids)
        w = np.exp(soft_beta * (infids - m))
        w = w / np.sum(w)
        # d score / d fids[ci,si] = -1/54 - soft_beta * w[ci,si]
        dfid = -1.0 / 54.0 - soft_beta * w
        # d score / d rho_final[ci,si] (vector) = dfid[ci,si] * conj(fid_vec[si])  (since fid=Re[vdot(fid_vec,rho)])
        # vdot(a,b)=conj(a)·b, d Re[vdot(a,b)]/db = Re[conj(a)] as a real linear functional -> gradient vector = conj(a).conj()?? 
        # Treat complex: score real. d score = Re[ sum d(rho)^* ... ]. We'll compute dL contribution using
        # d score / d vec(rho) as a complex vector g where score += Re[ g^H vec(rho) ] => g = conj(fid_vec)*dfid
        grad_rho = np.zeros((9, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                grad_rho[ci, si] = dfid[ci, si] * np.conj(fid_vec[si])

        # Adjoint propagation backward
        # adj[ci, step] is 9-vector = d score / d vec(rho at start of step)
        # rho after step = U[step] @ rho_before
        # d score / d rho_before = U[step]^H (as acting on vec) ... but U acts on vec via matrix mult:
        # vec_after = U @ vec_before  =>  d score/d vec_before = U^H @ d score/d vec_after  (using standard inner product on vec space)
        # We'll store adj as 9-vectors.
        adj = np.zeros((9, N, 9), dtype=complex)
        for ci in range(9):
            acc = np.zeros((6, 9), dtype=complex)
            for si in range(6):
                acc[si] = grad_rho[ci, si]
            # acc[si] = d score / d vec(rho_final[ci,si])
            for step in range(N - 1, -1, -1):
                # d score / d vec(rho_before) for each si = U[step]^H @ acc[si]
                U = U_all[ci, step]
                new_acc = np.zeros((6, 9), dtype=complex)
                for si in range(6):
                    new_acc[si] = U.conj().T @ acc[si]
                # store adj at this step (d score / d rho at start of step)
                adj[ci, step] = new_acc.sum(axis=0)   # not needed per-step for dU; we need d score/d U
                acc = new_acc
        # Now compute d score / d U[ci,step] (9x9):
        # vec_after = U @ vec_before ; d score = Re[ sum_si acc_after[si]^H (U @ vec_before[si]) ]
        # => d score / d U (as matrix) : dU contribution = Re[ sum_si acc_after[si] vec_before[si]^H ]^T ... 
        # We'll compute dL/dU via: score += Re[ Tr(M^H U) ] where M = sum_si vec_before[si] acc_after[si]^H (as 3x3)
        # => d score / d U_{ij} (real) = Re[ conj(M_{ij}) ]  for the real part; but U is complex and score real,
        # the complex gradient (Wirtinger) is conj(M). We'll use d score/d U = conj(M) as the complex gradient.
        # Then d U / d pulse via dL: U = expm(dt L), dU = dt * integral expm ... approx dt * U @ dL for small dt (first order).
        # d score / d L = U^H @ M  (since d score = Re[ Tr(M^H U) ], dU = dt U dL => d score = Re[ Tr(M^H dt U dL) ] = Re[ dt Tr(dL^H U^H M) ] => d score/dL = dt * U^H M)
        # Then dL/dI, dL/dQ are simple (linear).
        grad_pulse = np.zeros((N, 2))
        for ci in range(9):
            sc = case_scale[ci]
            for step in range(N):
                # vec_before at start of step for each si: propagate rho_init forward to start of step
                # We need vec_before[si] = (U[step-1]...U[0]) @ rho_init[si]
                # Recompute forward cumulative (cheap enough: N*N*6*9x9 ~ 80*80*6*81 ~ 3.1M mults per ci -> 28M total, ok)
                # Instead, precompute forward states F[ci, step, si] = vec at start of step
                pass
        # To avoid O(N^2) recompute, precompute forward states once
        F = np.zeros((9, N, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                vec = rho_init[si].copy()
                F[ci, 0, si] = vec
                for step in range(N - 1):
                    vec = U_all[ci, step] @ vec
                    F[ci, step + 1, si] = vec
        for ci in range(9):
            sc = case_scale[ci]
            for step in range(N):
                # M = sum_si vec_before[si] * acc_after[si]^H   (3x3)
                # acc_after[si] = adj[ci, step+1]?? We stored adj[ci,step]=d score/d rho at start of step.
                # We need d score / d rho_after(step) = sum over si of d score/d rho_final propagated back to after step.
                # Let's recompute a clean backward adjoint B[ci, step, si] = d score / d vec(rho after step)
                pass
        # Recompute backward adjoint per (ci,si) cleanly
        B = np.zeros((9, N, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                # at final: d score / d vec(rho_final[ci,si]) = grad_rho[ci,si]
                b = grad_rho[ci, si].copy()
                B[ci, N - 1, si] = b   # d score / d vec after last step
                for step in range(N - 1, 0, -1):
                    # vec_after(step) = U[step] @ vec_before(step)
                    # d score / d vec_before(step) = U[step]^H @ b
                    b = U_all[ci, step].conj().T @ b
                    B[ci, step - 1, si] = b   # d score / d vec after step-1
        # Now for each (ci,step): d score / d U[ci,step]:
        # vec_after = U @ vec_before ; d score = Re[ sum_si B_after[si]^H (U @ vec_before[si]) ]
        # = Re[ Tr( (sum_si vec_before[si] B_after[si]^H)^H U ) ]  ... define M = sum_si vec_before[si] B_after[si]^H (3x3)
        # then d score/dU (complex gradient) = conj(M)
        # d score/dL = dt * U^H M  (from dU = dt U dL first order)
        # dL/dI: L = case_static - 1j (kron(I3,Hd) - kron(Hd.T,I3)) ; Hd = 0.5 sc (I (a+adag) + i Q (a-adag))
        # dL/dI = -1j ( kron(I3, dHd/dI) - kron((dHd/dI).T, I3) )
        # dHd/dI = 0.5 sc (a + a_dag) ; dHd/dQ = 0.5 sc * i (a - a_dag)
        dHd_dI = 0.5 * sc * (a + a_dag)
        dHd_dQ = 0.5 * sc * 1j * (a - a_dag)
        dL_dI = -1j * (np.kron(I3, dHd_dI) - np.kron(dHd_dI.T, I3))
        dL_dQ = -1j * (np.kron(I3, dHd_dQ) - np.kron(dHd_dQ.T, I3))
        for step in range(N):
            M = np.zeros((3, 3), dtype=complex)
            for si in range(6):
                vb = F[ci, step, si].reshape(3, 3)
                ba = B[ci, step, si].reshape(3, 3)
                M += vb @ ba.conj().T
            dU = np.conj(M)
            dL = dt * (U_all[ci, step].conj().T @ M)
            dI = np.real(np.vdot(dL, dL_dI))   # Re[ dL^H dL_dI ]
            dQ = np.real(np.vdot(dL, dL_dQ))
            grad_pulse[step, 0] += dI
            grad_pulse[step, 1] += dQ
        return grad_pulse

    def model_objective(params, soft_beta=50.0, need_grad=False):
        if time.time() > deadline - 5:
            return (1e10, np.zeros(6)) if need_grad else 1e10
        p = build_pulse(params)
        if not check_validity(p):
            return (1e10, np.zeros(6)) if need_grad else 1e10
        score, U_all, rho_final = simulate_and_score(p, soft_beta)
        if not need_grad:
            return score
        grad_p = adjoint_grad(p, U_all, rho_final, soft_beta)
        # d score / d params via d p / d params (finite diff on the small model is cheap: 6 params)
        eps = 1e-7
        grad = np.zeros(6)
        for j in range(6):
            pp = params.copy(); pm = params.copy()
            pp[j] += eps; pm[j] -= eps
            p_plus = build_pulse(pp); p_minus = build_pulse(pm)
            sp = 1e10; sm = 1e10
            if check_validity(p_plus):
                sp, _, _ = simulate_and_score(p_plus, soft_beta)
            if check_validity(p_minus):
                sm, _, _ = simulate_and_score(p_minus, soft_beta)
            grad[j] = (sp - sm) / (2 * eps)
        return score, grad
=======
    # ---------------- Simulation ----------------
    def simulate_and_score(pulse_IQ, soft_beta=50.0):
        """
        Evolve all (case, state) and return (score, U_all, rho_final).
        """
        U_all = np.zeros((9, N, 9, 9), dtype=complex)
        for ci in range(9):
            sc = case_scale[ci]
            for step in range(N):
                Iv = pulse_IQ[step, 0]
                Qv = pulse_IQ[step, 1]
                H_drive = 0.5 * sc * (Iv * (a + a_dag) + 1j * Qv * (a - a_dag))
                H_left = np.kron(I3, H_drive)
                H_right = np.kron(H_drive.T, I3)
                L = case_static_L[ci] - 1j * (H_left - H_right)
                U_all[ci, step] = expm(dt * L)
        rho_final = np.zeros((9, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                vec = rho_init[si].copy()
                for step in range(N):
                    vec = U_all[ci, step] @ vec
                rho_final[ci, si] = vec
        fids = np.zeros((9, 6))
        for ci in range(9):
            for si in range(6):
                fids[ci, si] = np.real(np.vdot(fid_vec[si], rho_final[ci, si]))
        infids = 1.0 - fids
        if soft_beta is None:
            score = np.mean(infids) + np.max(infids)
        else:
            m = np.max(infids)
            soft_max = m + np.log(np.sum(np.exp(soft_beta * (infids - m)))) / soft_beta
            score = np.mean(infids) + soft_max
        return score, U_all, rho_final

    def exact_score(pulse_IQ):
        fids = np.zeros((9, 6))
        for ci in range(9):
            sc = case_scale[ci]
            for si in range(6):
                vec = rho_init[si].copy()
                for step in range(N):
                    Iv = pulse_IQ[step, 0]
                    Qv = pulse_IQ[step, 1]
                    H_drive = 0.5 * sc * (Iv * (a + a_dag) + 1j * Qv * (a - a_dag))
                    H_left = np.kron(I3, H_drive)
                    H_right = np.kron(H_drive.T, I3)
                    L = case_static_L[ci] - 1j * (H_left - H_right)
                    vec = expm(dt * L) @ vec
                fids[ci, si] = np.real(np.vdot(fid_vec[si], vec))
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)

    def model_objective(params, soft_beta=50.0, need_grad=False):
        if time.time() > deadline - 5:
            return (1e10, np.zeros(6)) if need_grad else 1e10
        p = build_pulse(params)
        if not check_validity(p):
            return (1e10, np.zeros(6)) if need_grad else 1e10
        score, _, _ = simulate_and_score(p, soft_beta)
        if not need_grad:
            return score
        
        # Finite difference gradient for the 6 parameters
        eps = 1e-5
        grad = np.zeros(6)
        for j in range(6):
            pp = params.copy(); pm = params.copy()
            pp[j] += eps; pm[j] -= eps
            p_plus = build_pulse(pp); p_minus = build_pulse(pm)
            sp = 1e10; sm = 1e10
            if check_validity(p_plus):
                sp, _, _ = simulate_and_score(p_plus, soft_beta)
            if check_validity(p_minus):
                sm, _, _ = simulate_and_score(p_minus, soft_beta)
            grad[j] = (sp - sm) / (2 * eps)
        return score, grad
>>>>>>> REPLACE

I also need to remove the Phase 3 free-waveform optimization because it relies on the removed `adjoint_grad` and is too slow/complex for the remaining budget if not careful. I will replace it with a second L-BFGS run on the DRAG parameters with different initial conditions or just extend the first one.

