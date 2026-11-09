import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 4, 6
    nm, mp, np_ = n*m, m*p, n*p  # 8, 24, 12

    T_target = np.zeros((n, m, m, p, n, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, j, k, i, k] = 1.0
    T_flat = T_target.flatten()

    def compute_tensor(U, V, W, rank):
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(n, p, rank)
        return np.einsum('ijc,jkc,ikc->ijj kik', Ur, Vr, Wr)

    def check_solution(U, V, W, rank):
        T = compute_tensor(U, V, W, rank)
        return np.allclose(T, T_target, atol=1e-8)

    # Schoolbook rank 48
    def schoolbook():
        U = np.zeros((nm, 48)); V = np.zeros((mp, 48)); W = np.zeros((np_, 48))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W

    U_best, V_best, W_best = schoolbook()
    best_rank = 48

    # Strassen 2x2x2 rank 7
    def strassen_2x2x2():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        # M1=(A11+A22)(B11+B22)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=-1
        # M2=(A21+A22)B11
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1; W[3,1]=-1
        # M3=A11(B21-B22)
        U[0,2]=1
        V[2,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        # M4=A22(B12-B11)
        U[3,3]=1
        V[1,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        # M5=(A11+A12)B22
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[1,4]=1
        # M6=(A21-A11)(B11+B12)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        # M7=(A12-A22)(B21+B22)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1; W[3,6]=1
        return U, V, W

    # Build 2x4x6 from 6 Strassen 2x2x2 blocks
    # Split j into {0,1} and {2,3}, split k into {0,1}, {2,3}, {4,5}
    # For each (j_group, k_group), we have a 2x2x2 sub-problem
    # j_group 0: j in {0,1}, j_group 1: j in {2,3}
    # k_group 0: k in {0,1}, k_group 1: k in {2,3}, k_group 2: k in {4,5}
    #
    # For sub-problem (jg, kg):
    # A sub: rows i=0,1; cols j=2*jg, 2*jg+1
    # B sub: rows jp=2*jg, 2*jg+1; cols k=2*kg, 2*kg+1
    # C sub: rows ip=0,1; cols kp=2*kg, 2*kg+1
    #
    # U[i, j, c] where j is global col index
    # V[jp, k, c] where jp is global row index of B, k is global col index
    # W[ip, kp, c] where ip is global row of C, kp is global col

    def build_from_strassen():
        U = np.zeros((nm, 42)); V = np.zeros((mp, 42)); W = np.zeros((np_, 42))
        Su, Sv, Sw = strassen_2x2x2()
        col = 0
        for jg in range(2):
            for kg in range(3):
                for c in range(7):
                    # U: i in 0,1; j in 2*jg, 2*jg+1
                    for i in range(2):
                        for dj in range(2):
                            j = 2*jg + dj
                            U[i*m + j, col] = Su[i*2 + dj, c]
                    # V: jp in 2*jg, 2*jg+1; k in 2*kg, 2*kg+1
                    for djp in range(2):
                        jp = 2*jg + djp
                        for dk in range(2):
                            k = 2*kg + dk
                            V[jp*p + k, col] = Sv[djp*2 + dk, c]
                    # W: ip in 0,1; kp in 2*kg, 2*kg+1
                    for ip in range(2):
                        for dkp in range(2):
                            kp = 2*kg + dkp
                            W[kp*n + ip, col] = Sw[ip*2 + dkp, c]
                    col += 1
        return U, V, W

    U_s, V_s, W_s = build_from_strassen()
    if check_solution(U_s, V_s, W_s, 42):
        U_best, V_best, W_best = U_s, V_s, W_s
        best_rank = 42

    # Try to optimize down from best_rank
    def try_optimize(rank, n_restarts=8, max_nfev=300):
        nonlocal U_best, V_best, W_best, best_rank
        if rank >= best_rank:
            return False
        if time.time() > deadline - 25:
            return False

        def objective(x):
            U = x[:nm*rank].reshape(nm, rank)
            V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W = x[nm*rank+mp*rank:].reshape(np_, rank)
            T = compute_tensor(U, V, W, rank)
            return T.flatten() - T_flat

        total_size = (nm + mp + np_) * rank
        found = False

        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break

            if restart == 0 and best_rank <= 48:
                # Start from current best, zero-pad to rank
                U0 = np.zeros((nm, rank))
                V0 = np.zeros((mp, rank))
                W0 = np.zeros((np_, rank))
                if best_rank <= 48:
                    U0[:, :min(rank, best_rank)] = U_best[:, :min(rank, best_rank)]
                    V0[:, :min(rank, best_rank)] = V_best[:, :min(rank, best_rank)]
                    W0[:, :min(rank, best_rank)] = W_best[:, :min(rank, best_rank)]
            else:
                scale = 0.05 + 0.05 * (restart % 5)
                U0 = np.random.randn(nm, rank) * scale
                V0 = np.random.randn(mp, rank) * scale
                W0 = np.random.randn(np_, rank) * scale

            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

            try:
                from scipy.optimize import least_squares
                result = least_squares(objective, x0, method='trf', max_nfev=max_nfev, xtol=1e-12, ftol=1e-12)

                if result.cost < 1e-10:
                    U_sol = result.x[:nm*rank].reshape(nm, rank)
                    V_sol = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_sol = result.x[nm*rank+mp*rank:].reshape(np_, rank)

                    # Try rounding
                    for rf in [2, 4, 1]:
                        Ur = np.round(U_sol * rf) / rf
                        Vr = np.round(V_sol * rf) / rf
                        Wr = np.round(W_sol * rf) / rf
                        if check_solution(Ur, Vr, Wr, rank):
                            U_best, V_best, W_best = Ur, Vr, Wr
                            best_rank = rank
                            found = True
                            break
                    if found:
                        break

                    # Check if nearly exact
                    T_check = compute_tensor(U_sol, V_sol, W_sol, rank)
                    if np.max(np.abs(T_check - T_target)) < 1e-10:
                        U_best, V_best, W_best = U_sol, V_sol, W_sol
                        best_rank = rank
                        found = True
                        break

            except Exception:
                pass

        return found

    # Try to reduce rank
    for rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 35:
            break
        if try_optimize(rank, n_restarts=6, max_nfev=200):
            continue
        elif time.time() < deadline - 50:
            try_optimize(rank, n_restarts=4, max_nfev=150)

    # Final check
    if not check_solution(U_best, V_best, W_best, best_rank):
        U_best, V_best, W_best = schoolbook()
        best_rank = 48

    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
