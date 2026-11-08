import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin before 150s kill

    # Schoolbook solution (guaranteed valid, rank = 18)
    U0 = np.zeros((n * m, rank_schoolbook))
    V0 = np.zeros((m * p, rank_schoolbook))
    W0 = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    # Verify schoolbook
    U0_3d = U0.reshape(n, m, rank_schoolbook)
    V0_3d = V0.reshape(m, p, rank_schoolbook)
    W0_3d = W0.reshape(p, n, rank_schoolbook)
    T0 = np.einsum('ijc,jkc,kic->ijk', U0_3d, V0_3d, W0_3d)
    if not np.allclose(T0, 1.0, atol=1e-9):
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    def compute_error(U, V, W, r):
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return float(np.sum((T - 1.0) ** 2))

    def build_A(U, V, W, r):
        """A[j,k,c] = sum_i U[i,j,c]*W[k,i,c]"""
        U_3d = U.reshape(n, m, r)
        W_3d = W.reshape(p, n, r)
        # A[j,k,c] = sum_i U[i,j,c] * W[k,i,c]
        # U[i,j,c]: indices (i,j,c); W[k,i,c]: indices (k,i,c)
        # Contract over i -> (j,k,c)
        A = np.einsum('ijc,kic->jkc', U_3d, W_3d)
        return A

    def local_search(U, V, W, r, max_steps, deadline):
        A = build_A(U, V, W, r)
        err = float(np.sum((A - 1.0) ** 2))
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            mat = np.random.randint(0, 3)
            delta = np.random.choice([-0.5, 0.5])
            
            if mat == 0:  # Perturb U[i,j,c]
                row = np.random.randint(0, n * m)
                c = np.random.randint(0, r)
                i, j = divmod(row, m)  # row = i*m + j
                old = U[row, c]
                U[row, c] = old + delta
                # A[j,k,c] += delta * W[k,i,c]
                # W[k,i,c]: row = k*n + i
                W_col = np.array([W[k * n + i, c] for k in range(p)])
                A_jk = A[j, :, c].copy()
                sum1 = np.dot(W_col, A_jk - 1.0)
                sum2 = np.dot(W_col, W_col)
                new_err = err + 2.0 * delta * sum1 + delta * delta * sum2
                if new_err < err - 1e-12:
                    err = new_err
                    A[j, :, c] += delta * W_col
                else:
                    U[row, c] = old
                    
            elif mat == 1:  # Perturb V[j,k,c]
                row = np.random.randint(0, m * p)
                c = np.random.randint(0, r)
                j, k = divmod(row, p)  # row = j*p + k
                old = V[row, c]
                V[row, c] = old + delta
                # Error change: 2*delta * sum_i U[i,j,c]*W[k,i,c]*(A[j,k,c]-1) + delta^2 * sum_i U[i,j,c]^2*W[k,i,c]^2
                U_col = np.array([U[i * m + j, c] for i in range(n)])
                W_col = np.array([W[k * n + i, c] for i in range(n)])
                A_jk = A[j, k, c]
                sum1 = np.dot(U_col * W_col, A_jk - 1.0)
                sum2 = np.dot(U_col * U_col, W_col * W_col)
                new_err = err + 2.0 * delta * sum1 + delta * delta * sum2
                if new_err < err - 1e-12:
                    err = new_err
                else:
                    V[row, c] = old
                    
            else:  # Perturb W[k,i,c]
                row = np.random.randint(0, p * n)
                c = np.random.randint(0, r)
                k, i = divmod(row, n)  # row = k*n + i
                old = W[row, c]
                W[row, c] = old + delta
                # A[j,k,c] += delta * U[i,j,c]
                # Error change: 2*delta * sum_j U[i,j,c]*V[j,k,c]*(A[j,k,c]-1) + delta^2 * sum_j U[i,j,c]^2*V[j,k,c]^2
                U_col = np.array([U[i * m + j, c] for j in range(m)])
                V_col = np.array([V[j * p + k, c] for j in range(m)])
                A_jk = A[:, k, c].copy()
                sum1 = np.dot(U_col * V_col, A_jk - 1.0)
                sum2 = np.dot(U_col * U_col, V_col * V_col)
                new_err = err + 2.0 * delta * sum1 + delta * delta * sum2
                if new_err < err - 1e-12:
                    err = new_err
                    A[:, k, c] += delta * U_col
                else:
                    W[row, c] = old
            
            if err < 1e-9:
                break
        
        return U, V, W, err

    # Try ranks from 15 up to 17
    for r in range(15, min(18, rank_schoolbook + 1)):
        if time.time() > deadline - 10:
            break
        
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Structured initialization
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            for c in range(r):
                num_cols = np.random.randint(2, 4)
                chosen = np.random.choice(rank_schoolbook, size=num_cols, replace=False)
                signs = np.random.choice([-1, 1], size=num_cols)
                for sc, sign in zip(chosen, signs):
                    k = sc % p
                    j = (sc // p) % m
                    i = sc // (m * p)
                    U[i * m + j, c] += sign
                    V[j * p + k, c] += sign
                    W[k * n + i, c] += sign
            
            # Optional random noise
            if np.random.random() < 0.3:
                U += (np.random.randint(-1, 2, size=(n*m, r)) * 0.5)
                V += (np.random.randint(-1, 2, size=(m*p, r)) * 0.5)
                W += (np.random.randint(-1, 2, size=(p*n, r)) * 0.5)
            
            U, V, W, err = local_search(U, V, W, r, max_steps=8000, deadline=deadline - 3)
            
            if err < 1e-7:
                true_err = compute_error(U, V, W, r)
                if true_err < 1e-6:
                    U_clean = np.round(U * 2) / 2
                    V_clean = np.round(V * 2) / 2
                    W_clean = np.round(W * 2) / 2
                    verify_err = compute_error(U_clean, V_clean, W_clean, r)
                    if verify_err < 1e-6:
                        best_rank = r
                        best_U, best_V, best_W = U_clean, V_clean, W_clean
                        break
        
        if best_rank <= r:
            break
    
    if best_rank < rank_schoolbook:
        U_out = np.round(best_U * 2) / 2
        V_out = np.round(best_V * 2) / 2
        W_out = np.round(best_W * 2) / 2
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    else:
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
