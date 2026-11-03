import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 3, 4, 6
    
    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
        
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    target[i, k, l] = 1.0
                    
        if not np.allclose(R, target, atol=1e-9):
            return False
        return True

    def get_strassen_66():
        rank = 66
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1

            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1

            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1

            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1

            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1

        add_1x2_2x2(0, 0, 0, 0, 0, 0)
        add_1x2_2x2(0, 2, 2, 0, 0, 0)
        add_1x2_2x2(0, 0, 0, 2, 0, 2)
        add_1x2_2x2(0, 2, 2, 2, 0, 2)
        add_1x2_2x2(0, 0, 0, 4, 0, 4)
        add_1x2_2x2(0, 2, 2, 4, 0, 4)

        add_strassen(1, 0, 0, 0, 1, 0)
        add_strassen(1, 2, 2, 0, 1, 0)
        add_strassen(1, 0, 0, 2, 1, 2)
        add_strassen(1, 2, 2, 2, 1, 2)
        add_strassen(1, 0, 0, 4, 1, 4)
        add_strassen(1, 2, 2, 4, 1, 4)
        
        return U, V, W

    def try_local_search(U, V, W, max_time=30.0):
        """Try to reduce rank by finding linear dependencies among terms."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        
        # Build the tensor
        T = np.einsum('ic,kc,lc->iklc', Ua, Va, Wa)
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    target[i, k, l] = 1.0
        
        # Try removing terms one at a time
        removed = []
        for _ in range(50):  # Try up to 50 removals
            if time.time() > deadline - 10.0:
                break
            current_rank = rank - len(removed)
            if current_rank <= 0:
                break
            
            # Build current factor matrices
            keep = [i for i in range(rank) if i not in removed]
            if len(keep) < 1:
                break
            
            U_cur = Ua[:, keep]
            V_cur = Va[:, keep]
            W_cur = Wa[:, keep]
            
            # Try each remaining term for removal
            found_removal = False
            for j in range(len(keep)):
                if time.time() > deadline - 10.0:
                    break
                    
                new_keep = keep[:j] + keep[j+1:]
                if len(new_keep) < 1:
                    continue
                    
                U_new = Ua[:, new_keep]
                V_new = Va[:, new_keep]
                W_new = Wa[:, new_keep]
                
                # Check if this term can be absorbed
                # The term contributes: U[:,j] outer V[:,j] outer W[:,j]
                # We need to see if this is already in the span of other terms
                u_j = Ua[:, keep[j]]
                v_j = Va[:, keep[j]]
                w_j = Wa[:, keep[j]]
                
                # Try to find coefficients alpha_i such that
                # u_j = sum alpha_i * u_i for i in new_keep
                # and v_j = sum beta_i * v_i and w_j = sum gamma_i * w_i
                # with alpha_i * beta_i * gamma_i giving the right tensor contribution
                
                # Simpler: just check if the tensor contribution of term j
                # is in the span of other terms' tensor contributions
                T_j = np.einsum('c,c,c->ikl', u_j, v_j, w_j)
                
                # Build matrix of other terms' contributions
                other_terms = []
                for idx in new_keep:
                    u_i = Ua[:, idx]
                    v_i = Va[:, idx]
                    w_i = Wa[:, idx]
                    T_i = np.einsum('c,c,c->ikl', u_i, v_i, w_i)
                    other_terms.append(T_i.flatten())
                
                if other_terms:
                    T_mat = np.array(other_terms).T  # shape (n*m*m*p*p*n, num_other)
                    T_j_flat = T_j.flatten()
                    
                    # Check if T_j is in the column space of T_mat
                    # Use least squares
                    try:
                        coeffs, residuals, _, _ = np.linalg.lstsq(T_mat, T_j_flat, rcond=1e-10)
                        T_recon = T_mat @ coeffs
                        if np.allclose(T_recon, T_j_flat, atol=1e-9):
                            # This term is redundant
                            removed.append(keep[j])
                            found_removal = True
                            break
                    except:
                        pass
            
            if not found_removal:
                break
        
        if removed:
            keep = [i for i in range(rank) if i not in removed]
            U_out = Ua[:, keep].tolist()
            V_out = Va[:, keep].tolist()
            W_out = Wa[:, keep].tolist()
            return U_out, V_out, W_out
        
        return None, None, None

    # Generate the base 66-rank solution
    U66, V66, W66 = get_strassen_66()
    
    # Verify it's correct
    if not verify(U66, V66, W66):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n*m)]
        V = [[0] * rank for _ in range(m*p)]
        W = [[0] * rank for _ in range(p*n)]
        col = 0
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    U[i][col] = 1
                    V[k][col] = 1
                    W[l][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try local search to reduce rank
    U_best, V_best, W_best = U66, V66, W66
    rank_best = 66
    
    if time.time() < deadline - 15.0:
        U_ls, V_ls, W_ls = try_local_search(U66, V66, W66, max_time=25.0)
        if U_ls is not None:
            new_rank = len(U_ls[0])
            if new_rank < rank_best and verify(U_ls, V_ls, W_ls):
                rank_best = new_rank
                U_best, V_best, W_best = U_ls, V_ls, W_ls
    
    return {"U": U_best, "V": V_best, "W": W_best}
