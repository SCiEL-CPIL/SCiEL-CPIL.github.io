import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 8
    N = n * m  # 16
    M = m * p  # 32
    P = p * n  # 32

    def verify(U, V, W, tol=1e-9):
        """Verify: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]"""
        for a in range(N):
            i_a, j_a = divmod(a, m)
            for b in range(M):
                j_b, k_b = divmod(b, p)
                for c in range(P):
                    k_c, i_c = divmod(c, n)
                    expected = 1.0 if (j_a == j_b and k_b == k_c and i_a == i_c) else 0.0
                    actual = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(actual - expected) > tol:
                        return False
        return True

    def strassen_2x2x2():
        """Rank-7 decomposition for 2x2x2 matrix multiplication."""
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1; W2[3, 1] = -1
        
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[1, 2] = 1; W2[3, 2] = 1
        
        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[2, 3] = 1
        
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[1, 4] = 1
        
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1
        
        return U2, V2, W2

    def build_block_strassen():
        """Build rank-98 block Strassen for 4x4x8."""
        U2, V2, W2 = strassen_2x2x2()
        rank = 98
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        
        A_specs = [
            [(1, 0), (1, 3)],
            [(1, 2), (1, 3)],
            [(1, 0)],
            [(1, 3)],
            [(1, 0), (1, 1)],
            [(1, 2), (-1, 0)],
            [(1, 1), (-1, 3)],
        ]
        
        B_specs = [
            [(1, 0), (1, 3)],
            [(1, 0)],
            [(1, 1), (-1, 3)],
            [(1, 2), (-1, 0)],
            [(1, 3)],
            [(1, 0), (1, 1)],
            [(1, 2), (1, 3)],
        ]
        
        C_specs = [
            [(1, 0), (1, 3)],
            [(1, 2), (-1, 3)],
            [(1, 1), (1, 3)],
            [(1, 0), (1, 2)],
            [(-1, 0), (1, 1)],
            [(1, 3)],
            [(1, 0)],
        ]
        
        for half in range(2):
            for k_p in range(7):
                for m_s in range(7):
                    col = half * 49 + k_p * 7 + m_s
                    
                    for coeff_a, block_idx in A_specs[k_p]:
                        bi, bj = divmod(block_idx, 2)
                        for r in range(2):
                            for c in range(2):
                                global_r = bi * 2 + r
                                global_c = bj * 2 + c
                                global_a_idx = global_r * 4 + global_c
                                a_eff_idx = r * 2 + c
                                U[global_a_idx, col] += coeff_a * U2[a_eff_idx, m_s]
                    
                    for coeff_b, block_idx in B_specs[k_p]:
                        bi, bj = divmod(block_idx, 2)
                        for r in range(2):
                            for c in range(2):
                                global_r = bi * 2 + r
                                global_c = bj * 2 + c
                                global_k = global_c + half * 4
                                global_b_idx = global_r * 8 + global_k
                                b_eff_idx = r * 2 + c
                                V[global_b_idx, col] += coeff_b * V2[b_eff_idx, m_s]
                    
                    for coeff_c, c_block_idx in C_specs[k_p]:
                        bi, bj = divmod(c_block_idx, 2)
                        for r in range(2):
                            for c in range(2):
                                global_i = bi * 2 + r
                                global_k = bj * 2 + c + half * 4
                                global_w_idx = global_k * 4 + global_i
                                p_eff_idx = r * 2 + c
                                W[global_w_idx, col] += coeff_c * W2[p_eff_idx, m_s]
        
        return U, V, W

    def try_reduce_rank_linear(U, V, W):
        """Try to reduce rank by finding linear dependencies among columns."""
        current_rank = U.shape[1]
        U_cur = U.copy()
        V_cur = V.copy()
        W_cur = W.copy()
        
        for _ in range(100):
            if time.time() > deadline - 20:
                break
            merged = False
            for i in range(current_rank - 1):
                for j in range(i + 1, current_rank):
                    # Check if column j is a scalar multiple of column i in all three matrices
                    if np.allclose(U_cur[:, i], 0, atol=1e-12):
                        continue
                    nz = np.where(np.abs(U_cur[:, i]) > 1e-12)[0]
                    if len(nz) == 0:
                        continue
                    idx = nz[0]
                    if abs(U_cur[idx, i]) < 1e-12:
                        continue
                    alpha = U_cur[idx, j] / U_cur[idx, i]
                    
                    if (np.allclose(U_cur[:, j], alpha * U_cur[:, i], atol=1e-9, rtol=1e-9) and
                        np.allclose(V_cur[:, j], alpha * V_cur[:, i], atol=1e-9, rtol=1e-9) and
                        np.allclose(W_cur[:, j], alpha * W_cur[:, i], atol=1e-9, rtol=1e-9)):
                        U_cur = np.delete(U_cur, j, axis=1)
                        V_cur = np.delete(V_cur, j, axis=1)
                        W_cur = np.delete(W_cur, j, axis=1)
                        current_rank -= 1
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break
        
        return U_cur, V_cur, W_cur

    # Build initial solution
    U, V, W = build_block_strassen()
    
    # Verify
    if not verify(U, V, W):
        # Fallback to naive if block Strassen is wrong
        rank = N * M  # very high, but should work
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        for a in range(N):
            i_a, j_a = divmod(a, m)
            for b in range(M):
                j_b, k_b = divmod(b, p)
                if j_a == j_b:
                    c = k_b * n + i_a
                    idx = a * M + b
                    U[a, idx] = 1
                    V[b, idx] = 1
                    W[c, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    U_red, V_red, W_red = try_reduce_rank_linear(U, V, W)
    
    if verify(U_red, V_red, W_red):
        U, V, W = U_red, V_red, W_red
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
