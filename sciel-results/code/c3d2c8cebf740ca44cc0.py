import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def construct():
    n, m, p = 3, 5, 7
    
    def get_splits(dim):
        """All compositions of dim using parts 1 and 2"""
        if dim == 1:
            return [[1]]
        if dim == 2:
            return [[2], [1, 1]]
        if dim == 3:
            return [[2, 1], [1, 2], [1, 1, 1]]
        if dim == 5:
            return [[2,2,1],[2,1,2],[1,2,2],[2,1,1,1],[1,2,1,1],[1,1,2,1],[1,1,1,2],[1,1,1,1,1]]
        if dim == 7:
            results = []
            def gen(rem, cur):
                if rem == 0:
                    results.append(cur[:])
                    return
                if rem >= 1:
                    gen(rem - 1, cur + [1])
                if rem >= 2:
                    gen(rem - 2, cur + [2])
            gen(dim, [])
            return results
        return []
    
    def block_rank(ni, mj, pk):
        if ni == 2 and mj == 2 and pk == 2:
            return 7
        return ni * mj * pk
    
    # Search for best partition
    best_rank = n * m * p
    best_splits = ([1]*n, [1]*m, [1]*p)
    
    for ns in get_splits(n):
        for ms in get_splits(m):
            for ps in get_splits(p):
                total = 0
                for ni in ns:
                    for mj in ms:
                        for pk in ps:
                            total += block_rank(ni, mj, pk)
                if total < best_rank:
                    best_rank = total
                    best_splits = (ns[:], ms[:], ps[:])
    
    ns, ms, ps = best_splits
    rank_total = best_rank
    
    # Time check
    if time.time() - START_TIME > TIME_LIMIT - 5:
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][idx] = 1
                    V_s[j*p+k][idx] = 1
                    W_s[k*n+i][idx] = 1
                    idx += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    def add_strassen(rank_offset, i0, j0, k0):
        u_loc = [[0]*7 for _ in range(4)]
        v_loc = [[0]*7 for _ in range(4)]
        w_loc = [[0]*7 for _ in range(4)]
        
        u_loc[0][0] = 1; u_loc[3][0] = 1
        v_loc[0][0] = 1; v_loc[3][0] = 1
        w_loc[0][0] = 1; w_loc[3][0] = 1
        
        u_loc[2][1] = 1; u_loc[3][1] = 1
        v_loc[0][1] = 1
        w_loc[1][1] = 1; w_loc[3][1] = -1
        
        u_loc[0][2] = 1
        v_loc[1][2] = 1; v_loc[3][2] = -1
        w_loc[2][2] = 1; w_loc[3][2] = 1
        
        u_loc[3][3] = 1
        v_loc[2][3] = 1; v_loc[0][3] = -1
        w_loc[0][3] = 1; w_loc[1][3] = 1
        
        u_loc[0][4] = 1; u_loc[1][4] = 1
        v_loc[3][4] = 1
        w_loc[0][4] = -1; w_loc[2][4] = 1
        
        u_loc[2][5] = 1; u_loc[0][5] = -1
        v_loc[0][5] = 1; v_loc[1][5] = 1
        w_loc[3][5] = 1
        
        u_loc[1][6] = 1; u_loc[3][6] = -1
        v_loc[2][6] = 1; v_loc[3][6] = 1
        w_loc[0][6] = 1
        
        for i_l in range(2):
            for j_l in range(2):
                u_g = (i0 + i_l) * m + (j0 + j_l)
                for r in range(7):
                    U[u_g][rank_offset + r] += u_loc[i_l*2 + j_l][r]
        for j_l in range(2):
            for k_l in range(2):
                v_g = (j0 + j_l) * p + (k0 + k_l)
                for r in range(7):
                    V[v_g][rank_offset + r] += v_loc[j_l*2 + k_l][r]
        for k_l in range(2):
            for i_l in range(2):
                w_g = (k0 + k_l) * n + (i0 + i_l)
                for r in range(7):
                    W[w_g][rank_offset + r] += w_loc[k_l*2 + i_l][r]
    
    def add_schoolbook(rank_offset, i0, j0, k0, ni, mj, pk):
        for i_l in range(ni):
            for j_l in range(mj):
                for k_l in range(pk):
                    r = (i_l * mj + j_l) * pk + k_l
                    U[(i0 + i_l) * m + (j0 + j_l)][rank_offset + r] += 1
                    V[(j0 + j_l) * p + (k0 + k_l)][rank_offset + r] += 1
                    W[(k0 + k_l) * n + (i0 + i_l)][rank_offset + r] += 1
    
    # Build all blocks
    rank_offset = 0
    i0 = 0
    for ni in ns:
        j0 = 0
        for mj in ms:
            k0 = 0
            for pk in ps:
                if ni == 2 and mj == 2 and pk == 2:
                    add_strassen(rank_offset, i0, j0, k0)
                    rank_offset += 7
                else:
                    add_schoolbook(rank_offset, i0, j0, k0, ni, mj, pk)
                    rank_offset += ni * mj * pk
                k0 += pk
            j0 += mj
        i0 += ni
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build tensor from factors: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Using trace convention: W[k*n+i, r]
    A = np.random.RandomState(42).rand(n, m)
    B = np.random.RandomState(42).rand(m, p)
    C_expected = A @ B
    
    # Compute C from factors
    # C[i,k] = sum_j A[i,j]*B[j,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
    # Actually: C[i,k] = sum_r U_part[i,k,r] * V_part[i,k,r] * W[k*n+i,r]
    # where U_part[i,k,r] = sum_j U[i*m+j,r] * A[i,j]
    # and V_part[i,k,r] = sum_j V[j*p+k,r] * B[j,k]
    
    U_reshaped = U_np.reshape(n, m, rank_total)  # U[i*m+j, r] -> [i,j,r]
    V_reshaped = V_np.reshape(m, p, rank_total)  # V[j*p+k, r] -> [j,k,r]
    W_reshaped = W_np.reshape(p, n, rank_total)  # W[k*n+i, r] -> [k,i,r]
    
    # C[i,k] = sum_r (sum_j U_reshaped[i,j,r] * A[i,j]) * (sum_j V_reshaped[j,k,r] * B[j,k]) * W_reshaped[k,i,r]
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            u_val = np.sum(U_reshaped[i, :, :] * A[i, :].reshape(1, m, 1), axis=1)  # shape (rank,)
            v_val = np.sum(V_reshaped[:, k, :] * B[:, k].reshape(m, 1), axis=0)  # shape (rank,)
            C_computed[i, k] = np.sum(u_val * v_val * W_reshaped[k, i, :])
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][idx] = 1
                    V_s[j*p+k][idx] = 1
                    W_s[k*n+i][idx] = 1
                    idx += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
