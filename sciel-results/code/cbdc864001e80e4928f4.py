"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Precompute indices for the tensor
    ii = np.array([i for i in range(n) for j in range(m) for k in range(p)])
    jj = np.array([j for i in range(n) for j in range(m) for k in range(p)])
    kk = np.array([k for i in range(n) for j in range(m) for k in range(p)])
    
    U_idx = ii * m + jj
    V_idx = jj * p + kk
    W_idx = kk * n + ii
    
    def verify(U, V, W):
        U_sel = U[U_idx]
        V_sel = V[V_idx]
        W_sel = W[W_idx]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return np.allclose(vals, 1.0, atol=1e-9)
    
    # Strassen's algorithm for 2x2x2 matrix multiplication (rank 7)
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 matrix mult with rank 7 (Strassen)."""
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 products for 2x2:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # c11 = M1+M4-M5+M7
        # c12 = M3+M5
        # c21 = M2+M4
        # c22 = M1-M2+M3+M6
        
        # U factor: coefficients for a_ij
        # V factor: coefficients for b_jk
        # W factor: coefficients for c_ik
        
        # M1: (a11+a22)(b11+b22)
        U[0,0] = 1; U[3,0] = 1  # a11 + a22
        V[0,0] = 1; V[3,0] = 1  # b11 + b22
        W[0,0] = 1; W[3,0] = 1  # c11 + c22
        
        # M2: (a21+a22)b11
        U[2,1] = 1; U[3,1] = 1  # a21 + a22
        V[0,1] = 1  # b11
        W[2,1] = 1; W[3,1] = 1  # c21 + c22
        
        # M3: a11(b12-b22)
        U[0,2] = 1  # a11
        V[1,2] = 1; V[3,2] = -1  # b12 - b22
        W[1,2] = 1; W[3,2] = 1  # c12 + c22
        
        # M4: a22(b21-b11)
        U[3,3] = 1  # a22
        V[2,3] = 1; V[0,3] = -1  # b21 - b11
        W[0,3] = 1; W[2,3] = 1  # c11 + c21
        
        # M5: (a11+a12)b22
        U[0,4] = 1; U[1,4] = 1  # a11 + a12
        V[3,4] = 1  # b22
        W[0,4] = -1; W[1,4] = 1  # -c11 + c12
        
        # M6: (a21-a11)(b11+b12)
        U[2,5] = 1; U[0,5] = -1  # a21 - a11
        V[0,5] = 1; V[1,5] = 1  # b11 + b12
        W[3,5] = 1  # c22
        
        # M7: (a12-a22)(b21+b22)
        U[1,6] = 1; U[3,6] = -1  # a12 - a22
        V[2,6] = 1; V[3,6] = 1  # b21 + b22
        W[0,6] = 1  # c11
        
        return U, V, W
    
    # Schoolbook for 2x2x1 (rank 4)
    def schoolbook_2x2x1():
        """Return U, V, W for 2x2x1 matrix mult with rank 4."""
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        for i in range(2):
            for j in range(2):
                col = i*2 + j
                U[i*2+j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
        return U, V, W
    
    # Schoolbook for 1x2x2 (rank 4)
    def schoolbook_1x2x2():
        """Return U, V, W for 1x2x2 matrix mult with rank 4."""
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        for j in range(2):
            for k in range(2):
                col = j*2 + k
                U[j, col] = 1
                V[j*2+k, col] = 1
                W[k, col] = 1
        return U, V, W
    
    # Schoolbook for 1x2x1 (rank 2)
    def schoolbook_1x2x1():
        """Return U, V, W for 1x2x1 matrix mult with rank 2."""
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        for j in range(2):
            U[j, j] = 1
            V[j, j] = 1
            W[0, j] = 1
        return U, V, W
    
    # Combine blocks for 3x4x7
    # Split: n=3 into [2,1], m=4 into [2,2], p=7 into [2,2,2,1]
    n_blocks = [2, 1]
    m_blocks = [2, 2]
    p_blocks = [2, 2, 2, 1]
    
    # Build full U, V, W
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    for ni, ns in enumerate(n_blocks):
        for mi, ms in enumerate(m_blocks):
            for pi, ps in enumerate(p_blocks):
                if ns == 2 and ms == 2 and ps == 2:
                    u, v, w = strassen_2x2x2()
                elif ns == 2 and ms == 2 and ps == 1:
                    u, v, w = schoolbook_2x2x1()
                elif ns == 1 and ms == 2 and ps == 2:
                    u, v, w = schoolbook_1x2x2()
                elif ns == 1 and ms == 2 and ps == 1:
                    u, v, w = schoolbook_1x2x1()
                else:
                    # Fallback: schoolbook
                    rank = ns * ms * ps
                    u = np.eye(ns * ms, rank).reshape(ns*ms, ns*ms*ps)[:, :rank]
                    v = np.eye(ms * ps, rank).reshape(ms*ps, ms*ps*ps)[:, :rank]
                    w = np.eye(ps * ns, rank).reshape(ps*ns, ps*ns*ps)[:, :rank]
                    # This is wrong, let me fix
                    u = np.zeros((ns*ms, rank))
                    v = np.zeros((ms*ps, rank))
                    w = np.zeros((ps*ns, rank))
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                col = i*ms*ps + j*ps + k
                                u[i*ms+j, col] = 1
                                v[j*ps+k, col] = 1
                                w[k*ns+i, col] = 1
                
                rank = u.shape[1]
                total_rank += rank
                
                # Offset indices
                n_offset = sum(n_blocks[:ni])
                m_offset = sum(m_blocks[:mi])
                p_offset = sum(p_blocks[:pi])
                
                u_offset = np.zeros_like(u)
                v_offset = np.zeros_like(v)
                w_offset = np.zeros_like(w)
                
                for r in range(rank):
                    u_offset[:, r] = u[:, r]
                    v_offset[:, r] = v[:, r]
                    w_offset[:, r] = w[:, r]
                
                U_list.append((n_offset, ns, u_offset))
                V_list.append((m_offset, ms, v_offset))
                W_list.append((p_offset, ps, w_offset))
    
    # Now build full matrices
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    col_offset = 0
    for idx in range(len(U_list)):
        n_off, ns, u = U_list[idx]
        m_off, ms, v = V_list[idx]
        p_off, ps, w = W_list[idx]
        rank = u.shape[1]
        
        for i in range(ns):
            for j in range(ms):
                U_full[(n_off+i)*m + (m_off+j), col_offset:col_offset+rank] = u[i*ms+j, :]
        
        for j in range(ms):
            for k in range(ps):
                V_full[(m_off+j)*p + (p_off+k), col_offset:col_offset+rank] = v[j*ps+k, :]
        
        for k in range(ps):
            for i in range(ns):
                W_full[(p_off+k)*n + (n_off+i), col_offset:col_offset+rank] = w[k*ns+i, :]
        
        col_offset += rank
    
    U = U_full
    V = V_full
    W = W_full
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
