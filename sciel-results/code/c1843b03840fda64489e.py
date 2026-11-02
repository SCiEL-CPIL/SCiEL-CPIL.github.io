import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return U, V, W for a correct rank-7 Strassen 2x2x2 decomposition.
    
    U: shape (4, 7) - U[i*2+j, c] = coeff of a_ij in term c
    V: shape (4, 7) - V[j*2+k, c] = coeff of b_jk in term c  
    W: shape (4, 7) - W[k*2+i, c] = coeff of M_c in C_ik
    
    Verification: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k in {0,1}
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 0: M1 = (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1  # C00+=M1, C11+=M1
    
    # Term 1: M2 = (a10+a11)b00
    U[2,1]=1; U[3,1]=1; V[0,1]=1
    W[2,1]=1; W[3,1]=-1  # C10+=M2, C11-=M2
    
    # Term 2: M3 = a00(b10-b11)
    U[0,2]=1; V[2,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1  # C01+=M3, C11+=M3
    
    # Term 3: M4 = a11(b01-b00)
    U[3,3]=1; V[1,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1  # C00+=M4, C10+=M4
    
    # Term 4: M5 = (a00+a01)b11
    U[0,4]=1; U[1,4]=1; V[3,4]=1
    W[0,4]=-1; W[1,4]=1  # C00-=M5, C01+=M5
    
    # Term 5: M6 = (a10-a00)(b00+b01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1
    W[1,5]=-1; W[2,5]=1  # C01-=M6, C10+=M6
    
    # Term 6: M7 = (a01-a11)(b10+b11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1
    W[0,6]=-1; W[3,6]=1  # C00-=M7, C11+=M7
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    r = U.shape[1]
    D = np.zeros((n, m, p))
    for c in range(r):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        D += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
    return np.allclose(D, 1.0, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Verify Strassen decomposition first
    U_s, V_s, W_s = _strassen_222()
    
    D_test = np.zeros((2, 2, 2))
    for c in range(7):
        Uc = U_s[:, c].reshape(2, 2)
        Vc = V_s[:, c].reshape(2, 2)
        Wc = W_s[:, c].reshape(2, 2)
        D_test += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
    
    strassen_valid = np.allclose(D_test, 1.0, atol=1e-10)
    print(f"Strassen valid: {strassen_valid}, max_err={np.max(np.abs(D_test-1.0)):.2e}")
    
    if not strassen_valid:
        # Fall back to schoolbook
        U = np.zeros((n*m, n*m*p), dtype=float)
        V = np.zeros((m*p, n*m*p), dtype=float)
        W = np.zeros((p*n, n*m*p), dtype=float)
        for c, (i, j, k) in enumerate([(i,j,k) for i in range(n) for j in range(m) for k in range(p)]):
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Block partitioning: n=3 -> [2,1], m=4 -> [2,2], p=6 -> [2,2,2]
    n_blocks = [(0, 2), (2, 3)]
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [(0, 2), (2, 4), (4, 6)]
    
    # Count total rank needed
    total_rank = 0
    for (i_s, i_e) in n_blocks:
        for (j_s, j_e) in m_blocks:
            for (k_s, k_e) in p_blocks:
                ns, ms, ps = i_e-i_s, j_e-j_s, k_e-k_s
                if ns==2 and ms==2 and ps==2:
                    total_rank += 7
                else:
                    total_rank += ns*ms*ps
    
    print(f"Total rank: {total_rank}")
    
    # Build decomposition
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    col = 0
    for (i_s, i_e) in n_blocks:
        for (j_s, j_e) in m_blocks:
            for (k_s, k_e) in p_blocks:
                ns, ms, ps = i_e-i_s, j_e-j_s, k_e-k_s
                
                if ns==2 and ms==2 and ps==2:
                    # Apply Strassen to this 2x2x2 block
                    for c in range(7):
                        # U entries: local (li,lj) -> global (i_s+li, j_s+lj)
                        for li in range(2):
                            for lj in range(2):
                                val = U_s[li*2+lj, c]
                                if val != 0:
                                    U[(i_s+li)*m + (j_s+lj), col] = val
                        # V entries: local (lj,lk) -> global (j_s+lj, k_s+lk)
                        for lj in range(2):
                            for lk in range(2):
                                val = V_s[lj*2+lk, c]
                                if val != 0:
                                    V[(j_s+lj)*p + (k_s+lk), col] = val
                        # W entries: local (lk,li) -> global (k_s+lk, i_s+li)
                        for lk in range(2):
                            for li in range(2):
                                val = W_s[lk*2+li, c]
                                if val != 0:
                                    W[(k_s+lk)*n + (i_s+li), col] = val
                        col += 1
                else:
                    # Schoolbook for this block
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                U[(i_s+i)*m + (j_s+j), col] = 1.0
                                V[(j_s+j)*p + (k_s+k), col] = 1.0
                                W[(k_s+k)*n + (i_s+i), col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Rank mismatch: {col} vs {total_rank}"
    
    # Verify the full decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        print(f"Decomposition verified! Rank = {total_rank}")
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        print("Decomposition FAILED verification, falling back to schoolbook")
        U_fb = np.zeros((n*m, n*m*p), dtype=float)
        V_fb = np.zeros((m*p, n*m*p), dtype=float)
        W_fb = np.zeros((p*n, n*m*p), dtype=float)
        for c, (i, j, k) in enumerate([(i,j,k) for i in range(n) for j in range(m) for k in range(p)]):
            U_fb[i*m+j, c] = 1.0
            V_fb[j*p+k, c] = 1.0
            W_fb[k*n+i, c] = 1.0
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
