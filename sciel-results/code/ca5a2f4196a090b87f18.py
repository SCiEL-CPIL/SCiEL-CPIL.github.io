"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    The decomposition satisfies:
      U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] summed over c
      equals (A * B)[i, k] for all i, k.
    
    For n=m=p=2:
      U has shape (4, 7)
      V has shape (4, 7)  
      W has shape (4, 7)
    
    Indexing:
      U row index: i*m + j = i*2 + j  (i in 0..1, j in 0..1)
      V row index: j*p + k = j*2 + k  (j in 0..1, k in 0..1)
      W row index: k*n + i = k*2 + i  (k in 0..1, i in 0..1)
    
    Strassen's 7 products for 2x2:
      Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
      
      m1 = (a00 + a11) * (b00 + b11)
      m2 = (a10 + a11) * b00
      m3 = a00 * (b01 - b11)
      m4 = a11 * (b10 - b00)
      m5 = (a00 + a01) * b11
      m6 = (a10 - a00) * (b00 + b01)
      m7 = (a01 - a11) * (b10 + b11)
      
      C00 = m1 + m4 - m5 + m7
      C01 = m3 + m5
      C10 = m2 + m4
      C11 = m1 - m2 + m3 + m6
    
    We need to express each product mc as sum over factor c of U[i,j,c]*V[j,k,c]*W[k,i,c].
    Since each mc is a single product of a linear form in A and a linear form in B,
    we can set for each c: U[:,c] encodes the A-side, V[:,c] encodes the B-side,
    and W[:,c] encodes how that product contributes to each C entry.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    rank = 7
    
    # U[i*m + j, c]: coefficient of A[i,j] in the c-th factor
    # V[j*p + k, c]: coefficient of B[j,k] in the c-th factor
    # W[k*n + i, c]: coefficient of the c-th product in C[i,k]
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to set index
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # m1 = (a00 + a11) * (b00 + b11)
    c = 0
    U[u_idx(0,0), c] = 1
    U[u_idx(1,1), c] = 1
    V[v_idx(0,0), c] = 1
    V[v_idx(1,1), c] = 1
    # C00 += m1, C11 += m1
    W[w_idx(0,0), c] = 1   # k=0, i=0 -> C[0,0]
    W[w_idx(1,1), c] = 1   # k=1, i=1 -> C[1,1]
    
    # m2 = (a10 + a11) * b00
    c = 1
    U[u_idx(1,0), c] = 1
    U[u_idx(1,1), c] = 1
    V[v_idx(0,0), c] = 1
    # C10 += m2
    W[w_idx(0,1), c] = 1   # k=0, i=1 -> C[1,0]
    
    # m3 = a00 * (b01 - b11)
    c = 2
    U[u_idx(0,0), c] = 1
    V[v_idx(0,1), c] = 1
    V[v_idx(1,1), c] = -1
    # C01 += m3, C11 += m3
    W[w_idx(1,0), c] = 1   # k=1, i=0 -> C[0,1]
    W[w_idx(1,1), c] = 1   # k=1, i=1 -> C[1,1]
    
    # m4 = a11 * (b10 - b00)
    c = 3
    U[u_idx(1,1), c] = 1
    V[v_idx(1,0), c] = 1
    V[v_idx(0,0), c] = -1
    # C00 += m4, C10 += m4
    W[w_idx(0,0), c] = 1   # k=0, i=0 -> C[0,0]
    W[w_idx(0,1), c] = 1   # k=0, i=1 -> C[1,0]
    
    # m5 = (a00 + a01) * b11
    c = 4
    U[u_idx(0,0), c] = 1
    U[u_idx(0,1), c] = 1
    V[v_idx(1,1), c] = 1
    # C00 -= m5, C01 += m5
    W[w_idx(0,0), c] = -1  # k=0, i=0 -> C[0,0]
    W[w_idx(1,0), c] = 1   # k=1, i=0 -> C[0,1]
    
    # m6 = (a10 - a00) * (b00 + b01)
    c = 5
    U[u_idx(1,0), c] = 1
    U[u_idx(0,0), c] = -1
    V[v_idx(0,0), c] = 1
    V[v_idx(0,1), c] = 1
    # C11 += m6
    W[w_idx(1,1), c] = 1   # k=1, i=1 -> C[1,1]
    
    # m7 = (a01 - a11) * (b10 + b11)
    c = 6
    U[u_idx(0,1), c] = 1
    U[u_idx(1,1), c] = -1
    V[v_idx(1,0), c] = 1
    V[v_idx(1,1), c] = 1
    # C00 += m7
    W[w_idx(0,0), c] = 1   # k=0, i=0 -> C[0,0]
    
    # Verify the decomposition
    # The tensor T[i,j,k] = delta (should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])
    # Actually the multiplication tensor has T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # i.e., T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] term appears in C[i,k])
    
    # Let's verify: for each (i,k), sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 
    # sum_j A[i,j]*B[j,k] coefficient, which is 1 for each j.
    # More precisely, the reconstructed C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
    # Wait, that's not quite right. Let me think again.
    
    # The tensor decomposition means: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where M[i,j,k] is the matrix multiplication tensor, which equals 1 for all i,j,k in {0,1}^3.
    
    # Let's verify this.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # This should equal 1
                if abs(val - 1) > 1e-9:
                    # Fallback: return schoolbook
                    rank_sb = n * m * p
                    U_sb = np.zeros((n*m, rank_sb))
                    V_sb = np.zeros((m*p, rank_sb))
                    W_sb = np.zeros((p*n, rank_sb))
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_sb[ii*m+jj, col] = 1
                                V_sb[jj*p+kk, col] = 1
                                W_sb[kk*n+ii, col] = 1
                                col += 1
                    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
