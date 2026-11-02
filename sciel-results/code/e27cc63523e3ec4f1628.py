import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit
    
    # Build the rank-56 Strassen solution
    rank_56 = 56
    U56 = [[0] * rank_56 for _ in range(n * m)]
    V56 = [[0] * rank_56 for _ in range(m * p)]
    W56 = [[0] * rank_56 for _ in range(p * n)]
    
    # Strassen's 7-term algorithm for 2x2 matrix multiplication
    # U vectors (for A, shape 4 = 2x2)
    u_vecs = [
        [1, 0, 0, 1],   # A11 + A22
        [0, 0, 1, 1],   # A21 + A22
        [1, 0, 0, 0],   # A11
        [0, 0, 0, 1],   # A22
        [1, 1, 0, 0],   # A11 + A12
        [-1, 0, 1, 0],  # A21 - A11
        [0, 1, 0, -1]   # A12 - A22
    ]
    
    # V vectors (for B, shape 4 = 2x2)
    v_vecs = [
        [1, 0, 0, 1],   # B11 + B22
        [1, 0, 0, 0],   # B11
        [0, 1, 0, -1],  # B12 - B22
        [-1, 0, 1, 0],  # B21 - B11
        [0, 0, 0, 1],   # B22
        [1, 1, 0, 0],   # B11 + B12
        [0, 0, 1, 1]    # B21 + B22
    ]
    
    # W vectors (for C, shape 4 = 2x2)
    # Output convention: W[k*n+i, r] where C[i,j] = sum_k A[i,k] * B[k,j]
    # For 2x2: C[0,0]=M1-M3+M5-M6, C[0,1]=M4+M5, C[1,0]=M2+M3, C[1,1]=M1-M2+M4+M6
    # Flat index: C[i,j] -> j*n+i, so C[0,0]->0, C[0,1]->1, C[1,0]->2, C[1,1]->3
    w_vecs = [
        [1, 0, 0, 1],   # M1: C[0,0]+C[1,1]
        [0, 0, 1, -1],  # M2: C[1,0]-C[0,1]... let me re-derive
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]
    
    # Let me re-derive Strassen carefully.
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # 
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 - M3 + M5 - M7
    # C12 = M4 + M5
    # C21 = M2 + M3
    # C22 = M1 - M2 + M4 + M6
    #
    # Flat indexing for 2x2: [C11, C12, C21, C22] = [0, 1, 2, 3]
    # W vectors (each of length 4, corresponding to [C11, C12, C21, C22]):
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, -1]
    # M3: [-1, 0, 1, 0]
    # M4: [0, 1, 0, 1]
    # M5: [1, 1, 0, 0]
    # M6: [0, 0, 0, 1]
    # M7: [-1, 0, 0, 0]
    
    w_vecs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [-1, 0, 1, 0],  # M3
        [0, 1, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [0, 0, 0, 1],   # M6
        [-1, 0, 0, 0]   # M7
    ]
    
    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            # U: 4 entries (2x2 A matrix, row-major: A[0,0],A[0,1],A[1,0],A[1,1])
            for li in range(4):
                if u_vecs[s][li] != 0:
                    U56[li][col] = u_vecs[s][li]
            
            # V: 32 entries (2x16 B matrix)
            # B is 2x16, flat index: B[k,j] -> k*16+j
            # Block r corresponds to columns 2*r and 2*r+1 of B
            # B[k, 2r] is at index k*16 + 2r, B[k, 2r+1] is at index k*16 + 2r+1
            v_indices = [2*r, 2*r+1, 16+2*r, 16+2*r+1]  # B[0,2r], B[0,2r+1], B[1,2r], B[1,2r+1]
            for li in range(4):
                if v_vecs[s][li] != 0:
                    V56[v_indices[li]][col] = v_vecs[s][li]
            
            # W: 32 entries (16x2 C matrix)
            # C is 16x2, flat index: C[j,i] -> j*2+i (since W[k*n+i,r] with n=2)
            # Wait, the convention is W[k*n+i,r] where output is C[i,j] = sum_k A[i,k]*B[k,j]
            # So C has shape (n, p) = (2, 16). Flat index: C[i,j] -> j*n + i = j*2 + i
            # Block r corresponds to columns 2r and 2r+1 of C
            # C[0, 2r] is at index (2r)*2 + 0 = 4r
            # C[0, 2r+1] is at index (2r+1)*2 + 0 = 4r+2
            # C[1, 2r] is at index (2r)*2 + 1 = 4r+1
            # C[1, 2r+1] is at index (2r+1)*2 + 1 = 4r+3
            w_indices = [4*r, 4*r+2, 4*r+1, 4*r+3]  # C[0,2r], C[0,2r+1], C[1,2r], C[1,2r+1]
            for li in range(4):
                if w_vecs[s][li] != 0:
                    W56[w_indices[li]][col] = w_vecs[s][li]
    
    # Verify the solution
    def _verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        T_recon = np.einsum('as,bs,cs->abc', Ua, Va, Wa)
        # Build target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    T[a, b, c] = 1.0
        return np.allclose(T_recon, T, atol=1e-8)
    
    if not _verify(U56, V56, W56):
        # Fallback: try schoolbook as last resort
        rank_school = n * m * p
        U_sch = [[0] * rank_school for _ in range(n * m)]
        V_sch = [[0] * rank_school for _ in range(m * p)]
        W_sch = [[0] * rank_school for _ in range(p * n)]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    idx = i * m * p + k * p + j
                    U_sch[i * m + k][idx] = 1
                    V_sch[k * p + j][idx] = 1
                    W_sch[j * n + i][idx] = 1
        return {"U": U_sch, "V": V_sch, "W": W_sch}
    
    # The rank-56 solution is optimal (rank of direct sum = sum of ranks)
    # Return immediately to avoid wasting time on futile sub-optimal searches
    return {"U": U56, "V": V56, "W": W56}
# EVOLVE-BLOCK-END
