"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The multiplication tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    must equal 1 for all i,j,k.
    """
    rank = len(U[0])
    # Build the tensor via Einstein summation
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # Compute sum over c
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # Use broadcasting: U_r[i,j,:] * V_r[j,k,:] * W_r[k,i,:]
    # This is tricky with pure broadcasting, so use einsum
    # We need: for each i,j,k, sum over c of U_r[i,j,c]*V_r[j,k,c]*W_r[k,i,c]
    
    # Build full tensor
    # U_r: (n, m, rank)
    # V_r: (m, p, rank)
    # W_r: (p, n, rank)
    
    # T[i,j,k] = <U_r[i,j,:], V_r[j,k,:], W_r[k,i,:]>
    # = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    
    # Let's compute this step by step
    # First, compute A[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c]
    # U_r[i,j,c] * V_r[j,k,c] -> sum over c gives (n, m, p) tensor
    # U_r: (n, m, rank) -> expand to (n, m, 1, rank)
    # V_r: (m, p, rank) -> expand to (1, m, p, rank)
    # Product: (n, m, p, rank), sum over last axis -> (n, m, p)
    A = np.einsum('ijc,jkc->ijk', U_r, V_r)
    
    # Now T[i,j,k] = A[i,j,k] * W_r[k,i,c] summed over c... no wait
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # This is not (sum_c U_r[i,j,c]*V_r[j,k,c]) * W_r[k,i,c]
    # It's the triple product summed over c.
    
    # Let me redo: T[i,j,k] = sum_c (U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c])
    # This is a trilinear form.
    
    # Use einsum properly:
    # U_r: indices i,j,c
    # V_r: indices j,k,c  -- wait, V_r is (m,p,rank) so indices are j,k,c where j is row (m), k is col (p)
    # W_r: indices k,i,c -- W_r is (p,n,rank) so indices are k,i,c where k is row (p), i is col (n)
    
    # So T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    # Check if T is all ones
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-9)


def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # U[i*2+j, c] = coefficient of A_ij in product c
    # V[j*2+k, c] = coefficient of B_jk in product c
    # W[k*2+i, c] = coefficient in C_ki... wait, W[k*n+i] where n=2, so W[k*2+i]
    # C[i,j] = sum_k A[i,k] B[k,j]
    # The tensor is T[i,j,k] = 1 where i=row of A, j=col of A (row of B), k=col of B
    # Output C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] corresponds to C[i,k] getting contribution from A[i,j]*B[j,k]
    # W index: k*n + i = k*2 + i, which is the output position (i,k)
    
    # M1: (A00+A11)(B00+B11)
    # A00: U[0,0]=1, A11: U[3,0]=1
    # B00: V[0,0]=1, B11: V[3,0]=1
    # C00: W[0*2+0,0]=W[0,0]=1, C11: W[1*2+1,0]=W[3,0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00
    # A10: U[2,1]=1, A11: U[3,1]=1
    # B00: V[0,1]=1
    # C10: W[0*2+1,1]=W[1,1]=1, C11: W[1*2+1,1]=W[3,1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = 1
    
    # M3: A00(B01-B11)
    # A00: U[0,2]=1
    # B01: V[1,2]=1, B11: V[3,2]=-1
    # C01: W[0*2+0,2]=W[0,2]=1, C11: W[1*2+1,2]=W[3,2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[0][2] = 1; W[3][2] = 1
    
    # M4: A11(B10-B00)
    # A11: U[3,3]=1
    # B10: V[2,3]=1, B00: V[0,3]=-1
    # C00: W[0,3]=1, C10: W[1,3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A00+A01)B11
    # A00: U[0,4]=1, A01: U[1,4]=1
    # B11: V[3,4]=1
    # C00: W[0,4]=-1, C01: W[0,5]=... wait
    # C00 = M1 + M4 - M5 + M7, so M5 contributes -1 to C00
    # C01 = M3 + M5, so M5 contributes +1 to C01
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[0][5] = 1  # Wait, W index for C01 is k*2+i where k=1,i=0 -> W[2,4]=1
    # Let me re-check: C[i,k], W[k*2+i]
    # C00: i=0,k=0 -> W[0*2+0]=W[0]
    # C01: i=0,k=1 -> W[1*2+0]=W[2]
    # C10: i=1,k=0 -> W[0*2+1]=W[1]
    # C11: i=1,k=1 -> W[1*2+1]=W[3]
    
    # So for M5: contributes to C00 with -1 and C01 with +1
    W[0][4] = -1; W[2][4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # A10: U[2,5]=1, A00: U[0,5]=-1
    # B00: V[0,5]=1, B01: V[1,5]=1
    # C11: W[3,5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11)
    # A01: U[1,6]=1, A11: U[3,6]=-1
    # B10: V[2,6]=1, B11: V[3,6]=1
    # C00: W[0,6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _verify_strassen():
    """Verify the Strassen decomposition is correct."""
    n, m, p = 2, 2, 2
    U, V, W = _strassen_2x2x2()
    return _verify_decomposition(U, V, W, n, m, p)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 10
    
    # First, verify Strassen works
    strassen_ok = _verify_strassen()
    
    if not strassen_ok:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Strategy: Use Strassen for 2x2x2 blocks
    # The 2x10 matrix B can be viewed as 5 independent 2x2 blocks
    # B[:, 2t:2t+2] for t=0..4
    # 
    # For each block t, we multiply A (2x2) * B_block (2x2) using Strassen
    # This gives C_block (2x2)
    # 
    # The total rank would be 5 * 7 = 35
    #
    # Let's construct this:
    # For each block t (0..4), and each Strassen product c (0..6):
    #   U[i*2+j, t*7+c] = U_strassen[i*2+j, c]
    #   V[j*2+k, t*7+c] = V_strassen[j*2+k, c] where k is the local column (0 or 1)
    #   But V's global column index is j*10 + (2*t + k_local)
    #   W[k_global*2+i, t*7+c] = W_strassen[k_local*2+i, c]
    #   where k_global = 2*t + k_local
    
    U_str, V_str, W_str = _strassen_2x2x2()
    rank = 5 * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 35)
    V = [[0] * rank for _ in range(m * p)]  # (20, 35)
    W = [[0] * rank for _ in range(p * n)]  # (20, 35)
    
    for t in range(5):
        for c in range(7):
            col = t * 7 + c
            # U: same for all blocks
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = U_str[i * m + j][c]
            
            # V: global index is j*p + k_global = j*10 + (2*t + k_local)
            for j in range(m):
                for k_local in range(2):
                    k_global = 2 * t + k_local
                    V[j * p + k_global][col] = V_str[j * 2 + k_local][c]
            
            # W: global index is k_global*n + i = (2*t + k_local)*2 + i
            for k_local in range(2):
                for i in range(n):
                    k_global = 2 * t + k_local
                    W[k_global * n + i][col] = W_str[k_local * 2 + i][c]
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
