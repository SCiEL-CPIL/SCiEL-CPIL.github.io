"""Rewritten program for matrix multiplication tensor rank minimization (2x4 * 4x4)."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=2, m=4, p=4, verbose=False):
    """Verify that the decomposition correctly computes matrix multiplication.
    
    The tensor for matrix multiplication satisfies:
    T[i,j,k] = 1 for all valid i, j, k
    where i in [0,n), j in [0,m), k in [0,p)
    
    The decomposition requires:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                    if verbose and err > 1e-9:
                        print(f"Error at i={i}, j={j}, k={k}: got {val}, expected 1.0")
    
    return max_err


def _build_strassen_28():
    """Build a rank-28 Strassen decomposition for 2x4 * 4x4.
    
    Strategy: 
    A (2x4) = [A0 | A1] where A0, A1 are 2x2
    B (4x4) = [[B00, B01], [B10, B11]] where Bij are 2x2
    C (2x4) = [A0*B00 + A1*B10, A0*B01 + A1*B11]
    
    Each 2x2 * 2x2 product uses 7 multiplications (Strassen).
    Total rank = 4 * 7 = 28.
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 28
    V = np.zeros((m * p, rank), dtype=float)  # 16 x 28
    W = np.zeros((p * n, rank), dtype=float)  # 8 x 28
    
    def add_strassen_block(U_indices, V_indices, W_indices, col_offset):
        """
        Adds 7 Strassen terms for a 2x2 * 2x2 multiplication.
        
        U_indices: [a, b, c, d] corresponding to A = [[a,b],[c,d]]
        V_indices: [e, f, g, h] corresponding to B = [[e,f],[g,h]]
        W_indices: [q, r, s, t] corresponding to C = [[q,r],[s,t]]
        col_offset: starting column index in U, V, W
        """
        a, b, c, d = U_indices
        e, f, g, h = V_indices
        q, r, s, t = W_indices
        c_off = col_offset
        
        # M1 = (a+d)(e+h)
        U[a, c_off] += 1; U[d, c_off] += 1
        V[e, c_off] += 1; V[h, c_off] += 1
        W[q, c_off] += 1; W[t, c_off] += 1
        W[r, c_off] -= 1; W[s, c_off] -= 1
        c_off += 1
        
        # M2 = (c+d)e
        U[c, c_off] += 1; U[d, c_off] += 1
        V[e, c_off] += 1
        W[s, c_off] += 1; W[t, c_off] -= 1
        c_off += 1
        
        # M3 = a(f-h)
        U[a, c_off] += 1
        V[f, c_off] += 1; V[h, c_off] -= 1
        W[r, c_off] += 1; W[t, c_off] += 1
        c_off += 1
        
        # M4 = d(g-f)
        U[d, c_off] += 1
        V[g, c_off] += 1; V[f, c_off] -= 1
        W[q, c_off] += 1; W[s, c_off] += 1
        c_off += 1
        
        # M5 = (a+c)h
        U[a, c_off] += 1; U[c, c_off] += 1
        V[h, c_off] += 1
        W[q, c_off] -= 1; W[r, c_off] += 1
        c_off += 1
        
        # M6 = (b-c)(e+g)
        U[b, c_off] += 1; U[c, c_off] -= 1
        V[e, c_off] += 1; V[g, c_off] += 1
        W[s, c_off] += 1; W[t, c_off] += 1
        c_off += 1
        
        # M7 = (b-d)(f+h)
        U[b, c_off] += 1; U[d, c_off] -= 1
        V[f, c_off] += 1; V[h, c_off] += 1
        W[q, c_off] += 1; W[r, c_off] += 1
        c_off += 1
        
        return c_off
    
    # Indices for A blocks
    # A0: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # A1: A[0,2]=4, A[0,3]=5, A[1,2]=6, A[1,3]=7
    A0 = [0, 1, 2, 3]
    A1 = [4, 5, 6, 7]
    
    # Indices for B blocks
    # B00: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # B01: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    # B10: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    # B11: B[2,2]=10, B[2,3]=11, B[3,2]=14, B[3,3]=15
    B00 = [0, 1, 4, 5]
    B01 = [2, 3, 6, 7]
    B10 = [8, 9, 12, 13]
    B11 = [10, 11, 14, 15]
    
    # Indices for C blocks
    # C is 2x4. W index is k*n+i.
    # C_left (cols 0,1):
    # C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    # C_right (cols 2,3):
    # C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
    C_left = [0, 2, 1, 3]
    C_right = [4, 6, 5, 7]
    
    # Block 1: A0 * B00 -> C_left
    add_strassen_block(A0, B00, C_left, 0)
    
    # Block 2: A1 * B10 -> C_left
    add_strassen_block(A1, B10, C_left, 7)
    
    # Block 3: A0 * B01 -> C_right
    add_strassen_block(A0, B01, C_right, 14)
    
    # Block 4: A1 * B11 -> C_right
    add_strassen_block(A1, B11, C_right, 21)
    
    return U.tolist(), V.tolist(), W.tolist()


def _build_schoolbook_32():
    """Build the trivial rank-32 schoolbook decomposition."""
    n, m, p = 2, 4, 4
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    
    # Try rank-28 Strassen first
    U, V, W = _build_strassen_28()
    err = _verify_decomposition(U, V, W)
    
    if err < 1e-9:
        # Success with rank 28
        pass
    else:
        # Fall back to schoolbook rank-32
        U, V, W = _build_schoolbook_32()
    
    # Deadline guard
    if time.time() - start_time > 140:
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
