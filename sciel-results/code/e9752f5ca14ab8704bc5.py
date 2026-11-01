import time
import numpy as np

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns (0,1), (2,3), and (4,5).
    - Use schoolbook (rank 4) for column 6.
    Total rank: 7 + 7 + 7 + 4 = 25.
    
    The tensor identity is:
    Sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all valid (i,j,k).
    Here n=2, m=2, p=7.
    U shape: (4, 25)
    V shape: (14, 25)
    W shape: (14, 25)
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    # Initialize factor matrices as numpy arrays for efficient vectorized verification
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_strassen_block(col_start, k_start):
        """
        Add Strassen rank-7 terms for a 2x2 block of B (columns k_start, k_start+1).
        col_start: starting column index in U, V, W for this block's terms.
        k_start: starting column index in B (and C).
        """
        
        # Strassen Formulas for C = A * B
        # m1 = (A00 + A11) * (B00 + B11)
        # m2 = (A10 + A11) * B00
        # m3 = A00 * (B01 - B11)
        # m4 = A11 * (B10 - B00)
        # m5 = (A00 + A01) * B11
        # m6 = (A10 - A00) * (B00 + B01)
        # m7 = (A01 - A11) * (B10 + B11)
        
        # C00 = m1 - m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 - m3 + m6
        
        # Indices in U (A-space):
        # A00: 0, A01: 1, A10: 2, A11: 3
        
        # Indices in V (B-space) for this block:
        # B00 (j=0, k=k_start): 0*p + k_start = k_start
        # B01 (j=0, k=k_start+1): 0*p + k_start+1 = k_start+1
        # B10 (j=1, k=k_start): 1*p + k_start = p + k_start
        # B11 (j=1, k=k_start+1): 1*p + k_start+1 = p + k_start+1
        
        v_00 = 0 * p + k_start
        v_01 = 0 * p + k_start + 1
        v_10 = 1 * p + k_start
        v_11 = 1 * p + k_start + 1
        
        # Indices in W (C-space) for this block:
        # C00 (i=0, k=k_start): k_start*n + 0 = 2*k_start
        # C01 (i=1, k=k_start): k_start*n + 1 = 2*k_start + 1
        # C10 (i=0, k=k_start+1): (k_start+1)*n + 0 = 2*(k_start+1)
        # C11 (i=1, k=k_start+1): (k_start+1)*n + 1 = 2*(k_start+1) + 1
        
        w_00 = k_start * n + 0
        w_10 = k_start * n + 1  # Wait, W[k*n+i]. i=0, k=k_start -> 2*k_start. i=1, k=k_start -> 2*k_start+1.
        # Let's re-verify W indexing.
        # W row index is k*n + i.
        # For k=k_start:
        #   i=0 -> 2*k_start
        #   i=1 -> 2*k_start + 1
        # For k=k_start+1:
        #   i=0 -> 2*(k_start+1)
        #   i=1 -> 2*(k_start+1) + 1
        
        # So:
        # C[0, k_start] (i=0, col=k_start) -> W[2*k_start]
        # C[1, k_start] (i=1, col=k_start) -> W[2*k_start + 1]
        # C[0, k_start+1] (i=0, col=k_start+1) -> W[2*k_start + 2]
        # C[1, k_start+1] (i=1, col=k_start+1) -> W[2*k_start + 3]
        
        w_00 = 2 * k_start       # C[0, k_start]
        w_10 = 2 * k_start + 1   # C[1, k_start]
        w_01 = 2 * k_start + 2   # C[0, k_start+1]
        w_11 = 2 * k_start + 3   # C[1, k_start+1]
        
        # Term 1: m1 = (A00 + A11) * (B00 + B11)
        # Contributes to C[0, k_start] (+1), C[1, k_start+1] (+1)
        r = col_start + 0
        U[0, r] += 1
        U[3, r] += 1
        V[v_00, r] += 1
        V[v_11, r] += 1
        W[w_00, r] += 1
        W[w_11, r] += 1
        
        # Term 2: m2 = (A10 + A11) * B00
        # Contributes to C[1, k_start] (+1), C[1, k_start+1] (-1)
        r = col_start + 1
        U[2, r] += 1
        U[3, r] += 1
        V[v_00, r] += 1
        W[w_10, r] += 1
        W[w_11, r] -= 1
        
        # Term 3: m3 = A00 * (B01 - B11)
        # Contributes to C[0, k_start+1] (+1), C[1, k_start+1] (-1)
        r = col_start + 2
        U[0, r] += 1
        V[v_01, r] += 1
        V[v_11, r] -= 1
        W[w_01, r] += 1
        W[w_11, r] -= 1
        
        # Term 4: m4 = A11 * (B10 - B00)
        # Contributes to C[1, k_start] (+1), C[0, k_start] (-1)
        r = col_start + 3
        U[3, r] += 1
        V[v_10, r] += 1
        V[v_00, r] -= 1
        W[w_10, r] += 1
        W[w_00, r] -= 1
        
        # Term 5: m5 = (A00 + A01) * B11
        # Contributes to C[0, k_start+1] (+1), C[0, k_start] (-1)
        r = col_start + 4
        U[0, r] += 1
        U[1, r] += 1
        V[v_11, r] += 1
        W[w_01, r] += 1
        W[w_00, r] -= 1
        
        # Term 6: m6 = (A10 - A00) * (B00 + B01)
        # Contributes to C[1, k_start+1] (+1)
        r = col_start + 5
        U[2, r] += 1
        U[0, r] -= 1
        V[v_00, r] += 1
        V[v_01, r] += 1
        W[w_11, r] += 1
        
        # Term 7: m7 = (A01 - A11) * (B10 + B11)
        # Contributes to C[0, k_start] (+1)
        r = col_start + 6
        U[1, r] += 1
        U[3, r] -= 1
        V[v_10, r] += 1
        V[v_11, r] += 1
        W[w_00, r] += 1

    # Apply Strassen to pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    
    # Apply schoolbook to column 6
    # C[0,6] = A[0,0]B[0,6] + A[0,1]B[1,6]
    # C[1,6] = A[1,0]B[0,6] + A[1,1]B[1,6]
    
    col_start = 21
    k = 6
    
    # Term 8: A[0,0] * B[0,6] -> C[0,6]
    r = col_start + 0
    U[0, r] += 1
    V[0 * p + k, r] += 1
    W[k * n + 0, r] += 1
    
    # Term 9: A[0,1] * B[1,6] -> C[0,6]
    r = col_start + 1
    U[1, r] += 1
    V[1 * p + k, r] += 1
    W[k * n + 0, r] += 1
    
    # Term 10: A[1,0] * B[0,6] -> C[1,6]
    r = col_start + 2
    U[2, r] += 1
    V[0 * p + k, r] += 1
    W[k * n + 1, r] += 1
    
    # Term 11: A[1,1] * B[1,6] -> C[1,6]
    r = col_start + 3
    U[3, r] += 1
    V[1 * p + k, r] += 1
    W[k * n + 1, r] += 1

    # Verification
    # Check that for all i, j, k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                
                # Vectorized check for this specific (a,b,c)
                s = np.sum(U[a_idx] * V[b_idx] * W[c_idx])
                
                if abs(s - 1.0) > 1e-9:
                    valid = False
    
    if not valid:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=int)
        V_sb = np.zeros((m*p, rank_sb), dtype=int)
        W_sb = np.zeros((p*n, rank_sb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
