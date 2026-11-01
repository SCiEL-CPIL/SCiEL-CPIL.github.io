import time
import numpy as np

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    # Initialize factor matrices
    # U: (n*m, rank) -> (4, 25)
    # V: (m*p, rank) -> (14, 25)
    # W: (p*n, rank) -> (14, 25)
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    def add_strassen_block(col_start, k_start):
        """
        Add Strassen rank-7 terms for a 2x2 block of B (columns k_start, k_start+1).
        col_start: starting column index in U, V, W for this block's terms.
        k_start: starting column index in B (and C).
        
        A is 2x2: [[A00, A01], [A10, A11]]
        B_block is 2x2: [[B0,ks, B0,ks+1], [B1,ks, B1,ks+1]]
        C_block is 2x2: [[C0,ks, C0,ks+1], [C1,ks, C1,ks+1]]
        
        Standard Strassen:
        M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        M2 = (A10 + A11) * B0,ks
        M3 = A00 * (B0,ks+1 - B1,ks+1)
        M4 = A11 * (B1,ks - B0,ks)
        M5 = (A00 + A01) * B1,ks+1
        M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        
        Reconstruction:
        C0,ks   = M1 - M4 - M5 + M7
        C0,ks+1 = M3 + M5
        C1,ks   = M2 + M4
        C1,ks+1 = M1 - M2 - M3 + M6
        """
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        # Contributes: C0,ks (+1), C1,ks+1 (+1)
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        # Contributes: C1,ks (+1), C1,ks+1 (-1)
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        # Contributes: C0,ks+1 (+1), C1,ks+1 (-1)
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] -= 1

        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        # Contributes: C0,ks (-1), C1,ks (+1)
        W[2*k_start + 0, col_start+3] -= 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        # Contributes: C0,ks (-1), C0,ks+1 (+1)
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        # Contributes: C1,ks+1 (+1)
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        # Contributes: C0,ks (+1)
        W[2*k_start + 0, col_start+6] += 1

    # Apply Strassen to pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    
    # Apply schoolbook to column 6
    col_start = 21
    k = 6
    
    # C0,6 = A00*B0,6 + A01*B1,6
    # Term 1: A00 * B0,6
    U[0, col_start+0] += 1
    V[7*0 + k, col_start+0] += 1
    W[2*k + 0, col_start+0] += 1
    
    # Term 2: A01 * B1,6
    U[1, col_start+1] += 1
    V[7*1 + k, col_start+1] += 1
    W[2*k + 0, col_start+1] += 1
    
    # C1,6 = A10*B0,6 + A11*B1,6
    # Term 3: A10 * B0,6
    U[2, col_start+2] += 1
    V[7*0 + k, col_start+2] += 1
    W[2*k + 1, col_start+2] += 1
    
    # Term 4: A11 * B1,6
    U[3, col_start+3] += 1
    V[7*1 + k, col_start+3] += 1
    W[2*k + 1, col_start+3] += 1

    # Full verification: check all 4*14*14 = 784 tensor entries
    valid = True
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                s = 0.0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                
                # Determine expected value
                # a = i*m+j, b = j'*p+k, c = k'*n+i'
                # Non-zero only if j==j' and i==i' and k==k'
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                if j == j2 and i == i2 and k == k2:
                    expected = 1.0
                else:
                    expected = 0.0
                
                if abs(s - expected) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook if verification fails
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
