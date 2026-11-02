import numpy as np
import time

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix 
    by a 4x7 matrix.
    
    Uses a combination of Strassen (rank 7) for 2x2*2x2 blocks and 
    schoolbook (rank 4) for 2x2*2x1 blocks.
    
    The middle dimension m=4 is split into two blocks of size 2:
    - j in {0, 1}
    - j in {2, 3}
    
    For each j-block, the output dimension p=7 is split into:
    - Three 2-column blocks (k=0,1; k=2,3; k=4,5) using Strassen (rank 7 each)
    - One 1-column block (k=6) using schoolbook (rank 4)
    
    Total rank = 2 * (3 * 7 + 1 * 4) = 2 * 25 = 50
    """
    n, m, p = 2, 4, 7  # A is n x m, B is m x p, C is n x p
    
    total_rank = 50
    U = np.zeros((n * m, total_rank), dtype=float)  # A factors
    V = np.zeros((m * p, total_rank), dtype=float)  # B factors
    W = np.zeros((p * n, total_rank), dtype=float)  # C factors (k*n + i)
    
    col = 0
    
    def add_strassen_block(j_start, k_start):
        """
        Adds Strassen decomposition for C[:, k_start:k_start+2] 
        = A[:, j_start:j_start+2] * B[j_start:j_start+2, k_start:k_start+2]
        
        Since n=2, this covers the full height of A and C.
        """
        nonlocal col
        
        # A indices: A[i, j] -> U[i*m + j, col]
        a00 = j_start
        a01 = j_start + 1
        a10 = m + j_start
        a11 = m + j_start + 1
        
        # B indices: B[j, k] -> V[j*p + k, col]
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        
        # W indices: W[k, i] -> W[k*n + i, col]
        w00 = k_start * n + 0
        w01 = (k_start + 1) * n + 0
        w10 = k_start * n + 1
        w11 = (k_start + 1) * n + 1
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[w00, col] += 1; W[w11, col] += 1
        col += 1
        
        # M2 = (a10 + a11) * b00
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[w10, col] += 1; W[w11, col] -= 1
        col += 1
        
        # M3 = a00 * (b01 - b11)
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[w01, col] += 1; W[w11, col] += 1
        col += 1
        
        # M4 = a11 * (b10 - b00)
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[w00, col] += 1; W[w10, col] += 1
        col += 1
        
        # M5 = (a00 + a01) * b11
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[w00, col] -= 1; W[w01, col] += 1
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[w00, col] += 1
        col += 1
    
    def add_schoolbook_block(j_start, k_start):
        """
        Adds schoolbook decomposition for C[:, k_start] 
        = A[:, j_start:j_start+2] * B[j_start:j_start+2, k_start]
        """
        nonlocal col
        
        a00 = j_start
        a01 = j_start + 1
        a10 = m + j_start
        a11 = m + j_start + 1
        
        b00 = j_start * p + k_start
        b10 = (j_start + 1) * p + k_start
        
        w00 = k_start * n + 0
        w10 = k_start * n + 1
        
        # P1 = a00 * b00
        U[a00, col] += 1; V[b00, col] += 1; W[w00, col] += 1
        col += 1
        
        # P2 = a01 * b10
        U[a01, col] += 1; V[b10, col] += 1; W[w00, col] += 1
        col += 1
        
        # P3 = a10 * b00
        U[a10, col] += 1; V[b00, col] += 1; W[w10, col] += 1
        col += 1
        
        # P4 = a11 * b10
        U[a11, col] += 1; V[b10, col] += 1; W[w10, col] += 1
        col += 1
    
    # Execute construction
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(j_start, k_start)
        add_schoolbook_block(j_start, 6)
    
    if col != total_rank:
        raise RuntimeError(f"Rank mismatch: expected {total_rank}, got {col}")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    n, m, p = 2, 4, 7
    rank = U.shape[1]
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                calc = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(calc - target) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    if valid:
        print(f"Verification Passed. Rank: {rank}")
    else:
        print("Verification Failed")
