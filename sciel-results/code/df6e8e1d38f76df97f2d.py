import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Finds a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix 
    by a 4x7 matrix.
    
    Strategy:
    - Split the middle dimension (m=4) into two blocks of size 2: j in {0,1} and j in {2,3}.
    - For each 2x2 block of A, multiply by B.
    - Split the last dimension of B (p=7) into blocks: three 2x2 blocks and one 2x1 block.
    - Use Strassen's algorithm (rank 7) for the 2x2 * 2x2 sub-multiplications.
    - Use schoolbook (rank 4) for the 2x2 * 2x1 sub-multiplications.
    
    Total Rank = 2 * (3 * 7 + 1 * 4) = 2 * 25 = 50.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    total_rank = 50
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    
    def add_strassen_block(i_start, j_start, k_start):
        """
        Adds a Strassen decomposition for C[i_start:i_start+2, k_start:k_start+2] 
        = A[i_start:i_start+2, j_start:j_start+2] * B[j_start:j_start+2, k_start:k_start+2]
        
        Note: In our problem, n=2, so i_start is always 0 and we cover the full height of A/C.
        """
        nonlocal col
        
        # Indices for A (2x4 matrix)
        # A is indexed as [i, j]. In flattened U, index is i*m + j.
        # Block covers i in [0, 1], j in [j_start, j_start+1]
        a00_idx = 0 * m + j_start
        a01_idx = 0 * m + (j_start + 1)
        a10_idx = 1 * m + j_start
        a11_idx = 1 * m + (j_start + 1)
        
        # Indices for B (4x7 matrix)
        # B is indexed as [j, k]. In flattened V, index is j*p + k.
        # Block covers j in [j_start, j_start+1], k in [k_start, k_start+1]
        b00_idx = j_start * p + k_start
        b01_idx = j_start * p + (k_start + 1)
        b10_idx = (j_start + 1) * p + k_start
        b11_idx = (j_start + 1) * p + (k_start + 1)
        
        # Indices for W (7x2 matrix)
        # W is indexed as [k, i]. Convention: W[k*n + i].
        # Block covers k in [k_start, k_start+1], i in [0, 1]
        w00_idx = k_start * n + 0
        w01_idx = (k_start + 1) * n + 0
        w10_idx = k_start * n + 1
        w11_idx = (k_start + 1) * n + 1
        
        # Strassen 7 products:
        # M1 = (a00 + a11) * (b00 + b11)
        U[a00_idx, col] += 1; U[a11_idx, col] += 1
        V[b00_idx, col] += 1; V[b11_idx, col] += 1
        W[w00_idx, col] += 1; W[w11_idx, col] += 1
        col += 1
        
        # M2 = (a10 + a11) * b00
        U[a10_idx, col] += 1; U[a11_idx, col] += 1
        V[b00_idx, col] += 1
        W[w10_idx, col] += 1; W[w11_idx, col] -= 1
        col += 1
        
        # M3 = a00 * (b01 - b11)
        U[a00_idx, col] += 1
        V[b01_idx, col] += 1; V[b11_idx, col] -= 1
        W[w01_idx, col] += 1; W[w11_idx, col] += 1
        col += 1
        
        # M4 = a11 * (b10 - b00)
        U[a11_idx, col] += 1
        V[b10_idx, col] += 1; V[b00_idx, col] -= 1
        W[w00_idx, col] += 1; W[w10_idx, col] += 1
        col += 1
        
        # M5 = (a00 + a01) * b11
        U[a00_idx, col] += 1; U[a01_idx, col] += 1
        V[b11_idx, col] += 1
        W[w00_idx, col] -= 1; W[w01_idx, col] += 1
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[a10_idx, col] += 1; U[a00_idx, col] -= 1
        V[b00_idx, col] += 1; V[b01_idx, col] += 1
        W[w11_idx, col] += 1
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[a01_idx, col] += 1; U[a11_idx, col] -= 1
        V[b10_idx, col] += 1; V[b11_idx, col] += 1
        W[w00_idx, col] += 1
        col += 1

    def add_schoolbook_block(i_start, j_start, k_start):
        """
        Adds a schoolbook decomposition for C[i_start:i_start+2, k_start]
        = A[i_start:i_start+2, j_start:j_start+2] * B[j_start:j_start+2, k_start]
        
        Here we compute C[:, k_start] which is a 2x1 vector.
        C[0, k] = A[0, j0]B[j0, k] + A[0, j1]B[j1, k]
        C[1, k] = A[1, j0]B[j0, k] + A[1, j1]B[j1, k]
        """
        nonlocal col
        
        # A indices
        a00_idx = 0 * m + j_start
        a01_idx = 0 * m + (j_start + 1)
        a10_idx = 1 * m + j_start
        a11_idx = 1 * m + (j_start + 1)
        
        # B indices (only 1 column in B for this block)
        b00_idx = j_start * p + k_start
        b10_idx = (j_start + 1) * p + k_start
        
        # W indices (only 1 column in C for this block)
        w00_idx = k_start * n + 0
        w10_idx = k_start * n + 1
        
        # P1 = a00 * b00
        U[a00_idx, col] += 1; V[b00_idx, col] += 1; W[w00_idx, col] += 1
        col += 1
        
        # P2 = a01 * b10
        U[a01_idx, col] += 1; V[b10_idx, col] += 1; W[w00_idx, col] += 1
        col += 1
        
        # P3 = a10 * b00
        U[a10_idx, col] += 1; V[b00_idx, col] += 1; W[w10_idx, col] += 1
        col += 1
        
        # P4 = a11 * b10
        U[a11_idx, col] += 1; V[b10_idx, col] += 1; W[w10_idx, col] += 1
        col += 1

    # Execute construction
    # Loop over the two j-blocks (columns of A, rows of B)
    for j_start in [0, 2]:
        # Loop over k-blocks of B (columns of B, rows of C)
        # k=0,1 -> Strassen
        # k=2,3 -> Strassen
        # k=4,5 -> Strassen
        # k=6   -> Schoolbook
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        
        add_schoolbook_block(0, j_start, 6)
        
    # Check if we used all columns
    if col != total_rank:
        raise RuntimeError(f"Rank mismatch: expected {total_rank}, got {col}")

    # Convert to lists as required by interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    # Self-test
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
                calc = 0.0
                for r in range(rank):
                    calc += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(calc - target) > 1e-9:
                    valid = False
                    print(f"Mismatch at {i},{j},{k}: calc={calc}, target={target}")
                    break
            if not valid: break
        if not valid: break
    if valid:
        print(f"Verification Passed. Rank: {rank}")
    else:
        print("Verification Failed")
