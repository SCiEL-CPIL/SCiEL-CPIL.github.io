import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We use 4 Strassen blocks for 2x2x2 sub-matrices.
    # Dimensions: n=3, m=5, p=5
    # Block 1: i in [0,1], j in [0,1], k in [0,1]
    # Block 2: i in [0,1], j in [2,3], k in [0,1]
    # Block 3: i in [0,1], j in [0,1], k in [2,3]
    # Block 4: i in [0,1], j in [2,3], k in [2,3]
    
    # Each Strassen block has rank 7.
    # Total Strassen rank = 4 * 7 = 28.
    
    # The remaining terms are handled by schoolbook multiplication.
    # Total terms = n * m * p = 3 * 5 * 5 = 75.
    # Terms covered by Strassen blocks = 4 * (2*2*2) = 32.
    # Remaining terms = 75 - 32 = 43.
    # Total Rank = 28 + 43 = 71.
    
    total_rank = 71
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    # Function to add a Strassen block at offset (i0, j0, k0)
    def add_strassen_block(i0, j0, k0, c_offset):
        # M1
        U[i0*m + j0, c_offset] += 1
        U[(i0+1)*m + (j0+1), c_offset] += 1
        V[j0*p + k0, c_offset] += 1
        V[(j0+1)*p + (k0+1), c_offset] += 1
        W[k0*n + i0, c_offset] += 1
        W[(k0+1)*n + (i0+1), c_offset] += 1
        
        # M2
        U[(i0+1)*m + j0, c_offset+1] += 1
        U[(i0+1)*m + (j0+1), c_offset+1] += 1
        V[j0*p + k0, c_offset+1] += 1
        W[k0*n + (i0+1), c_offset+1] += 1
        W[(k0+1)*n + (i0+1), c_offset+1] -= 1
        
        # M3
        U[i0*m + j0, c_offset+2] += 1
        V[j0*p + (k0+1), c_offset+2] += 1
        V[(j0+1)*p + (k0+1), c_offset+2] -= 1
        W[(k0+1)*n + i0, c_offset+2] += 1
        W[(k0+1)*n + (i0+1), c_offset+2] += 1
        
        # M4
        U[(i0+1)*m + (j0+1), c_offset+3] += 1
        V[(j0+1)*p + k0, c_offset+3] += 1
        V[j0*p + k0, c_offset+3] -= 1
        W[k0*n + i0, c_offset+3] += 1
        W[k0*n + (i0+1), c_offset+3] += 1
        
        # M5
        U[i0*m + j0, c_offset+4] += 1
        U[i0*m + (j0+1), c_offset+4] += 1
        V[(j0+1)*p + (k0+1), c_offset+4] += 1
        W[k0*n + i0, c_offset+4] -= 1
        W[(k0+1)*n + i0, c_offset+4] += 1
        
        # M6
        U[(i0+1)*m + j0, c_offset+5] += 1
        U[i0*m + j0, c_offset+5] -= 1
        V[j0*p + k0, c_offset+5] += 1
        V[j0*p + (k0+1), c_offset+5] += 1
        W[(k0+1)*n + (i0+1), c_offset+5] += 1
        
        # M7
        U[i0*m + (j0+1), c_offset+6] += 1
        U[(i0+1)*m + (j0+1), c_offset+6] -= 1
        V[(j0+1)*p + k0, c_offset+6] += 1
        V[(j0+1)*p + (k0+1), c_offset+6] += 1
        W[k0*n + i0, c_offset+6] += 1

    # Define the 4 blocks
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    # Track which (i,j,k) tuples are covered
    covered = np.zeros((n, m, p), dtype=bool)
    
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0+di, j0+dj, k0+dk] = True

    # Fill remaining with schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i, j, k]:
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    
    # Verification
    # T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
    # We check if this matches the delta tensor
    is_valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(U.shape[1]):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if val != 1:
                    is_valid = False
                    print(f"Invalid at ({i},{j},{k}): {val}")
    
    print(f"Rank: {U.shape[1]}")
    print(f"Valid: {is_valid}")
