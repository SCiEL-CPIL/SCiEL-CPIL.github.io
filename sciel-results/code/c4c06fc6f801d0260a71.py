import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We will use a block decomposition:
    # 1. 3x3x3 block at i=0..2, j=0..2, k=0..2. Rank 26 (Strassen 7 + Schoolbook 19)
    # 2. 3x2x5 block at i=0..2, j=3..4, k=0..4. Rank 21
    # 3. 3x3x2 block at i=0..2, j=0..2, k=3..4. Rank 12
    
    total_rank = 26 + 21 + 12
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(i0, j0, k0, c_start):
        # Term 0
        U[i0*m + j0][c_start] += 1
        U[(i0+1)*m + (j0+1)][c_start] += 1
        V[j0*p + k0][c_start] += 1
        V[(j0+1)*p + (k0+1)][c_start] += 1
        W[k0*n + i0][c_start] += 1
        W[(k0+1)*n + (i0+1)][c_start] += 1
        # Term 1
        c = c_start + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        # Term 2
        c = c_start + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        # Term 3
        c = c_start + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        # Term 4
        c = c_start + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        # Term 5
        c = c_start + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        # Term 6
        c = c_start + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Block 1: 3x3x3 (i=0..2, j=0..2, k=0..2)
    # Use Strassen on 2x2x2 at (0,0,0)
    add_strassen(0, 0, 0, col)
    col += 7
    
    # Cover remaining 19 terms in 3x3x3 with Schoolbook
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if not (i < 2 and j < 2 and k < 2):
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Block 2: 3x2x5 (i=0..2, j=3..4, k=0..4)
    # Decompose into 3x2x3 (k=0..2) and 3x2x2 (k=3..4)
    # 3x2x3: i=0..2, j=3..4, k=0..2. Rank 12.
    # Use Strassen on 2x2x2 part?
    # i=0..1, j=3..4, k=0..1 is 2x2x2. Rank 7.
    add_strassen(0, 3, 0, col)
    col += 7
    
    # Remaining in 3x2x3:
    # i=2, j=3..4, k=0..2 (6 terms)
    # i=0..1, j=3..4, k=2 (4 terms)
    for i in range(3):
        for j in range(3, 5):
            for k in range(3):
                if not (i < 2 and k < 2):
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # 3x2x2: i=0..2, j=3..4, k=3..4. Rank 9.
    # Use Strassen on 2x2x2 part?
    # i=0..1, j=3..4, k=3..4 is 2x2x2. Rank 7.
    add_strassen(0, 3, 3, col)
    col += 7
    
    # Remaining in 3x2x2:
    # i=2, j=3..4, k=3..4 (4 terms)
    for i in range(3):
        for j in range(3, 5):
            for k in range(3, 5):
                if not (i < 2):
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Block 3: 3x3x2 (i=0..2, j=0..2, k=3..4). Rank 12.
    # Use Strassen on 2x2x2 part?
    # i=0..1, j=0..1, k=3..4 is 2x2x2. Rank 7.
    add_strassen(0, 0, 3, col)
    col += 7
    
    # Remaining in 3x3x2:
    # i=2, j=0..2, k=3..4 (6 terms)
    # i=0..1, j=2, k=3..4 (4 terms)
    for i in range(3):
        for j in range(3):
            for k in range(3, 5):
                if not (i < 2 and j < 2):
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
