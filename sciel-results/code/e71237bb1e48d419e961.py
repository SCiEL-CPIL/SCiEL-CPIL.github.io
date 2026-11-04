"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use 6 blocks of 2x2x2 Strassen decomposition.
    # Total rank = 6 * 7 = 42.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Define the 6 blocks: (i_start, j_start, k_start)
    # i is 0..1, j is 0..3, k is 0..5
    # We split j into 0..1 and 2..3
    # We split k into 0..1, 2..3, 4..5
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4)
    ]
    
    for i0, j0, k0 in blocks:
        # Local indices for A (2x2), B (2x2), C (2x2)
        # A[i, j] global index: (i0+i)*m + (j0+j)
        # B[j, k] global index: (j0+j)*p + (k0+k)
        # C[i, k] global index: (k0+k)*n + (i0+i)
        
        # Strassen terms
        # 1. (A00+A11)(B00+B11)
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # 2. (A10+A11)B00
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # 3. A00(B10-B11)
        U[i0*m + j0][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[k0*n + i0][col] += 1
        col += 1
        
        # 4. A11(B01-B00)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[j0*p + k0][col] -= 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # 5. (A00+A10)B11
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + j0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # 6. (A10-A00)(B00+B01)
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # 7. (A01-A11)(B10+B11)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
