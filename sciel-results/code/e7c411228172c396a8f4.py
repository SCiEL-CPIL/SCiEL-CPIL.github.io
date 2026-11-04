"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Rank 50 decomposition:
    # 6 blocks of 2x2x2 (Strassen, rank 7 each) -> 42
    # 8 rank-1 terms for 2x4x1 -> 8
    # Total 50
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for 2x2x2 matrix multiplication
    # A block: rows i0, i0+1; cols j0, j0+1
    # B block: rows j0, j0+1; cols k0, k0+1
    # C block: rows i0, i0+1; cols k0, k0+1
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Strassen mults M1..M7
        # M1: (A11+A22)(B11+B22)
        # M2: (A21+A22)B11
        # M3: A11(B12-B22)
        # M4: A22(B21-B11)
        # M5: (A11+A12)B22
        # M6: (A21-A11)(B11+B12)
        # M7: (A12-A22)(B21+B22)
        
        # U indices: i*m + j
        # V indices: j*p + k
        # W indices: k*n + i
        
        # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        # C[i0,k0] += M1, C[i0+1,k0+1] += M1
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        # C[i0+1,k0] += M2, C[i0+1,k0+1] -= M2
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        # M3: A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        # C[i0,k0+1] += M3, C[i0+1,k0+1] += M3
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4: A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        # C[i0,k0] += M4, C[i0+1,k0] += M4
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        # C[i0,k0] -= M5, C[i0,k0+1] += M5
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        # C[i0+1,k0+1] += M6
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        # C[i0,k0] += M7
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Tile the 4x7 B matrix into 2x2 blocks (columns 0-1, 2-3, 4-5) and one 2x1 block (column 6)
    # A is 2x4, tiled into 2x2 blocks (columns 0-1, 2-3)
    
    # For each 2x2 block of A (2 blocks: cols 0-1, cols 2-3)
    # and each 2x2 block of B (3 blocks: cols 0-1, 2-3, 4-5)
    # we get a 2x2x2 Strassen multiplication
    # Total: 2 * 3 = 6 Strassen blocks = 42 rank
    
    # A block 0: cols 0-1 (j0=0)
    # A block 1: cols 2-3 (j0=2)
    
    # B block 0: cols 0-1 (k0=0)
    # B block 1: cols 2-3 (k0=2)
    # B block 2: cols 4-5 (k0=4)
    
    # 6 Strassen blocks
    add_strassen(0, 0, 0)  # A(0:2,0:2) * B(0:2,0:2) -> C(0:2,0:2)
    add_strassen(0, 0, 2)  # A(0:2,0:2) * B(0:2,2:4) -> C(0:2,2:4)
    add_strassen(0, 0, 4)  # A(0:2,0:2) * B(0:2,4:6) -> C(0:2,4:6)
    add_strassen(0, 2, 0)  # A(0:2,2:4) * B(2:4,0:2) -> C(0:2,0:2)
    add_strassen(0, 2, 2)  # A(0:2,2:4) * B(2:4,2:4) -> C(0:2,2:4)
    add_strassen(0, 2, 4)  # A(0:2,2:4) * B(2:4,4:6) -> C(0:2,4:6)
    
    # Remaining: column 6 of B (2x1 block)
    # This requires 2*4*1 = 8 rank-1 terms
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    # Verify decomposition if time permits
    if time.time() - start_time < DEADLINE - 10:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all valid (i,j,k)
            # For matmul: C[i,k] = sum_j A[i,j]*B[j,k]
            # The tensor entry T[i,j,k] = 1 for all valid (i,j,k)
            
            # Verify on all entries
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        lhs = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                        err = abs(lhs - 1.0)
                        if err > max_err:
                            max_err = err
                            if err > 1e-9:
                                print(f"Verification failed at i={i},j={j},k={k}: lhs={lhs}")
            
            if max_err < 1e-9:
                print(f"Verification passed with max error: {max_err}")
            else:
                print(f"Verification failed with max error: {max_err}")
        except Exception as e:
            print(f"Verification error: {e}")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
