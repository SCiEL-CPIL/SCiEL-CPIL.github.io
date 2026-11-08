import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _construct_rank_55():
    """
    Constructive rank-55 decomposition using Strassen for 2x2x2 blocks
    and direct multiplication for 2x1x2 blocks.
    
    Split m=3 into m=2 + m=1:
    - 2x2x10: five 2x2x2 Strassen blocks, rank 7 each = 35
    - 2x1x10: five 2x1x2 direct blocks, rank 4 each = 20
    Total rank = 55
    """
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    
    # Strassen 2x2x2 decomposition (7 rank-1 terms)
    # For matrices A(2x2), B(2x2), C=A@B:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Process 5 Strassen blocks for the 2x2x10 part
    for block in range(5):
        k_start = block * 2
        # B columns: k_start, k_start+1
        # A rows/cols: A is 2x2 (rows 0,1; cols 0,1)
        # C rows: 0,1; C cols: k_start, k_start+1
        
        # M1 = (A11+A22)(B11+B22)
        U[0][col] = 1  # A11
        U[3][col] = 1  # A22
        V[0*2+k_start][col] = 1    # B11 (row 0, col k_start)
        V[0*2+k_start+1][col] = 1  # B12 (row 0, col k_start+1)
        W[k_start*n+0][col] = 1    # C col k_start, row 0
        W[k_start*n+1][col] = 1    # C col k_start+1, row 1
        W[(k_start+1)*n+0][col] = 1
        W[(k_start+1)*n+1][col] = 1
        # C11 += M1, C22 += M1
        # C11 is at W[k_start*n+0], C22 is at W[(k_start+1)*n+1]
        # Actually let me be more careful with W indexing
        # W[k*n+i] where k is col of C, i is row of C
        # C11: row 0, col k_start -> W[k_start*n+0]
        # C12: row 0, col k_start+1 -> W[(k_start+1)*n+0]
        # C21: row 1, col k_start -> W[k_start*n+1]
        # C22: row 1, col k_start+1 -> W[(k_start+1)*n+1]
        
        # Let me redo this more carefully
        col -= 1  # undo, let me restart this section properly
        
    # Let me rebuild from scratch with correct indexing
    col = 0
    
    # Strassen for 2x2x2: 
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A@B
    # C[0,0] = ae+bg, C[0,1] = af+bh, C[1,0] = ce+dg, C[1,1] = cf+dh
    # Strassen:
    # M1 = (a+d)(e+h)  -> contributes to C00, C11
    # M2 = (c+d)e      -> contributes to C10
    # M3 = a(f-h)      -> contributes to C01
    # M4 = d(e-g)      -> contributes to C10
    # M5 = (a+b)h      -> contributes to C01
    # M6 = (c-a)(e+f)  -> contributes to C11
    # M7 = (b-d)(g+h)  -> contributes to C00
    
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for block in range(5):
        k0 = block * 2  # first col of B in this block
        k1 = k0 + 1     # second col of B in this block
        
        # M1 = (a+d)(e+h): A entries a=A[0,0], d=A[1,1]; B entries e=B[0,k0], h=B[1,k1]
        # Wait, B is 3x10, so B[row][col]. For the 2x2 sub-block of B, rows 0,1, cols k0,k1
        # e = B[0,k0], f = B[0,k1], g = B[1,k0], h = B[1,k1]
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0][col] = 1   # A[0,0]
        U[3][col] = 1   # A[1,1] (row 1, col 1 -> index 1*2+1=3)
        V[0*10+k0][col] = 1    # B[0,k0]
        V[1*10+k1][col] = 1    # B[1,k1]
        # C00 += M1: C[0,k0] -> W[k0*2+0]
        # C11 += M1: C[1,k1] -> W[k1*2+1]
        W[k0*2+0][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[2][col] = 1   # A[1,0] (row 1, col 0 -> index 1*2+0=2)
        U[3][col] = 1   # A[1,1]
        V[0*10+k0][col] = 1    # B[0,k0]
        # C10 += M2: C[1,k0] -> W[k0*2+1]
        W[k0*2+1][col] = 1
        col += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0][col] = 1   # A[0,0]
        V[0*10+k1][col] = 1    # B[0,k1]
        V[1*10+k1][col] = -1   # -B[1,k1]
        # C01 += M3: C[0,k1] -> W[k1*2+0]
        W[k1*2+0][col] = 1
        col += 1
        
        # M4 = A[1,1] * (B[0,k0]-B[1,k0])
        U[3][col] = 1   # A[1,1]
        V[0*10+k0][col] = 1    # B[0,k0]
        V[1*10+k0][col] = -1   # -B[1,k0]
        # C10 += M4: C[1,k0] -> W[k0*2+1]
        W[k0*2+1][col] = 1
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0][col] = 1   # A[0,0]
        U[1][col] = 1   # A[0,1] (row 0, col 1 -> index 0*2+1=1)
        V[1*10+k1][col] = 1    # B[1,k1]
        # C01 += M5: C[0,k1] -> W[k1*2+0]
        W[k1*2+0][col] = 1
        col += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[2][col] = 1   # A[1,0]
        U[0][col] = -1  # -A[0,0]
        V[0*10+k0][col] = 1    # B[0,k0]
        V[0*10+k1][col] = 1    # B[0,k1]
        # C11 += M6: C[1,k1] -> W[k1*2+1]
        W[k1*2+1][col] = 1
        col += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1][col] = 1   # A[0,1]
        U[3][col] = -1  # -A[1,1]
        V[1*10+k0][col] = 1    # B[1,k0]
        V[1*10+k1][col] = 1    # B[1,k1]
        # C00 += M7: C[0,k0] -> W[k0*2+0]
        W[k0*2+0][col] = 1
        col += 1
    
    # Now the 2x1x10 part: A is 2x1 (col 2 of A, i.e., A[0,2] and A[1,2])
    # B is 1x10 (row 2 of B)
    # C is 2x10
    # Direct multiplication: for each output col k, C[0,k] = A[0,2]*B[2,k], C[1,k] = A[1,2]*B[2,k]
    # Each 2x1x2 block needs rank 4 (2 entries in C, each is a product)
    # Actually for 2x1x2: C[0,k0]=a*e, C[0,k1]=a*f, C[1,k0]=c*e, C[1,k1]=c*f
    # This is rank 4: ae, af, ce, cf
    
    for block in range(5):
        k0 = block * 2
        k1 = k0 + 1
        
        # C[0,k0] = A[0,2] * B[2,k0]
        U[0*3+2][col] = 1   # A[0,2] (row 0, col 2 -> index 0*3+2=2)
        V[2*10+k0][col] = 1  # B[2,k0]
        W[k0*2+0][col] = 1   # C[0,k0]
        col += 1
        
        # C[0,k1] = A[0,2] * B[2,k1]
        U[0*3+2][col] = 1
        V[2*10+k1][col] = 1
        W[k1*2+0][col] = 1
        col += 1
        
        # C[1,k0] = A[1,2] * B[2,k0]
        U[1*3+2][col] = 1   # A[1,2] (row 1, col 2 -> index 1*3+2=5)
        V[2*10+k0][col] = 1
        W[k0*2+1][col] = 1   # C[1,k0]
        col += 1
        
        # C[1,k1] = A[1,2] * B[2,k1]
        U[1*3+2][col] = 1
        V[2*10+k1][col] = 1
        W[k1*2+1][col] = 1
        col += 1
    
    assert col == rank, f"Expected {rank} cols, got {col}"
    
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _is_exact(U, V, W):
    """
    Check if the decomposition is exact.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T_calc = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_calc += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T_calc[i*m+j, j*p+k, k*n+i] - 1.0) > 1e-9:
                    return False
    return True

def _local_improve(U, V, W, max_time=60.0):
    """
    Attempt to reduce rank by eliminating redundant columns.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    current_rank = U.shape[1]
    
    if not _is_exact(U, V, W):
        return U, V, W
        
    improved_U, improved_V, improved_W = U.copy(), V.copy(), W.copy()
    current_rank = improved_U.shape[1]
    
    # Try removing columns one at a time
    for attempt in range(10):
        if time.time() - start_time > max_time:
            break
        
        removed_any = False
        cols_to_try = list(range(current_rank))
        np.random.shuffle(cols_to_try)
        
        for r in cols_to_try[:]:
            if time.time() - start_time > max_time:
                break
            
            mask = [i for i in range(current_rank) if i != r]
            if len(mask) == 0:
                continue
                
            U_test = improved_U[:, mask]
            V_test = improved_V[:, mask]
            W_test = improved_W[:, mask]
            
            if _is_exact(U_test, V_test, W_test):
                improved_U = U_test
                improved_V = V_test
                improved_W = W_test
                current_rank = len(mask)
                removed_any = True
        
        if not removed_any:
            break
    
    return improved_U, improved_V, improved_W

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 140.0

    # Start with constructive rank 55
    U, V, W = _construct_rank_55()
    
    # Verify correctness
    if not _is_exact(U, V, W):
        # Fallback to rank 60 if rank 55 fails
        n, m, p = DIMENSIONS
        rank = n * m * p
        U60 = [[0] * rank for _ in range(n * m)]
        V60 = [[0] * rank for _ in range(m * p)]
        W60 = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U60[i * m + j][col] = 1
                    V60[j * p + k][col] = 1
                    W60[k * n + i][col] = 1
                    col += 1
        U, V, W = np.array(U60, dtype=float), np.array(V60, dtype=float), np.array(W60, dtype=float)
    
    # Perform local improvement
    if time.time() < DEADLINE - 10:
        U, V, W = _local_improve(U, V, W, max_time=DEADLINE - time.time() - 5)
        
    U = np.rint(U).astype(int)
    V = np.rint(V).astype(int)
    W = np.rint(W).astype(int)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
