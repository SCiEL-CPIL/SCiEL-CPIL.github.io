import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    # Build the multiplication tensor T[i*m+j, j*p+k, k*n+i]
    # For each term c: sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal
    # 1 if the product entry (i,k) gets contribution from j, i.e., for all i,j,k the
    # sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for each (i,j,k) triple.
    # Actually the tensor is: T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # and the decomposition gives: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal T[i,j,k]=1
    
    # More precisely, the matrix multiplication tensor has T[i,j,k]=1 for all valid i,j,k,
    # and the decomposition requires: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's verify using the matrix form: C = A * B
    # A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: the tensor T has T[i,j,k] = 1 (all ones for valid indices)
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Build U, V, W as numpy arrays
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Start with schoolbook as baseline
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best_rank = rank_sb
    best_result = {"U": U, "V": V, "W": W}
    
    # Try Strassen-based approach on 2x4x8 subproblems
    # Split the 3x4x8 into: first 2 rows (2x4x8) + last 1 row (1x4x8)
    # For 2x4x8: use Strassen on 2x2x2 blocks
    # 4 = 2*2, 8 = 4*2
    # Split 4x8 B into four 2x2 blocks: B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4]
    # A[0:2,:] is 2x4, split into A[0:2,0:2] (2x2) and A[0:2,2:4] (2x2)
    # C[0:2,0:4] = A[0:2,0:2]*B[0:2,0:4] + A[0:2,2:4]*B[2:4,0:4]
    # This is a 2x2x4 problem twice, or we can do it more carefully.
    
    # Let's think about 2x4x8 more carefully.
    # A is 2x4, B is 4x8.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k] for i in {0,1}, k in {0,...,7}
    
    # Split j into two groups: j in {0,1} and j in {2,3}
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    # Let A1 = A[:,0:2] (2x2), A2 = A[:,2:4] (2x2)
    # Let B1 = B[0:2,:] (2x8), B2 = B[2:4,:] (2x8)
    # C = A1*B1 + A2*B2, where A1*B1 is 2x8 and A2*B2 is 2x8.
    
    # For A1*B1: A1 is 2x2, B1 is 2x8.
    # Split B1 into four 2x2 blocks: B1[0:2,0:2], B1[0:2,2:4], B1[0:2,4:6], B1[0:2,6:8]
    # A1*B1 = A1 * [B11 B12 B13 B14] = [A1*B11, A1*B12, A1*B13, A1*B14]
    # Each A1*B1r is a 2x2 * 2x2 multiplication, which can be done in rank 7 (Strassen).
    # So A1*B1 uses 4 * 7 = 28 rank.
    # Similarly A2*B2 uses 4 * 7 = 28 rank.
    # Total for 2x4x8: 56 rank (vs schoolbook 64).
    
    # For 1x4x8: A is 1x4, B is 4x8.
    # C[k] = sum_j A[j]*B[j,k] for k in 0..7
    # This is just a dot product, schoolbook rank = 4*8 = 32.
    # Can we do better? 1x4x8 tensor... this is essentially a vector-matrix product.
    # The rank of 1xm x p is m*p (no known improvement).
    
    # So total: 56 + 32 = 88 rank vs schoolbook 96.
    
    # Let's build this decomposition.
    # Strassen 2x2x2: 
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M3 + M6 - M7
    
    # For a 2x2 * 2x2 multiplication at positions (i0, j0) in A, (j0, k0) in B, output at (i0, k0):
    # We need to add terms to U, V, W.
    
    # Let's define a helper to add a Strassen 2x2x2 block.
    # A indices: (i0+0, j0+0), (i0+0, j0+1), (i0+1, j0+0), (i0+1, j0+1)
    # B indices: (j0+0, k0+0), (j0+0, k0+1), (j0+1, k0+0), (j0+1, k0+1)
    # C indices: (i0+0, k0+0), (i0+0, k0+1), (i0+1, k0+0), (i0+1, k0+1)
    
    def add_strassen_2x2x2(U, V, W, i0, j0, k0, start_col):
        """Add Strassen decomposition for 2x2 A block at (i0,j0), 2x2 B block at (j0,k0).
        Returns next available column."""
        col = start_col
        # M1 = (A00+A11)(B00+B11)
        # U: A00 at (i0,j0) gets +1, A11 at (i0+1,j0+1) gets +1
        # V: B00 at (j0,k0) gets +1, B11 at (j0+1,k0+1) gets +1
        # W: C00 at (k0,i0) gets +1, C01 at (k0+1,i0) gets +1, C10 at (k0,i0+1) gets +1, C11 at (k0+1,i0+1) gets +1
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M3 = A00(B10-B11)
        U[i0*m + j0][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += -1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M4 = A11(B01-B10)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[j0*p + k0][col] += -1
        W[k0*n + i0][col] += -1
        W[(k0+1)*n + i0][col] += -1
        W[k0*n + (i0+1)][col] += -1
        W[(k0+1)*n + (i0+1)][col] += -1
        col += 1
        
        # M5 = (A00+A01)B11
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B10)
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] += -1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B01+B11)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += -1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += -1
        W[(k0+1)*n + i0][col] += -1
        col += 1
        
        return col
    
    # Build the 88-rank decomposition
    if time.time() < deadline:
        rank_new = 88
        U2 = [[0] * rank_new for _ in range(n * m)]
        V2 = [[0] * rank_new for _ in range(m * p)]
        W2 = [[0] * rank_new for _ in range(p * n)]
        col = 0
        
        # First 2 rows: 2x4x8 using Strassen
        # A is 2x4, B is 4x8
        # Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
        # Split B into B1 (2x8, rows 0-1) and B2 (2x8, rows 2-3)
        # C = A1*B1 + A2*B2
        # A1*B1: A1 is 2x2, B1 is 2x8. Split B1 into four 2x2 blocks.
        for kb in range(4):  # k block 0..3, each 2 wide
            col = add_strassen_2x2x2(U2, V2, W2, 0, 0, kb*2, col)
        # A2*B2: A2 is 2x2 (A cols 2-3), B2 is 2x8 (B rows 2-3)
        for kb in range(4):
            col = add_strassen_2x2x2(U2, V2, W2, 0, 2, kb*2, col)
        
        # Last row: 1x4x8 schoolbook (rank 32)
        for j in range(4):
            for k in range(8):
                U2[2*m + j][col] = 1
                V2[j*p + k][col] = 1
                W2[k*n + 2][col] = 1
                col += 1
        
        if col == rank_new:
            # Verify
            if _verify_decomposition(U2, V2, W2, n, m, p):
                best_rank = rank_new
                best_result = {"U": U2, "V": V2, "W": W2}
            else:
                print("Strassen decomposition failed verification, using schoolbook")
        else:
            print(f"Column mismatch: {col} vs {rank_new}, using schoolbook")
    
    # Try even better: use Strassen for the 1x4x8 part?
    # 1x4x8: A is 1x4, B is 4x8.
    # C[k] = sum_j A[j]*B[j,k]
    # This is rank 32 schoolbook. Can we do better?
    # Actually, this is a "thin" matrix product. The tensor 1x4x8 has rank exactly 32
    # (it's equivalent to a 4x8 identity-like structure). No improvement possible.
    
    # What about using Strassen on the full 3x4x8 differently?
    # We could also try: split 3x4x8 into 2x4x4 + 2x4x4 + 1x4x8
    # Or use a different partitioning.
    
    # Try: 3x4x8 as three 1x4x8 blocks. Each is rank 32, total 96. Same as schoolbook.
    
    # Try: use Laderman's algorithm for 3x3x3 (rank 23) on subblocks?
    # 3x4x8: split 3x4 into 3x2 + 3x2, and 4x8 into 4x4 + 4x4.
    # Each 3x2x4 block: schoolbook rank 24. Two blocks: 48. Plus cross terms?
    # No, this doesn't work directly because the splits don't align for tensor decomposition.
    
    # Actually, let me reconsider. The 3x4x8 tensor can be viewed as:
    # For i in 0..2, k in 0..7: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # This is 24 independent "lines" in the tensor.
    # Schoolbook: rank 96 (each (i,j,k) triple is one term).
    # Strassen approach above: rank 88.
    
    # Can we do better by applying Strassen to the 3x2x4 subproblems?
    # Split 3x4x8 into 3x2x8 + 3x2x8 (split j into 0-1 and 2-3).
    # 3x2x8: A is 3x2, B is 2x8.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Schoolbook rank: 3*2*8 = 48. Two such blocks: 96. Same as before.
    # Can we improve 3x2x8? 
    # 3x2x8: this is like two 3x1x8 sums. Not obviously improvable.
    
    # I'll stick with the 88-rank Strassen decomposition.
    
    return best_result
