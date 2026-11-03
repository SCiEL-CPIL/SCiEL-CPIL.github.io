"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x4 matrix by a 4x8 matrix.
    
    Strategy:
    - Start with 8 independent Strassen 2x2 blocks (rank 56).
    - Use a time budget to attempt to find lower-rank decompositions via
      randomized search over integer/half-integer entries.
    - Return the best (lowest rank) valid decomposition found before deadline.
    """
    t0 = time.time()
    DEADLINE = 120.0  # seconds
    SAFETY_MARGIN = 10.0  # stop searching this many seconds before deadline
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    
    # The multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # U has row index (i, j) for A[i,j] -> row = i*m + j
    # V has row index (j, k) for B[j,k] -> row = j*p + k
    # W has row index (k, i) for C[i,k] -> row = k*n + i  (trace convention)
    
    def verify(U, V, W):
        """Check that the decomposition is exact: 
        For all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] == delta_{i,k in W}
        More precisely: the tensor T_{i,j,k} = sum_c U[i*m+j,c]*V[j*p+k,c] 
        should equal the matrix multiplication tensor, which has T_{i,j,k} = 1 
        if the output C[i,k] receives contribution from A[i,j]*B[j,k].
        
        Actually, the correct condition: 
        C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] * (something)
        
        The tensor decomposition means:
        For the bilinear form: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_k' W[k*n+i,c] ... )
        
        Let me think more carefully. The standard tensor decomposition of matrix multiplication:
        T_{i,j,k} is the tensor such that C[i,k] = sum_j A[i,j] * B[j,k].
        
        A decomposition of rank r means:
        T_{i,j,k} = sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[k,i,c]
        
        Wait, that's not quite right either. Let me use the standard formulation.
        
        The matrix multiplication tensor M_{i,j,k} satisfies:
        C_{i,k} = sum_j A_{i,j} M_{i,j,k} B_{j,k} ... no.
        
        Actually, C_{i,k} = sum_j A_{i,j} B_{j,k}.
        
        A rank-r decomposition means:
        C_{i,k} = sum_{c=1}^r (sum_j U_{i,j,c} A_{i,j}) (sum_k' ... ) 
        
        Hmm, let me use the standard convention from the problem statement:
        U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
        
        The decomposition is valid if for all i, j, k:
        sum_c U[i*m+j, c] * V[j*p+k, c] = [contribution to C]
        
        But C[i,k] = sum_j A[i,j] B[j,k], so the "tensor" is:
        T[i,j,k] = 1 for all i,j,k (since every A[i,j] and B[j,k] pair contributes to C[i,k])
        
        No wait. The tensor for matrix multiplication (n x m) * (m x p) -> (n x p) is:
        T[i, j, k] = delta where the indices are: i in [0,n), j in [0,m), k in [0,p)
        and T[i,j,k] = 1 always (it's a constant tensor of ones in the bilinear sense).
        
        Actually no. Let me reconsider. The matrix multiplication can be written as:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        This is a trilinear map (A, B) -> C. The "tensor" is the structure tensor.
        
        A rank-r decomposition means:
        C[i,k] = sum_{c=1}^r (sum_j U[i*m+j, c] * A[i,j]) * (sum_k' ... )
        
        I think the correct interpretation is:
        C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_k2 V[j*p+k2, c] * B[j,k2])
        
        Hmm, that doesn't dimension-check cleanly. Let me look at the schoolbook example:
        
        In the schoolbook: for each (i,j,k): U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
        So C[i,k] = sum_c W[k*n+i,c] * ... 
        
        I think the reconstruction is:
        C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * ... 
        
        No. Let me just verify numerically with the schoolbook.
        
        Actually, I think the decomposition means:
        For the output C[i,k]:
        C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_k2 V[j*p+k2, c] * B[j,k2])
        
        That can't be right because j is summed in two places.
        
        Let me think about it differently. The standard formulation:
        A tensor T with indices (i, j, k) can be decomposed as:
        T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        
        But here we have U with 2 indices (i,j) flattened, V with (j,k) flattened, W with (k,i) flattened.
        
        So the tensor is T[i,j,k] where:
        - i indexes rows of A (and C): 0..n-1
        - j indexes cols of A / rows of B: 0..m-1  
        - k indexes cols of B (and C): 0..p-1
        
        And T[i,j,k] = 1 (the matrix multiplication tensor is all ones in this indexing).
        
        The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        Wait, but that would mean for schoolbook:
        For a given c corresponding to (i0,j0,k0):
        U[i*m+j, c] = 1 if i=i0 and j=j0
        V[j*p+k, c] = 1 if j=j0 and k=k0
        W[k*n+i, c] = 1 if k=k0 and i=i0
        
        So the product is 1 only when i=i0, j=j0, k=k0. Summing over all c=(i0,j0,k0),
        T[i,j,k] = 1 if there exists c with i=i0,j=j0,k=k0, i.e., T[i,j,k]=1. Correct!
        
        So the verification is: for all i, j, k:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        """
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def random_decomposition(rank, seed=None):
        """Generate a random decomposition with integer entries in {-1, 0, 1}."""
        rng = random.Random(seed)
        U = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m*p)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p*n)]
        return U, V, W
    
    def try_random(rank, num_trials, seed=0):
        """Try random decompositions of given rank."""
        for t in range(num_trials):
            U, V, W = random_decomposition(rank, seed + t)
            if verify(U, V, W):
                return U, V, W
        return None
    
    # Start with Strassen-based rank 56 decomposition
    def strassen_decomposition():
        """Construct rank-56 decomposition using 8 Strassen 2x2 blocks."""
        rank = 56
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        # Strassen 2x2: C = AB where A,B are 2x2
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b22)
        # M4 = a22(b12-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # For our case: A is 2x4, B is 4x8.
        # Split A into left 2x2 and right 2x2: A_left = A[:,0:2], A_right = A[:,2:4]
        # Split B into 4 blocks of 2x2 along rows: B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4]
        # And similarly for the right half of B.
        # 
        # Actually, let me think about this differently.
        # A (2x4) * B (4x8) = C (2x8)
        # 
        # We can write: C[:, 0:2] = A * B[0:4, 0:2]
        # C[:, 2:4] = A * B[0:4, 2:4]
        # etc.
        # 
        # For C[:, 0:2] = A * B[:, 0:2] where A is 2x4 and B[:,0:2] is 4x2.
        # Split A into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
        # Split B[:,0:2] into B_T (2x2, rows 0-1) and B_B (2x2, rows 2-3).
        # Then A * B[:,0:2] = A_L * B_T + A_R * B_B.
        # Each product is 2x2 * 2x2 = 2x2, which can be done with Strassen (rank 7).
        # So C[:,0:2] needs rank 7+7 = 14.
        # Similarly for each of the 4 output column pairs.
        # Total rank = 4 * 14 = 56.
        
        # For each output column pair (col_pair in 0..3):
        #   Output columns: 2*col_pair, 2*col_pair+1
        #   B block rows: all 4 rows, cols 2*col_pair..2*col_pair+1
        #   Split B block into top 2 rows and bottom 2 rows
        #   A_L = A[:, 0:2], A_R = A[:, 2:4]
        #   C_out = A_L * B_top + A_R * B_bot
        #   Each term is Strassen rank 7.
        
        strassen_entries = [
            # (A_indices, B_indices, coeff) for each of 7 multiplications
            # M1 = (a11+a22)(b11+b22)
            [(0,0,1), (1,1,1), (0,0,1), (1,1,1)],  # not quite, let me use explicit lists
        ]
        
        # Let me define Strassen more carefully.
        # For 2x2 matrices X (indices x00,x01,x10,x11) and Y (y00,y01,y10,y11):
        # M1 = (x00+x11)*(y00+y11)
        # M2 = (x10+x11)*y00
        # M3 = x00*(y10-y11)
        # M4 = x11*(y01-y00)
        # M5 = (x00+x01)*y11
        # M6 = (x10-x00)*(y00+y01)
        # M7 = (x01-x11)*(y10+y11)
        # Z00 = M1+M4-M5+M7
        # Z01 = M3+M5
        # Z10 = M2+M4
        # Z11 = M1-M2+M3+M6
        
        # Each M_c contributes to U, V, W.
        # For a product X*Y where X has row index (a_i, a_j) in A and Y has row index (b_j, b_k) in B:
        # U entries: for X, the row is a_i*m + a_j for the element x_ij
        # V entries: for Y, the row is b_j*p + b_k for the element y_jk
        # W entries: for Z, the row is k*n + i for the element z_ik
        
        # Let me build this up.
        
        def add_strassen_block(a_row_start, a_col_start, b_row_start, b_col_start, c_col_start, offset):
            """Add a Strassen decomposition for A[a_row_start:a_row_start+2, a_col_start:a_col_start+2] * B[b_row_start:b_row_start+2, b_col_start:b_col_start+2] -> C[a_row_start:a_row_start+2, c_col_start:c_col_start+2]
            
            a_row_start: starting row in A (0 or 1, since n=2)
            a_col_start: starting col in A (0, 2)
            b_row_start: starting row in B (0, 2)
            b_col_start: starting col in B (0, 2, 4, 6)
            c_col_start: starting col in C (0, 2, 4, 6)
            offset: rank offset
            """
            # X = A[a_row_start:a_row_start+2, a_col_start:a_col_start+2]
            # Y = B[b_row_start:b_row_start+2, b_col_start:b_col_start+2]
            # Z = C[a_row_start:a_row_start+2, c_col_start:c_col_start+2]
            
            # X indices: x00=A[a0,ac0], x01=A[a0,ac1], x10=A[a1,ac0], x11=A[a1,ac1]
            # where a0=a_row_start, a1=a_row_start+1, ac0=a_col_start, ac1=a_col_start+1
            # Y indices: y00=B[br0,bc0], y01=B[br0,bc1], y10=B[br1,bc0], y11=B[br1,bc1]
            # where br0=b_row_start, br1=b_row_start+1, bc0=b_col_start, bc1=b_col_start+1
            # Z indices: z00=C[a0,cc0], z01=C[a0,cc1], z10=C[a1,cc0], z11=C[a1,cc1]
            # where cc0=c_col_start, cc1=c_col_start+1
            
            a0, a1 = a_row_start, a_row_start + 1
            ac0, ac1 = a_col_start, a_col_start + 1
            br0, br1 = b_row_start, b_row_start + 1
            bc0, bc1 = b_col_start, b_col_start + 1
            cc0, cc1 = c_col_start, c_col_start + 1
            
            # Define the 7 Strassen multiplications
            # Each is: (list of (a_row, a_col, coeff) for X, list of (b_row, b_col, coeff) for Y)
            strassen = [
                # M1 = (x00+x11)*(y00+y11)
                ([(a0, ac0, 1), (a1, ac1, 1)], [(br0, bc0, 1), (br1, bc1, 1)]),
                # M2 = (x10+x11)*y00
                ([(a1, ac0, 1), (a1, ac1, 1)], [(br0, bc0, 1)]),
                # M3 = x00*(y10-y11)
                ([(a0, ac0, 1)], [(br1, bc0, 1), (br1, bc1, -1)]),
                # M4 = x11*(y01-y00)
                ([(a1, ac1, 1)], [(br0, bc1, 1), (br0, bc0, -1)]),
                # M5 = (x00+x01)*y11
                ([(a0, ac0, 1), (a0, ac1, 1)], [(br1, bc1, 1)]),
                # M6 = (x10-x00)*(y00+y01)
                ([(a1, ac0, 1), (a0, ac0, -1)], [(br0, bc0, 1), (br0, bc1, 1)]),
                # M7 = (x01-x11)*(y10+y11)
                ([(a0, ac1, 1), (a1, ac1, -1)], [(br1, bc0, 1), (br1, bc1, 1)]),
            ]
            
            # Output contributions:
            # Z00 = M1+M4-M5+M7
            # Z01 = M3+M5
            # Z10 = M2+M4
            # Z11 = M1-M2+M3+M6
            output_coeffs = {
                (a0, cc0): [(0, 1), (3, 1), (4, -1), (6, 1)],   # M1, M4, -M5, M7
                (a0, cc1): [(2, 1), (4, 1)],                     # M3, M5
                (a1, cc0): [(1, 1), (3, 1)],                     # M2, M4
                (a1, cc1): [(0, 1), (1, -1), (2, 1), (5, 1)],   # M1, -M2, M3, M6
            }
            
            for s in range(7):
                c = offset + s
                a_terms, b_terms = strassen[s]
                for (ar, ac, coeff) in a_terms:
                    U[ar*m + ac][c] += coeff
                for (br, bc, coeff) in b_terms:
                    V[br*p + bc][c] += coeff
                # W will be set based on output contributions
            
            # Set W entries
            for (out_ar, out_cc), coeffs in output_coeffs.items():
                for (s, coeff) in coeffs:
                    c = offset + s
                    W[out_cc*n + out_ar][c] += coeff
            
            return offset + 7
        
        offset = 0
        for col_pair in range(4):  # 4 pairs of output columns
            b_col = 2 * col_pair
            c_col = 2 * col_pair
            # Top 2 rows of B block
            offset = add_strassen_block(0, 0, 0, b_col, c_col, offset)  # A[:,0:2] * B[0:2, b_col:b_col+2]
            offset = add_strassen_block(0, 2, 2, b_col, c_col, offset)  # A[:,2:4] * B[2:4, b_col:b_col+2]
        
        return U, V, W, rank
    
    U, V, W, current_rank = strassen_decomposition()
    best_U, best_V, best_W, best_rank = U, V, W, current_rank
    
    # Verify the Strassen decomposition
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_U, best_V, best_W, best_rank = U, V, W, rank
        current_rank = rank
    
    # Try to find lower rank decompositions using random search
    # Try ranks from current_rank-1 down to some minimum
    # Known lower bound: at least n*m + m*p + p*n - ... actually the tensor rank of 2x4x8 matmul
    # is unknown exactly, but we know it's at most 56.
    
    if time.time() - t0 < DEADLINE - SAFETY_MARGIN:
        # Try random searches for lower ranks
        for rank in range(best_rank - 1, max(best_rank - 20, 20), -1):
            if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
                break
            
            num_trials = 500
            result = try_random(rank, num_trials)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W, best_rank = U, V, W, rank
                break
    
    # Check if we're still within time and try a few more
    if time.time() - t0 < DEADLINE - SAFETY_MARGIN:
        for rank in range(max(best_rank - 5, 20), best_rank):
            if time.time() - t0 > DEADLINE - SAFETY_MARGIN:
                break
            num_trials = 200
            result = try_random(rank, num_trials)
            if result is not None:
                best_rank = rank
                best_U, best_V, best_W = result
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
