"""
Combined program: Uses the structural decomposition from Inspiration 1 
(2x1x6 + 3*2x2x2 Strassen blocks) which yields rank 33.
Includes rigorous verification and a fallback to schoolbook if verification fails.
Removes extensive random search in favor of deterministic correctness and speed.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Deterministic structural decomposition of rank 33.
    Decomposition:
    Split A (2x3) into A1 (2x1) and A2 (2x2).
    Split B (3x6) into B1 (1x6) and B2 (2x6).
    C = A1*B1 + A2*B2.
    A1*B1: 2x1x6, rank 12 (schoolbook).
    A2*B2: 2x2x6, split into three 2x2x2 blocks, each rank 7 (Strassen), total 21.
    Total rank: 33.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12
    
    # Build the target tensor T[a, b, c] for verification
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def structural_decomposition():
        """
        Return rank-33 decomposition based on block splitting.
        U: (6, 33), V: (18, 33), W: (12, 33)
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        col = 0
        
        # Part 1: A1 (2x1) * B1 (1x6) => 2x6 output
        # A1 uses column j=0 of A.
        # B1 uses row j=0 of B.
        # Indices:
        # U: i*m + 0
        # V: 0*p + k = k
        # W: k*n + i
        for k in range(p):
            for i in range(n):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        # col is now 12
        
        # Part 2: A2 (2x2) * B2 (2x6) => 2x6 output
        # A2 uses columns j=1,2 of A.
        # B2 uses rows j=1,2 of B.
        # Split B2 into three 2x2 blocks: cols (0,1), (2,3), (4,5) of B.
        # For each block t (0,1,2), use Strassen 7 multiplications.
        
        # Strassen formulas for 2x2x2:
        # Let A_sub be 2x2, B_sub be 2x2, C_sub be 2x2.
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Mapping to global indices:
        # A2 rows i=0,1; cols j=1,2.
        #   U index: i*m + j.
        #   A00 -> U[0*3+1] = U[1]
        #   A01 -> U[0*3+2] = U[2]
        #   A10 -> U[1*3+1] = U[4]
        #   A11 -> U[1*3+2] = U[5]
        
        # B2 rows j=1,2; cols k=2t, 2t+1.
        #   V index: j*p + k.
        #   B00 (j=1, k=2t) -> V[1*6+2t] = V[6+2t]
        #   B01 (j=1, k=2t+1) -> V[1*6+2t+1] = V[7+2t]
        #   B10 (j=2, k=2t) -> V[2*6+2t] = V[12+2t]
        #   B11 (j=2, k=2t+1) -> V[2*6+2t+1] = V[13+2t]
        
        # C submatrix rows i=0,1; cols k=2t, 2t+1.
        #   W index: k*n + i.
        #   C00 (i=0, k=2t) -> W[2t*2+0] = W[4t]
        #   C01 (i=0, k=2t+1) -> W[(2t+1)*2+0] = W[4t+2]
        #   C10 (i=1, k=2t) -> W[2t*2+1] = W[4t+1]
        #   C11 (i=1, k=2t+1) -> W[(2t+1)*2+1] = W[4t+3]

        for t in range(3):
            # M1 = (A00+A11)(B00+B11)
            U[1, col] = 1  # A00
            U[5, col] = 1  # A11
            V[6+2*t, col] = 1  # B00
            V[13+2*t, col] = 1  # B11
            W[4*t, col] = 1  # C00
            W[4*t+3, col] = 1  # C11
            col += 1
            
            # M2 = (A10+A11)B00
            U[4, col] = 1  # A10
            U[5, col] = 1  # A11
            V[6+2*t, col] = 1  # B00
            W[4*t+1, col] = 1  # C10
            W[4*t+3, col] = -1  # C11
            col += 1
            
            # M3 = A00(B01-B11)
            U[1, col] = 1  # A00
            V[7+2*t, col] = 1  # B01
            V[13+2*t, col] = -1  # B11
            W[4*t+2, col] = 1  # C01
            W[4*t+3, col] = 1  # C11
            col += 1
            
            # M4 = A11(B10-B00)
            U[5, col] = 1  # A11
            V[12+2*t, col] = 1  # B10
            V[6+2*t, col] = -1  # B00
            W[4*t, col] = 1  # C00
            W[4*t+1, col] = 1  # C10
            col += 1
            
            # M5 = (A00+A01)B11
            U[1, col] = 1  # A00
            U[2, col] = 1  # A01
            V[13+2*t, col] = 1  # B11
            W[4*t, col] = -1  # C00
            W[4*t+2, col] = 1  # C01
            col += 1
            
            # M6 = (A10-A00)(B00+B01)
            U[4, col] = 1  # A10
            U[1, col] = -1  # A00
            V[6+2*t, col] = 1  # B00
            V[7+2*t, col] = 1  # B01
            W[4*t+3, col] = 1  # C11
            col += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[2, col] = 1  # A01
            U[5, col] = -1  # A11
            V[12+2*t, col] = 1  # B10
            V[13+2*t, col] = 1  # B11
            W[4*t, col] = 1  # C00
            col += 1
        
        return U, V, W

    # Start with the structural decomposition
    best_U, best_V, best_W = structural_decomposition()
    best_rank = 33
    
    # Verify the baseline
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook if structural construction is wrong
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p

    # Optional: Short random search for improvement (unlikely to succeed for rank < 33)
    # We only do this if we have significant time left and the structural one failed or we want to be thorough.
    # Given the stagnation, we trust the structural rank 33.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
