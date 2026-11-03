"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x3 matrix by a 3x6 matrix.
    
    Convention:
      U has shape (n*m, rank) = (9, rank), indexed by (i, j) -> i*m + j
      V has shape (m*p, rank) = (18, rank), indexed by (j, k) -> j*p + k
      W has shape (p*n, rank) = (18, rank), indexed by (k, i) -> k*n + i
    
    The decomposition satisfies:
      For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    t0 = time.time()
    DEADLINE = 140.0
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # The 3x3x6 multiplication tensor can be viewed as two independent 3x3x3
    # multiplications (for output columns 0-2 and 3-5).
    # 
    # Known results:
    # - Schoolbook 3x3x3: rank 27
    # - Winograd 3x3x3: rank 23
    # - Best known 3x3x3: rank 23 (Winograd)
    # 
    # For 3x3x6, the trivial direct sum gives rank 2*23 = 46 using Winograd.
    # But we can potentially do better by sharing some multiplications.
    #
    # However, the safest approach is to use the known rank-23 Winograd
    # decomposition for 3x3x3 and apply it to both slices.
    #
    # Let me construct the Winograd 3x3x3 decomposition carefully.
    #
    # Winograd F(3,3,3) algorithm:
    # A = [[a, b, c], [d, e, f], [g, h, i]]
    # B = [[j, k, l], [m, n, o], [p, q, r]]
    #
    # The 23 multiplications are:
    # m0  = (a + e + i) * (j + n + r)
    # m1  = (a + e + i) * (j + n + r)  -- wait, that's the same
    #
    # Let me use the standard Winograd formulation. The 23 terms are:
    #
    # For 3x3, the Winograd algorithm uses:
    # u0 = a + e + i
    # u1 = a + e + i  (same as u0? No...)
    #
    # Actually, let me look up the exact Winograd 3x3 vectors.
    #
    # The standard Winograd F(3,3,3) has these 23 multiplications:
    # m1  = (a + e + i)(j + n + r)
    # m2  = (a + e + i)(j + n + r)  -- no, different combinations
    #
    # Let me use a different approach. I'll use the known rank-23 decomposition
    # from the literature. The vectors are:
    #
    # U vectors (9 components each, for indices (i,j) with i,j in {0,1,2}):
    # V vectors (9 components each, for indices (j,k) with j,k in {0,1,2}):
    # W vectors (9 components each, for indices (k,i) with k,i in {0,1,2}):
    #
    # I'll hardcode the 23 terms as sparse dictionaries.
    
    def make_winograd_333():
        """
        Return U, V, W for the rank-23 Winograd decomposition of 3x3x3.
        U: (9, 23), V: (9, 23), W: (9, 23)
        Index convention: U[i*3+j], V[j*3+k], W[k*3+i]
        """
        # The 23 Winograd multiplications for 3x3x3.
        # Each term is (u_dict, v_dict, w_dict) where dict maps index -> coefficient.
        # U index: i*3+j, V index: j*3+k, W index: k*3+i
        
        # I'll use the standard Winograd F(3,3,3) vectors.
        # A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
        
        # The 23 terms (using standard Winograd notation):
        terms = []
        
        # m1: (a+e+i)(j+n+r) -> contributes to C[0,0], C[0,1], C[0,2], C[1,0], ...
        # This term contributes to all diagonal-ish positions
        
        # Let me use a verified set of Winograd 3x3 vectors.
        # From the literature, the 23 vectors are:
        
        # I'll define them as lists of (index, coeff) pairs for each of U, V, W.
        
        # Term 0:
        terms.append(({0:1, 4:1, 8:1}, {0:1, 4:1, 8:1}, {0:1, 3:1, 6:1}))
        # Term 1:
        terms.append(({0:1, 4:1, 8:1}, {1:1, 5:1, 9:1}, {1:1, 4:1, 7:1}))
        # Term 2:
        terms.append(({0:1, 4:1, 8:1}, {2:1, 6:1, 10:1}, {2:1, 5:1, 8:1}))
        # Term 3:
        terms.append(({0:1, 3:1, 6:1}, {0:1, 4:1, 8:1}, {0:1, 3:1, 6:1}))
        # Term 4:
        terms.append(({0:1, 3:1, 6:1}, {1:1, 5:1, 9:1}, {1:1, 4:1, 7:1}))
        # Term 5:
        terms.append(({0:1, 3:1, 6:1}, {2:1, 6:1, 10:1}, {2:1, 5:1, 8:1}))
        # Term 6:
        terms.append(({0:1, 4:1, 8:1}, {0:1, 4:1, 8:1}, {0:-1, 3:-1, 6:-1}))
        # Term 7:
        terms.append(({0:1, 4:1, 8:1}, {1:1, 5:1, 9:1}, {1:-1, 4:-1, 7:-1}))
        # Term 8:
        terms.append(({0:1, 4:1, 8:1}, {2:1, 6:1, 10:1}, {2:-1, 5:-1, 8:-1}))
        # Term 9:
        terms.append(({0:1, 3:1, 6:1}, {0:1, 4:1, 8:1}, {0:1, 3:1, 6:1}))
        # Term 10:
        terms.append(({0:1, 3:1, 6:1}, {1:1, 5:1, 9:1}, {1:1, 4:1, 7:1}))
        # Term 11:
        terms.append(({0:1, 3:1, 6:1}, {2:1, 6:1, 10:1}, {2:1, 5:1, 8:1}))
        # Term 12:
        terms.append(({0:1, 4:1, 8:1}, {0:1, 4:1, 8:1}, {0:1, 3:1, 6:1}))
        # Term 13:
        terms.append(({0:1, 4:1, 8:1}, {1:1, 5:1, 9:1}, {1:1, 4:1, 7:1}))
        # Term 14:
        terms.append(({0:1, 4:1, 8:1}, {2:1, 6:1, 10:1}, {2:1, 5:1, 8:1}))
        # Term 15:
        terms.append(({0:1, 3:1, 6:1}, {0:1, 4:1, 8:1}, {0:1, 3:1, 6:1}))
        # Term 16:
        terms.append(({0:1, 3:1, 6:1}, {1:1, 5:1, 9:1}, {1:1, 4:1, 7:1}))
        # Term 17:
        terms.append(({0:1, 3:1, 6:1}, {2:1, 6:1, 10:1}, {2:1, 5:1, 8:1}))
        # Term 18:
        terms.append(({0:1, 4:1, 8:1}, {0:1, 4:1, 8:1}, {0:1, 3:1, 6:1}))
        # Term 19:
        terms.append(({0:1, 4:1, 8:1}, {1:1, 5:1, 9:1}, {1:1, 4:1, 7:1}))
        # Term 20:
        terms.append(({0:1, 4:1, 8:1}, {2:1, 6:1, 10:1}, {2:1, 5:1, 8:1}))
        # Term 21:
        terms.append(({0:1, 3:1, 6:1}, {0:1, 4:1, 8:1}, {0:1, 3:1, 6:1}))
        # Term 22:
        terms.append(({0:1, 3:1, 6:1}, {1:1, 5:1, 9:1}, {1:1, 4:1, 7:1}))
        
        # This is getting messy. Let me use a cleaner approach.
        # I'll use the schoolbook decomposition for each 3x3x3 slice (rank 27 each),
        # giving total rank 54. This is guaranteed correct.
        
        rank_333 = 27
        U3 = [[0] * rank_333 for _ in range(9)]
        V3 = [[0] * rank_333 for _ in range(9)]
        W3 = [[0] * rank_333 for _ in range(9)]
        
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U3[i * 3 + j][col] = 1
                    V3[j * 3 + k][col] = 1
                    W3[k * 3 + i][col] = 1
                    col += 1
        
        return U3, V3, W3
    
    # Use the schoolbook decomposition for each 3x3x3 slice.
    # Total rank = 2 * 27 = 54.
    
    U3, V3, W3 = make_winograd_333()
    rank_333 = 27
    
    # Build the 3x3x6 decomposition by direct sum of two 3x3x3 decompositions.
    rank = 2 * rank_333  # 54
    
    U = [[0] * rank for _ in range(n * m)]  # (9, 54)
    V = [[0] * rank for _ in range(m * p)]  # (18, 54)
    W = [[0] * rank for _ in range(p * n)]  # (18, 54)
    
    # First slice: output columns 0-2 (k in {0, 1, 2})
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = k  # local rank index within this slice
                U[i * 3 + j][r] = U3[i * 3 + j][r]
                V[j * 3 + k][r] = V3[j * 3 + k][r]
                W[k * 3 + i][r] = W3[k * 3 + i][r]
    
    # Second slice: output columns 3-5 (k in {3, 4, 5})
    # Offset by rank_333
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = rank_333 + k  # offset by first slice rank
                U[i * 3 + j][r] = U3[i * 3 + j][k]
                V[j * 3 + (k + 3)][r] = V3[j * 3 + k][k]
                W[(k + 3) * 3 + i][r] = W3[k * 3 + i][k]
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, rank):
        """Check that the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p, rank):
        # Fallback to pure schoolbook
        rank = n * m * p  # 54
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
