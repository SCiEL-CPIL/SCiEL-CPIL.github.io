import time
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    deadline = time.time() + 140  # 140s safety margin
    
    # Strassen steps for 2x2x2:
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
    
    strassen = [
        # u, v, w coefficients (local indices 0..3 for 2x2)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),   # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),        # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),        # M3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),        # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),        # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),        # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),        # M7
    ]
    
    def emit_block(a_cs, b_rs, b_cs, c_cs, off, U, V, W, rank):
        for s in range(7):
            col = off + s
            if col >= rank:
                continue
            uc, vc, wc = strassen[s]
            for li, c in uc.items():
                r, cc = li // 2, li % 2
                U[r * m + a_cs + cc][col] += c
            for li, c in vc.items():
                r, cc = li // 2, li % 2
                V[(b_rs + r) * p + b_cs + cc][col] += c
            for li, c in wc.items():
                r, cc = li // 2, li % 2
                W[(c_cs + cc) * n + r][col] += c
    
    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        emit_block(0, 0, 0, 0, 0, U, V, W, rank)
        emit_block(0, 0, 2, 2, 7, U, V, W, rank)
        emit_block(2, 2, 0, 0, 14, U, V, W, rank)
        emit_block(2, 2, 2, 2, 21, U, V, W, rank)
        return U, V, W
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        # Build the tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # and check it matches the multiplication tensor
        # A is n x m, B is m x p, C = A*B is n x p
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] should be 1 if this contributes to C[i,k], else 0
        # Actually: T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication entry
        # The multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
        # where i in [0,n), j in [0,m), k in [0,p)
        # and the decomposition is: sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j,k
        
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    U28, V28, W28 = build_strassen_28()
    
    # Verify the Strassen-28 decomposition
    if verify_decomposition(U28, V28, W28, n, m, p):
        # Try to find rank-27 by looking for a linear combination of the 28 factors
        # that can be eliminated
        # The idea: if there's a linear dependence among the rank-1 terms,
        # we can reduce the rank
        
        # Try randomized approach: pick a random linear combination of columns
        # and see if we can express one column as a combination of others
        # while maintaining the decomposition property
        
        # Alternative: use the fact that if we have a valid rank-28 decomposition,
        # we can try to find a rank-27 by solving for coefficients
        # such that one factor becomes a linear combination of others
        
        # For each pair of columns (r1, r2), check if there exists a scalar
        # alpha such that column r2 = alpha * column r1 + (adjustment)
        # This is complex, so let's try a different approach
        
        # Try to find a non-trivial linear relation among the 28 rank-1 tensors
        # If such a relation exists, we can reduce the rank
        
        # Use randomized linear algebra: sample random vectors and check
        # if the 28 rank-1 tensors are linearly independent
        
        # The rank-1 tensors are: T_r = U_col_r ⊗ V_col_r ⊗ W_col_r
        # We want to check if there exist c_1, ..., c_28 not all zero
        # such that sum_r c_r * T_r = 0
        
        # This is equivalent to checking if the 28 vectors in the tensor space
        # are linearly independent
        
        # Let's try a simpler approach: use the border method or
        # check if we can merge two factors
        
        # For now, return the rank-28 solution as the best we have
        return {"U": U28, "V": V28, "W": W28}
    
    # Fallback: return rank-32 trivial decomposition
    rank = n * m * p  # 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
