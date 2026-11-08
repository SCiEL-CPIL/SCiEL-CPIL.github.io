"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify the tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 if valid matmul term"""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # The tensor T[i,j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For matmul, T[i,j,k] = 1 for all i,j,k (it's the full tensor)
    # Actually T[i,j,k] = 1 for the matmul tensor: C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T(i,j,k) = 1 for all i in [n], j in [m], k in [p]
    # Wait no - the matmul tensor has T(i,j,k) = 1 for all valid i,j,k
    # The decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    total = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i,j,k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    expected = np.ones((n, m, p))
    if not np.allclose(total, expected, atol=1e-6):
        diff = np.max(np.abs(total - expected))
        return False, diff
    return True, 0.0

def _try_rank(n, m, p, target_rank, timeout=10):
    """Try to find a decomposition of given rank (placeholder for future optimization)"""
    pass

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def _build_rank54():
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                U[0*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                U[1*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                W[k0*n + 1][col] += 1
                W[k1*n + 1][col] -= 1
                col += 1
                
                U[0*m + j0][col] += 1
                V[j0*p + k1][col] += 1
                V[j1*p + k1][col] -= 1
                W[k1*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                U[1*m + j1][col] += 1
                V[j1*p + k0][col] += 1
                V[j0*p + k0][col] -= 1
                W[k0*n + 0][col] += 1
                W[k0*n + 1][col] += 1
                col += 1
                
                U[0*m + j0][col] += 1
                U[0*m + j1][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] -= 1
                W[k1*n + 0][col] += 1
                col += 1
                
                U[1*m + j0][col] += 1
                U[0*m + j0][col] -= 1
                V[j0*p + k0][col] += 1
                V[j0*p + k1][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                
                U[0*m + j1][col] += 1
                U[1*m + j1][col] -= 1
                V[j1*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                col += 1
        
        for k in range(p):
            U[0*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            U[1*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def _build_rank42():
        """Try a 42-rank decomposition using different blocking.
        
        Strategy: 
        - For j in {0,1}: use 7-term for each of 3 k-pairs = 21 terms
        - For j in {2,3}: use 7-term for each of 3 k-pairs = 21 terms  
        - For j=4: use 12 terms
        Total = 54. That's the same.
        
        Alternative: Try to use a 6-term decomposition for 2x2x2 blocks.
        The minimal rank for 2x2x2 is 7 (Strassen), so we can't go below 7 per block.
        
        Let me try a different approach: use a 3x3 blocking.
        Split j into {0,1,2} and {3,4}.
        For {0,1,2} x {0,1,2,3,4,5}: need to decompose 2x3x6
        For {3,4} x {0,1,2,3,4,5}: need to decompose 2x2x6
        
        For 2x2x6: use 7-term for each of 3 k-pairs = 21
        For 2x3x6: split into 2x2x3 + 2x1x3 = 7*2 + 2*3 = 14+6 = 20? 
        Actually for 2xa xb, the rank can be computed.
        
        Let me just try rank 42 with a different structure.
        """
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # This is a placeholder - the actual construction needs to be verified
        # For now, return the rank-54 solution
        return _build_rank54()
    
    # Build baseline
    baseline = _build_rank54()
    valid, diff = _verify(baseline["U"], baseline["V"], baseline["W"], n, m, p)
    
    if not valid:
        return baseline
    
    best = baseline
    best_rank = 54
    
    # Try to find a lower rank by searching
    # The key insight: for 2xm x pxn matmul, we can use the fact that
    # the tensor is T[i,j,k] = 1 for all i,j,k
    
    # Try rank 30 (theoretical lower bound is max(2*5, 5*6, 2*6) = 30)
    # This requires a very efficient decomposition
    
    if time.time() < deadline - 30:
        # Try a simple approach: use the fact that for small cases,
        # we can sometimes find lower rank decompositions
        
        # For 2x5x6, one approach is to use:
        # C[0,k] = sum_j A[0,j] * B[j,k]
        # C[1,k] = sum_j A[1,j] * B[j,k]
        
        # We can write this as:
        # Let S_j = A[0,j] + A[1,j], D_j = A[0,j] - A[1,j]
        # Then C[0,k] = sum_j (S_j + D_j)/2 * B[j,k]
        #      C[1,k] = sum_j (S_j - D_j)/2 * B[j,k]
        
        # This doesn't directly help with rank reduction.
        
        # Let me try a different approach: use a 2x2x2 Strassen-like
        # decomposition but applied more cleverly.
        
        pass
    
    return best
# EVOLVE-BLOCK-END
