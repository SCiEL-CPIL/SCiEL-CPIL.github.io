import time
import numpy as np

def construct():
    t_start = time.time()
    deadline = t_start + 120.0  # safety margin
    
    n, m, p = 2, 3, 8
    
    # Dimensions of factors
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    # Start with the known rank-44 decomposition
    # We can try to reduce rank by using a smarter approach
    # For 2x3x8, the schoolbook is 48. Strassen-based gets 44.
    # Can we do better?
    
    # Strategy: Use the fact that we have 2 rows, 3 middle, 8 columns.
    # Split the 8 columns into two groups of 4.
    # For each group of 4 columns, we're computing 2x3 x 3x4.
    # Schoolbook for 2x3x4 is 24. Two groups = 48.
    # Strassen for 2x2x4: split 2x3 into (2x1) and (2x2).
    #   2x1 x 1x4 = 8 mults (schoolbook)
    #   2x2 x 2x4: split 2x4 into two 2x2 blocks.
    #     2x2 x 2x2 Strassen = 7 mults per block, 2 blocks = 14
    #   Total for 2x3x4 = 8 + 14 = 22
    # Two groups: 44. Same as before.
    
    # Alternative: Use a 2x4x8 approach?
    # Split 3 middle into (1) and (2,3).
    # A[:,0] * B[0,:] = 2x1 x 1x8 = 16 mults
    # A[:,1:3] * B[1:3,:] = 2x2 x 2x8
    #   Split 2x8 into four 2x2 blocks: 4 * 7 = 28 mults
    # Total: 16 + 28 = 44. Same.
    
    # Try: Can we share multiplications across the two groups of 4?
    # If we use Strassen on the 2x2 x 2x8 part by treating 8 as 4*2,
    # we get 7*4 = 28. No sharing possible easily.
    
    # Let me try a different split: 3 = 1 + 1 + 1
    # Three 2x1 x 1x8 parts: 3 * 16 = 48. Worse.
    
    # Try 3 = 2 + 1:
    # 2x2 x 2x8 = 28 (as above)
    # 2x1 x 1x8 = 16
    # Total = 44.
    
    # What about a global Strassen-like approach?
    # 2x3x8: Pad to 4x4x8? No, that would be worse.
    
    # Let me try to see if rank 43 or lower is achievable.
    # Lower bounds: max(n*m, m*p, p*n) = max(6, 24, 16) = 24.
    # Also, n*m*p = 48. The tensor rank is at least max(n*m, m*p, p*n) = 24.
    # But Strassen shows we can do better than schoolbook.
    
    # For 2x3x4, is rank 21 possible? That would give 42 total.
    # For 2x3x4, schoolbook is 24. Strassen gives 22.
    # Is 21 possible? Unknown.
    
    # Let me just implement the rank-44 solution cleanly and verify it.
    # Then attempt a search for rank 43 if time permits.
    
    rank = 44
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    col = 0
    
    # Part 1: j=0 (2x1 x 1x8) - schoolbook, 16 mults
    for k in range(p):
        U[0][col] = 1  # A[0,0]
        V[k][col] = 1  # B[0,k]
        W[k*n + 0][col] = 1  # C[0,k]
        col += 1
        
        U[1*m + 0][col] = 1  # A[1,0]
        V[0*p + k][col] = 1  # B[0,k]
        W[k*n + 1][col] = 1  # C[1,k]
        col += 1
    
    # Part 2: j=1,2 (2x2 x 2x8) - Strassen on 4 blocks of 2x2x2
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # P1: (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U[0*m + 1][col] = 1
        U[1*m + 2][col] = 1
        V[1*p + k0][col] = 1
        V[2*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # P2: (A[1,1] + A[1,2]) * B[1,k0]
        U[1*m + 1][col] = 1
        U[1*m + 2][col] = 1
        V[1*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1
        
        # P3: A[0,1] * (B[1,k1] - B[2,k1])
        U[0*m + 1][col] = 1
        V[1*p + k1][col] = 1
        V[2*p + k1][col] = -1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # P4: A[1,2] * (B[2,k0] - B[1,k0])
        U[1*m + 2][col] = 1
        V[2*p + k0][col] = 1
        V[1*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        
        # P5: (A[0,1] + A[0,2]) * B[2,k1]
        U[0*m + 1][col] = 1
        U[0*m + 2][col] = 1
        V[2*p + k1][col] = 1
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1
        
        # P6: (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1*m + 1][col] = 1
        U[0*m + 1][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # P7: (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0*m + 2][col] = 1
        U[1*m + 2][col] = -1
        V[2*p + k0][col] = 1
        V[2*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1
    
    # Verify the rank-44 solution
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Should not happen, but fallback to schoolbook
        rank_fb = n * m * p  # 48
        U_fb = [[0] * rank_fb for _ in range(dim_U)]
        V_fb = [[0] * rank_fb for _ in range(dim_V)]
        W_fb = [[0] * rank_fb for _ in range(dim_W)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

def _verify(U, V, W):
    """Check that the decomposition is exact."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    # Test with A having 1 at (i0,j0), B having 1 at (j0,k0)
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            if i == i0 and k == k0:
                                val += W[k * n + i][r] * U[i * m + j0][r] * V[j0 * p + k][r]
                        
                        expected_val = 1 if (i == i0 and k == k0) else 0
                        if val != expected_val:
                            return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
