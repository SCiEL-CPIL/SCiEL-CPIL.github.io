import time
import numpy as np

def construct():
    _t0 = time.time()
    n, m, p = 2, 3, 10
    
    # Try to find a rank-minimal decomposition for 2x3x10 matrix multiplication.
    # The tensor T_{i,j,k} = A_{i,k} * B_{k,j} where A is 2x3, B is 3x10, C is 2x10.
    # Output convention: W[k*n+i, r] where k is column of C (0..9), i is row of C (0,1).
    # So W index = j*2 + i for C[i,j].
    
    # Strategy: Use a global decomposition. For 2x3 matrix multiplication, rank 10 is known.
    # For 2x3x10, we can try to use 10 "base" rank terms for the A-side pattern,
    # and replicate/adapt for the 10 columns of B.
    
    # Actually, let's try a systematic approach: 
    # Each rank-1 term is (u^T a) * (v^T b) * w where u in R^6, v in R^30, w in R^20.
    # We need sum_r u_r * v_r * w_r = T (the multiplication tensor).
    
    # Let's try to construct a rank-50 decomposition by finding a better 2x3x2 block.
    # For 2x3x2, can we do rank 10?
    
    # Let me try to verify if the current 55 is valid first, then try to reduce.
    
    # First, let's try a rank-50 approach: use 10 terms per 2-column block.
    # This requires a rank-10 2x3x2 decomposition.
    
    # Known: 2x3x2 has trivial rank 12. Can we do 10 or 11?
    # Let's try to construct one with 10 terms.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 6 x 50
    V = [[0] * rank for _ in range(m * p)]  # 30 x 50
    W = [[0] * rank for _ in range(p * n)]  # 20 x 50
    col = 0

    # Try a rank-10 decomposition for each 2x3x2 block
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # We need to produce C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        # C[0,k0] = A[0,0]*B[0,k0] + A[0,1]*B[1,k0] + A[0,2]*B[2,k0]
        # C[0,k1] = A[0,0]*B[0,k1] + A[0,1]*B[1,k1] + A[0,2]*B[2,k1]
        # C[1,k0] = A[1,0]*B[0,k0] + A[1,1]*B[1,k0] + A[1,2]*B[2,k0]
        # C[1,k1] = A[1,0]*B[0,k1] + A[1,1]*B[1,k1] + A[1,2]*B[2,k1]
        
        # Term 1: (A00+A11)*(B0k0+B1k1) contributes to C0k0 and C1k1
        U[0][col] += 1; U[4][col] += 1
        V[0*10+k0][col] += 1; V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        
        # Term 2: (A10+A11)*B0k0 contributes to C1k0 and C1k1 (with -1)
        U[3][col] += 1; U[4][col] += 1
        V[0*10+k0][col] += 1
        W[k0*2+1][col] += 1; W[k1*2+1][col] -= 1
        col += 1
        
        # Term 3: A00*(B0k1-B1k1) contributes to C0k1 and C1k1
        U[0][col] += 1
        V[0*10+k1][col] += 1; V[1*10+k1][col] -= 1
        W[k1*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        
        # Term 4: A11*(B1k0-B0k0) contributes to C0k0 and C1k0
        U[4][col] += 1
        V[1*10+k0][col] += 1; V[0*10+k0][col] -= 1
        W[k0*2+0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        
        # Term 5: (A00+A01)*B1k1 contributes to C0k0(-1) and C0k1
        U[0][col] += 1; U[1][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] -= 1; W[k1*2+0][col] += 1
        col += 1
        
        # Term 6: (A10-A00)*(B0k0+B0k1) contributes to C1k1
        U[3][col] += 1; U[0][col] -= 1
        V[0*10+k0][col] += 1; V[0*10+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1
        
        # Term 7: (A01-A11)*(B1k0+B1k1) contributes to C0k0
        U[1][col] += 1; U[4][col] -= 1
        V[1*10+k0][col] += 1; V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1
        col += 1
        
        # Terms 8-10: handle A[0,2]*B[2,*] and A[1,2]*B[2,*] with 3 terms instead of 4
        # Term 8: A02*(B2k0+B2k1) -> C0k0, C0k1
        U[2][col] += 1
        V[2*10+k0][col] += 1; V[2*10+k1][col] += 1
        W[k0*2+0][col] += 1; W[k1*2+0][col] += 1
        col += 1
        
        # Term 9: A12*(B2k0+B2k1) -> C1k0, C1k1
        U[5][col] += 1
        V[2*10+k0][col] += 1; V[2*10+k1][col] += 1
        W[k0*2+1][col] += 1; W[k1*2+1][col] += 1
        col += 1
        
        # Term 10: We need to fix the double-counting from terms 8 and 9.
        # Actually, terms 8 and 9 as written produce:
        # C0k0 += A02*B2k0 + A02*B2k1  (but we only want A02*B2k0)
        # C0k1 += A02*B2k0 + A02*B2k1  (but we only want A02*B2k1)
        # So this doesn't work directly.
        
        # Let me reconsider. The 4 direct terms were:
        # A02*B2k0 -> C0k0
        # A02*B2k1 -> C0k1
        # A12*B2k0 -> C1k0
        # A12*B2k1 -> C1k1
        # These are 4 independent rank-1 terms. Can we do it in 3?
        # (A02+A12)*(B2k0+B2k1) gives cross terms we don't want.
        # So we likely need 4 terms for these, making total 11 per block.
        
        # Let me revert to 11 terms per block (rank 55) but check if we can merge across blocks.
        pass

    # Since the above approach doesn't easily reduce to 10 per block,
    # let's stick with 55 but verify correctness and see if cross-block sharing helps.
    
    # Actually, let's just return the original 55-rank solution which is known to be valid.
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    col = 0

    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        # Term 1: (A00+A11)*(B0k0+B1k1) -> C0k0, C1k1
        U[0][col] += 1; U[4][col] += 1
        V[0*10+k0][col] += 1; V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        # Term 2: (A10+A11)*B0k0 -> C1k0, C1k1
        U[3][col] += 1; U[4][col] += 1
        V[0*10+k0][col] += 1
        W[k0*2+1][col] += 1; W[k1*2+1][col] -= 1
        col += 1
        # Term 3: A00*(B0k1-B1k1) -> C0k1, C1k1
        U[0][col] += 1
        V[0*10+k1][col] += 1; V[1*10+k1][col] -= 1
        W[k1*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        # Term 4: A11*(B1k0-B0k0) -> C0k0, C1k0
        U[4][col] += 1
        V[1*10+k0][col] += 1; V[0*10+k0][col] -= 1
        W[k0*2+0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        # Term 5: (A00+A01)*B1k1 -> C0k0(-1), C0k1
        U[0][col] += 1; U[1][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] -= 1; W[k1*2+0][col] += 1
        col += 1
        # Term 6: (A10-A00)*(B0k0+B0k1) -> C1k1
        U[3][col] += 1; U[0][col] -= 1
        V[0*10+k0][col] += 1; V[0*10+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1
        # Term 7: (A01-A11)*(B1k0+B1k1) -> C0k0
        U[1][col] += 1; U[4][col] -= 1
        V[1*10+k0][col] += 1; V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1
        col += 1
        # Terms 8-11: direct terms for row 2 of B
        U[2][col] += 1; V[2*10+k0][col] += 1; W[k0*2+0][col] += 1
        col += 1
        U[2][col] += 1; V[2*10+k1][col] += 1; W[k1*2+0][col] += 1
        col += 1
        U[5][col] += 1; V[2*10+k0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        U[5][col] += 1; V[2*10+k1][col] += 1; W[k1*2+1][col] += 1
        col += 1

    if time.time() - _t0 > 100:
        return {"U": U, "V": V, "W": W}
    return {"U": U, "V": V, "W": W}
