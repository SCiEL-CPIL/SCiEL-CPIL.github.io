import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 3, 4, 8  # A is n x m, B is m x p, C is n x p
    
    # Decompose: M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # M_{1,1,2} has rank 2 (schoolbook: sum over k=0,1)
    # M_{3,4,4} has rank 3*4*4 = 48 (schoolbook)
    # Total rank = 48 * 2 = 96... that's not better.
    
    # Let me think differently.
    # 3*4*8 = 96 is the schoolbook rank.
    # Can we do better?
    # 
    # Key insight: 8 = 4 * 2, so we can write B as [B1; B2] where B1 is 4x4 and B2 is 4x4.
    # C = A*B1 + A*B2
    # Each A*B_i is 3x4 * 4x4 = 3*4*4 = 48 rank schoolbook.
    # Total: 96. Same.
    #
    # Alternative: Use the fact that 3 = 2+1.
    # A = [A_top; A_bottom] where A_top is 2x4, A_bottom is 1x4.
    # C_top = A_top * B (2x4 * 4x8) -> rank 2*4*8 = 64 schoolbook
    # C_bottom = A_bottom * B (1x4 * 4x8) -> rank 1*4*8 = 32 schoolbook
    # Total: 96. Same.
    #
    # Let me try a different block decomposition.
    # View A as 3x4, B as 4x8.
    # Split columns of B into two 4-column blocks: B = [B_left | B_right]
    # C = [A*B_left | A*B_right]
    # Each A*B_left is 3x4 * 4x4 -> rank 48 schoolbook
    # Total: 96. Same.
    #
    # What about using Strassen-like ideas?
    # For 2x2 * 2x2, rank 7 is known (Strassen).
    # 
    # Let me try: split into 2x2 blocks where possible.
    # A (3x4): rows [0:2], [2:3]; cols [0:2], [2:4]
    # B (4x8): rows [0:2], [2:4]; cols [0:2], [2:4], [4:6], [6:8]
    #
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    #   = (2x2 * 2x2) + (2x2 * 2x2) -> rank 7+7=14 if Strassen works
    #
    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    #   = rank 7+7=14
    #
    # C[0:2, 4:6] = A[0:2,0:2]*B[0:2,4:6] + A[0:2,2:4]*B[2:4,4:6]
    #   = rank 7+7=14
    #
    # C[0:2, 6:8] = A[0:2,0:2]*B[0:2,6:8] + A[0:2,2:4]*B[2:4,6:8]
    #   = rank 7+7=14
    #
    # C[2:3, 0:2] = A[2:3,0:2]*B[0:2,0:2] + A[2:3,2:4]*B[2:4,0:2]
    #   = (1x2 * 2x2) + (1x2 * 2x2) -> rank 4+4=8 schoolbook
    #
    # C[2:3, 2:4] = A[2:3,0:2]*B[0:2,2:4] + A[2:3,2:4]*B[2:4,2:4]
    #   = rank 4+4=8
    #
    # C[2:3, 4:6] = A[2:3,0:2]*B[0:2,4:6] + A[2:3,2:4]*B[2:4,4:6]
    #   = rank 4+4=8
    #
    # C[2:3, 6:8] = A[2:3,0:2]*B[0:2,6:8] + A[2:3,2:4]*B[2:4,6:8]
    #   = rank 4+4=8
    #
    # Total: 4*14 + 4*8 = 56 + 32 = 88
    # 
    # This is better than 96! But I need a valid rank-7 decomposition for 2x2*2x2.
    #
    # The standard Strassen algorithm does NOT directly give a tensor decomposition
    # because some terms mix A and B elements. However, there ARE known rank-7
    # tensor decompositions for 2x2x2 matrix multiplication.
    #
    # One such decomposition (from the literature) uses these 7 products:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    #
    # m1 = (a+b)*(e+f)
    # m2 = (c+d)*(e+f)  
    # m3 = a*(h-f)
    # m4 = d*(g-e)
    # m5 = (a+e)*(h+f)  -- PROBLEM: this mixes a and e!
    #
    # The issue is that m5 = (a+e)*(h+f) is not separable.
    #
    # Actually, I recall that the rank-7 decomposition for 2x2x2 is NOT known to
    # be explicitly constructible in a simple way. The lower bound is 7 and upper
    # bound was 7 (Alon-Boro), but the explicit construction is complex.
    #
    # Let me just use schoolbook for everything to be safe. Rank 96.
    #
    # Actually, wait. Let me reconsider the Kronecker product approach more carefully.
    #
    # M_{3,4,8} can be viewed as M_{3,4,4} ⊗ M_{1,1,2}
    # where the tensor product combines:
    # - U_total[i*m1 + j, c1*R2 + c2] = U1[i*m1+j, c1] * U2[j, c2]  ... no this doesn't work directly.
    
    # Let me just do schoolbook for now - it's guaranteed correct.
    rank = n * m * p  # 96
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {s}")
    
    elapsed = time.time() - start_time
    if elapsed > 140:
        raise TimeoutError("Too slow")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
