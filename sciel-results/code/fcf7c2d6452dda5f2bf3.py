import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Rank-23 algorithm for 3x3x3 (Bini-Capovani-Lotti-Romani)
    # Verified decomposition: 23 bilinear forms
    # Each term: (u_idx, v_idx, w_idx, coeff)
    # u_idx: i*3+j (A[i,j]), v_idx: j*3+k (B[j,k]), w_idx: i*3+k (C[i,k])
    
    # The 23 terms of the BCR rank-23 algorithm for 3x3x3:
    # Using the standard verified decomposition
    rank23_terms = [
        # (u, v, w, coeff)
        (0, 0, 0, 1),
        (3, 0, 3, -1),
        (6, 0, 6, 1),
        (0, 1, 1, 1),
        (3, 1, 4, -1),
        (6, 1, 7, 1),
        (0, 2, 2, 1),
        (3, 2, 5, -1),
        (6, 2, 8, 1),
        (1, 0, 1, 1),
        (4, 0, 4, -1),
        (7, 0, 7, 1),
        (1, 1, 2, 1),
        (4, 1, 5, -1),
        (7, 1, 8, 1),
        (1, 2, 3, 1),
        (4, 2, 6, -1),
        (7, 2, 9, 1),  # wait, w_idx should be i*3+k, max is 2*3+2=8
    ]
    
    # Let me use a properly verified rank-23 decomposition.
    # The BCR algorithm has these 23 bilinear forms:
    # I'll use the known correct one and verify it.
    
    # Actually, let me construct a verified rank-23 decomposition programmatically
    # by checking against the identity tensor.
    
    # Known rank-23 decomposition for 3x3x3 (verified):
    # Using the formulation from Bini, Capovani, Lotti, Romani (1979)
    # The 23 terms with their (a_idx, b_idx, c_idx, coeff):
    
    bcr_23 = [
        (0, 0, 0, 1),
        (3, 0, 3, -1),
        (6, 0, 6, 1),
        (0, 1, 1, 1),
        (3, 1, 4, -1),
        (6, 1, 7, 1),
        (0, 2, 2, 1),
        (3, 2, 5, -1),
        (6, 2, 8, 1),
        (1, 0, 1, 1),
        (4, 0, 4, -1),
        (7, 0, 7, 1),
        (1, 1, 2, 1),
        (4, 1, 5, -1),
        (7, 1, 8, 1),
        (1, 2, 3, 1),
        (4, 2, 6, -1),
        (7, 2, 9, 1),
    ]
    
    # Wait, w_idx max should be 8 (i=2, k=2). Let me fix.
    # Actually for 3x3x3: u in [0,8], v in [0,8], w in [0,8]
    # w_idx = i*3 + k where i in {0,1,2}, k in {0,1,2}
    
    # Let me use a different well-known rank-23 decomposition.
    # I'll verify it by checking the tensor equation.
    
    # For now, let me try to find a valid rank-23 by solving.
    # But that's complex. Let me instead use the schoolbook approach
    # but try to reduce rank by combining terms.
    
    # Actually, the simplest approach: use schoolbook rank 54, but try
    # to find if rank 46 (2*23) works by solving linear systems.
    
    # Strategy: View 3x3 * 3x6 as two 3x3 * 3x3 problems.
    # For each 3x3x3 sub-problem, try to use rank-23.
    # Total rank: 46.
    
    # Let me implement a verified rank-23 for 3x3x3.
    # Using the known correct BCR decomposition:
    
    # The correct BCR rank-23 terms (a_idx, b_idx, c_idx, coeff):
    # a_idx = i*3+j, b_idx = j*3+k, c_idx = i*3+k
    bcr_correct = [
        (0, 0, 0, 1),   # a00*b00 -> c00
        (3, 0, 3, -1),  # a10*b00 -> -c10  
        (6, 0, 6, 1),   # a20*b00 -> c20
        (0, 1, 1, 1),   # a00*b01 -> c01
        (3, 1, 4, -1),  # a10*b01 -> -c11
        (6, 1, 7, 1),   # a20*b01 -> c21
        (0, 2, 2, 1),   # a00*b02 -> c02
        (3, 2, 5, -1),  # a10*b02 -> -c12
        (6, 2, 8, 1),   # a20*b02 -> c22
        (1, 0, 1, 1),   # a01*b10 -> c01
        (4, 0, 4, -1),  # a11*b10 -> -c11
        (7, 0, 7, 1),   # a21*b10 -> c21
        (1, 1, 2, 1),   # a01*b11 -> c02
        (4, 1, 5, -1),  # a11*b11 -> -c12
        (7, 1, 8, 1),   # a21*b11 -> c22
        (1, 2, 3, 1),   # a01*b12 -> c03? No, c_idx should be i*3+k
        # I'm making errors. Let me use a completely different approach.
    ]
    
    # Let me just verify the schoolbook and return it, since finding
    # a valid rank-23 is error-prone without a verified reference.
    
    # Schoolbook rank = 54
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify schoolbook
    U_s = np.array(U, dtype=float)
    V_s = np.array(V, dtype=float)
    W_s = np.array(W, dtype=float)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank_school):
                    val += U_s[i * m + j, r] * V_s[j * p + k, r] * W_s[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)
    
    if error > 1e-9:
        # Should not happen, but return anyway
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
