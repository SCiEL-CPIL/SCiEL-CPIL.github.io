import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 5
    
    def verify_exact(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Strassen 2x2x2 rank-7 decomposition
    def strassen_7():
        """Return (U, V, W) for 2x2 * 2x2 with rank 7.
        U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
        Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # Strassen's 7 products for (a11 a12; a21 a22) * (b11 b12; b21 b22)
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b21-b22)
        # M4 = a22(b12-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        #
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        terms = [
            # (U entries, V entries, W entries)
            # M1 = (a11+a22)(b11+b22)
            # U: a11->(0,0)=1, a22->(1,1)=1
            # V: b11->(0,0)=1, b22->(1,1)=1
            # W: c11->(0,0)=1, c22->(1,1)=1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2 = (a21+a22)b11
            # U: a21->(1,0)=1, a22->(1,1)=1
            # V: b11->(0,0)=1
            # W: c21->(0,1)=1, c22->(1,1)=-1
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
            # M3 = a11(b21-b22)
            # U: a11->(0,0)=1
            # V: b21->(1,0)=1, b22->(1,1)=-1
            # W: c12->(0,1)=1, c22->(1,1)=1
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # M4 = a22(b12-b11)
            # U: a22->(1,1)=1
            # V: b12->(0,1)=1, b11->(0,0)=-1
            # W: c11->(0,0)=1, c21->(0,1)=1
            ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (0,1):1}),
            # M5 = (a11+a12)b22
            # U: a11->(0,0)=1, a12->(0,1)=1
            # V: b22->(1,1)=1
            # W: c11->(0,0)=-1, c12->(0,1)=1
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6 = (a21-a11)(b11+b12)
            # U: a21->(1,0)=1, a11->(0,0)=-1
            # V: b11->(0,0)=1, b12->(0,1)=1
            # W: c22->(1,1)=1
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # M7 = (a12-a22)(b21+b22)
            # U: a12->(0,1)=1, a22->(1,1)=-1
            # V: b21->(1,0)=1, b22->(1,1)=1
            # W: c11->(0,0)=1
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for r, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), val in u_dict.items():
                U[i*2+j][r] += val
            for (j, k), val in v_dict.items():
                V[j*2+k][r] += val
            for (k, i), val in w_dict.items():
                W[k*2+i][r] += val
        
        return U, V, W

    # 2x2x3 rank-9 decomposition: 2x2 * 2x3
    # U: (4, rank), V: (6, rank), W: (6, rank)
    # This handles A[2x2] * B[2x3] -> C[2x3]
    def strassen_2x2x3():
        """
        Decompose 2x2 * 2x3 multiplication tensor.
        U: shape (4, rank) - indices (i,j) for i in 0..1, j in 0..1
        V: shape (6, rank) - indices (j,k) for j in 0..1, k in 0..2
        W: shape (6, rank) - indices (k,i) for k in 0..2, i in 0..1
        
        We can view this as two 2x2x2 problems sharing the A matrix:
        C[:, 0:2] = A * B[:, 0:2]  (rank 7)
        C[:, 2:2] = A * B[:, 2:3]  (rank 4 schoolbook)
        Total: 11. But we can do better.
        
        Actually, let's use a direct approach:
        The tensor has 2*2*3 = 12 non-zero entries.
        We need at least... let's try rank 9.
        
        Approach: Split B into B1 (2x2) and B2 (2x1).
        C1 = A * B1 (2x2 result) - rank 7 via Strassen
        C2 = A * B2 (2x1 result) - rank 4 via schoolbook
        Total: 11
        
        Alternative: Use the fact that we can share some computations.
        Let me try a different split.
        
        Actually, for 2x2 * 2x3, we can use:
        - 7 Strassen terms for the first two columns of B
        - 2 additional terms for the third column (since A is 2x2, multiplying by a 2x1 vector needs 2 multiplications, but we can use existing Strassen intermediates)
        
        Hmm, let me just do 7 + 4 = 11 for simplicity, or try to find rank 9.
        
        For now, let's use a simple construction:
        Rank 9: 7 Strassen for columns 0-1, and 2 extra for column 2.
        Wait, that's 9! Let me verify.
        
        For column 2 of B (a 2x1 vector b = [b0, b1]):
        c0 = a00*b0 + a01*b1
        c1 = a10*b0 + a11*b1
        
        We need 2 scalar multiplications. But in the tensor decomposition,
        each "term" r contributes U[i,j]*V[j,k]*W[k,i].
        
        For k=2:
        c0 += U[i,0]*V[0,2]*W[2,i] + U[i,1]*V[1,2]*W[2,i]
        
        So for the third column, we need:
        c0 = a00*b0 + a01*b1
        c1 = a10*b0 + a11*b1
        
        This requires 4 terms in general (schoolbook for 2x2*2x1 = 4).
        But wait, 2x2 * 2x1 has rank at most 4 (schoolbook).
        Can we do it in 2? No, because it's a 2x2x1 tensor which has rank 4 in general.
        
        So total would be 7 + 4 = 11.
        
        Hmm, that gives total rank = 7 + 7 + 11 + 11 = 36. Same as before.
        
        Let me reconsider. Maybe a different partitioning helps.
        
        Actually, let's try: partition m=4 as 2+2, and p=5 as 3+2.
        Then we have 4 blocks:
        - (2x2) * (2x3): rank ?
        - (2x2) * (2x2): rank 7
        - (2x2) * (2x3): rank ?
        - (2x2) * (2x2): rank 7
        
        If 2x2*2x3 has rank 9, total = 9+7+9+7 = 32.
        
        Let me try to find a rank-9 decomposition for 2x2 * 2x3.
        
        Tensor T[i,j,k] for i in 0..1, j in 0..1, k in 0..2.
        T[i,j,k] = 1 for all valid (i,j,k).
        
        We want U (4x9), V (6x9), W (6x9) such that
        sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k.
        
        Let me try to construct this. We can think of it as:
        For each k, we have a 2x2 matrix multiplication.
        
        Approach: Use 7 Strassen terms for k=0,1 (treating B as 2x2 for these cols),
        and 2 additional terms for k=2.
        
        For k=2, we need:
        c0 = a00*b02 + a01*b12
        c1 = a10*b02 + a11*b12
        
        In the tensor framework, for k=2:
        sum_j U[i,j,r] * V[j,2,r] * W[2,i,r] should give the right c_i.
        
        If we add 2 new terms (r=7, r=8):
        Term 7: U[i,j,7] nonzero only for specific (i,j), V[j,2,7] nonzero for specific j, W[2,i,7] nonzero for specific i.
        
        Let's say:
        Term 7: U[0,0,7]=1, V[0,2,7]=1, W[2,0,7]=1 -> contributes a00*b02 to c0
        Term 8: U[0,1,7]=1, V[1,2,7]=1, W[2,0,7]=1 -> contributes a01*b12 to c0
        
        But we also need c1 = a10*b02 + a11*b12.
        That would need 2 more terms. So 4 total for k=2, giving rank 11.
        
        Unless we can share. Let me think differently.
        
        What if we use a different structure?
        
        Let's try: for all 3 columns, use a unified set of 9 terms.
        
        Actually, I recall that the rank of 2x2 * 2x3 matrix multiplication is 9.
        Let me look for a known decomposition.
        
        One approach: 
        M1 = (a00+a11)(b00+b11)  -- for col 0
        M2 = (a10+a11)b00       -- for col 0
        M3 = a00(b10-b11)       -- for col 0
        M4 = a11(b01-b00)       -- for col 0
        M5 = (a00+a01)b11       -- for col 0
        M6 = (a10-a00)(b00+b01) -- for col 0
        M7 = (a01-a11)(b10+b11) -- for col 0
        
        These give c00 and c10.
        
        For col 1, we can reuse some structure.
        
        This is getting complex. Let me just implement the 7+4=11 approach for now,
        which gives total rank 36, and see if I can improve.
        
        Actually, wait. Let me reconsider the partitioning.
        
        What if I partition p=5 as 2+3 instead of 3+2?
        Then:
        - (2x2)*(2x2): rank 7
        - (2x2)*(2x3): rank ?
        
        Same issue.
        
        Let me try a completely different approach: partition m=4 as 1+3.
        - (2x1)*(1x5): This is just outer product, rank 5 (each of the 5 outputs needs 1 mult)
          Actually, (2x1)*(1x5) = 2x5 result. Each entry c[i,k] = a[i,0]*b[0,k].
          This is a rank-5 tensor (5 independent products).
        - (2x3)*(3x5): This is 2x3 * 3x5. Rank of this?
          Schoolbook: 2*3*5 = 30.
          With Strassen on 3x3 blocks... 3x3*3x3 has rank 23 (known).
          But we have 2x3 * 3x5, not 3x3*3x3.
          
        This seems worse.
        
        Let me stick with the 2+2 split on m and try to get 2x2*2x3 to rank 9.
        
        After more thought, I believe the rank of the 2x2*2x3 matrix multiplication tensor is indeed 9.
        Here's a construction:
        
        We have A (2x2), B (2x3), C = A*B (2x3).
        
        Let's define 9 terms:
        
        For the first 2 columns of B (B[:, 0:2]), use Strassen (7 terms).
        For the 3rd column, we need 2 more terms.
        
        But as I noted, 2x2 * 2x1 has rank 4, not 2.
        
        UNLESS we can share with the Strassen intermediates.
        
        Key insight: In Strassen, we compute intermediates like (a00+a11), (b00+b11), etc.
        For the third column, we need:
        c0 = a00*b02 + a01*b12
        c1 = a10*b02 + a11*b12
        
        If we had access to a00, a01, a10, a11 individually, we'd need 4 terms.
        But in the tensor decomposition, each term r provides U[i,j,r]*V[j,k,r]*W[k,i,r].
        
        For k=2, the contribution to c[i] is sum_j U[i,j,r]*V[j,2,r]*W[2,i,r].
        
        If we set W[2,0,r] and W[2,1,r] appropriately, we can get different combinations.
        
        Let me try:
        Term 7: U[0,0,7]=1, U[1,0,7]=0, U[0,1,7]=0, U[1,1,7]=0
                V[0,2,7]=1, V[1,2,7]=0
                W[2,0,7]=1, W[2,1,7]=0
                -> contributes a00*b02 to c0
                
        Term 8: U[0,0,8]=0, U[1,0,8]=0, U[0,1,8]=1, U[1,1,8]=0
                V[0,2,8]=0, V[1,2,8]=1
                W[2,0,8]=1, W[2,1,8]=0
                -> contributes a01*b12 to c0
                
        Term 9: U[0,0,9]=0, U[1,0,9]=1, U[0,1,9]=0, U[1,1,9]=0
                V[0,2,9]=1, V[1,2,9]=0
                W[2,0,9]=0, W[2,1,9]=1
                -> contributes a10*b02 to c1
                
        Term 10: U[0,0,10]=0, U[1,0,10]=0, U[0,1,10]=0, U[1,1,10]=1
                V[0,2,10]=0, V[1,2,10]=1
                W[2,0,10]=0, W[2,1,10]=1
                -> contributes a11*b12 to c1
                
        That's 4 terms for column 2, total 7+4=11.
        
        To get rank 9, I need to find 2 terms that handle all 4 products for column 2.
        That seems impossible with just 2 terms since we have 4 independent products.
        
        So rank of 2x2*2x3 is at least 9? Let me check: the tensor has 12 entries.
        The border rank might be 9, but the rank could be higher.
        
        Actually, I think the rank is 11. Let me just go with 11 and see if the total works out differently.
        
        Total with 2+2 split on m and 3+2 split on p:
        - Block (i in 0..1, j in 0..1, k in 0..2): 2x2*2x3, rank 11
        - Block (i in 0..1, j in 0..1, k in 3..4): 2x2*2x2, rank 7
        - Block (i in 0..1, j in 2..3, k in 0..2): 2x2*2x3, rank 11
        - Block (i in 0..1, j in 2..3, k in 3..4): 2x2*2x2, rank 7
        Total: 11+7+11+7 = 36.
        
        Same as before! The 2+2 split on m with Strassen gives 36 regardless.
        
        What if I use a different split on m? Say m=4 as 3+1:
        - (2x3)*(3x5): rank ?
        - (2x1)*(1x5): rank 5
        Total: rank(2x3*3x5) + 5
        
        For 2x3*3x5, schoolbook is 30. With Strassen-like optimizations...
        This is getting complicated.
        
        Let me try yet another approach: use the known result that 2x4*4x5 can be done with rank 32.
        Actually, I'm not sure that's known.
        
        Given the complexity, let me just implement the current 36-rank solution properly and ensure it works.
        The current code seems to have a bug in the Strassen construction. Let me rewrite it cleanly.
        """
        rank = 11
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        
        # First 7 terms: Strassen for columns 0,1
        strassen_terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (0,1):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for r, (u_dict, v_dict, w_dict) in enumerate(strassen_terms):
            for (i, j), val in u_dict.items():
                U[i*2+j][r] += val
            for (j, k), val in v_dict.items():
                V[j*2+k][r] += val
            for (k, i), val in w_dict.items():
                W[k*2+i][r] += val
        
        # Last 4 terms: schoolbook for column 2
        # c0 = a00*b02 + a01*b12
        # c1 = a10*b02 + a11*b12
        schoolbook_terms = [
            ({(0,0):1}, {(0,2):1}, {(2,0):1}),
            ({(0,1):1}, {(1,2):1}, {(2,0):1}),
            ({(1,0):1}, {(0,2):1}, {(2,1):1}),
            ({(1,1):1}, {(1,2):1}, {(2,1):1}),
        ]
        
        for dr, (u_dict, v_dict, w_dict) in enumerate(schoolbook_terms):
            r = 7 + dr
            for (i, j), val in u_dict.items():
                U[i*2+j][r] += val
            for (j, k), val in v_dict.items():
                V[j*2+k][r] += val
            for (k, i), val in w_dict.items():
                W[k*2+i][r] += val
        
        return U, V, W

    def build_decomposition():
        """Build the full 2x4*4x5 decomposition using 2+2 split on m and 3+2 split on p."""
        rank = 36  # 11 + 7 + 11 + 7
        U = [[0]*rank for _ in range(n*m)]  # (8, 36)
        V = [[0]*rank for _ in range(m*p)]  # (20, 36)
        W = [[0]*rank for _ in range(p*n)]  # (10, 36)
        
        # Get sub-decompositions
        s7 = strassen_7()       # 2x2*2x2, rank 7
        s11 = strassen_2x2x3()  # 2x2*2x3, rank 11
        
        col = 0
        # Block 1: i in 0..1, j in 0..1, k in 0..2 -> use s11
        for r in range(11):
            for i in range(2):
                for j in range(2):
                    U[i*4 + j][col] = s11[0][i*2+j][r]
            for j in range(2):
                for k in range(3):
                    V[j*5 + k][col] = s11[1][j*2+k][r]
            for k in range(3):
                for i in range(2):
                    W[k*2 + i][col] = s11[2][k*2+i][r]
            col += 1
        
        # Block 2: i in 0..1, j in 0..1, k in 3..4 -> use s7
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*4 + j][col] = s7[0][i*2+j][r]
            for j in range(2):
                for k in range(2):
                    V[j*5 + (3+k)][col] = s7[1][j*2+k][r]
            for k in range(2):
                for i in range(2):
                    W[(3+k)*2 + i][col] = s7[2][k*2+i][r]
            col += 1
        
        # Block 3: i in 0..1, j in 2..3, k in 0..2 -> use s11
        for r in range(11):
            for i in range(2):
                for j in range(2):
                    U[i*4 + (2+j)][col] = s11[0][i*2+j][r]
            for j in range(2):
                for k in range(3):
                    V[(2+j)*5 + k][col] = s11[1][j*2+k][r]
            for k in range(3):
                for i in range(2):
                    W[k*2 + i][col] = s11[2][k*2+i][r]
            col += 1
        
        # Block 4: i in 0..1, j in 2..3, k in 3..4 -> use s7
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*4 + (2+j)][col] = s7[0][i*2+j][r]
            for j in range(2):
                for k in range(2):
                    V[(2+j)*5 + (3+k)][col] = s7[1][j*2+k][r]
            for k in range(2):
                for i in range(2):
                    W[(3+k)*2 + i][col] = s7[2][k*2+i][r]
            col += 1
        
        return U, V, W, col

    U, V, W, rank = build_decomposition()
    
    if verify_exact(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
                
    return {"U": U_school, "V": V_school, "W": W_school}
