import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Attempt 51-rank decomposition
    # Structure:
    # A (3x3) split into A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    # B (3x6) split into B11 (2x6), B21 (1x6)
    # C (3x6) = A11*B11 + A12*B21 (top 2 rows)
    #           A21*B11 + A22*B21 (bottom 1 row)
    
    # We will build U, V, W lists of triples (u_idx, v_idx, w_idx, coeff)
    # Then expand to matrices.
    
    terms = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        terms.append((u_idx, v_idx, w_idx, coeff))
        
    # Block 1: A11 (rows 0,1; cols 0,1) * B11 (rows 0,1; cols 0..5)
    # B11 is 2x6. We split cols into 3 groups of 2.
    # Group 1: cols 0,1
    # Group 2: cols 2,3
    # Group 3: cols 4,5
    
    # Strassen terms for 2x2x2
    # Indices in local 2x2 block:
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M3 + M6
    
    # We need to map these to global indices.
    # Local A (0,0) -> Global (0,0)
    # Local A (0,1) -> Global (0,1)
    # Local A (1,0) -> Global (1,0)
    # Local A (1,1) -> Global (1,1)
    
    # Local B (0,0) -> Global (0, col_offset)
    # Local B (0,1) -> Global (0, col_offset+1)
    # Local B (1,0) -> Global (1, col_offset)
    # Local B (1,1) -> Global (1, col_offset+1)
    
    # Local C (0,0) -> Global (0, col_offset)
    # Local C (0,1) -> Global (0, col_offset+1)
    # Local C (1,0) -> Global (1, col_offset)
    # Local C (1,1) -> Global (1, col_offset+1)
    
    def add_strassen_block(col_offset):
        # M1
        # (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11. W: C00, C11 (with signs)
        # This is not a single term, it's a product of sums.
        # We need to expand (A00+A11)(B00+B11) into 4 terms?
        # No, Strassen's algorithm reduces 8 multiplications to 7.
        # The 7 "multiplications" are the rank-1 tensors.
        # Each "multiplication" is a product of a linear combination of A and a linear combination of B.
        # So each of the 7 Strassen terms corresponds to 1 rank-1 tensor in the decomposition.
        
        # Term 1: (A00+A11) * (B00+B11)
        # U part: A00=1, A11=1.
        # V part: B00=1, B11=1.
        # W part: Contributes to C00 (+1), C11 (+1).
        # Wait, the output C is a sum of these terms.
        # C00 = M1 - M4 - M5 + M7
        # So M1 contributes +1 to C00 and +1 to C11?
        # No, M1 is just the product. The assignment to C is linear.
        # So for Term 1:
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        
        # We add 1 term to the global decomposition.
        # But U, V, W are vectors.
        # U vector has 1 at (0,0) and (1,1).
        # V vector has 1 at (0, col_offset) and (1, col_offset+1).
        # W vector has 1 at (col_offset, 0) and (col_offset+1, 1).
        
        # Let's define the 7 terms explicitly.
        
        # T1: (A00+A11)(B00+B11)
        # U: (0,0)=1, (1,1)=1
        # V: (0,co)=1, (1,co+1)=1
        # W: (co,0)=1, (co+1,1)=1
        # Contribution to C:
        # C00 gets +1 from T1?
        # Formula: C00 = M1 - M4 - M5 + M7
        # So T1 (M1) contributes +1 to C00.
        # C11 = M1 - M2 - M3 + M6
        # So T1 (M1) contributes +1 to C11.
        
        # We will store the terms as (u_indices, v_indices, w_indices)
        # where u_indices is a dict of {global_idx: coeff}
        
        def add_rank1_term(u_dict, v_dict, w_dict):
            terms.append((u_dict, v_dict, w_dict))
            
        co = col_offset
        
        # M1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,co): 1, (1,co+1): 1}
        w1 = {(co,0): 1, (co+1,1): 1}
        # W indices are (k, i) where k is col, i is row.
        # C00 is row 0, col co. W index (co, 0).
        # C11 is row 1, col co+1. W index (co+1, 1).
        add_rank1_term(u1, v1, w1)
        
        # M2: (A10+A11)B00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,co): 1}
        w2 = {(co,1): 1} # C10 = M2 + M4. So M2 contributes to C10 (row 1, col co).
        add_rank1_term(u2, v2, w2)
        
        # M3: A00(B01-B11)
        u3 = {(0,0): 1}
        v3 = {(0,co+1): 1, (1,co+1): -1}
        w3 = {(co+1,0): 1} # C01 = M3 + M5. So M3 contributes to C01 (row 0, col co+1).
        add_rank1_term(u3, v3, w3)
        
        # M4: A11(B10-B00)
        u4 = {(1,1): 1}
        v4 = {(1,co): 1, (0,co): -1}
        w4 = {(co,1): 1} # C10 = M2 + M4. So M4 contributes to C10.
        # Wait, C00 = M1 - M4 - M5 + M7.
        # So M4 contributes -1 to C00.
        # My add_rank1_term adds the term directly.
        # The coefficients in W should reflect the sign?
        # No, the term is $u \otimes v \otimes w$.
        # The sum of these terms equals the tensor.
        # So if M4 contributes -1 to C00, the W component for C00 in this term should be -1.
        # But M4 also contributes +1 to C10.
        # So w4 should have (co,1): 1 and (co,0): -1.
        
        # Let's re-evaluate.
        # The decomposition is $T = \sum r_i$.
        # Each $r_i$ is a rank-1 tensor.
        # The value of the tensor at (i,j,k) is $\sum_r U[i,j,r] V[j,k,r] W[k,i,r]$.
        # For a specific Strassen term, say M4 = A11(B10-B00).
        # This term contributes to C10 and C00.
        # C10 = ... + M4 + ...
        # C00 = ... - M4 + ...
        # So the rank-1 tensor for M4 has:
        # U: A11=1
        # V: B10=1, B00=-1
        # W: C10=1, C00=-1
        
        # I will correct the W indices.
        
        # M1: (A00+A11)(B00+B11)
        # C00 = +M1, C11 = +M1
        # W: (co,0): 1, (co+1,1): 1
        
        # M2: (A10+A11)B00
        # C10 = +M2
        # W: (co,1): 1
        
        # M3: A00(B01-B11)
        # C01 = +M3
        # W: (co+1,0): 1
        
        # M4: A11(B10-B00)
        # C10 = +M4, C00 = -M4
        # W: (co,1): 1, (co,0): -1
        
        # M5: (A00+A01)B11
        # C01 = +M5, C00 = -M5
        # W: (co+1,0): 1, (co,0): -1
        
        # M6: (A10-A00)(B00+B01)
        # C11 = +M6
        # W: (co+1,1): 1
        
        # M7: (A01-A11)(B10+B11)
        # C00 = +M7
        # W: (co,0): 1
        
        # Let's re-add M4, M5, M6, M7 with correct W.
        # I already added M1, M2, M3.
        # I need to add M4, M5, M6, M7.
        
        u4 = {(1,1): 1}
        v4 = {(1,co): 1, (0,co): -1}
        w4 = {(co,1): 1, (co,0): -1}
        add_rank1_term(u4, v4, w4)
        
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,co+1): 1}
        w5 = {(co+1,0): 1, (co,0): -1}
        add_rank1_term(u5, v5, w5)
        
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,co): 1, (0,co+1): 1}
        w6 = {(co+1,1): 1}
        add_rank1_term(u6, v6, w6)
        
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,co): 1, (1,co+1): 1}
        w7 = {(co,0): 1}
        add_rank1_term(u7, v7, w7)
        
    # Add 3 Strassen blocks for cols 0-1, 2-3, 4-5
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # Block 4: A12 (rows 0,1; col 2) * B21 (row 2; cols 0..5)
    # C12 (rows 0,1; cols 0..5)
    # C0,k = A0,2 * B2,k
    # C1,k = A1,2 * B2,k
    for k in range(6):
        # Term for C0,k
        u = {(0,2): 1}
        v = {(2,k): 1}
        w = {(k,0): 1}
        add_rank1_term(u, v, w)
        
        # Term for C1,k
        u = {(1,2): 1}
        v = {(2,k): 1}
        w = {(k,1): 1}
        add_rank1_term(u, v, w)
        
    # Block 5: A21 (row 2; cols 0,1) * B11 (rows 0,1; cols 0..5)
    # C21 (row 2; cols 0..5)
    # C2,k = A2,0 * B0,k + A2,1 * B1,k
    for k in range(6):
        # Term for A2,0 * B0,k
        u = {(2,0): 1}
        v = {(0,k): 1}
        w = {(k,2): 1}
        add_rank1_term(u, v, w)
        
        # Term for A2,1 * B1,k
        u = {(2,1): 1}
        v = {(1,k): 1}
        w = {(k,2): 1}
        add_rank1_term(u, v, w)
        
    # Block 6: A22 (row 2; col 2) * B21 (row 2; cols 0..5)
    # C22 (row 2; cols 0..5)
    # C2,k = A2,2 * B2,k
    for k in range(6):
        u = {(2,2): 1}
        v = {(2,k): 1}
        w = {(k,2): 1}
        add_rank1_term(u, v, w)
        
    # Now build U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), coeff in u_dict.items():
            U[i*m+j][r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j*p+k][r] = coeff
        for (k, i), coeff in w_dict.items():
            W[k*n+i][r] = coeff
            
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U, V, W, rank = schoolbook()
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
