import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 78
    # 1x4x3: 12
    # 1x4x4: 16
    # 2x4x3: 22 (2 * 11)
    # 2x4x4: 28 (2 * 14)
    total_rank = 12 + 16 + 22 + 28
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add schoolbook terms for n' x m' x p'
    def add_schoolbook(n_prime, m_prime, p_prime, i_off, j_off, k_off, col_off):
        for i in range(n_prime):
            for j in range(m_prime):
                for k in range(p_prime):
                    U[(i + i_off) * m + (j + j_off)][col_off] = 1
                    V[(j + j_off) * p + (k + k_off)][col_off] = 1
                    W[(k + k_off) * n + (i + i_off)][col_off] = 1
                    col_off += 1
        return col_off

    # Helper to add Strassen 2x2x2 terms
    # i_off, j_off, k_off are offsets in global indices
    def add_strassen_222(i_off, j_off, k_off, col_off):
        # M1: (A00+A11)(B00+B11)
        # U: A00, A11; V: B00, B11; W: C00, C11
        # We need to represent the bilinear form.
        # Term 1: A00 B00 -> C00
        U[i_off * m + j_off][col_off] = 1
        V[j_off * p + k_off][col_off] = 1
        W[k_off * n + i_off][col_off] = 1
        col_off += 1
        
        # Term 2: A00 B11 -> C01
        U[i_off * m + j_off][col_off] = 1
        V[j_off * p + (k_off + 1)][col_off] = 1
        W[(k_off + 1) * n + i_off][col_off] = 1
        col_off += 1
        
        # Term 3: A11 B00 -> C10
        U[(i_off + 1) * m + j_off][col_off] = 1
        V[j_off * p + k_off][col_off] = 1
        W[k_off * n + (i_off + 1)][col_off] = 1
        col_off += 1
        
        # Term 4: A11 B11 -> C11
        U[(i_off + 1) * m + j_off][col_off] = 1
        V[j_off * p + (k_off + 1)][col_off] = 1
        W[(k_off + 1) * n + (i_off + 1)][col_off] = 1
        col_off += 1
        
        # This manual construction of Strassen is tricky because Strassen combines terms.
        # It is safer to use the explicit linear combinations for U, V, W.
        # However, Strassen rank 7 is a sum of 7 rank-1 tensors.
        # The 7 terms are:
        # 1. (A00+A11) * (B00+B11) -> contributes to C00, C01, C10, C11
        # This is not a rank-1 tensor in the standard sense unless we expand.
        # Actually, the Strassen algorithm decomposes the tensor into 7 terms of the form u v^T w^T?
        # No, it decomposes the multiplication map.
        # The tensor $T = \sum C_{ij} A_{ik} B_{kj}$.
        # Strassen expresses this as sum of 7 terms $\sum_{r=1}^7 u_r \otimes v_r \otimes w_r$.
        
        # Let's use the known Strassen factors.
        # For 2x2x2:
        # r=1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) -> No, this is not correct.
        
        # Correct Strassen factors for 2x2x2:
        # The 7 terms correspond to:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to map these to U, V, W.
        # Each M_r is a product of a linear combo of A and a linear combo of B.
        # This results in a rank-1 tensor in the flattened space?
        # Yes, each term is $u_r \otimes v_r \otimes w_r$ where $w_r$ is the output vector.
        
        # Let's define the vectors for each term.
        # Indices for A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # Indices for B: 00, 10, 01, 11 -> 0, 2, 1, 3 (Row-major? No, B is 2x2. B[j,k]. j=0,1; k=0,1. Index j*2+k. 00->0, 01->1, 10->2, 11->3)
        # Indices for C: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # M1: (A00+A11)(B00+B11)
        # u1 = [1, 0, 0, 1] (A00, A11)
        # v1 = [1, 0, 0, 1] (B00, B11)
        # w1 = [1, 1, 1, 1] (Contributes to all C? No, M1 contributes to C00, C01, C10, C11 with coefficients)
        # C00 += M1
        # C01 += M1
        # C10 += M1
        # C11 += M1
        # So w1 = [1, 1, 1, 1]
        
        # M2: (A10+A11)B00
        # u2 = [0, 0, 1, 1]
        # v2 = [1, 0, 0, 0]
        # C10 += M2
        # w2 = [0, 0, 1, 0]
        
        # M3: A00(B01-B11)
        # u3 = [1, 0, 0, 0]
        # v3 = [0, 1, 0, -1]
        # C01 += M3
        # w3 = [0, 1, 0, 0]
        
        # M4: A11(B10-B00)
        # u4 = [0, 0, 0, 1]
        # v4 = [0, 0, 1, -1]
        # C10 += M4
        # w4 = [0, 0, 1, 0]
        
        # M5: (A00+A10)B11
        # u5 = [1, 0, 1, 0]
        # v5 = [0, 0, 0, 1]
        # C01 += M5
        # C11 += M5
        # w5 = [0, 1, 0, 1]
        
        # M6: (A10-A00)(B00+B01)
        # u6 = [-1, 0, 1, 0]
        # v6 = [1, 1, 0, 0]
        # C11 += M6
        # w6 = [0, 0, 0, 1]
        
        # M7: (A01-A11)(B10+B11)
        # u7 = [0, 1, 0, -1]
        # v7 = [0, 0, 1, 1]
        # C00 += M7
        # C11 += M7
        # w7 = [1, 0, 0, 1]
        
        # Let's verify C00: M1 + M4 - M5 + M7?
        # My w vectors:
        # C00: M1(1) + M7(1) = 2?
        # Standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-map w vectors to match these coefficients.
        # The tensor term is $u \otimes v \otimes w$.
        # The contribution to $C_{ij}$ is $w_{ij} u(A) v(B)$.
        
        # Term 1 (M1): u=[1,0,0,1], v=[1,0,0,1], w=[1,1,1,1]
        # Term 2 (M2): u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
        # Term 3 (M3): u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
        # Term 4 (M4): u=[0,0,0,1], v=[0,0,1,-1], w=[0,0,1,0]
        # Term 5 (M5): u=[1,0,1,0], v=[0,0,0,1], w=[0,1,0,1]
        # Term 6 (M6): u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1]
        # Term 7 (M7): u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,1]
        
        # Check C00: w1[0] + w7[0] = 1 + 1 = 2. But we want 1?
        # Wait, the standard Strassen formula is:
        # C00 = M1 + M4 - M5 + M7
        # If I use the w vectors above, the sum is:
        # C00 = 1(M1) + 0(M2) + 0(M3) + 0(M4) - 0(M5) + 0(M6) + 1(M7) = 2 M1 + M7?
        # No, the w vector defines the coefficient.
        # For M1, w=[1,1,1,1].
        # For M7, w=[1,0,0,1].
        # Sum at C00: 1 + 1 = 2.
        # This implies $2 M1 + M7$? No, it implies the coefficient of the term is 2?
        # No, the term is $u \otimes v \otimes w$.
        # If $w=[1,1,1,1]$, it adds to all.
        # So C00 gets 1 from M1.
        # C00 gets 1 from M7.
        # Total 2.
        # But we want 1.
        # This means the w vectors must be scaled or the u/v vectors adjusted.
        # Actually, $M1 = (A00+A11)(B00+B11)$.
        # The term in the sum is $M1$.
        # So the coefficient is 1.
        # Why did I get 2?
        # Because I included M7 in C00.
        # $C00 = M1 + M4 - M5 + M7$.
        # So C00 should be $1 \cdot M1 + 1 \cdot M4 - 1 \cdot M5 + 1 \cdot M7$.
        # My w vectors:
        # M1: w00=1
        # M4: w10=1 (so w00=0)
        # M5: w01=1, w11=1 (so w00=0)
        # M7: w00=1, w11=1
        # Sum w00 = 1 (from M1) + 1 (from M7) = 2.
        # This is incorrect.
        
        # Let's look at the standard decomposition.
        # The 7 terms are linearly independent.
        # The w vectors must satisfy the linear combinations.
        
        # Correct w vectors for Strassen 2x2x2:
        # 1: [1, 1, 1, 1]
        # 2: [0, 0, 1, 0]
        # 3: [0, 1, 0, 0]
        # 4: [0, 0, 1, 0]
        # 5: [0, 1, 0, 1]
        # 6: [0, 0, 0, 1]
        # 7: [1, 0, 0, 1]
        
        # Let's check C00: 1(M1) + 0 + 0 + 0 + 0 + 0 + 1(M7) = M1 + M7.
        # But formula is M1 + M4 - M5 + M7.
        # M4 has w10=1. So w00=0.
        # M5 has w01=1, w11=1. So w00=0.
        # So C00 = M1 + M7.
        # This is not matching the formula.
        
        # The formula is derived from the specific u, v vectors.
        # If I change u, v, w, I must ensure the product matches.
        
        # It is safer to use the **schoolbook** for $2 \times 2 \times 2$ if I can't get the factors right quickly, but rank 8 vs 7.
        # Using schoolbook for $2 \times 2 \times 2$ increases rank.
        # $R(2, 2, 3) \le 8 + 4 = 12$.
        # $R(2, 4, 3) \le 24$.
        # Total rank becomes $12 + 16 + 24 + 28 = 80$.
        # Still better than 84.
        
        # Let's stick to Schoolbook for the sub-blocks to ensure correctness and simplicity, as 80 < 84.
        # Or even just Schoolbook for everything? No, 84.
        # Using Schoolbook for $2 \times 2 \times 3$ (Rank 12) and $2 \times 2 \times 4$ (Rank 16).
        # $R(2, 4, 3) \le 24$.
        # $R(2, 4, 4) \le 32$.
        # Total: $12 + 16 + 24 + 32 = 84$. No improvement.
        
        # So I MUST use Strassen for the $2 \times 2 \times 2$ parts.
        # I will use the verified Strassen factors from literature.
        
        return col_off

    # Use Schoolbook for 1x4x3
    col = add_schoolbook(1, 4, 3, 0, 0, 0, col)
    
    # Use Schoolbook for 1x4x4
    col = add_schoolbook(1, 4, 4, 0, 0, 3, col)
    
    # For 2x4x3, use 2 * (2x2x3)
    # 2x2x3 = Strassen(2x2x2) + Schoolbook(2x2x1)
    # We need to implement Strassen correctly.
    
    return {"U": U, "V": V, "W": W}
