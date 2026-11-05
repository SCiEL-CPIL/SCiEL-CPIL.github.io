"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # We decompose M_{3,3,7} into M_{3,3,3} + M_{3,3,3} + M_{3,3,1}
    # We use a 2+1 block split for M_{3,3,3} to get rank 26.
    # M_{3,3,1} uses schoolbook rank 9.
    # Total rank = 26 + 26 + 9 = 61.
    
    terms = []
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        # u_idx, v_idx, w_idx are lists of (index, coeff) or just indices if coeff is 1?
        # Better to pass lists of (index, coeff) for U, V, W
        pass

    # Helper to get Strassen terms for 2x2x2
    # Returns list of (u_list, v_list, w_list)
    # u, v, w are dicts or lists of (idx, coeff)
    def get_strassen_222():
        # Indices for 2x2: 0,1,2,3
        # A: 00,01,10,11 -> 0,1,2,3
        # B: 00,01,10,11 -> 0,1,2,3
        # C: 00,01,10,11 -> 0,1,2,3
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to output U, V, W vectors for each of the 7 multiplications.
        # Each multiplication is a rank 1 tensor in the sub-space.
        # U_k, V_k, W_k for k=1..7
        
        s_terms = []
        
        # M1
        u1 = [(0, 1), (3, 1)]
        v1 = [(0, 1), (3, 1)]
        # C00 += M1, C11 += M1
        # Wait, Strassen computes intermediates M1..M7.
        # The final C is a linear combination.
        # The tensor decomposition terms correspond to the M_k's?
        # No, the rank is the number of multiplications M_k.
        # So we add 7 terms to the global list.
        # For each M_k, we define the U, V, W vectors in the local 2x2 space.
        # Then we map them to the global space.
        
        # Local indices 0,1,2,3 correspond to:
        # A: 00, 01, 10, 11
        # B: 00, 01, 10, 11
        # C: 00, 01, 10, 11
        
        # M1: U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1) ?
        # No, W is the output coefficient.
        # The term is U_k \otimes V_k \otimes W_k.
        # The multiplication is $U_k \cdot A \cdot V_k^T$? No.
        # The term contributes to C.
        # $C_{ij} += W_{ij} (U \cdot A) (V \cdot B)$?
        # Standard tensor decomposition: $T = \sum r U_r \otimes V_r \otimes W_r$.
        # $C_{ik} = \sum_j \sum_r U_{ir} V_{jr} W_{kj} A_{ij} B_{jk}$.
        # Actually, $C = \sum_r (U_r A) (V_r B)^T$? No.
        # $C_{ik} = \sum_r U_{ir} W_{kr} (A \cdot ?)$
        
        # Let's stick to the explicit index mapping.
        # For a term r, we have vectors $u_r$ (size $n \times m$), $v_r$ (size $m \times p$), $w_r$ (size $p \times n$).
        # The contribution is $\sum_{i,j,k} u_{ij} v_{jk} w_{ki} A_{ij} B_{jk}$.
        # Wait, the standard form is $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # Tensor $T_{ijk} = \delta_{ij} \delta_{jk}$? No.
        # $T = \sum_{i,j,k} e_i \otimes f_j \otimes g_k$ where $A_{ij} B_{jk} C_{ki}$.
        # The rank 1 term is $u \otimes v \otimes w$.
        # $u$ has 1 at $(i,j)$, $v$ has 1 at $(j,k)$, $w$ has 1 at $(k,i)$.
        # This is the schoolbook term.
        
        # For Strassen, we have linear combinations.
        # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
        # This corresponds to a term where:
        # $u$ has 1 at $A_{00}$ and 1 at $A_{11}$.
        # $v$ has 1 at $B_{00}$ and 1 at $B_{11}$.
        # $w$ has 1 at $C_{00}$ and 1 at $C_{11}$?
        # No, $M_1$ contributes to $C_{00}$ and $C_{11}$.
        # $C_{00} += M_1$. $C_{11} += M_1$.
        # So $w$ has 1 at $C_{00}$ and 1 at $C_{11}$.
        
        # Let's verify.
        # Term: $u = e_{A00} + e_{A11}$, $v = e_{B00} + e_{B11}$, $w = e_{C00} + e_{C11}$.
        # Value: $(A00+A11)(B00+B11)(C00+C11 \text{ coeff?})$
        # The tensor contraction is $\sum u_{ij} v_{jk} w_{ki} A_{ij} B_{jk}$.
        # If $w$ has 1 at $C00$ and $C11$, then the term adds to $C00$ and $C11$.
        # Yes.
        
        # So for each Strassen intermediate $M_k$, we define u, v, w.
        
        # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
        # M2: (A10+A11)B00. u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0] (C10)
        # M3: A00(B01-B11). u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0] (C01)
        # M4: A11(B10-B00). u=[0,0,0,1], v=[0,0,1,-1], w=[1,0,1,0] (C00, C10)
        # M5: (A00+A01)B11. u=[1,1,0,0], v=[0,0,0,1], w=[0,1,-1,0] (C01, -C10)
        # M6: (A10-A00)(B00+B01). u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1] (C11)
        # M7: (A01-A11)(B10+B11). u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,-1] (C00, -C11)
        
        # Check C00: M1 + M4 - M5 + M7
        # C00 from M1: 1
        # C00 from M4: 1
        # C00 from M5: -1
        # C00 from M7: 1
        # Sum: 1+1-1+1 = 2? 
        # Wait.
        # Strassen:
        # C00 = M1 + M4 - M5 + M7
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11(B10-B00) = A11B10 - A11B00
        # M5 = (A00+A01)B11 = A00B11 + A01B11
        # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
        
        # Sum for C00 terms:
        # A00B00: M1(1)
        # A00B11: M1(1) - M5(1) = 0
        # A11B00: M1(1) - M4(1) = 0
        # A11B11: M1(1) - M7(1) = 0
        # A11B10: M4(1) - M7(1) = 0
        # A01B11: -M5(1) + M7(1) = 0
        # A01B10: M7(1)
        
        # This is not C00. C00 = A00B00 + A01B10.
        # My M1..M7 definitions or combination is wrong.
        
        # Correct Strassen:
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
        
        # Let's re-evaluate C00 with this.
        # M1: A00B00 + A00B11 + A11B00 + A11B11
        # M4: A11B10 - A11B00
        # M5: A00B11 + A01B11
        # M7: A01B10 + A01B11 - A11B10 - A11B11
        
        # Sum:
        # A00B00: 1
        # A00B11: 1 - 1 = 0
        # A11B00: 1 - 1 = 0
        # A11B11: 1 - 1 = 0
        # A11B10: 1 - 1 = 0
        # A01B11: -1 + 1 = 0
        # A01B10: 1
        
        # Result: A00B00 + A01B10. Correct.
        
        # So the w vectors for the 7 terms are:
        # w1: [1, 0, 0, 1] (C00, C11)
        # w2: [0, 0, 1, 0] (C10)
        # w3: [0, 1, 0, 0] (C01)
        # w4: [1, 0, 1, 0] (C00, C10)
        # w5: [0, 1, -1, 0] (C01, -C10)
        # w6: [0, 0, 0, 1] (C11)
        # w7: [1, 0, 0, -1] (C00, -C11)
        
        # u vectors:
        # u1: [1, 0, 0, 1]
        # u2: [0, 0, 1, 1]
        # u3: [1, 0, 0, 0]
        # u4: [0, 0, 0, 1]
        # u5: [1, 1, 0, 0]
        # u6: [-1, 0, 1, 0]
        # u7: [0, 1, 0, -1]
        
        # v vectors:
        # v1: [1, 0, 0, 1]
        # v2: [1, 0, 0, 0]
        # v3: [0, 1, 0, -1]
        # v4: [0, 0, 1, -1]
        # v5: [0, 0, 0, 1]
        # v6: [1, 1, 0, 0]
        # v7: [0, 0, 1, 1]
        
        return [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
    
    # We will implement the 2+1 split for M_{3,3,3}
    # and schoolbook for M_{3,3,1}
    
    # This is getting long. I will structure the code to build the global U, V, W lists.
    
    # Global dimensions:
    # U: 9
    # V: 21
    # W: 21
    
    U = []
    V = []
    W = []
    
    # Add terms for M_{3,3,3} block 1 (cols 0,1,2 of B)
    # Add terms for M_{3,3,3} block 2 (cols 3,4,5 of B)
    # Add terms for M_{3,3,1} block 3 (col 6 of B)
    
    # Helper to map local 2x2 indices to global 3x3 indices
    # Local 0->0, 1->1, 2->2, 3->3
    # For A (3x3):
    # A22: 0,1,2,3 -> Global 0,1,3,4
    # A21: 0,1 -> Global 2,5
    # A12: 0,1 -> Global 6,7
    # A11: 0 -> Global 8
    
    # For B (3x3):
    # B22: 0,1,2,3 -> Global 0,1,3,4
    # B21: 0,1 -> Global 2,5
    # B12: 0,1 -> Global 6,7
    # B11: 0 -> Global 8
    
    # For C (3x3):
    # C22: 0,1,2,3 -> Global 0,1,3,4
    # C21: 0,1 -> Global 2,5
    # C12: 0,1 -> Global 6,7
    # C11: 0 -> Global 8
    
    # We need to add 7 Strassen terms for A22*B22.
    # Map local u (4) to global u (9).
    # local 0->0, 1->1, 2->3, 3->4.
    
    def map_u_22_to_33(local_u):
        global_u = [0]*9
        global_u[0] = local_u[0]
        global_u[1] = local_u[1]
        global_u[3] = local_u[2]
        global_u[4] = local_u[3]
        return global_u

    def map_v_22_to_33(local_v):
        global_v = [0]*9
        global_v[0] = local_v[0]
        global_v[1] = local_v[1]
        global_v[3] = local_v[2]
        global_v[4] = local_v[3]
        return global_v

    def map_w_22_to_33(local_w):
        global_w = [0]*9
        global_w[0] = local_w[0]
        global_w[1] = local_w[1]
        global_w[3] = local_w[2]
        global_w[4] = local_w[3]
        return global_w

    # Add Strassen terms for A22*B22
    strassen_terms = get_strassen_222()
    for u, v, w in strassen_terms:
        U.append(map_u_22_to_33(u))
        V.append(map_v_22_to_33(v))
        W.append(map_w_22_to_33(w))
        
    # Now handle the other 5 sub-products for M_{3,3,3}.
    # A21*B12 (2x1 * 1x2 -> 2x2). Rank 4.
    # A22*B21 (2x2 * 2x1 -> 2x1). Rank 4.
    # A21*B11 (2x1 * 1x1 -> 2x1). Rank 2.
    # A12*B22 (1x2 * 2x2 -> 1x2). Rank 4.
    # A12*B21 (1x2 * 2x1 -> 1x1). Rank 2.
    # A11*B12 (1x1 * 1x2 -> 1x2). Rank 2.
    # A11*B11 (1x1 * 1x1 -> 1x1). Rank 1.
    
    # Total 7+4+4+2+4+2+2+1 = 26.
    
    # This is too complex to hardcode all mappings in the thought process.
    # I will implement the schoolbook for the non-2x2 parts and Strassen for 2x2.
    
    # Schoolbook for A21*B12:
    # A21 indices: 2,5. B12 indices: 6,7. C22 indices: 0,1,3,4.
    # Terms:
    # (2, 6, 0), (2, 7, 1), (5, 6, 3), (5, 7, 4)
    
    # Schoolbook for A22*B21:
    # A22: 0,1,3,4. B21: 2,5. C21: 2,5.
    # Terms:
    # (0,2,2), (1,2,5), (3,2,2), (4,2,5)
    
    # Schoolbook for A21*B11:
    # A21: 2,5. B11: 8. C21: 2,5.
    # Terms:
    # (2,8,2), (5,8,5)
    
    # Schoolbook for A12*B22:
    # A12: 6,7. B22: 0,1,3,4. C12: 6,7.
    # Terms:
    # (6,0,6), (6,1,7), (7,3,6), (7,4,7)
    
    # Schoolbook for A12*B21:
    # A12: 6,7. B21: 2,5. C11: 8.
    # Terms:
    # (6,2,8), (7,5,8)
    
    # Schoolbook for A11*B12:
    # A11: 8. B12: 6,7. C12: 6,7.
    # Terms:
    # (8,6,6), (8,7,7)
    
    # Schoolbook for A11*B11:
    # A11: 8. B11: 8. C11: 8.
    # Terms:
    # (8,8,8)
    
    # We need to add these to U, V, W.
    # Note: V indices for B are $j \times 3 + k$.
    # B12: j=2, k=0,1 -> 6,7.
    # B21: j=0,1, k=2 -> 2,5.
    # B11: j=2, k=2 -> 8.
    # B22: j=0,1, k=0,1 -> 0,1,3,4.
    
    # W indices for C are $k \times 3 + i$.
    # C21: k=2, i=0,1 -> 2,5.
    # C12: k=0,1, i=2 -> 6,7.
    # C11: k=2, i=2 -> 8.
    # C22: k=0,1, i=0,1 -> 0,1,3,4.
    
    # U indices for A are $i \times 3 + j$.
    # A21: i=0,1, j=2 -> 2,5.
    # A12: i=2, j=0,1 -> 6,7.
    # A11: i=2, j=2 -> 8.
    # A22: i=0,1, j=0,1 -> 0,1,3,4.
    
    # All consistent.
    
    # Add schoolbook terms for the mixed parts
    mixed_terms = [
        # A21*B12
        (2, 6, 0), (2, 7, 1), (5, 6, 3), (5, 7, 4),
        # A22*B21
        (0, 2, 2), (1, 2, 5), (3, 2, 2), (4, 2, 5),
        # A21*B11
        (2, 8, 2), (5, 8, 5),
        # A12*B22
        (6, 0, 6), (6, 1, 7), (7, 3, 6), (7, 4, 7),
        # A12*B21
        (6, 2, 8), (7, 5, 8),
        # A11*B12
        (8, 6, 6), (8, 7, 7),
        # A11*B11
        (8, 8, 8)
    ]
    
    for u_idx, v_idx, w_idx in mixed_terms:
        u = [0]*9
        v = [0]*9
        w = [0]*9
        u[u_idx] = 1
        v[v_idx] = 1
        w[w_idx] = 1
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Now we have 26 terms for the first M_{3,3,3} block (B cols 0,1,2).
    # We need to repeat this for the second M_{3,3,3} block (B cols 3,4,5).
    # B indices shift by 3.
    # B22 (cols 3,4) -> j=0,1, k=3,4 -> 3,4,6,7.
    # B21 (col 5) -> j=0,1, k=5 -> 5,8.
    # B12 (cols 3,4) -> j=2, k=3,4 -> 9,10.
    # B11 (col 5) -> j=2, k=5 -> 11.
    
    # C indices for this block?
    # C is 3x7.
    # Block 1 C cols 0,1,2.
    # Block 2 C cols 3,4,5.
    # Block 3 C col 6.
    
    # So for Block 2, W indices shift by 3.
    # C22 -> 3,4,6,7.
    # C21 -> 5,8.
    # C12 -> 9,10.
    # C11 -> 11.
    
    # U is same.
    
    # We can copy the 26 terms from Block 1 and shift V and W indices.
    
    # V shift: +3
    # W shift: +3
    
    U_block1 = U.copy()
    V_block1 = V.copy()
    W_block1 = W.copy()
    
    # Clear U, V, W? No, we are building the list.
    # We already appended 26 terms.
    # Now append 26 more for Block 2.
    
    for i in range(26):
        u = U_block1[i]
        v = [0]*21 # V size is 3*7=21
        w = [0]*21 # W size is 7*3=21
        
        # Map V
        for j in range(9):
            if V_block1[i][j] != 0:
                v[j+3] = V_block1[i][j]
                
        # Map W
        for j in range(9):
            if W_block1[i][j] != 0:
                w[j+3] = W_block1[i][j]
                
        U.append(u)
        V.append(v)
        W.append(w)
        
    # Now Block 3: M_{3,3,1} (B col 6).
    # Schoolbook rank 9.
    # B indices: j=0,1,2, k=6 -> 6,9,12.
    # C indices: k=6, i=0,1,2 -> 18,19,20.
    # U indices: 0..8.
    
    # Terms:
    # (i, j, k=6) -> U[i*3+j], V[j*3+6], W[6*3+i]
    
    for i in range(3):
        for j in range(3):
            u_idx = i*3 + j
            v_idx = j*3 + 6
            w_idx = 6*3 + i
            
            u = [0]*9
            v = [0]*21
            w = [0]*21
            u[u_idx] = 1
            v[v_idx] = 1
            w[w_idx] = 1
            U.append(u)
            V.append(v)
            W.append(w)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
