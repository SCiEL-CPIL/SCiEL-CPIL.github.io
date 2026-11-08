"""
Task: Matrix multiplication 4x4
Goal: Find rank-minimal exact tensor decomposition.
Strategy: Use Strassen's algorithm to get rank 49 (vs schoolbook rank 64).
"""

import time

DIMENSIONS = [4, 4, 4]

def _build_strassen_decomp():
    """
    Constructs the tensor decomposition for 4x4 matrix multiplication
    using a 2-level Strassen algorithm.
    
    Returns:
    U: list of 16 lists of length 49 (input A coefficients)
    V: list of 16 lists of length 49 (input B coefficients)
    W: list of 16 lists of length 49 (output C coefficients)
    """
    n = 4
    rank = 49
    
    # Initialize with zeros
    U = [[0.0] * rank for _ in range(n * n)]
    V = [[0.0] * rank for _ in range(n * n)]
    W = [[0.0] * rank for _ in range(n * n)]
    
    # Strassen's 2x2 matrix multiplication formulas
    # Inputs are split into 2x2 blocks:
    # A = A11 A12
    #     A21 A22
    # B = B11 B12
    #     B21 B22
    #
    # C = C11 C12
    #     C21 C22
    #
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Define the 7 Strassen multipliers for the 2x2 case
    # Each multiplier is a tuple (coeff_A_block, coeff_B_block)
    # coeff is +1 or -1 or 0
    
    # M1: (A11+A22)*(B11+B22)
    strassen_m1 = [((0,0), 1), ((1,1), 1), ((0,0), 1), ((1,1), 1)] # Simplified view, need proper expansion
    # Actually, we need to expand the linear combinations.
    
    # Let's define the components of A and B required for each Strassen product M_k
    # M_k = (sum a_i * A_i) * (sum b_j * B_j)
    # where A_i, B_j are 2x2 blocks (indices 0..15).
    
    # Indices:
    # A:
    # 0: A11[0,0], 1: A11[0,1], 2: A12[0,0], 3: A12[0,1]
    # 4: A21[0,0], 5: A21[0,1], 6: A22[0,0], 7: A22[0,1]
    # 8: A11[1,0], 9: A11[1,1], 10: A12[1,0], 11: A12[1,1]
    # 12: A21[1,0], 13: A21[1,1], 14: A22[1,0], 15: A22[1,1]
    
    # B:
    # 0: B11[0,0], 1: B11[0,1], 2: B12[0,0], 3: B12[0,1]
    # 4: B21[0,0], 5: B21[0,1], 6: B22[0,0], 7: B22[0,1]
    # 8: B11[1,0], 9: B11[1,1], 10: B12[1,0], 11: B12[1,1]
    # 12: B21[1,0], 13: B21[1,1], 14: B22[1,0], 15: B22[1,1]
    
    # Output C indices (C = A * B):
    # 0: C11[0,0], 1: C11[0,1], 2: C12[0,0], 3: C12[0,1]
    # 4: C21[0,0], 5: C21[0,1], 6: C22[0,0], 7: C22[0,1]
    # 8: C11[1,0], 9: C11[1,1], 10: C12[1,0], 11: C12[1,1]
    # 12: C21[1,0], 13: C21[1,1], 14: C22[1,0], 15: C22[1,1]
    
    # Level 1: 4x4 decomposition into eight 2x2 blocks (indices 0..7)
    # We are doing 2x2 multiplication where elements are 2x2 blocks.
    # Block A indices in 4x4 (row major):
    # Block 0 (A11): 0..3
    # Block 1 (A12): 4..7
    # Block 2 (A21): 8..11
    # Block 3 (A22): 12..15
    
    # Block B indices:
    # Block 0 (B11): 0..3
    # Block 1 (B12): 4..7
    # Block 2 (B21): 8..11
    # Block 3 (B22): 12..15
    
    # Output C indices:
    # Block 0 (C11): 0..3
    # Block 1 (C12): 4..7
    # Block 2 (C21): 8..11
    # Block 3 (C22): 12..15

    # We will construct 49 terms corresponding to the 7 outer products * 7 inner products.
    # Actually, it's 49 rank-1 components in total for 4x4.
    # The recursion is: 4x4 = Strassen(2x2) applied to blocks which are 2x2.
    # Total rank = 7 (outer) * 7 (inner) = 49.
    
    # Let's define the Strassen coefficients for the 2x2 case (indices 0,1,2,3 for blocks)
    # A_blocks: 0=A11, 1=A12, 2=A21, 3=A22
    # B_blocks: 0=B11, 1=B12, 2=B21, 3=B22
    
    # M1 = (A11+A22)(B11+B22) -> A_idx: [0, 3], B_idx: [0, 3]
    # M2 = (A21+A22)B11     -> A_idx: [2, 3], B_idx: [0]
    # M3 = A11(B12-B22)     -> A_idx: [0], B_idx: [1, -1 (at 3)]
    # M4 = A22(B21-B11)     -> A_idx: [3], B_idx: [2, -1 (at 0)]
    # M5 = (A11+A12)B22     -> A_idx: [0, 1], B_idx: [3]
    # M6 = (A21-A11)(B11+B12)-> A_idx: [2, -1 (at 0)], B_idx: [0, 1]
    # M7 = (A12-A22)(B21+B22)-> A_idx: [1, -1 (at 3)], B_idx: [2, 3]
    
    outer_terms = [
        (([0, 3], [0, 3]), "M1"),  # (A0+A3) * (B0+B3)
        (([2, 3], [0]), "M2"),     # (A2+A3) * B0
        (([0], [1, -1]), "M3"),    # A0 * (B1 - B3)
        (([3], [2, -1]), "M4"),    # A3 * (B2 - B0)
        (([0, 1], [3]), "M5"),     # (A0+A1) * B3
        (([2, -1], [0, 1]), "M6"), # (A2-A0) * (B0+B1)
        (([1, -1], [2, 3]), "M7"), # (A1-A3) * (B2+B3)
    ]
    
    # For each of the 7 outer products, we decompose the 2x2 block multiplication into 7 inner products.
    # The inner indices are 0..3 for A-block-elements, 0..3 for B-block-elements, 0..3 for C-block-elements.
    # But wait, the "elements" are 2x2 matrices.
    # So when we say "multiply A_block_i by B_block_j", we are performing a 2x2 * 2x2 multiplication,
    # which itself requires rank 7 (Strassen).
    # So for each outer term M_k, we generate 7 inner terms.
    # Total 49 terms.
    
    # Let's define the inner Strassen coefficients (for 2x2 block multiplication).
    # These map block indices (0,1,2,3) of the 2x2 matrix to 4 elements (0,1,2,3).
    # Element indices for a 2x2 block:
    # 0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
    
    inner_strassen_A = [
        ([0, 3], "iM1"), # A_tl + A_br
        ([2, 3], "iM2"), # A_bl + A_br
        ([0], "iM3"),    # A_tl
        ([3], "iM4"),    # A_br
        ([0, 1], "iM5"), # A_tl + A_tr
        ([2, -1], "iM6"),# A_bl - A_tl (coeffs applied to indices)
        ([1, -1], "iM7") # A_tr - A_br
    ]
    inner_strassen_B = [
        ([0, 3], "iM1"), # B_tl + B_br
        ([0], "iM2"),    # B_tl
        ([1, -1], "iM3"),# B_tr - B_br
        ([2, -1], "iM4"),# B_bl - B_tl (Wait, formula is B21-B11 -> bl - tl)
        ([3], "iM5"),    # B_br
        ([0, 1], "iM6"), # B_tl + B_tr
        ([2, 3], "iM7")  # B_bl + B_br
    ]
    
    # Output combination for C = A*B (2x2)
    # C_tl = iM1 + iM4 - iM5 + iM7
    # C_tr = iM3 + iM5
    # C_bl = iM2 + iM4
    # C_br = iM1 - iM2 + iM3 + iM6
    
    inner_combos_C = {
        0: ([(0, 1), (3, 1), (4, -1), (6, 1)], "C0"), # C_tl
        1: ([(2, 1), (4, 1)], "C1"),                   # C_tr
        2: ([(1, 1), (3, 1)], "C2"),                   # C_bl
        3: ([(0, 1), (1, -1), (2, 1), (5, 1)], "C3")   # C_br
    }
    
    # Now, we combine outer and inner.
    # Term index = outer_idx * 7 + inner_idx
    
    for o_idx, (outer_def, _) in enumerate(outer_terms):
        outer_A_idx = outer_def[0][0] # List of coeffs? No, I put [indices] in list, need coeffs too.
        # Let's re-define outer terms properly with coeffs.
        
    # Correct Definition of Outer Terms with Coeffs
    outer_terms_defs = [
        # (List of (A_block_idx, coeff), List of (B_block_idx, coeff))
        ([((0, 1), (3, 1)), ((0, 1), (3, 1))]), # M1
        ([((2, 1), (3, 1)), ((0, 1))]),          # M2
        ([((0, 1)), ((1, 1), (3, -1))]),         # M3
        ([((3, 1)), ((2, 1), (0, -1))]),         # M4
        ([((0, 1), (1, 1)), ((3, 1))]),          # M5
        ([((2, 1), (0, -1)), ((0, 1), (1, 1))]), # M6
        ([((1, 1), (3, -1)), ((2, 1), (3, 1))])  # M7
    ]
    
    inner_terms_defs = [
        ([((0, 1), (3, 1)), ((0, 1), (3, 1))]), # iM1
        ([((2, 1), (3, 1)), ((0, 1))]),          # iM2
        ([((0, 1)), ((1, 1), (3, -1))]),         # iM3
        ([((3, 1)), ((2, 1), (0, -1))]),         # iM4
        ([((0, 1), (1, 1)), ((3, 1))]),          # iM5
        ([((2, 1), (0, -1)), ((0, 1), (1, 1))]), # iM6
        ([((1, 1), (3, -1)), ((2, 1), (3, 1))])  # iM7
    ]
    
    # Note: The inner terms operate on the elements of the 2x2 blocks.
    # An element in block b, local index l has global index b*4 + l.
    # Global index for A:
    # Block 0 (A11): 0,1,2,3 -> Local 0,1,2,3
    # Block 1 (A12): 4,5,6,7 -> Local 0,1,2,3
    # Block 2 (A21): 8,9,10,11 -> Local 0,1,2,3
    # Block 3 (A22): 12,13,14,15 -> Local 0,1,2,3
    # Mapping: global_idx = block_idx * 4 + local_idx
    
    # Same for B.
    
    # Global index for C:
    # Block 0 (C11): 0,1,2,3
    # Block 1 (C12): 4,5,6,7
    # Block 2 (C21): 8,9,10,11
    # Block 3 (C22): 12,13,14,15
    
    # We need to fill U, V, W.
    # U[i, r] is coeff of A_i in rank r.
    # V[j, r] is coeff of B_j in rank r.
    # W[k, r] is coeff of C_k in rank r.
    
    for o_idx in range(7):
        o_A, o_B = outer_terms_defs[o_idx]
        for i_idx in range(7):
            i_A, i_B = inner_terms_defs[i_idx]
            
            col = o_idx * 7 + i_idx
            
            # Accumulate U contributions
            # U depends on o_A and i_A.
            # Term is sum_{a_idx, a_coeff} sum_{l_idx, l_coeff} o_coeff * i_coeff * A_{global_a_idx}
            # Wait, Strassen is bilinear.
            # (Sum A_a o_a) * (Sum B_b o_b) = Sum_{a,b} o_a o_b A_a B_b.
            # Here, each "A_a" is a 2x2 matrix.
            # So we are expanding (Sum_{a in outer_A} o_a * BlockA_a) * (Sum_{b in outer_B} o_b * BlockB_b).
            # BlockA_a is a 2x2 matrix with elements A_{a*4 + l}.
            # The multiplication of these two "block" matrices is computed via inner Strassen.
            # Inner Strassen decomposes the product of two 2x2 matrices (which are the sum of outer blocks)
            # No, that's not right.
            
            # Correct Interpretation:
            # The rank-1 terms of the tensor product are products of rank-1 terms of the factors.
            # Let $T_{4x4}$ be the multiplication tensor.
            # $T_{4x4} = T_{2x2} \otimes T_{2x2}$. (Kronecker product structure)
            # If $T_{2x2} = \sum_{r=1}^7 \mathbf{u}_r \otimes \mathbf{v}_r \otimes \mathbf{w}_r$,
            # then $T_{4x4} = \sum_{r=1}^7 \sum_{s=1}^7 (\mathbf{u}_r \otimes \mathbf{u}_s) \otimes (\mathbf{v}_r \otimes \mathbf{v}_s) \otimes (\mathbf{w}_r \otimes \mathbf{w}_s)$.
            
            # So, for each pair (r, s) in 1..7 x 1..7:
            # $\mathbf{u}_{rs} = \mathbf{u}_r \otimes \mathbf{u}_s$
            # $\mathbf{v}_{rs} = \mathbf{v}_r \otimes \mathbf{v}_s$
            # $\mathbf{w}_{rs} = \mathbf{w}_r \otimes \mathbf{w}_s$
            
            # We need to calculate the vectors $\mathbf{u}_r, \mathbf{v}_r, \mathbf{w}_r$ for the 2x2 case.
            # Then take their Kronecker products.
            
            # Let's define the 2x2 vectors.
            # Dimension 16.
            # Indices 0..15.
            
            # u_r corresponds to A.
            # v_r corresponds to B.
            # w_r corresponds to C.
            
            u_vecs = [None] * 7
            v_vecs = [None] * 7
            w_vecs = [None] * 7
            
            for r in range(7):
                u_vecs[r] = [0.0] * 16
                v_vecs[r] = [0.0] * 16
                w_vecs[r] = [0.0] * 16
                
                # Use the definitions from above
                # Note: The definitions `outer_terms_defs` were actually for the 2x2 structure!
                # Let's re-verify.
                # Yes, `outer_terms_defs` defined the coefficients for A_blocks, B_blocks.
                # But in the context of T_2x2, the "blocks" are just scalars (indices 0..3).
                # So for the base 2x2 case:
                # A indices: 0,1,2,3.
                # B indices: 0,1,2,3.
                # C indices: 0,1,2,3.
                
                # My `outer_terms_defs` used block indices 0..3.
                # So `outer_terms_defs[r][0]` gives list of (idx, coeff) for A.
                # `outer_terms_defs[r][1]` gives list of (idx, coeff) for B.
                
                for (a_idx, coeff) in outer_terms_defs[r][0]:
                    u_vecs[r][a_idx] = coeff
                for (b_idx, coeff) in outer_terms_defs[r][1]:
                    v_vecs[r][b_idx] = coeff
                
                # For W, we use the C combinations derived from inner Strassen logic.
                # Wait, `inner_combos_C` was defined for the 2x2 case output.
                # Let's map r (0..6) to the output C indices (0..3) using the Strassen output formulas.
                
                # C0 (0) = M1 + M4 - M5 + M7
                # C1 (1) = M3 + M5
                # C2 (2) = M2 + M4
                # C3 (3) = M1 - M2 + M3 + M6
                
                # We need to express w_r vector in terms of M's?
                # No, w_r is the vector for C such that C = sum M_k * C_k.
                # But in the tensor decomposition, we need $\mathbf{w}_r$ such that
                # $C_{i,j} = \sum_r U_{i,r} V_{j,r} W_{k,r}$.
                # Actually, the third factor in $T_{2x2}$ maps $M_r$ to $C$.
                # So $W$ vector for rank r has non-zero entries at the positions of C
                # with values equal to the coefficients of M_r in the linear combination forming C.
                
                # Let's check standard Strassen decomposition:
                # $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$
                # $\mathbf{u}_1$ has 1 at A11, 1 at A22.
                # $\mathbf{v}_1$ has 1 at B11, 1 at B22.
                # $\mathbf{w}_1$: $C_{11} += M_1$, $C_{22} += M_1$.
                # So $w_1[0] = 1, w_1[3] = 1$.
                
                # $M_2 = (A_{21}+A_{22})B_{11}$
                # $w_2$: $C_{21} += M_2, C_{22} += -M_2$.
                # $w_2[2] = 1, w_2[3] = -1$.
                
                # $M_3 = A_{11}(B_{12}-B_{22})$
                # $w_3$: $C_{12} += M_3, C_{22} += M_3$.
                # $w_3[1] = 1, w_3[3] = 1$.
                
                # $M_4 = A_{22}(B_{21}-B_{11})$
                # $w_4$: $C_{11} += M_4, C_{21} += M_4$.
                # $w_4[0] = 1, w_4[2] = 1$.
                
                # $M_5 = (A_{11}+A_{12})B_{22}$
                # $w_5$: $C_{11} += -M_5, C_{12} += M_5$.
                # $w_5[0] = -1, w_5[1] = 1$.
                
                # $M_6 = (A_{21}-A_{11})(B_{11}+B_{12})$
                # $w_6$: $C_{22} += M_6$.
                # $w_6[3] = 1$.
                
                # $M_7 = (A_{12}-A_{22})(B_{21}+B_{22})$
                # $w_7$: $C_{11} += M_7$.
                # $w_7[0] = 1$.

                # Let's verify this against the formulas:
                # C11 = M1 + M4 - M5 + M7 -> w1[0]=1, w4[0]=1, w5[0]=-1, w7[0]=1. OK.
                # C12 = M3 + M5 -> w3[1]=1, w5[1]=1. OK.
                # C21 = M2 + M4 -> w2[2]=1, w4[2]=1. OK.
                # C22 = M1 - M2 + M3 + M6 -> w1[3]=1, w2[3]=-1, w3[3]=1, w6[3]=1. OK.
                
                w_vals = [
                    {0: 1, 3: 1},       # M1
                    {2: 1, 3: -1},      # M2
                    {1: 1, 3: 1},       # M3
                    {0: 1, 2: 1},       # M4
                    {0: -1, 1: 1},      # M5
                    {3: 1},             # M6
                    {0: 1}              # M7
                ]
                
                for c_idx, coeff in w_vals[r].items():
                    w_vecs[r][c_idx] = coeff

            # Now we have u_vecs[0..6], v_vecs[0..6], w_vecs[0..6] of length 16 (no, length 4).
            # Wait, for 2x2 case, dimension is 4.
            # My u_vecs above were length 16?
            # No, indices were 0..3. So length 4.
            # But I initialized with 16. Let's fix that.
            for r in range(7):
                u_vecs[r] = u_vecs[r][:4]
                v_vecs[r] = v_vecs[r][:4]
                w_vecs[r] = w_vecs[r][:4]
            
            # Now for 4x4 case (rank 49):
            # Term rs (r, s in 0..6) has column c = r*7 + s.
            # U_col = u_vecs[r] \otimes u_vecs[s]
            # V_col = v_vecs[r] \otimes v_vecs[s]
            # W_col = w_vecs[r] \otimes w_vecs[s]
            
            # Kronecker product $a \otimes b$ where a,b are length 4:
            # Result length 16.
            # (a \otimes b)[i*4+j] = a[i] * b[j]
            
            for r in range(7):
                for s in range(7):
                    c = r * 7 + s
                    for i in range(4):
                        for j in range(4):
                            val_u = u_vecs[r][i] * u_vecs[s][j]
                            val_v = v_vecs[r][i] * v_vecs[s][j]
                            val_w = w_vecs[r][i] * w_vecs[s][j]
                            
                            idx = i * 4 + j
                            U[idx][c] = val_u
                            V[idx][c] = val_v
                            W[idx][c] = val_w

    return U, V, W

def construct():
    # Deadline guard (optional, though generation is fast)
    start_time = time.time()
    
    U, V, W = _build_strassen_decomp()
    
    # Convert to lists of lists (already are)
    # Ensure entries are integers or half-integers (they are integers +/- 1, 0)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple validation
    res = construct()
    U, V, W = res["U"], res["V"], res["W"]
    n, m, p = 4, 4, 4
    rank = len(U[0])
    print(f"Rank: {rank}")
    
    # Check a few random entries
    import random
    random.seed(42)
    valid = True
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        # T_{i,j,k} = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Wait, indices in U, V, W.
        # U is (n*m, rank) -> U[i*m + j, r]
        # V is (m*p, rank) -> V[j*p + k, r]
        # W is (p*n, rank) -> W[k*n + i, r]
        
        val = 0
        for r in range(rank):
            val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
            
        # In matrix multiplication tensor T_{i,j,k}, 
        # T_{i,j,k} = 1 if we are computing C_{i,k} which depends on A_{i,j} and B_{j,k}.
        # The tensor is usually defined such that T_{a,b,c} is the coefficient.
        # Here T_{i,j,k} = A_{i,j} B_{j,k}.
        # So for all i,j,k, the tensor component is 1?
        # Yes, the bilinear form is sum A_{ij} B_{jk} for each output C_{ik}.
        # The tensor for the multiplication map is such that:
        # (A \otimes B \otimes T) = sum C_{ik} A_{ij} B_{jk}
        # T_{i,j,k} = 1 for all i,j,k? No.
        # T is 3rd order.
        # Contract U (left) and V (mid) with A and B.
        # Sum_r U[i,j,r] V[j,k,r] should equal W[k,i,r] ??? No.
        
        # The decomposition is: T = sum_r u_r \otimes v_r \otimes w_r.
        # So T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        # For matrix multiplication, T[i,j,k] = 1 if we consider the term A_{i,j} B_{j,k}.
        # Wait, the standard tensor $M_{n,n,n}$ has entries:
        # $T_{i,j,k} = 1$ if ... actually $T$ is defined by $C = A B \iff C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
        # This implies $T_{i,j,k} = \delta_{j, \text{link}}$.
        # Specifically, the component $T_{i,j,k}$ (indices for A, B, C) is 1 if they match the pattern?
        # No, the indices are:
        # i: row of A (0..n-1)
        # j: col of A (0..n-1) AND row of B (0..n-1)
        # k: col of B (0..n-1) AND row of C (0..n-1)
        # So the index tuple is $(i, j, k)$.
        # The term in the sum is $A_{i,j} B_{j,k}$. This produces $C_{i,k}$.
        # So the tensor component at index $(i, j, k)$ corresponds to the monomial $A_{i,j} B_{j,k}$.
        # Thus, $T_{i,j,k} = 1$ for all $i, j, k \in \{0..n-1\}$.
        
        if val != 1:
            print(f"Failed at {i},{j},{k}: {val}")
            valid = False
    if valid:
        print("Validation Passed")
    else:
        print("Validation Failed")

# EVOLVE-BLOCK-START
def construct():
    return construct()
# EVOLVE-BLOCK-END

