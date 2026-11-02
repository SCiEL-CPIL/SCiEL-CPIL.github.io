import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 114:
    # 49 (Strassen 4x4) + 49 (Strassen 4x4) + 16 (Schoolbook 4x1)
    
    def strassen_2x2_terms():
        """
        Returns list of 7 tuples (U, V, W) for 2x2 matrix multiplication.
        U, V, W are length 4 lists representing flattened 2x2 matrices (row-major).
        """
        terms = []
        # M1 = (A11+A22)(B11+B22)
        # U: 1,0,0,1 ; V: 1,0,0,1 ; W: 1,0,0,1 (C11, C22)
        terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2 = (A21+A22)B11
        # U: 0,0,1,1 ; V: 1,0,0,0 ; W: 0,0,1,-1 (C21, C22)
        terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # M3 = A11(B12-B22)
        # U: 1,0,0,0 ; V: 0,1,0,-1 ; W: 0,1,0,1 (C12, C22)
        terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # M4 = A22(B21-B11)
        # U: 0,0,0,1 ; V: -1,0,1,0 ; W: 1,0,1,0 (C11, C21)
        terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        # M5 = (A11+A12)B22
        # U: 1,1,0,0 ; V: 0,0,0,1 ; W: -1,1,0,0 (C11, C12)
        terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # M6 = (A21-A11)(B11+B12)
        # U: -1,0,1,0 ; V: 1,1,0,0 ; W: 0,0,0,1 (C22)
        terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7 = (A12-A22)(B21+B22)
        # U: 0,1,-1,0 ; V: 0,0,1,1 ; W: 1,0,0,0 (C11)
        terms.append(([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0]))
        return terms

    # Strassen 4x4 decomposition
    # A, B are 4x4.
    # A = [[A11, A12], [A21, A22]] where Aij are 2x2.
    # B = [[B11, B12], [B21, B22]] where Bij are 2x2.
    # M1 = (A11+A22)(B11+B22) -> 2x2 matrix
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map the 4x4 indices (0..15) to the 2x2 block indices.
    # A flattened:
    # 0:1:2:3 (A11)
    # 4:5:6:7 (A12)
    # 8:9:10:11 (A21)
    # 12:13:14:15 (A22)
    
    # For each outer term k (1..7), we define:
    # U_k: 4x4 vector (16 entries)
    # V_k: 4x4 vector (16 entries)
    # W_k: 4x4 vector (16 entries) - BUT W_k is not a 4x4 vector in the same way.
    # M_k is a 2x2 matrix. It contributes to C11, C12, C21, C22.
    # The contribution of M_k to C is a 4x4 matrix where:
    # Block (1,1) is coeff_M1_11 * M_k
    # etc.
    
    # Actually, it's easier to think in terms of the 7 inner terms for each M_k.
    # M_k = sum_{l=1}^7 U_{kl} \otimes V_{kl} \otimes W_{kl} (2x2 terms)
    # U_{kl} is 2x2. V_{kl} is 2x2. W_{kl} is 2x2.
    
    # We need to expand U_{kl} to 4x4 by placing it in the appropriate block(s) of A.
    # And V_{kl} to 4x4 by placing it in the appropriate block(s) of B.
    # And W_{kl} to 4x4 by placing it in the appropriate block(s) of C, scaled by the coefficient of M_k in that block of C.

    # Helper to expand a 2x2 vector (length 4) to 4x4 vector (length 16) given a block location (0-3) and scale
    def expand_2x2_to_4x4(vec2x2, block_idx, scale=1.0):
        res = [0.0] * 16
        # block_idx 0: A11 (0-3), 1: A12 (4-7), 2: A21 (8-11), 3: A22 (12-15)
        offset = block_idx * 4
        for i in range(4):
            res[offset + i] = vec2x2[i] * scale
        return res

    # Definitions of U_k, V_k for the 7 outer terms in 4x4 Strassen
    # Each is a dict {block_idx: vec2x2} or list of (block_idx, vec2x2)
    # U_k: linear combo of A blocks
    # V_k: linear combo of B blocks
    
    # M1: (A11+A22)(B11+B22)
    U_1 = [(0, [1,0,0,0]), (3, [1,0,0,0])] # A11 + A22? No, the vec2x2 is the identity scaled?
    # Wait, U_k should be the 2x2 matrix X_k.
    # X_1 = A11 + A22.
    # So U_1 is not a single 2x2 vector. It is a sum.
    # In the decomposition, U_k is a 16-dim vector.
    # U_1 has 1s in A11 and A22 positions.
    # U_1 = expand(A11=1, A22=1)
    
    def make_U(blocks):
        """blocks: dict {block_idx: vec2x2}"""
        res = [0.0] * 16
        for b, v in blocks.items():
            res = [r + v[i] for r, i in zip(res, range(16))] # This is wrong.
        # Simpler:
        res = [0.0] * 16
        for b, v in blocks.items():
            offset = b * 4
            for i in range(4):
                res[offset + i] += v[i]
        return res

    def make_V(blocks):
        res = [0.0] * 16
        for b, v in blocks.items():
            offset = b * 4
            for i in range(4):
                res[offset + i] += v[i]
        return res

    # Identity 2x2 vec: [1,0,0,1]
    I = [1, 0, 0, 1]
    Z = [0, 0, 0, 0]

    # M1: X1 = A11 + A22. Y1 = B11 + B22.
    U1 = make_U({0: I, 3: I})
    V1 = make_V({0: I, 3: I})
    # M1 contributes to C11 (1), C22 (1)
    W1_blocks = {0: I, 3: I} # C11, C22

    # M2: X2 = A21 + A22. Y2 = B11.
    U2 = make_U({2: I, 3: I})
    V2 = make_V({0: I})
    # M2 contributes to C21 (1), C22 (-1)
    W2_blocks = {2: I, 3: [-1, 0, 0, -1]} # Scale -1 for C22?
    # Wait, W_k should be the 4x4 vector representing the contribution of M_k to C.
    # If M_k is a 2x2 matrix, and it goes to C21 with coeff 1, then the C21 block of C is M_k.
    # So W2 has I in block 2 (C21) and -I in block 3 (C22).
    
    # M3: X3 = A11. Y3 = B12 - B22.
    U3 = make_U({0: I})
    V3 = make_V({1: I, 3: [-1, 0, 0, -1]})
    # M3 contributes to C12 (1), C22 (1)
    W3_blocks = {1: I, 3: I}

    # M4: X4 = A22. Y4 = B21 - B11.
    U4 = make_U({3: I})
    V4 = make_V({2: I, 0: [-1, 0, 0, -1]})
    # M4 contributes to C11 (1), C21 (1)
    W4_blocks = {0: I, 2: I}

    # M5: X5 = A11 + A12. Y5 = B22.
    U5 = make_U({0: I, 1: I})
    V5 = make_V({3: I})
    # M5 contributes to C11 (-1), C12 (1)
    W5_blocks = {0: [-1, 0, 0, -1], 1: I}

    # M6: X6 = A21 - A11. Y6 = B11 + B12.
    U6 = make_U({2: I, 0: [-1, 0, 0, -1]})
    V6 = make_V({0: I, 1: I})
    # M6 contributes to C22 (1)
    W6_blocks = {3: I}

    # M7: X7 = A12 - A22. Y7 = B21 + B22.
    U7 = make_U({1: I, 3: [-1, 0, 0, -1]})
    V7 = make_V({2: I, 3: I})
    # M7 contributes to C11 (1)
    W7_blocks = {0: I}

    outer_terms = [
        (U1, V1, W1_blocks),
        (U2, V2, W2_blocks),
        (U3, V3, W3_blocks),
        (U4, V4, W4_blocks),
        (U5, V5, W5_blocks),
        (U6, V6, W6_blocks),
        (U7, V7, W7_blocks)
    ]

    inner_terms = strassen_2x2_terms()

    # Build 49 terms for 4x4
    terms_44 = []
    for U_k, V_k, W_k_blocks in outer_terms:
        for u_inner, v_inner, w_inner in inner_terms:
            # u_inner is 2x2 vec. It multiplies X_k.
            # X_k is 4x4 vec.
            # The product U_total = U_k \otimes u_inner?
            # No.
            # X_k is formed by A blocks.
            # u_inner is a linear combo of X_k's entries.
            # So U_total[i] = sum_j U_k[j] * (u_inner maps j to i)?
            # Actually, U_k is the 4x4 matrix X_k (flattened).
            # u_inner is a 2x2 matrix (flattened) that defines the linear form on X_k.
            # Wait, the inner term is $M_{kl} = X_{kl} Y_{kl}$.
            # $X_{kl}$ is a 2x2 matrix formed from $X_k$.
            # In the 2x2 Strassen decomposition, $U_l$ is the 2x2 matrix for $X$.
            # So $X_{kl}$ is obtained by applying the same linear transformation as $U_l$ to $X_k$.
            # This is equivalent to:
            # $U_{total} = U_k \circledast U_l$?
            # No, $U_k$ is 16-dim. $U_l$ is 4-dim.
            # $U_k$ has 4 blocks of 2x2.
            # $U_l$ has 4 entries.
            # The entry $U_{total}$ for a specific $A_{ij}$ is $U_k[A_{ij}] \times U_l[\text{position of } A_{ij} \text{ in } X_k \text{ block}]$?
            # No.
            # $X_k$ is a 2x2 matrix of 2x2 blocks? No, $X_k$ is a 2x2 matrix (in the outer loop).
            # But $X_k$ is formed from 4x4 A.
            # Example $X_1 = A_{11} + A_{22}$.
            # $A_{11}$ is a 2x2 matrix.
            # So $X_1$ is a 2x2 matrix.
            # The inner Strassen acts on $X_1$ (2x2) and $Y_1$ (2x2).
            # The inner $U_l$ is a 2x2 vector (length 4).
            # The inner $U_l$ defines a linear form on the 4 entries of $X_1$.
            # $X_1$ entries are sums of $A$ blocks.
            # So $U_{total}$ is the vector of coefficients for $A$'s 16 entries.
            # For each of the 4 positions in $X_1$ (say pos 0), the coefficient from $U_l$ is $U_l[0]$.
            # The value at pos 0 of $X_1$ is $A_{11}$ (2x2 matrix).
            # So $U_{total}$ gets $U_l[0] \times (\text{coeffs of } A_{11} \text{ in } U_k)$.
            
            # How to get coeffs of $A_{11}$ in $U_k$?
            # $U_k$ is 16-dim.
            # $A_{11}$ is indices 0-3.
            # So we just take the first 4 entries of $U_k$ and scale by $U_l[0]$.
            # Take next 4 entries of $U_k$ (A12) and scale by $U_l[1]$.
            # etc.
            
            U_total = [0.0] * 16
            V_total = [0.0] * 16
            W_total = [0.0] * 16
            
            for pos in range(4):
                scale_u = u_inner[pos]
                scale_v = v_inner[pos]
                scale_w = w_inner[pos]
                
                # U_k block pos
                offset = pos * 4
                for i in range(4):
                    U_total[offset + i] = U_k[offset + i] * scale_u
                    V_total[offset + i] = V_k[offset + i] * scale_v
                    
            # W_total:
            # W_k_blocks is a dict {c_block_idx: vec2x2}
            # The inner w_inner is a 2x2 vec (4 entries) representing the position in M_k.
            # M_k is a 2x2 matrix.
            # The contribution of M_k to C is:
            # For each C block c_block, the contribution is scale_w_inner[entry] * (W_k_blocks[c_block][entry])?
            # No.
            # $W_{total}$ is 16-dim (C is 4x4).
            # $M_k$ is 2x2.
            # $w_inner$ is the 2x2 matrix of coefficients for $M_k$'s entries?
            # No, $w_inner$ from $strassen\_2x2\_terms$ is the vector $W$ for the 2x2 output.
            # It has 4 entries.
            # These 4 entries correspond to the 4 positions in the 2x2 matrix $M_k$.
            # Let $w_inner = [w_{00}, w_{01}, w_{10}, w_{11}]$.
            # The contribution to $C$ is:
            # $C_{c\_block, r, c} += M_{k, r, c} \times (\text{coeff of } M_k \text{ in } C_{c\_block})$.
            # Wait, $W_{k\_blocks}[c\_block]$ is the 2x2 matrix of coefficients?
            # In my definition, $W_{k\_blocks}$ maps C block index to a 2x2 vec (the coefficient matrix).
            # Actually, for M1, $W_{1\_blocks} = \{0: I, 3: I\}$.
            # This means $C_{11} += 1 \cdot M_1$.
            # So the coefficient for $M_{1, 00}$ in $C_{11, 00}$ is $1 \cdot 1 = 1$.
            # The coefficient for $M_{1, 01}$ in $C_{11, 01}$ is $1 \cdot 1 = 1$?
            # No, $M_1$ is a 2x2 matrix.
            # $C_{11}$ is a 2x2 matrix.
            # $C_{11} = M_1$.
            # So $C_{11, 00} = M_{1, 00}$.
            # $C_{11, 01} = M_{1, 01}$.
            # So the coefficient is 1 for all entries of $M_1$.
            # My $W_{1\_blocks}[0] = I = [1,0,0,1]$.
            # This is wrong. $I$ is the identity matrix.
            # The coefficient matrix should be all 1s if $C_{11} = M_1$.
            # Let's fix $W_{k\_blocks}$.
            # If $C_{11} = M_1$, then the 2x2 coefficient matrix is $J = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$.
            # Vec: [1,1,1,1].
            
            # Let's redefine $W_{k\_blocks}$ with J (all ones) or 0 or -J.
            J = [1, 1, 1, 1]
            
            W1_blocks = {0: J, 3: J}
            W2_blocks = {2: J, 3: [-1, -1, -1, -1]}
            W3_blocks = {1: J, 3: J}
            W4_blocks = {0: J, 2: J}
            W5_blocks = {0: [-1, -1, -1, -1], 1: J}
            W6_blocks = {3: J}
            W7_blocks = {0: J}
            
            # Recompute outer terms W parts
            outer_terms[0] = (U1, V1, W1_blocks)
            outer_terms[1] = (U2, V2, W2_blocks)
            outer_terms[2] = (U3, V3, W3_blocks)
            outer_terms[3] = (U4, V4, W4_blocks)
            outer_terms[4] = (U5, V5, W5_blocks)
            outer_terms[5] = (U6, V6, W6_blocks)
            outer_terms[6] = (U7, V7, W7_blocks)
            
            # Now compute W_total
            for c_block, w_k_vec in W_k_blocks.items():
                offset_c = c_block * 4
                for i in range(4):
                    # w_k_vec[i] is the coefficient for M_k[entry i] in C[c_block, entry i]
                    # w_inner[i] is the coefficient for the inner term in M_k[entry i]
                    W_total[offset_c + i] += w_k_vec[i] * w_inner[i]

            terms_44.append((U_total, V_total, W_total))
            
    return terms_44

def construct():
    # Get the 49 terms for 4x4 multiplication
    terms_44 = construct_44()
    
    # Rank 114
    rank = 114
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Part 1: B1 (cols 0-3), C1 (rows 0-3)
    for t, (u, v, w) in enumerate(terms_44):
        for i in range(16):
            U[i][t] = u[i]
        for i in range(16):
            # v is 16-dim (4x4). Map to B (4x9).
            # B row j, col k (0-3). Index j*9 + k.
            # v index i -> j = i//4, k = i%4.
            j = i // 4
            k = i % 4
            V[j * 9 + k][t] = v[i]
        for i in range(16):
            # w is 16-dim (4x4). Map to C (9x4).
            # C row k (0-3), col i (0-3). Index k*4 + i.
            k = i // 4
            i_c = i % 4
            W[k * 4 + i_c][t] = w[i]
            
    # Part 2: B2 (cols 4-7), C2 (rows 4-7)
    offset_rank = 49
    for t, (u, v, w) in enumerate(terms_44):
        col = t + offset_rank
        for i in range(16):
            U[i][col] = u[i]
        for i in range(16):
            j = i // 4
            k = i % 4
            V[j * 9 + (k + 4)][col] = v[i]
        for i in range(16):
            k = i // 4
            i_c = i % 4
            W[(k + 4) * 4 + i_c][col] = w[i]
            
    # Part 3: B3 (col 8), C3 (row 8)
    offset_rank = 98
    for i in range(4):
        for j in range(4):
            col = offset_rank + i * 4 + j
            U[i * 4 + j][col] = 1.0
            V[j * 9 + 8][col] = 1.0
            W[8 * 4 + i][col] = 1.0
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
