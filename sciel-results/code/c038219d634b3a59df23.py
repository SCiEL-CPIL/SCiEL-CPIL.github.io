"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    Uses Strassen's algorithm for the 4x4 block to reduce rank.
    """
    n, m, p = DIMENSIONS
    
    # Generate 7 terms for 2x2x2 Strassen
    # Returns list of (u, v, w) where u, v, w are lists of 4 floats
    def get_strassen_2x2_terms():
        terms = []
        # 1
        terms.append((
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1]
        ))
        # 2
        terms.append((
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ))
        # 3
        terms.append((
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, 0]
        ))
        # 4
        terms.append((
            [0, 0, 0, 1],
            [0, 0, 1, -1],
            [1, 1, 0, 0]
        ))
        # 5
        terms.append((
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [-1, 0, 1, 0]
        ))
        # 6
        terms.append((
            [-1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 1]
        ))
        # 7
        terms.append((
            [0, 1, 0, -1],
            [0, 0, 1, 1],
            [1, 0, 0, 1]
        ))
        return terms

    # Generate 49 terms for 4x4x4 Strassen
    def get_strassen_4x4_terms():
        base_terms = get_strassen_2x2_terms()
        terms = []
        
        # The 7 high-level products for 4x4
        # Each product is a linear combination of A and B blocks, then multiplied
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map the 2x2 terms to the 4x4 context
        # For each of the 7 high-level products, we have a 2x2 matrix multiplication
        # We iterate through the 7 base terms and apply the linear combinations
        
        # Helper to combine base terms with coefficients
        # u, v, w are length 4.
        # The high-level combination applies coefficients to the blocks.
        
        # M1: A_block = A11+A22, B_block = B11+B22
        # u_new[i,j] = u_base[i,j] * (coeff for A11 or A22)
        # This is tricky because u_base is for 2x2.
        # Let's just construct the 49 terms explicitly by expanding the definitions.
        
        # Actually, it's easier to just sum the contributions.
        # But we need rank-1 terms.
        # Each of the 7 high-level M's is a 2x2 product.
        # So we take each of the 7 base terms, and for each high-level M,
        # we create a new term by scaling the u and v vectors according to the linear combination.
        
        # Example M1:
        # u_M1[i,j] = u_base[i,j] if (i,j) in A11 or A22?
        # No, A11 and A22 are 2x2 blocks.
        # In the 4x4 matrix, A11 is top-left 2x2, A22 is bottom-right 2x2.
        # u_base corresponds to a 2x2 matrix.
        # For M1, the input to the 2x2 product is A' = A11+A22.
        # So u_new should have u_base values in A11 positions and A22 positions?
        # Yes.
        
        def expand_term(u_base, v_base, w_base, a_coeffs, b_coeffs):
            # a_coeffs: dict mapping block index (0:11, 1:12, 2:21, 3:22) to coeff
            # b_coeffs: same
            u_new = [0.0] * 16
            v_new = [0.0] * 16
            w_new = [0.0] * 16
            
            # Map base 2x2 indices to 4x4 indices for each block
            # Block 0 (11): rows 0-1, cols 0-1
            # Block 1 (12): rows 0-1, cols 2-3
            # Block 2 (21): rows 2-3, cols 0-1
            # Block 3 (22): rows 2-3, cols 2-3
            
            block_offsets = [
                (0, 0), # 11
                (0, 2), # 12
                (2, 0), # 21
                (2, 2)  # 22
            ]
            
            for bi, (br, bc) in enumerate(block_offsets):
                coeff_a = a_coeffs.get(bi, 0)
                coeff_b = b_coeffs.get(bi, 0)
                
                if coeff_a != 0:
                    for i in range(2):
                        for j in range(2):
                            idx_base = i * 2 + j
                            idx_new = (br + i) * 4 + (bc + j)
                            u_new[idx_new] += u_base[idx_base] * coeff_a
                
                if coeff_b != 0:
                    for i in range(2):
                        for j in range(2):
                            idx_base = i * 2 + j
                            idx_new = (br + i) * 4 + (bc + j)
                            v_new[idx_new] += v_base[idx_base] * coeff_b
            
            # w is for C. The structure of C is the same for all M's?
            # No, w depends on which C block the M contributes to.
            # But in Strassen, the M's are intermediate scalars (or matrices in recursive case).
            # Here M is a 2x2 matrix.
            # Wait, in the 4x4 case, M1 is a 2x2 matrix product.
            # So w_base is the w for the 2x2 result.
            # The result of M1 is a 2x2 matrix.
            # This 2x2 matrix is then added to C blocks.
            # But the rank-1 term must correspond to the whole 4x4 C.
            # So w_new should be w_base expanded into the appropriate C block?
            # No, the w in the rank-1 term $u \otimes v \otimes w$ determines the contribution to C.
            # If we use the same w_base for all 7 high-level terms, we are saying that all M's contribute to the same C positions?
            # That's not right.
            #
            # Let's look at the structure again.
            # $C = \sum_{r=1}^{49} u_r \otimes v_r \otimes w_r$.
            # The 49 terms come from 7 groups of 7.
            # Group k corresponds to high-level product $M_k$.
            # For group k, the 7 terms are the Strassen terms for the 2x2 product that computes $M_k$.
            # The u and v for these terms are the linear combinations of A and B blocks.
            # The w for these terms must be such that the sum of the 7 terms in group k equals $M_k \cdot W_k$, where $W_k$ is the matrix that distributes $M_k$ to C.
            # Actually, $M_k$ is a 2x2 matrix.
            # And $C = \sum_k M_k \otimes W_k$? No.
            # $C_{block} = \sum_k c_{k, block} M_k$.
            # So the w vector for the terms in group k should be $W_k$ expanded?
            # No, $w_r$ is a vector of length 16 (for 4x4 C).
            # For group k, all 7 terms share the same w?
            # No, the 7 terms in group k have different w's corresponding to the 2x2 Strassen w's.
            # But the 2x2 Strassen w's are for a 2x2 C.
            # So we need to map the 2x2 w to the 4x4 w.
            # The 2x2 C of $M_k$ corresponds to which part of the 4x4 C?
            # It doesn't. $M_k$ is an intermediate matrix.
            # The final C is formed by summing $M_k$ with coefficients.
            # $C = M_1 W_1 + M_2 W_2 + ... + M_7 W_7$?
            # No, $C_{11} = M_1 + M_4 - M_5 + M_7$.
            # So $M_1$ contributes to $C_{11}$ (top-left 2x2 block of C).
            # So for group 1, the w vectors should be non-zero only in the top-left 2x2 block of C.
            # And they should follow the pattern of the 2x2 Strassen w's.
            
            # So, for each group k, we determine which C blocks it contributes to and with what sign.
            # Then we expand the base w's into those blocks.
            
            return u_new, v_new, w_new

        # This is getting complicated to generalize.
        # Let's just hardcode the 49 terms by iterating the 7 groups.
        
        # Group 1: M1 = (A11+A22)(B11+B22)
        # Contributes to C11 (+1) and C22 (+1)
        # Wait, $C_{11} = M_1 + ...$ and $C_{22} = M_1 - ...$
        # So M1 contributes +1 to C11 and +1 to C22.
        
        # Let's define the contribution of each M_k to the 4 C blocks (11, 12, 21, 22).
        # C11: M1(+1), M4(+1), M5(-1), M7(+1)
        # C12: M3(+1), M5(+1)
        # C21: M2(+1), M4(+1)
        # C22: M1(+1), M2(-1), M3(+1), M6(+1)
        
        contributions = {
            0: {0: 1, 3: 1}, # M1 -> C11, C22
            1: {2: 1, 3: -1}, # M2 -> C21, C22
            2: {1: 1, 3: 1},  # M3 -> C12, C22
            3: {0: 1, 2: 1},  # M4 -> C11, C21
            4: {0: -1, 1: 1}, # M5 -> C11, C12
            5: {3: 1},        # M6 -> C22
            6: {0: 1}         # M7 -> C11
        }
        
        # a_coeffs and b_coeffs for each M_k
        # M1: A11+A22, B11+B22
        # M2: A21+A22, B11
        # M3: A11, B12-B22
        # M4: A22, B21-B11
        # M5: A11+A12, B22
        # M6: A21-A11, B11+B12
        # M7: A12-A22, B21+B22
        
        ab_coeffs = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}),
            ({0: 1}, {1: 1, 3: -1}),
            ({3: 1}, {2: 1, 0: -1}),
            ({0: 1, 1: 1}, {3: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1})
        ]
        
        all_terms = []
        for k in range(7):
            a_c, b_c = ab_coeffs[k]
            w_contribs = contributions[k]
            
            for u_b, v_b, w_b in base_terms:
                u_new = [0.0] * 16
                v_new = [0.0] * 16
                w_new = [0.0] * 16
                
                # Expand u and v
                for bi in range(4):
                    br, bc = block_offsets[bi]
                    if bi in a_c:
                        for i in range(2):
                            for j in range(2):
                                idx_b = i * 2 + j
                                idx_n = (br + i) * 4 + (bc + j)
                                u_new[idx_n] += u_b[idx_b] * a_c[bi]
                    if bi in b_c:
                        for i in range(2):
                            for j in range(2):
                                idx_b = i * 2 + j
                                idx_n = (br + i) * 4 + (bc + j)
                                v_new[idx_n] += v_b[idx_b] * b_c[bi]
                
                # Expand w
                # w_b is for 2x2 C.
                # We need to place it in the C blocks specified by w_contribs.
                # The 2x2 C indices 0,1,2,3 map to local (0,0), (0,1), (1,0), (1,1).
                # In 4x4 C, block 0 is top-left, block 1 top-right, etc.
                # So for each C block ci in w_contribs:
                #   for each local index li in 0..3:
                #       w_new[global_index] += w_b[li] * w_contribs[ci]
                
                for ci, coeff in w_contribs.items():
                    cr, cc = block_offsets[ci]
                    for i in range(2):
                        for j in range(2):
                            idx_b = i * 2 + j
                            idx_n = (cr + i) * 4 + (cc + j)
                            w_new[idx_n] += w_b[idx_b] * coeff
                
                all_terms.append((u_new, v_new, w_new))
                
        return all_terms

    # Build the full 4x5x5 decomposition
    terms = []
    
    # Part 1: A11 * B11 (4x4 * 4x4)
    strassen_terms = get_strassen_4x4_terms()
    for u, v, w in strassen_terms:
        # u, v, w are lists of 16 floats.
        # Map to global indices.
        # u is A (4x5). A11 is cols 0-3.
        # v is B (5x5). B11 is rows 0-3, cols 0-3.
        # w is C (5x4). C11 is rows 0-3, cols 0-3.
        
        u_g = [0.0] * 20
        v_g = [0.0] * 25
        w_g = [0.0] * 20
        
        for idx in range(16):
            i = idx // 4
            j = idx % 4
            u_g[i * 5 + j] = u[idx]
            
            # v: B11 is top-left 4x4.
            # idx maps to (j, k) in 4x4.
            v_g[j * 5 + i] = v[idx] # Wait, v is B. B[j,k].
            # In get_strassen_4x4_terms, v was for B 4x4.
            # v[idx] where idx = j*4+k.
            # So v_g[j*5 + k] = v[idx].
            # Let's re-verify the index mapping in the helper.
            # In helper: idx_new = (br+i)*4 + (bc+j).
            # So idx is row*4 + col.
            # So v[idx] is B[row, col].
            # So v_g[row*5 + col] = v[idx].
            
            # w: C11 is top-left 4x4 of C (5x4).
            # w is C. C[k, i]? No, C is 4x5?
            # C is 4x5? No, A is 4x5, B is 5x5. C is 4x5.
            # Wait, the task says "multiplying a 4 by 5 matrix by a 5 by 5 matrix".
            # C is 4x5.
            # W is (p*n) x rank. p=5, n=4. W is 20x rank.
            # W index is k*n + i. k is col of C (0..4), i is row of C (0..3).
            # In helper, w is for C 4x4.
            # w[idx] where idx = k*4 + i?
            # In helper, w_new[idx_n] where idx_n = (cr+i)*4 + (cc+j).
            # Here i,j are local row, col of C block.
            # So idx is row*4 + col.
            # So w[idx] is C[row, col].
            # So w_g[col*4 + row] = w[idx].
            
            # Let's re-do the mapping carefully.
        
        # Redo mapping
        u_g = [0.0] * 20
        v_g = [0.0] * 25
        w_g = [0.0] * 20
        
        for idx in range(16):
            r = idx // 4
            c = idx % 4
            
            # u: A11[r, c] -> A[r, c]
            u_g[r * 5 + c] = u[idx]
            
            # v: B11[r, c] -> B[r, c]
            v_g[r * 5 + c] = v[idx]
            
            # w: C11[r, c] -> C[r, c]
            # W index is k*n + i = c*4 + r
            w_g[c * 4 + r] = w[idx]
        
        terms.append((u_g, v_g, w_g))
    
    # Part 2: A12 * B21 (4x1 * 1x4)
    # A12 is col 4 of A. B21 is row 4 of B.
    # C11 += A12 * B21.
    # C11[r, c] += A[r, 4] * B[4, c].
    for r in range(4):
        for c in range(4):
            u_g = [0.0] * 20
            v_g = [0.0] * 25
            w_g = [0.0] * 20
            u_g[r * 5 + 4] = 1.0
            v_g[4 * 5 + c] = 1.0
            w_g[c * 4 + r] = 1.0
            terms.append((u_g, v_g, w_g))
            
    # Part 3: A11 * B12 (4x4 * 4x1)
    # C12 (col 4 of C) += A11 * B12.
    # C[r, 4] += sum_j A[r, j] * B[j, 4].
    for r in range(4):
        for j in range(4):
            u_g = [0.0] * 20
            v_g = [0.0] * 25
            w_g = [0.0] * 20
            u_g[r * 5 + j] = 1.0
            v_g[j * 5 + 4] = 1.0
            w_g[4 * 4 + r] = 1.0
            terms.append((u_g, v_g, w_g))
            
    # Part 4: A12 * B22 (4x1 * 1x1)
    # C12 (col 4) += A12 * B22.
    # C[r, 4] += A[r, 4] * B[4, 4].
    for r in range(4):
        u_g = [0.0] * 20
        v_g = [0.0] * 25
        w_g = [0.0] * 20
        u_g[r * 5 + 4] = 1.0
        v_g[4 * 5 + 4] = 1.0
        w_g[4 * 4 + r] = 1.0
        terms.append((u_g, v_g, w_g))
        
    # Convert to matrices
    rank = len(terms)
    U = np.zeros((20, rank))
    V = np.zeros((25, rank))
    W = np.zeros((20, rank))
    
    for r_idx, (u_g, v_g, w_g) in enumerate(terms):
        U[:, r_idx] = u_g
        V[:, r_idx] = v_g
        W[:, r_idx] = w_g
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
