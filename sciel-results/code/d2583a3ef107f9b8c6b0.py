"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6
    
    deadline = time.time() + 140
    
    # Block size: 2x2 blocks of 3x3 matrices
    bs = 3
    rank3 = 23  # BCLT rank for 3x3
    rank = 7 * rank3  # 161 total rank
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen definitions for 2x2 block multiplication
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_defs = [
        # M0: (A00 + A11)(B00 + B11)
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        # M1: (A10 + A11)B00
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        # M2: A00(B01 - B11)
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        # M3: A11(B10 - B00)
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        # M4: (A00 + A01)B11
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        # M5: (A10 - A00)(B00 + B01)
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        # M6: (A01 - A11)(B10 + B11)
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    
    # Reconstruction: C_block = sum gamma_s * M_s
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    recon_coeffs = [
        # C00 = M0 + M3 - M4 + M6
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        # C01 = M2 + M4
        [(2, 1.0), (4, 1.0)],
        # C10 = M1 + M3
        [(1, 1.0), (3, 1.0)],
        # C11 = M0 - M1 + M2 + M5
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]
    
    # BCLT decomposition for 3x3 matrix multiplication (rank 23)
    # Each term t has:
    #   U3[t]: list of (global_idx_in_3x3, coeff) for A matrix
    #   V3[t]: list of (global_idx_in_3x3, coeff) for B matrix  
    #   W3[t]: list of (global_idx_in_3x3, coeff) for C matrix
    # Global index in 3x3: row*3 + col (0..8)
    
    # The BCLT algorithm uses 23 bilinear forms.
    # I'll construct them using a systematic approach based on the known BCLT structure.
    # 
    # For 3x3, the BCLT decomposition can be described as follows:
    # We partition the 3x3 matrices into overlapping 2x2 blocks and apply Strassen,
    # but with careful bookkeeping to get rank 23.
    #
    # Actually, let me use a verified BCLT construction.
    # The standard BCLT uses the following 23 terms (using 0-indexed positions in 3x3):
    # Position (i,j) has index i*3+j
    
    bclt_U = []
    bclt_V = []
    bclt_W = []
    
    # BCLT 23 terms - these are the standard Bini-Capovani-Lotti-Turletti coefficients
    # Each entry is a dict mapping position index (0-8) to coefficient
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    bclt_U.append({0: 1})
    bclt_V.append({0: 1})
    bclt_W.append({0: 1})
    
    # Term 1: A[0,1]*B[1,0] -> C[0,0]
    bclt_U.append({1: 1})
    bclt_V.append({3: 1})
    bclt_W.append({0: 1})
    
    # Term 2: A[1,0]*B[0,1] -> C[0,1]
    bclt_U.append({3: 1})
    bclt_V.append({1: 1})
    bclt_W.append({1: 1})
    
    # Term 3: A[1,1]*B[1,1] -> C[1,1]
    bclt_U.append({4: 1})
    bclt_V.append({4: 1})
    bclt_W.append({4: 1})
    
    # Term 4: A[2,2]*B[2,2] -> C[2,2]
    bclt_U.append({8: 1})
    bclt_V.append({8: 1})
    bclt_W.append({8: 1})
    
    # Term 5: A[0,2]*B[2,0] -> C[0,2]
    bclt_U.append({2: 1})
    bclt_V.append({6: 1})
    bclt_W.append({2: 1})
    
    # Term 6: A[2,0]*B[0,2] -> C[2,0]
    bclt_U.append({6: 1})
    bclt_V.append({2: 1})
    bclt_W.append({6: 1})
    
    # Term 7: A[1,2]*B[2,1] -> C[1,2]
    bclt_U.append({5: 1})
    bclt_V.append({7: 1})
    bclt_W.append({5: 1})
    
    # Term 8: A[2,1]*B[1,2] -> C[2,1]
    bclt_U.append({7: 1})
    bclt_V.append({5: 1})
    bclt_W.append({7: 1})
    
    # Now the Strassen-like terms for overlapping 2x2 blocks
    # Block TL: rows 0-1, cols 0-1 (positions 0,1,3,4)
    # Block TR: rows 0-1, cols 1-2 (positions 1,2,4,5)
    # Block BL: rows 1-2, cols 0-1 (positions 3,4,6,7)
    # Block BR: rows 1-2, cols 1-2 (positions 4,5,7,8)
    
    # For each 2x2 block, Strassen gives 7 terms.
    # But we need to be careful about overlaps.
    # 
    # Let me use a different approach: use the known BCLT coefficients directly.
    # The BCLT algorithm is well-documented. Let me reconstruct it properly.
    
    # Actually, let me use a simpler verified approach:
    # Use 2x2 Strassen on overlapping blocks with proper combination.
    
    # I'll use the following strategy:
    # - 4 overlapping 2x2 blocks, each with 7 Strassen terms = 28 terms
    # - But some terms can be shared/combined to reduce to 23
    
    # For now, let me use a correct but possibly not minimal construction.
    # I'll use the schoolbook rank 9 for 3x3 as a fallback if BCLT is complex.
    
    # Let me use a verified BCLT construction from literature.
    # The BCLT 23-term algorithm for 3x3:
    
    # I'll define it using the standard BCLT paper coefficients.
    # This is complex, so let me use a simpler approach:
    # Use 2x2 Strassen (7 terms) on the top-left 2x2, and handle the rest separately.
    
    # Actually, for correctness and simplicity, let me use the schoolbook 9-term 
    # decomposition for 3x3, giving total rank 7*9=63... wait that's less than 189.
    
    # Hmm, 7*9 = 63 which is way below 189. But that can't be right because
    # the schoolbook rank of 6x6 is 216, and 7*9=63 < 216 which is impossible.
    
    # The issue: when we do Strassen at the top level, each "multiplication" M_s
    # is a 3x3 matrix multiplication, which has rank 9 (schoolbook).
    # So total rank = 7 * 9 = 63? No, that's wrong.
    
    # Wait, I think I'm confusing myself. Let me reconsider.
    
    # In the 2x2 Strassen decomposition of 6x6:
    # - We split into 2x2 blocks of 3x3 matrices
    # - Each Strassen term M_s involves multiplying two 3x3 matrices
    # - The rank of 3x3 matrix multiplication (schoolbook) is 9
    # - So total rank = 7 Strassen terms * 9 (rank of 3x3) = 63?
    
    # But the schoolbook rank of 6x6 is 36, not 216. I think I was wrong about 216.
    # 6x6 schoolbook = 6*6*6 = 216. Yes, 216 is correct.
    
    # So 7*9=63 is indeed less than 216, which means this IS an improvement.
    # But wait, is the rank of the tensor product the product of ranks?
    
    # For tensor decomposition, if we have a decomposition of size n x m x p with rank r,
    # and we decompose each "scalar" multiplication into a rank-r' decomposition,
    # the total rank is r * r'.
    
    # So 7 (Strassen) * 9 (3x3 schoolbook) = 63. That should work!
    
    # But the current program uses rank 189 = 27*7, where 27 is the schoolbook rank
    # of 3x3... wait, 3x3 schoolbook is 9, not 27.
    
    # I think there's confusion. Let me re-examine.
    # 3x3 schoolbook: 3*3*3 = 27 scalar multiplications. Yes! I was wrong.
    # 3x3 matrix multiplication requires 27 scalar multiplications in schoolbook.
    
    # So 7 * 27 = 189. That matches the current program.
    
    # And BCLT reduces 3x3 from rank 27 to rank 23.
    # So 7 * 23 = 161.
    
    # OK so I need the BCLT 23-term decomposition for 3x3.
    
    # Let me use a verified BCLT construction.
    
    # The BCLT algorithm for 3x3 matrix multiplication:
    # Reference: Bini, Capovani, Lotti, Turletti (1979)
    
    # I'll use the following approach: define the 23 bilinear forms explicitly.
    
    # After careful consideration, here's a correct BCLT construction:
    
    bclt_U.clear()
    bclt_V.clear()
    bclt_W.clear()
    
    # I'll use a systematic construction based on overlapping Strassen blocks.
    # For 3x3, we can use 2x2 Strassen on overlapping 2x2 submatrices.
    
    # Define 2x2 Strassen terms for a given 2x2 block starting at (r0, c0):
    # The 4 positions are: (r0,c0), (r0,c0+1), (r0+1,c0), (r0+1,c0+1)
    # Indices: p00=r0*3+c0, p01=r0*3+c0+1, p10=(r0+1)*3+c0, p11=(r0+1)*3+c0+1
    
    def strassen_2x2_terms(r0, c0, scale=1.0):
        """Return 7 Strassen terms for 2x2 block at (r0,c0) in 3x3 matrix."""
        p00 = r0 * 3 + c0
        p01 = r0 * 3 + c0 + 1
        p10 = (r0 + 1) * 3 + c0
        p11 = (r0 + 1) * 3 + c0 + 1
        
        terms = []
        # M0 = (A00+A11)(B00+B11)
        terms.append(
            ({p00: scale, p11: scale},
             {p00: scale, p11: scale},
             {p00: scale, p11: scale})
        )
        # M1 = (A10+A11)B00
        terms.append(
            ({p10: scale, p11: scale},
             {p00: scale},
             {p10: scale, p11: -scale})
        )
        # M2 = A00(B01-B11)
        terms.append(
            ({p00: scale},
             {p01: scale, p11: -scale},
             {p01: scale, p11: scale})
        )
        # M3 = A11(B10-B00)
        terms.append(
            ({p11: scale},
             {p10: scale, p00: -scale},
             {p00: scale, p10: scale})
        )
        # M4 = (A00+A01)B11
        terms.append(
            ({p00: scale, p01: scale},
             {p11: scale},
             {p00: -scale, p01: scale})
        )
        # M5 = (A10-A00)(B00+B01)
        terms.append(
            ({p10: scale, p00: -scale},
             {p00: scale, p01: scale},
             {p11: scale})
        )
        # M6 = (A01-A11)(B10+B11)
        terms.append(
            ({p01: scale, p11: -scale},
             {p10: scale, p11: scale},
             {p00: scale})
        )
        return terms
    
    # Use 2x2 Strassen on the top-left 2x2 block (rows 0-1, cols 0-1)
    for term in strassen_2x2_terms(0, 0):
        bclt_U.append(term[0])
        bclt_V.append(term[1])
        bclt_W.append(term[2])
    
    # Now handle the remaining elements: column 2 and row 2
    # Elements not covered by TL 2x2: (0,2), (1,2), (2,0), (2,1), (2,2)
    
    # For C[0,2] = sum_k A[0,k]*B[k,2]
    # = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    
    # For C[1,2] = sum_k A[1,k]*B[k,2]
    # = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # For C[2,0] = sum_k A[2,k]*B[k,0]
    # = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    
    # For C[2,1] = sum_k A[2,k]*B[k,1]
    # = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    
    # For C[2,2] = sum_k A[2,k]*B[k,2]
    # = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # That's 15 terms for the remaining 5 elements, but some can be combined.
    # Total would be 7 + 15 = 22, which is less than 23. But this isn't minimal.
    
    # For simplicity and correctness, let me just use schoolbook for the remaining terms.
    # This gives 7 + 15 = 22 terms, but I need exactly 23 for BCLT.
    
    # Actually, this approach is getting complicated. Let me use a simpler strategy:
    # Just use schoolbook rank 27 for 3x3, giving total rank 7*27=189 (same as current).
    # 
    # OR, let me try to get a valid lower rank.
    
    # Let me use the following: 2x2 Strassen on TL block (7 terms) + 
    # schoolbook for the cross terms (15 terms) = 22 terms total.
    # But I need to verify this is correct.
    
    # Actually, let me just use schoolbook 27 for now to ensure correctness,
    # and the rank will be 189 (same as current). The improvement will come from
    # finding a better 3x3 decomposition.
    
    # For now, use schoolbook 3x3:
    bclt_U.clear()
    bclt_V.clear()
    bclt_W.clear()
    
    for i in range(3):
        for k in range(3):
            for j in range(3):
                bclt_U.append({i * 3 + k: 1})
                bclt_V.append({k * 3 + j: 1})
                bclt_W.append({i * 3 + j: 1})
    
    rank3 = len(bclt_U)  # 27
    rank = 7 * rank3  # 189
    
    # Reinitialize with correct rank
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Now build the full decomposition
    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    # Update U
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        U[u_row][r_idx] += coeff_a
                    
                    # Update V
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row][r_idx] += coeff_b
                    
                    # Update W
                    for c_block_idx in range(4):
                        coeffs = recon_coeffs[c_block_idx]
                        gamma = 0.0
                        for (s_idx, g) in coeffs:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            w_row = j_glob * n + i_glob
                            W[w_row][r_idx] += gamma
                    
                    r_idx += 1
    
    assert r_idx == rank, f"Expected {rank}, got {r_idx}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
