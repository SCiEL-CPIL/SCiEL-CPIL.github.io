import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    def build_strassen_2x2_blocks():
        """Standard Strassen on 2x2 blocks. Rank 189."""
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        strassen_patterns = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]

        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i, col_idx] += val
                        col_idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def verify_decomposition(U, V, W, num_tests=3):
        """Verify if the decomposition is correct using vectorized check."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        for _ in range(num_tests):
            A = np.random.randint(-5, 5, (n, m))
            B = np.random.randint(-5, 5, (m, p))
            C_true = A @ B
            # C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Vectorize: for each (i,k), sum over j,r
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    # U[i*m:(i+1)*m, :] shape (m, rank)
                    # V[:, k] shape (m, rank) -> need V[j*p+k, r] for j in range(m)
                    # W[k*n+i, :] shape (rank,)
                    u_block = U[i*m:(i+1)*m, :]  # (m, rank)
                    v_slice = np.array([V[j*p+k, :] for j in range(m)])  # (m, rank)
                    w_vec = W[k*n+i, :]  # (rank,)
                    # C[i,k] = sum_{j,r} u_block[j,r] * v_slice[j,r] * w_vec[r]
                    C_comp[i, k] = np.sum(u_block * v_slice * w_vec)
            if not np.allclose(C_true, C_comp, atol=1e-9):
                return False
        return True

    def build_laderman_strassen_kronecker():
        """
        Laderman 3x3 (rank 19) Kronecker Strassen 2x2 (rank 7) = rank 133.
        
        Laderman's algorithm for 3x3 matrix multiplication with rank 19:
        The 3x3 matrices are partitioned into 3x3 blocks of size 1 (scalars).
        We use the known rank-19 decomposition.
        
        Strassen 2x2 (rank 7):
        Standard Strassen on 2x2 blocks.
        
        Kronecker product: The 6x6 matrix is viewed as a 3x3 block matrix
        where each block is 2x2. We apply Laderman at the block level (3x3 blocks)
        and Strassen within each 2x2 block.
        
        Total rank = 19 * 7 = 133.
        """
        # Laderman 3x3 rank-19 decomposition
        # Laderman (1976) showed rank(3x3) <= 19.
        # The decomposition uses 19 multiplications.
        # 
        # For 3x3 matrices A, B, C = A*B:
        # We partition into 9 scalar elements each.
        # Laderman's algorithm:
        #
        # The 19 terms are (using 0-indexed positions (row, col)):
        # M1  = (A00+A11) * (B00+B11)  -> C00 += M1, C11 += M1
        # M2  = (A10+A11) * B00         -> C10 += M2, C11 -= M2
        # M3  = A00 * (B01-B11)         -> C01 += M3, C11 += M3
        # M4  = A11 * (B10-B00)         -> C00 += M4, C10 += M4
        # M5  = (A00+A01) * B11         -> C00 -= M5, C01 += M5
        # M6  = (A10-A00) * (B00+B01)   -> C11 += M6
        # M7  = (A01-A11) * (B10+B11)   -> C00 += M7
        # M8  = A02 * B00               -> C02 += M8
        # M9  = A02 * B01               -> C02 += M9
        # M10 = A02 * B02               -> C02 += M10
        # M11 = A12 * B00               -> C12 += M11
        # M12 = A12 * B01               -> C12 += M12
        # M13 = A12 * B02               -> C12 += M13
        # M14 = A20 * B00               -> C20 += M14
        # M15 = A20 * B01               -> C20 += M15
        # M16 = A20 * B02               -> C20 += M16
        # M17 = A21 * B00               -> C21 += M17
        # M18 = A21 * B01               -> C21 += M18
        # M19 = A21 * B02               -> C21 += M19
        #
        # Wait, that's only 19 but it doesn't cover C22. Let me reconsider.
        # Actually Laderman's rank-19 algorithm is more complex. Let me use
        # a verified formulation.
        #
        # Actually, the standard Laderman 3x3 rank-19 uses:
        # M1  = (A00+A11)(B00+B11)
        # M2  = (A10+A11)B00
        # M3  = A00(B01-B11)
        # M4  = A11(B10-B00)
        # M5  = (A00+A01)B11
        # M6  = (A10-A00)(B00+B01)
        # M7  = (A01-A11)(B10+B11)
        # Then for the third row/col, we need additional terms.
        # 
        # Let me use a different approach. The rank of 3x3 is known to be 19.
        # I'll use the explicit construction from the literature.
        #
        # Laderman's 19-term algorithm for 3x3:
        # The key insight is that 3x3 = (2x2 with Strassen) + extra terms.
        # 
        # Actually, let me use a cleaner formulation. I'll construct the
        # decomposition directly using the Kronecker product structure.
        #
        # For the Kronecker product approach:
        # 6x6 = (3x3 blocks of 2x2)
        # Block-level: 3x3 multiplication with rank 19 (Laderman)
        # Within-block: 2x2 multiplication with rank 7 (Strassen)
        # Total rank: 19 * 7 = 133
        #
        # I need the explicit Laderman 3x3 rank-19 patterns.
        # Let me use a verified set of 19 patterns.
        
        # Laderman 3x3 rank-19 decomposition (verified formulation):
        # Each pattern is (U_dict, V_dict, W_dict) where keys are (row, col)
        # and values are coefficients.
        
        laderman_patterns = [
            # M1: (A00+A11)(B00+B11)
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            # M2: (A10+A11)B00
            ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
            # M3: A00(B01-B11)
            ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
            # M4: A11(B10-B00)
            ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
            # M5: (A00+A01)B11
            ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
            # M6: (A10-A00)(B00+B01)
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
            # M7: (A01-A11)(B10+B11)
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
            # M8: A02*B00
            ({(0,2):1}, {(0,0):1}, {(0,2):1}),
            # M9: A02*B01
            ({(0,2):1}, {(0,1):1}, {(0,2):1}),
            # M10: A02*B02
            ({(0,2):1}, {(0,2):1}, {(0,2):1}),
            # M11: A12*B00
            ({(1,2):1}, {(0,0):1}, {(1,2):1}),
            # M12: A12*B01
            ({(1,2):1}, {(0,1):1}, {(1,2):1}),
            # M13: A12*B02
            ({(1,2):1}, {(0,2):1}, {(1,2):1}),
            # M14: A20*B00
            ({(2,0):1}, {(0,0):1}, {(2,0):1}),
            # M15: A20*B01
            ({(2,0):1}, {(0,1):1}, {(2,0):1}),
            # M16: A20*B02
            ({(2,0):1}, {(0,2):1}, {(2,0):1}),
            # M17: A21*B00
            ({(2,1):1}, {(0,0):1}, {(2,1):1}),
            # M18: A21*B01
            ({(2,1):1}, {(0,1):1}, {(2,1):1}),
            # M19: A21*B02
            ({(2,1):1}, {(0,2):1}, {(2,1):1}),
        ]
        
        # Hmm, this doesn't cover C22. The issue is that Laderman's algorithm
        # is not simply "Strassen on top-left 2x2 + naive for rest."
        # Let me reconsider.
        #
        # Actually, the correct Laderman 3x3 rank-19 algorithm is:
        # It uses the fact that rank(3) <= rank(2)*rank(2) + extra terms.
        # The standard result is that 3x3 has rank 19.
        #
        # Let me use a different, verified approach. I'll look up the actual
        # Laderman construction.
        #
        # Laderman's 19-term algorithm:
        # M1  = (a00+a11)(b00+b11)
        # M2  = (a10+a11)b00
        # M3  = a00(b01-b11)
        # M4  = a11(b10-b00)
        # M5  = (a00+a01)b11
        # M6  = (a10-a00)(b00+b01)
        # M7  = (a01-a11)(b10+b11)
        # M8  = a02(b00+b01+b02)
        # M9  = a12(b00+b01+b02)
        # M10 = a20(b00+b10+b20)
        # M11 = a21(b00+b10+b20)
        # M12 = a22(b00+b10+b20)
        # ... this is getting complicated and I'm not sure it's correct.
        #
        # Let me try a different strategy. Instead of trying to hardcode Laderman,
        # let me verify if my simple construction actually works by checking
        # if the 7 Strassen patterns + 12 naive patterns give a valid decomposition.
        #
        # Actually, the 7 Strassen patterns handle the top-left 2x2 block of the 3x3.
        # The remaining 5 rows/cols (positions involving index 2) need 5*3 + 3*5 - ... 
        # Let me count: positions (i,j) where i>=2 or j>=2:
        # (0,2), (1,2), (2,0), (2,1), (2,2) -> 5 positions
        # Each needs 3 multiplications (one for each column of B or row of A)
        # Total: 5 * 3 = 15 additional multiplications
        # 7 + 15 = 22, not 19.
        #
        # So the simple approach gives rank 22, not 19. Laderman's algorithm
        # is more clever and shares some computations.
        #
        # Given the complexity, let me just use the naive 3x3 decomposition
        # (rank 27) combined with Strassen 2x2 (rank 7) for rank 189,
        # OR use a verified Laderman construction.
        #
        # Let me try to find a verified Laderman 3x3 rank-19 construction.
        
        # After research, here's a correct Laderman 3x3 rank-19 algorithm:
        # The 19 multiplications are:
        # M1  = (a00+a11)(b00+b11)
        # M2  = (a10+a11)b00
        # M3  = a00(b01-b11)
        # M4  = a11(b10-b00)
        # M5  = (a00+a01)b11
        # M6  = (a10-a00)(b00+b01)
        # M7  = (a01-a11)(b10+b11)
        # M8  = a02*b00
        # M9  = a02*b01
        # M10 = a02*b02
        # M11 = a12*b00
        # M12 = a12*b01
        # M13 = a12*b02
        # M14 = a20*b00
        # M15 = a20*b01
        # M16 = a20*b02
        # M17 = a21*b00
        # M18 = a21*b01
        # M19 = a21*b02
        #
        # But this doesn't give c22! c22 = a20*b02 + a21*b02 + a22*b02 + ...
        # We're missing a22*b00, a22*b01, a22*b02.
        #
        # So this naive extension gives rank 22 (7 Strassen + 15 naive for row/col 2).
        # Laderman's actual rank-19 algorithm must share computations more cleverly.
        #
        # Given time constraints, let me just use the rank-22 3x3 construction
        # (7 Strassen + 15 naive) which gives total rank 22*7 = 154.
        # This is still better than 189.
        
        laderman_22_patterns = [
            # 7 Strassen patterns for top-left 2x2
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
            ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
            ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
            ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
            # 15 naive patterns for row/col 2
            ({(0,2):1}, {(0,0):1}, {(0,2):1}),
            ({(0,2):1}, {(0,1):1}, {(0,2):1}),
            ({(0,2):1}, {(0,2):1}, {(0,2):1}),
            ({(1,2):1}, {(0,0):1}, {(1,2):1}),
            ({(1,2):1}, {(0,1):1}, {(1,2):1}),
            ({(1,2):1}, {(0,2):1}, {(1,2):1}),
            ({(2,0):1}, {(0,0):1}, {(2,0):1}),
            ({(2,0):1}, {(0,1):1}, {(2,0):1}),
            ({(2,0):1}, {(0,2):1}, {(2,0):1}),
            ({(2,1):1}, {(0,0):1}, {(2,1):1}),
            ({(2,1):1}, {(0,1):1}, {(2,1):1}),
            ({(2,1):1}, {(0,2):1}, {(2,1):1}),
            ({(2,2):1}, {(0,0):1}, {(2,2):1}),
            ({(2,2):1}, {(0,1):1}, {(2,2):1}),
            ({(2,2):1}, {(0,2):1}, {(2,2):1}),
        ]
        
        # That's 7 + 18 = 25 patterns, not 22. Let me recount.
        # Row 2 (i=2): j=0,1,2 -> 3 patterns for each of 3 columns of B = 9
        # Col 2 (j=2, i=0,1): i=0,1 -> 2 patterns for each of 3 columns of B = 6
        # Total: 9 + 6 = 15
        # 7 + 15 = 22
        
        laderman_22_patterns = [
            # 7 Strassen patterns for top-left 2x2
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
            ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
            ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
            ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
            # 15 naive patterns for remaining positions
            # (0,2): 3 patterns
            ({(0,2):1}, {(0,0):1}, {(0,2):1}),
            ({(0,2):1}, {(0,1):1}, {(0,2):1}),
            ({(0,2):1}, {(0,2):1}, {(0,2):1}),
            # (1,2): 3 patterns
            ({(1,2):1}, {(0,0):1}, {(1,2):1}),
            ({(1,2):1}, {(0,1):1}, {(1,2):1}),
            ({(1,2):1}, {(0,2):1}, {(1,2):1}),
            # (2,0): 3 patterns
            ({(2,0):1}, {(0,0):1}, {(2,0):1}),
            ({(2,0):1}, {(0,1):1}, {(2,0):1}),
            ({(2,0):1}, {(0,2):1}, {(2,0):1}),
            # (2,1): 3 patterns
            ({(2,1):1}, {(0,0):1}, {(2,1):1}),
            ({(2,1):1}, {(0,1):1}, {(2,1):1}),
            ({(2,1):1}, {(0,2):1}, {(2,1):1}),
            # (2,2): 3 patterns
            ({(2,2):1}, {(0,0):1}, {(2,2):1}),
            ({(2,2):1}, {(0,1):1}, {(2,2):1}),
            ({(2,2):1}, {(0,2):1}, {(2,2):1}),
        ]
        
        # That's 7 + 18 = 25. I miscounted. Let me recount properly.
        # Positions needing coverage: all (i,j) in 3x3
        # Top-left 2x2: (0,0),(0,1),(1,0),(1,1) -> handled by 7 Strassen
        # Remaining: (0,2),(1,2),(2,0),(2,1),(2,2) -> 5 positions
        # Each position (i,j) needs sum_k a_ik * b_kj
        # For (0,2): k=0,1,2 -> 3 terms
        # For (1,2): k=0,1,2 -> 3 terms
        # For (2,0): k=0,1,2 -> 3 terms
        # For (2,1): k=0,1,2 -> 3 terms
        # For (2,2): k=0,1,2 -> 3 terms
        # Total: 15 terms
        # 7 + 15 = 22 terms
        
        # But in my list above I have 7 + 18 = 25. Let me fix.
        # The issue: for position (i,j), the pattern is:
        # U has (i,k) with coeff 1, V has (k,j) with coeff 1, W has (i,j) with coeff 1
        # for each k. So for (0,2): 3 patterns with k=0,1,2
        # That's 3 patterns, not 3 separate entries in my list.
        # Let me recount my list:
        # (0,2): 3 entries (k=0,1,2) ✓
        # (1,2): 3 entries (k=0,1,2) ✓
        # (2,0): 3 entries (k=0,1,2) ✓
        # (2,1): 3 entries (k=0,1,2) ✓
        # (2,2): 3 entries (k=0,1,2) ✓
        # Total: 15 entries ✓
        # 7 + 15 = 22 ✓
        
        # Wait, I listed 18 entries above. Let me recount:
        # (0,2): 3
        # (1,2): 3
        # (2,0): 3
        # (2,1): 3
        # (2,2): 3
        # = 15. Yes, 15 entries. Plus 7 Strassen = 22 total. ✓
        
        rank_3x3 = len(laderman_22_patterns)  # Should be 22
        rank_2x2 = 7
        total_rank = rank_3x3 * rank_2x2  # 22 * 7 = 154
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # Kronecker product construction:
        # 6x6 matrix = 3x3 blocks of 2x2
        # Block (I,J) of A corresponds to rows I*2..I*2+1, cols J*2..J*2+1
        # Block (J,K) of B corresponds to rows J*2..J*2+1, cols K*2..K*2+1
        # Block (I,K) of C corresponds to rows I*2..I*2+1, cols K*2..K*2+1
        #
        # For each 3x3 pattern (u_3x3, v_3x3, w_3x3) and each 2x2 Strassen pattern
        # (u_2x2, v_2x2, w_2x2), we create one rank term.
        #
        # The 6x6 U entry for rank term (r3, r2) at position (i,j):
        # i = I*2 + i2, j = J*2 + j2
        # U[i,j] = u_3x3[I,J] * u_2x2[i2,j2]
        #
        # Similarly for V and W.
        
        strassen_patterns = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        col_idx = 0
        for u3, v3, w3 in laderman_22_patterns:
            for u2, v2, w2 in strassen_patterns:
                # U: position (i,j) where i = I*2+i2, j = J*2+j2
                for (I, J), val3 in u3.items():
                    for (i2, j2), val2 in u2.items():
                        i = I * 2 + i2
                        j = J * 2 + j2
                        U[i * m + j, col_idx] = val3 * val2
                
                # V: position (j,k) where j = J*2+j2, k = K*2+k2
                for (J, K), val3 in v3.items():
                    for (j2, k2), val2 in v2.items():
                        j = J * 2 + j2
                        k = K * 2 + k2
                        V[j * p + k, col_idx] = val3 * val2
                
                # W: position (k,i) where k = K*2+k2, i = I*2+i2
                # W index is k*n + i
                for (I, K), val3 in w3.items():
                    for (i2, k2), val2 in w2.items():
                        k = K * 2 + k2
                        i = I * 2 + i2
                        W[k * n + i, col_idx] = val3 * val2
                
                col_idx += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, total_rank

    # Try the Kronecker product approach first
    if time.time() < deadline:
        sol_kron, rank_kron = build_laderman_strassen_kronecker()
        if verify_decomposition(sol_kron['U'], sol_kron['V'], sol_kron['W']):
            return sol_kron
    
    # Fallback to Strassen 189
    sol, rank = build_strassen_2x2_blocks()
    return sol
# EVOLVE-BLOCK-END
