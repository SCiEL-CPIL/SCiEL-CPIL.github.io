"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen decomposition for 2x2x2: 7 rank-1 terms
    # For X(2x2) * Y(2x2) = Z(2x2)
    # X entries indexed (0,0),(0,1),(1,0),(1,1) -> flat index 0,1,2,3
    # Y entries indexed (0,0),(0,1),(1,0),(1,1) -> flat index 0,1,2,3
    # Z entries indexed (0,0),(0,1),(1,0),(1,1) -> flat index 0,1,2,3
    
    # The 7 Strassen products with their U (for X), V (for Y), W (for Z) coefficients
    # Each is a tuple (u_coeffs, v_coeffs, w_coeffs) where each is a dict mapping flat index to coeff
    strassen_terms = [
        # P1 = (x00+x11)(y00+y11)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1}),  # contributes to z00
        # P2 = (x10+x11)y00
        ({2:1, 3:1}, {0:1}, {2:1}),  # contributes to z10 (index 2 in Z)
        # P3 = x00(y10-y11)
        ({0:1}, {2:1, 3:-1}, {1:1}),  # contributes to z01 (index 1 in Z)
        # P4 = x11(y01-y00)
        ({3:1}, {1:1, 0:-1}, {0:1, 2:1}),  # contributes to z00 and z10
        # P5 = (x00+x01)y11
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),  # contributes to z00 and z01
        # P6 = (x10-x00)(y00+y01)
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),  # contributes to z11 (index 3 in Z)
        # P7 = (x01-x11)(y10+y11)
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:1}),  # contributes to z00 and z11
    ]
    
    # Wait, I need to be more careful. Let me redefine.
    # Z indices: z00=0, z01=1, z10=2, z11=3
    # 
    # P1 = (x00+x11)(y00+y11) -> z00 += 1, z11 += 1... no
    # Let me re-derive:
    # z00 = P1 + P4 - P5 + P7
    # z01 = P3 + P5
    # z10 = P2 + P4
    # z11 = P1 - P2 + P3 + P6
    
    # So each P contributes to specific Z entries with specific coefficients.
    
    # Let me redefine strassen_terms as list of (u_dict, v_dict, w_dict)
    # where w_dict maps Z flat index to coefficient
    
    strassen_terms = [
        # P1: u={0:1,3:1}, v={0:1,3:1}, w={0:1, 3:1}  # z00 and z11
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # P2: u={2:1,3:1}, v={0:1}, w={2:1}  # z10
        ({2:1, 3:1}, {0:1}, {2:1}),
        # P3: u={0:1}, v={2:1,3:-1}, w={1:1, 3:1}  # z01 and z11
        ({0:1}, {2:1, 3:-1}, {1:1, 3:1}),
        # P4: u={3:1}, v={1:1,0:-1}, w={0:1, 2:1}  # z00 and z10
        ({3:1}, {1:1, 0:-1}, {0:1, 2:1}),
        # P5: u={0:1,1:1}, v={3:1}, w={0:-1, 1:1}  # z00 and z01
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        # P6: u={2:1,0:-1}, v={0:1,1:1}, w={3:1}  # z11
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        # P7: u={1:1,3:-1}, v={2:1,3:1}, w={0:1, 3:1}  # z00 and z11
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1, 3:1}),
    ]
    
    # Now build the full decomposition for 4x4x6
    # A is 4x4, B is 4x6, C is 4x6
    # Block structure:
    # A: 2x2 blocks of 2x2, A[bi][bj] for bi,bj in {0,1}
    # B: 2x3 blocks of 2x2, B[bj][bk] for bj in {0,1}, bk in {0,1,2}
    # C: 2x3 blocks of 2x2, C[bi][bk] for bi in {0,1}, bk in {0,1,2}
    
    # Number of sub-products: for each (bi, bj, bk) in {0,1} x {0,1} x {0,1,2}
    # That's 2*2*3 = 12 sub-products
    # Each uses 7 Strassen terms, total rank = 12*7 = 84
    
    num_subproducts = 2 * 2 * 3  # bi, bj, bk
    rank = num_subproducts * 7
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84 (p*n = 6*4 = 24)
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for bk in range(3):
                # This sub-product computes A_block[bi][bj] * B_block[bj][bk]
                # and contributes to C_block[bi][bk]
                
                # A_block[bi][bj] has entries at A[2*bi + di][2*bj + dj] for di,dj in {0,1}
                # Flat index in A (n*m = 16): (2*bi+di)*m + (2*bj+dj) = (2*bi+di)*4 + (2*bj+dj)
                
                # B_block[bj][bk] has entries at B[2*bj + dj][2*bk + dk] for dj,dk in {0,1}
                # Flat index in B (m*p = 24): (2*bj+dj)*p + (2*bk+dk) = (2*bj+dj)*6 + (2*bk+dk)
                
                # C_block[bi][bk] has entries at C[2*bi + di][2*bk + dk] for di,dk in {0,1}
                # Flat index in C (p*n = 24): (2*bk+dk)*n + (2*bi+di) = (2*bk+dk)*4 + (2*bi+di)
                # Note: W[k*n+i, r] convention, so C[i][k] -> W[k*n+i]
                
                for s in range(7):
                    u_dict, v_dict, w_dict = strassen_terms[s]
                    
                    # Fill U column
                    for x_idx, coeff in u_dict.items():
                        # x_idx is flat index in 2x2 block: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                        di, dj = divmod(x_idx, 2)
                        a_row = 2 * bi + di
                        a_col = 2 * bj + dj
                        a_flat = a_row * m + a_col
                        U[a_flat][col] += coeff
                    
                    # Fill V column
                    for y_idx, coeff in v_dict.items():
                        # y_idx is flat index in 2x2 block: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                        dj, dk = divmod(y_idx, 2)
                        b_row = 2 * bj + dj
                        b_col = 2 * bk + dk
                        b_flat = b_row * p + b_col
                        V[b_flat][col] += coeff
                    
                    # Fill W column
                    for z_idx, coeff in w_dict.items():
                        # z_idx is flat index in 2x2 block: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                        di, dk = divmod(z_idx, 2)
                        c_row = 2 * bi + di
                        c_col = 2 * bk + dk
                        # W[k*n+i] where i=c_row, k=c_col
                        w_flat = c_col * n + c_row
                        W[w_flat][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
