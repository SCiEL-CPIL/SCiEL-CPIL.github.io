"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Strategy: Partition into blocks and use Strassen for square sub-blocks
    # 4x5x6 = (4x4x4) + (4x1x6) + (4x4x2)
    # Core 4x4x4: Strassen rank 49
    # Strip A: 4x1x6 (last column of A, all of B): schoolbook rank 24
    # Strip B: 4x4x2 (last 2 columns of B for first 4 rows of A): schoolbook rank 32
    # Total: 49 + 24 + 32 = 105
    
    total_rank = 105
    U = [[0] * total_rank for _ in range(n * m)]  # 20 x 105
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 105
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 105
    
    col = 0
    
    # --- Part 1: Core 4x4x4 using Strassen (rank 49) ---
    # Indices: i in 0..3, j in 0..3, k in 0..3
    # Strassen 2x2 factors
    U2 = [
        [1, 0, 1, 0, 1, 0, 0],   # A11
        [0, 0, 0, 0, 0, 1, 0],   # A12
        [0, 1, 0, 1, 0, 0, 1],   # A21
        [1, 1, 0, 1, 1, 0, 0]    # A22
    ]
    V2 = [
        [1, 1, 0, 1, 0, 1, 0],   # B11
        [0, 0, 1, 0, 1, 0, 1],   # B12
        [0, 1, 1, 0, 0, 1, 1],   # B21
        [1, 0, -1, -1, 1, 0, 0]  # B22
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0]   # C22
    ]
    
    # Compose Strassen for 4x4x4
    # 4x4 matrix split into 2x2 blocks of 2x2 matrices
    # Block indices: ib = i//2, il = i%2, etc.
    # For each of the 7 Strassen multiplications, we do a 2x2x2 Strassen
    # So we have 7 * 7 = 49 rank-1 terms
    
    for s in range(7):  # 7 Strassen terms for the 4x4 block
        for t in range(7):  # 7 Strassen terms for the inner 2x2
            # Determine which 2x2 block of A, B, C this corresponds to
            # Strassen term s uses specific blocks of A, B, C
            # Inner term t uses specific 2x2 sub-blocks within those
            
            # A block selection based on Strassen term s
            # s=0: A11+A22, s=1: A21+A22, s=2: A11, s=3: A22, 
            # s=4: A11+A12, s=5: A21-A11, s=6: A12-A22
            # B block selection
            # s=0: B11+B22, s=1: B11, s=2: B21-B22, s=3: B12-B11,
            # s=4: B22, s=5: B11+B12, s=6: B21+B22
            # C block: s=0: C11, s=1: C21, s=2: C12, s=3: C22,
            # s=4: C11, s=5: C22, s=6: C11
            
            # For each (s,t), we need to determine the contributions
            # This is complex; let me build it systematically
            
            # The 4x4 Strassen: M_s = A_block_s * B_block_s
            # where A_block_s and B_block_s are 2x2 matrices
            # Then M_s is computed via 2x2 Strassen with 7 rank-1 terms
            
            # A_block_s as a 2x2 matrix in terms of A's 2x2 sub-blocks:
            # s=0: A11+A22 -> [[A11_11+A22_11, A11_12+A22_12],[A11_21+A22_21, A11_22+A22_22]]
            # etc.
            
            # This is getting very complex. Let me use a different approach.
            # I'll directly compute the rank-1 factors by enumerating.
            
            pass  # Will implement below
    
    # Let me take a cleaner approach: directly generate the 49 rank-1 factors
    # for 4x4x4 Strassen by composing two levels of Strassen.
    
    # Level 1: 4x4 split into four 2x2 blocks
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # Strassen for the block matrix:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5
    
    # Each M_s is a 2x2 matrix product, computed via Strassen (rank 7)
    # So total rank = 7*7 = 49
    
    # For each s in 0..6, M_s = A_s * B_s where A_s, B_s are 2x2
    # A_s and B_s are linear combinations of the 2x2 blocks of A and B
    
    # Let me define the linear combinations explicitly
    # A blocks: A00, A01, A10, A11 (each 2x2, indexed by (ib,il) where ib in {0,1}, il in {0,1})
    # Global index: i = ib*2 + il
    
    # For s=0: A_s = A00 + A11
    # For s=1: A_s = A10 + A11
    # For s=2: A_s = A00
    # For s=3: A_s = A11
    # For s=4: A_s = A00 + A01
    # For s=5: A_s = A10 - A00
    # For s=6: A_s = A01 - A11
    
    # Similarly for B_s.
    
    # Now, for each s, A_s is a 2x2 matrix. We compute A_s * B_s via Strassen.
    # The 2x2 Strassen has 7 rank-1 terms with factors U2, V2, W2.
    
    # So for each (s, t) where t in 0..6:
    # The rank-1 term contributes to:
    # U: For each (i,j) in the 2x2 block of A_s, the coefficient from U2[i_block][t] * (coefficient of A_{i_block} in A_s)
    # V: Similarly for B
    # W: For each (i,j) in the 2x2 block of C, the coefficient from W2[i_block][t] * (coefficient of C_{i_block} in the final C)
    
    # This is intricate. Let me just hardcode it by careful enumeration.
    
    # Actually, let me use a simpler but still valid approach:
    # Use schoolbook for the 4x4x4 core (rank 64) but optimize the strips.
    # 64 + 24 + 32 = 120. Same as before.
    
    # To beat 120, I MUST use Strassen for at least part of it.
    
    # Let me try a different partition:
    # 4x5x6: Split the middle dimension (5) into 4+1
    # Part 1: 4x4x6 (rank 96 schoolbook, or better)
    # Part 2: 4x1x6 (rank 24)
    # For 4x4x6: Split the last dimension into 4+2
    # Part 1a: 4x4x4 (Strassen rank 49)
    # Part 1b: 4x4x2 (schoolbook rank 32)
    # Total: 49 + 32 + 24 = 105
    
    # This is the same partition. I need to correctly implement the 4x4x4 Strassen.
    
    # Let me carefully implement it.
    
    # Global indices for 4x4x4:
    # i in 0..3 (row of A, row of C)
    # j in 0..3 (col of A, row of B)
    # k in 0..3 (col of B, col of C)
    
    # Block structure:
    # i_block = i // 2, i_local = i % 2
    # j_block = j // 2, j_local = j % 2
    # k_block = k // 2, k_local = k % 2
    
    # Strassen level 1 (block level):
    # A blocks: A00 (i_block=0, j_block=0), A01 (i_block=0, j_block=1), A10 (i_block=1, j_block=0), A11 (i_block=1, j_block=1)
    # B blocks: B00 (j_block=0, k_block=0), B01 (j_block=0, k_block=1), B10 (j_block=1, k_block=0), B11 (j_block=1, k_block=1)
    # C blocks: C00 (i_block=0, k_block=0), C01 (i_block=0, k_block=1), C10 (i_block=1, k_block=0), C11 (i_block=1, k_block=1)
    
    # Strassen terms:
    # M0 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M1 = (A10+A11)B00 -> contributes to C10, C11
    # M2 = A00(B10-B11) -> contributes to C01, C11
    # M3 = A11(B01-B00) -> contributes to C00, C10
    # M4 = (A00+A01)B11 -> contributes to C01, C00
    # M5 = (A10-A00)(B00+B01) -> contributes to C11
    # M6 = (A01-A11)(B10+B11) -> contributes to C00
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For each M_s, it's a 2x2 matrix product. We compute it via 2x2 Strassen.
    # The 2x2 Strassen for M_s = A_s * B_s:
    # N0 = (A_s[0,0]+A_s[1,1])(B_s[0,0]+B_s[1,1])
    # N1 = (A_s[1,0]+A_s[1,1])B_s[0,0]
    # N2 = A_s[0,0](B_s[1,0]-B_s[1,1])
    # N3 = A_s[1,1](B_s[0,1]-B_s[0,0])
    # N4 = (A_s[0,0]+A_s[0,1])B_s[1,1]
    # N5 = (A_s[1,0]-A_s[0,0])(B_s[0,0]+B_s[0,1])
    # N6 = (A_s[0,1]-A_s[1,1])(B_s[1,0]+B_s[1,1])
    # M_s[0,0] = N0 + N3 - N4 + N6
    # M_s[0,1] = N2 + N4
    # M_s[1,0] = N1 + N3
    # M_s[1,1] = N0 - N1 + N2 + N5
    
    # Each N_t is a scalar product of two linear combinations.
    # The rank-1 factor for (s, t) is:
    # U contribution: for each (i,j) in 4x4, the coefficient of A[i,j] in A_s's relevant element
    # V contribution: for each (j,k) in 4x4, the coefficient of B[j,k] in B_s's relevant element
    # W contribution: for each (i,k) in 4x4, the coefficient of C[i,k] in the final C
    
    # This is very detailed. Let me just build it programmatically.
    
    # Define the A_s and B_s as linear combinations of the 2x2 blocks
    # A_s[i_local, j_local] for i_local, j_local in {0,1}
    # Each A_s[i_local, j_local] is a linear combination of A[ib*2+il, jb*2+jl]
    
    # For s=0: A_s = A00 + A11
    #   A_s[0,0] = A[0,0] + A[2,2]
    #   A_s[0,1] = A[0,1] + A[2,3]
    #   A_s[1,0] = A[1,0] + A[3,2]
    #   A_s[1,1] = A[1,1] + A[3,3]
    
    # I'll create helper functions to build the factors.
    
    def build_strassen_4x4():
        """Build U, V, W for 4x4x4 Strassen, each 16x49."""
        U_core = [[0]*49 for _ in range(16)]
        V_core = [[0]*49 for _ in range(16)]
        W_core = [[0]*49 for _ in range(16)]
        
        # 2x2 Strassen factors
        U2 = [
            [1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 1],
            [1, 1, 0, 1, 1, 0, 0]
        ]
        V2 = [
            [1, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, -1, -1, 1, 0, 0]
        ]
        W2 = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        
        # For each s in 0..6, define A_s and B_s as 2x2 matrices of linear combos
        # A_s[il][jl] = sum of coeffs * A[ib*2+il, jb*2+jl]
        # B_s[jl][kl] = sum of coeffs * B[jb*2+jl, kb*2+kl]
        
        # s=0: A_s = A00+A11, B_s = B00+B11
        # s=1: A_s = A10+A11, B_s = B00
        # s=2: A_s = A00, B_s = B10-B11
        # s=3: A_s = A11, B_s = B01-B00
        # s=4: A_s = A00+A01, B_s = B11
        # s=5: A_s = A10-A00, B_s = B00+B01
        # s=6: A_s = A01-A11, B_s = B10+B11
        
        # Represent A_s as dict: (ib, jb) -> coeff for each (il, jl)
        # Actually, A_s[il, jl] is a linear combo of A[ib*2+il, jb*2+jl]
        # For a given s, A_s[il, jl] = sum over (ib,jb) of coeff_s[il,jl,ib,jb] * A[ib*2+il, jb*2+jl]
        
        # Let me define for each s, the coefficients for A and B
        # A_coeffs[s][il][jl][ib][jb] = coefficient
        # B_coeffs[s][jl][kl][jb][kb] = coefficient
        
        A_coeffs = [[[ [ [0,0] for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(7)]
        B_coeffs = [[[ [ [0,0] for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(7)]
        
        # s=0: A_s = A00+A11
        for il in range(2):
            for jl in range(2):
                A_coeffs[0][il][jl][0][0] = 1  # A00
                A_coeffs[0][il][jl][1][1] = 1  # A11
        # B_s = B00+B11
        for jl in range(2):
            for kl in range(2):
                B_coeffs[0][jl][kl][0][0] = 1  # B00
                B_coeffs[0][jl][kl][1][1] = 1  # B11
        
        # s=1: A_s = A10+A11
        for il in range(2):
            for jl in range(2):
                A_coeffs[1][il][jl][1][0] = 1  # A10
                A_coeffs[1][il][jl][1][1] = 1  # A11
        # B_s = B00
        for jl in range(2):
            for kl in range(2):
                B_coeffs[1][jl][kl][0][0] = 1  # B00
        
        # s=2: A_s = A00
        for il in range(2):
            for jl in range(2):
                A_coeffs[2][il][jl][0][0] = 1
        # B_s = B10-B11
        for jl in range(2):
            for kl in range(2):
                B_coeffs[2][jl][kl][1][0] = 1  # B10
                B_coeffs[2][jl][kl][1][1] = -1 # B11
        
        # s=3: A_s = A11
        for il in range(2):
            for jl in range(2):
                A_coeffs[3][il][jl][1][1] = 1
        # B_s = B01-B00
        for jl in range(2):
            for kl in range(2):
                B_coeffs[3][jl][kl][0][1] = 1  # B01
                B_coeffs[3][jl][kl][0][0] = -1 # B00
        
        # s=4: A_s = A00+A01
        for il in range(2):
            for jl in range(2):
                A_coeffs[4][il][jl][0][0] = 1  # A00
                A_coeffs[4][il][jl][0][1] = 1  # A01
        # B_s = B11
        for jl in range(2):
            for kl in range(2):
                B_coeffs[4][jl][kl][1][1] = 1
        
        # s=5: A_s = A10-A00
        for il in range(2):
            for jl in range(2):
                A_coeffs[5][il][jl][1][0] = 1  # A10
                A_coeffs[5][il][jl][0][0] = -1 # A00
        # B_s = B00+B01
        for jl in range(2):
            for kl in range(2):
                B_coeffs[5][jl][kl][0][0] = 1  # B00
                B_coeffs[5][jl][kl][0][1] = 1  # B01
        
        # s=6: A_s = A01-A11
        for il in range(2):
            for jl in range(2):
                A_coeffs[6][il][jl][0][1] = 1  # A01
                A_coeffs[6][il][jl][1][1] = -1 # A11
        # B_s = B10+B11
        for jl in range(2):
            for kl in range(2):
                B_coeffs[6][jl][kl][1][0] = 1  # B10
                B_coeffs[6][jl][kl][1][1] = 1  # B11
        
        # Now, for each s, t (t in 0..6):
        # The rank-1 index is s*7 + t
        # U contribution: For global (i,j) where i=ib*2+il, j=jb*2+jl:
        #   U[i*4+j][s*7+t] += A_coeffs[s][il][jl][ib][jb] * U2[il*2+jl... wait
        # 
        # Actually, the 2x2 Strassen U2 is indexed by the local (il, jl) of the 2x2 matrix.
        # U2[il*2+jl][t] gives the coefficient for the t-th Strassen term.
        # But U2 is 4x7, indexed by row (il*2+jl) and column t.
        
        # Wait, U2[i_local*2 + j_local][t] where i_local in {0,1}, j_local in {0,1}
        # So for the 2x2 matrix, the element at (i_local, j_local) has coefficient U2[i_local*2+j_local][t]
        
        # For global (i,j) = (ib*2+il, jb*2+jl):
        # The coefficient in A_s[il, jl] for A[ib*2+il, jb*2+jl] is A_coeffs[s][il][jl][ib][jb]
        # The Strassen 2x2 factor for element (il, jl) is U2[il*2+jl][t]
        # So U[ib*2+il, jb*2+jl][s*7+t] += A_coeffs[s][il][jl][ib][jb] * U2[il*2+jl][t]
        
        # Similarly for V:
        # For global (j,k) = (jb*2+jl, kb*2+kl):
        # V[jb*2+jl, kb*2+kl][s*7+t] += B_coeffs[s][jl][kl][jb][kb] * V2[jl*2+kl][t]
        
        # For W:
        # The C block contribution: C_block[ib_c, kb_c] gets contribution from M_s with coefficient
        # C00: M0(+1), M3(+1), M4(-1), M6(+1)
        # C01: M2(+1), M4(+1)
        # C10: M1(+1), M3(+1)
        # C11: M0(+1), M1(-1), M2(+1), M5(+1)
        
        # So for C block (ib_c, kb_c), the coefficient for M_s is:
        C_block_coeff = {
            (0,0): {0:1, 3:1, 4:-1, 6:1},
            (0,1): {2:1, 4:1},
            (1,0): {1:1, 3:1},
            (1,1): {0:1, 1:-1, 2:1, 5:1}
        }
        
        # For global (i,k) = (ib_c*2+il_c, kb_c*2+kl_c):
        # The M_s[il_c, kl_c] gets coefficient W2[il_c*2+kl_c][t] from the inner Strassen
        # And C_block[ib_c, kb_c] gets coefficient C_block_coeff[(ib_c,kb_c)].get(s, 0) from M_s
        # So W[ib_c*2+il_c, kb_c*2+kl_c][s*7+t] += C_block_coeff[(ib_c,kb_c)].get(s,0) * W2[il_c*2+kl_c][t]
        
        for s in range(7):
            for t in range(7):
                idx = s * 7 + t
                # U
                for ib in range(2):
                    for jb in range(2):
                        for il in range(2):
                            for jl in range(2):
                                coeff = A_coeffs[s][il][jl][ib][jb]
                                if coeff != 0:
                                    i_global = ib * 2 + il
                                    j_global = jb * 2 + jl
                                    U_core[i_global * 4 + j_global][idx] += coeff * U2[il * 2 + jl][t]
                # V
                for jb in range(2):
                    for kb in range(2):
                        for jl in range(2):
                            for kl in range(2):
                                coeff = B_coeffs[s][jl][kl][jb][kb]
                                if coeff != 0:
                                    j_global = jb * 2 + jl
                                    k_global = kb * 2 + kl
                                    V_core[j_global * 4 + k_global][idx] += coeff * V2[jl * 2 + kl][t]
                # W
                for ib_c in range(2):
                    for kb_c in range(2):
                        coeff_c = C_block_coeff.get((ib_c, kb_c), {}).get(s, 0)
                        if coeff_c != 0:
                            for il_c in range(2):
                                for kl_c in range(2):
                                    i_global = ib_c * 2 + il_c
                                    k_global = kb_c * 2 + kl_c
                                    W_core[i_global * 4 + k_global][idx] += coeff_c * W2[il_c * 2 + kl_c][t]
        
        return U_core, V_core, W_core
    
    U_core, V_core, W_core = build_strassen_4x4()
    
    # Copy core into main matrices
    for i in range(16):
        for c in range(49):
            U[i][col + c] = U_core[i][c]
            V[i][col + c] = V_core[i][c]
            W[i][col + c] = W_core[i][c]
    col += 49
    
    # --- Part 2: Strip 4x4x2 (j in 0..3, k in 4..5) ---
    # Schoolbook: rank 32
    for i in range(4):
        for j in range(4):
            for k in range(2):
                k_global = 4 + k
                U[i * 5 + j][col] = 1
                V[j * 6 + k_global][col] = 1
                W[k_global * 4 + i][col] = 1
                col += 1
    
    # --- Part 3: Strip 4x1x6 (j=4, all k) ---
    # Schoolbook: rank 24
    for i in range(4):
        for k in range(6):
            U[i * 5 + 4][col] = 1
            V[4 * 6 + k][col] = 1
            W[k * 4 + i][col] = 1
            col += 1
    
    assert col == total_rank, f"Column count {col} != total_rank {total_rank}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
