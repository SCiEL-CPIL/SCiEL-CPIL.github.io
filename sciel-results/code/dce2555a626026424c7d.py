"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy: Decompose into 4x4*4x4 (Strassen, rank 49) + 4x4*4x1 (schoolbook, rank 16).
    Total rank: 65.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Build Strassen decomposition for 4x4 * 4x4
    def strassen_4x4_4x4():
        """Return U, V, W for 4x4*4x4 Strassen decomposition.
        U: 16x49, V: 16x49, W: 16x49
        Convention: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
        """
        # Strassen at the 4x4 level: 7 multiplications of 2x2 matrices
        # Each 2x2*2x2 uses Strassen (7 scalar mults)
        # Total: 7*7 = 49 scalar mults
        
        # Strassen formulas for 2x2:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For 4x4*4x4, split into 2x2 blocks:
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]] where each is 2x2
        # C = A*B, C is 4x4
        # C11 = A11*B11 + A12*B21 (2x2)
        # C12 = A11*B12 + A12*B22 (2x2)
        # C21 = A21*B11 + A22*B21 (2x2)
        # C22 = A21*B12 + A22*B22 (2x2)
        
        # Using Strassen at the 4x4 level:
        # M1 = (A11+A22)*(B11+B22) -> 2x2*2x2, rank 7
        # M2 = (A21+A22)*B11 -> 2x2*2x2, rank 7
        # M3 = A11*(B12-B22) -> 2x2*2x2, rank 7
        # M4 = A22*(B21-B11) -> 2x2*2x2, rank 7
        # M5 = (A11+A12)*B22 -> 2x2*2x2, rank 7
        # M6 = (A21-A11)*(B11+B12) -> 2x2*2x2, rank 7
        # M7 = (A12-A22)*(B21+B22) -> 2x2*2x2, rank 7
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Now I need to express each 2x2*2x2 Strassen decomposition in terms of the full 4x4 indices.
        
        # Let me define helper to get Strassen U,V,W for a 2x2*2x2 subproblem
        # where A is at rows (a1,a2) cols (a3,a4) and B is at rows (b1,b2) cols (b3,b4)
        # and output C is at rows (c1,c2) cols (c3,c4)
        
        def strassen_2x2_sub(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
            """Return lists of (U_indices, V_indices, W_indices, coeff) for 2x2*2x2 Strassen.
            Each entry is a scalar multiplication term.
            U_indices: (i,j) for A[i,j]
            V_indices: (j,k) for B[j,k]  
            W_indices: (k,i) for C[k,i]
            """
            # Strassen for 2x2:
            # A = [[A11, A12], [A21, A22]]
            # B = [[B11, B12], [B21, B22]]
            # C = A*B
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # M1 = (A11+A22)*(B11+B22)
            # M2 = (A21+A22)*B11
            # M3 = A11*(B12-B22)
            # M4 = A22*(B21-B11)
            # M5 = (A11+A12)*B22
            # M6 = (A21-A11)*(B11+B12)
            # M7 = (A12-A22)*(B21+B22)
            
            # A11 = A[a_rows[0], a_cols[0]]
            # A12 = A[a_rows[0], a_cols[1]]
            # A21 = A[a_rows[1], a_cols[0]]
            # A22 = A[a_rows[1], a_cols[1]]
            
            # B11 = B[b_rows[0], b_cols[0]]
            # B12 = B[b_rows[0], b_cols[1]]
            # B21 = B[b_rows[1], b_cols[0]]
            # B22 = B[b_rows[1], b_cols[1]]
            
            # C11 = C[c_rows[0], c_cols[0]]
            # C12 = C[c_rows[0], c_cols[1]]
            # C21 = C[c_rows[1], c_cols[0]]
            # C22 = C[c_rows[1], c_cols[1]]
            
            # For each Strassen multiplication, I need to expand into scalar mults.
            # Each 2x2*2x2 Strassen has 7 scalar mults.
            
            # Let me define the 7 Strassen terms for a generic 2x2*2x2:
            # Term 1: (A11+A22)*(B11+B22)
            #   U: A11 coeff +1, A22 coeff +1
            #   V: B11 coeff +1, B22 coeff +1
            #   W: contributes to C11 (+1), C22 (+1)
            
            # Actually, I need to be more careful. Let me think of this differently.
            
            # The multiplication tensor for 2x2*2x2 has rank 7.
            # I'll construct the 7 terms explicitly.
            
            # For a 2x2*2x2 with A indices (i,j) where i,j in {0,1}, 
            # B indices (j,k) where j,k in {0,1}, C indices (k,i) where k,i in {0,1}:
            
            # Strassen terms (using standard formulation):
            # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
            # M2 = (A[1,0]+A[1,1])*B[0,0]
            # M3 = A[0,0]*(B[0,1]-B[1,1])
            # M4 = A[1,1]*(B[1,0]-B[0,0])
            # M5 = (A[0,0]+A[0,1])*B[1,1]
            # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
            # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
            
            # C[0,0] = M1 + M4 - M5 + M7
            # C[0,1] = M3 + M5
            # C[1,0] = M2 + M4
            # C[1,1] = M1 - M2 + M3 + M6
            
            # Each M is a product of a linear form in A and a linear form in B.
            # So each M contributes to the tensor decomposition as:
            # sum over terms in A_form * sum over terms in B_form * W_coeff
            
            # For M1: A_form = A[0,0] + A[1,1], B_form = B[0,0] + B[1,1]
            # This gives 4 scalar mults: A[0,0]*B[0,0], A[0,0]*B[1,1], A[1,1]*B[0,0], A[1,1]*B[1,1]
            # But in the tensor decomposition, M1 is a single "multiplication" that we represent
            # as a rank-1 term in the tensor. Wait, no - Strassen reduces the number of 
            # multiplications, but each multiplication is still a scalar mult.
            
            # Actually, I think I'm confusing myself. Let me restart with a clearer approach.
            
            # The key insight: Strassen's algorithm for 2x2*2x2 uses 7 multiplications.
            # Each multiplication is of the form (linear combo of A entries) * (linear combo of B entries).
            # In the tensor decomposition, each such multiplication corresponds to one rank-1 term.
            
            # So for 2x2*2x2, I have 7 rank-1 terms.
            # For 4x4*4x4 using Strassen at the block level, I have 7 block multiplications,
            # each of which is a 2x2*2x2 Strassen (7 scalar mults), so 49 total.
            
            # Let me construct this step by step.
            
            # First, let me define the 7 Strassen multiplications for a 2x2*2x2:
            # Each is (u_form, v_form, w_contrib)
            # where u_form is a dict mapping (i,j) -> coeff for A
            # v_form is a dict mapping (j,k) -> coeff for B
            # w_contrib is a dict mapping (k,i) -> coeff for C
            
            strassen_terms_2x2 = [
                # M1 = (A00+A11)*(B00+B11)
                ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
                # M2 = (A10+A11)*B00
                ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
                # M3 = A00*(B01-B11)
                ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}),
                # M4 = A11*(B10-B00)
                ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
                # M5 = (A00+A01)*B11
                ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
                # M6 = (A10-A00)*(B00+B01)
                ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
                # M7 = (A01-A11)*(B10+B11)
                ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),
            ]
            
            # Wait, I need to double-check the Strassen formulas. Let me verify:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # Let me verify C00:
            # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
            # M4 = A11(B10-B00) = A11B10 - A11B00
            # M5 = (A00+A01)B11 = A00B11 + A01B11
            # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
            # M1 + M4 - M5 + M7 = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
            # = A00B00 + A01B10
            # But C00 should be A00B00 + A01B10. Yes! Correct.
            
            # Let me verify C01:
            # M3 = A00(B01-B11) = A00B01 - A00B11
            # M5 = (A00+A01)B11 = A00B11 + A01B11
            # M3 + M5 = A00B01 - A00B11 + A00B11 + A01B11 = A00B01 + A01B11
            # C01 should be A00B01 + A01B11. Yes! Correct.
            
            # Let me verify C10:
            # M2 = (A10+A11)B00 = A10B00 + A11B00
            # M4 = A11(B10-B00) = A11B10 - A11B00
            # M2 + M4 = A10B00 + A11B00 + A11B10 - A11B00 = A10B00 + A11B10
            # C10 should be A10B00 + A11B10. Yes! Correct.
            
            # Let me verify C11:
            # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
            # M2 = (A10+A11)B00 = A10B00 + A11B00
            # M3 = A00(B01-B11) = A00B01 - A00B11
            # M6 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
            # M1 - M2 + M3 + M6 = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B01 - A00B11 + A10B00 + A10B01 - A00B00 - A00B01
            # = A11B11 + A10B01
            # C11 should be A10B01 + A11B11. Yes! Correct.
            
            # Great, the Strassen formulas are correct.
            
            # Now I need to map these to the actual indices in the 4x4 problem.
            # For a subproblem with A at rows a_rows, cols a_cols, B at rows b_rows, cols b_cols,
            # and C at rows c_rows, cols c_cols:
            
            # A[i,j] in subproblem -> A[a_rows[i], a_cols[j]] in full
            # B[j,k] in subproblem -> B[b_rows[j], b_cols[k]] in full
            # C[k,i] in subproblem -> C[c_rows[k], c_cols[i]] in full
            # Wait, the convention is W[k*n+i, r] for C[k,i]
            
            # Actually, let me reconsider. In the tensor decomposition:
            # U[i*m+j, r] corresponds to A[i,j]
            # V[j*p+k, r] corresponds to B[j,k]
            # W[k*n+i, r] corresponds to C[k,i]
            
            # So for a 2x2*2x2 subproblem:
            # A sub-index (ia, ja) -> full A index (a_rows[ia], a_cols[ja]) -> U row = a_rows[ia]*m + a_cols[ja]
            # B sub-index (jb, kb) -> full B index (b_rows[jb], b_cols[kb]) -> V row = b_rows[jb]*p + b_cols[kb]
            # C sub-index (kc, ic) -> full C index (c_rows[kc], c_cols[ic]) -> W row = c_rows[kc]*n + c_cols[ic]
            
            # Wait, I need to be careful. The subproblem 2x2*2x2 has:
            # A is 2x2: A[ia, ja] where ia, ja in {0,1}
            # B is 2x2: B[jb, kb] where jb, kb in {0,1}
            # C is 2x2: C[kc, ic] where kc, ic in {0,1}
            
            # The mapping from sub to full:
            # A[ia, ja] -> A_full[a_rows[ia], a_cols[ja]]
            # B[jb, kb] -> B_full[b_rows[jb], b_cols[kb]]
            # C[kc, ic] -> C_full[c_rows[kc], c_cols[ic]]
            
            # But wait, in the Strassen formulas, the indices are:
            # A[i,j] where i is row, j is col
            # B[j,k] where j is row, k is col
            # C[k,i] where k is row, i is col (this is the trace convention)
            
            # So in the subproblem:
            # A sub (ia, ja): ia is row, ja is col
            # B sub (jb, kb): jb is row, kb is col
            # C sub (kc, ic): kc is row, ic is col
            
            # The Strassen formulas use:
            # A[0,0], A[0,1], A[1,0], A[1,1] -> these are (ia, ja)
            # B[0,0], B[0,1], B[1,0], B[1,1] -> these are (jb, kb)
            # C[0,0], C[0,1], C[1,0], C[1,1] -> these are (kc, ic)
            
            # So the mapping is:
            # A sub (ia, ja) -> full A (a_rows[ia], a_cols[ja]) -> U row = a_rows[ia]*m + a_cols[ja]
            # B sub (jb, kb) -> full B (b_rows[jb], b_cols[kb]) -> V row = b_rows[jb]*p + b_cols[kb]
            # C sub (kc, ic) -> full C (c_rows[kc], c_cols[ic]) -> W row = c_rows[kc]*n + c_cols[ic]
            
            # Now let me build the full 4x4*4x4 Strassen decomposition.
            
            # At the 4x4 level, we have 7 block multiplications:
            # M1 = (A11+A22)*(B11+B22)
            # M2 = (A21+A22)*B11
            # M3 = A11*(B12-B22)
            # M4 = A22*(B21-B11)
            # M5 = (A11+A12)*B22
            # M6 = (A21-A11)*(B11+B12)
            # M7 = (A12-A22)*(B21+B22)
            
            # Where A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
            # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
            
            # C11 = M1 + M4 - M5 + M7 -> C[0:2, 0:2]
            # C12 = M3 + M5 -> C[0:2, 2:4]
            # C21 = M2 + M4 -> C[2:4, 0:2]
            # C22 = M1 - M2 + M3 + M6 -> C[2:4, 2:4]
            
            # Each M is a 2x2*2x2 multiplication, which has rank 7.
            # So total rank = 7 * 7 = 49.
            
            # Let me define the 7 block multiplications and their contributions to C blocks.
            
            block_mults = [
                # (name, A_form, B_form, C_contrib)
                # A_form: dict mapping block_name -> coeff
                # B_form: dict mapping block_name -> coeff
                # C_contrib: dict mapping block_name -> coeff
                ('M1', {'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1}, {'C11': 1, 'C22': 1}),
                ('M2', {'A21': 1, 'A22': 1}, {'B11': 1}, {'C21': 1}),
                ('M3', {'A11': 1}, {'B12': 1, 'B22': -1}, {'C12': 1}),
                ('M4', {'A22': 1}, {'B21': 1, 'B11': -1}, {'C11': 1, 'C21': 1}),
                ('M5', {'A11': 1, 'A12': 1}, {'B22': 1}, {'C11': -1, 'C12': 1}),
                ('M6', {'A21': 1, 'A11': -1}, {'B11': 1, 'B12': 1}, {'C22': 1}),
                ('M7', {'A12': 1, 'A22': -1}, {'B21': 1, 'B22': 1}, {'C11': 1, 'C22': -1}),
            ]
            
            # Block index mapping:
            # A11: rows 0-1, cols 0-1
            # A12: rows 0-1, cols 2-3
            # A21: rows 2-3, cols 0-1
            # A22: rows 2-3, cols 2-3
            # B11: rows 0-1, cols 0-1
            # B12: rows 0-1, cols 2-3
            # B21: rows 2-3, cols 0-1
            # B22: rows 2-3, cols 2-3
            # C11: rows 0-1, cols 0-1
            # C12: rows 0-1, cols 2-3
            # C21: rows 2-3, cols 0-1
            # C22: rows 2-3, cols 2-3
            
            block_A_rows = {'A11': [0,1], 'A12': [0,1], 'A21': [2,3], 'A22': [2,3]}
            block_A_cols = {'A11': [0,1], 'A12': [2,3], 'A21': [0,1], 'A22': [2,3]}
            block_B_rows = {'B11': [0,1], 'B12': [0,1], 'B21': [2,3], 'B22': [2,3]}
            block_B_cols = {'B11': [0,1], 'B12': [2,3], 'B21': [0,1], 'B22': [2,3]}
            block_C_rows = {'C11': [0,1], 'C12': [0,1], 'C21': [2,3], 'C22': [2,3]}
            block_C_cols = {'C11': [0,1], 'C12': [2,3], 'C21': [0,1], 'C22': [2,3]}
            
            # Now I need to expand each block multiplication into scalar mults.
            # For each block mult M, I have:
            # A_form: linear combo of A blocks
            # B_form: linear combo of B blocks
            # C_contrib: linear combo of C blocks
            
            # The product (A_form * B_form) contributes to C_contrib.
            # To express this as rank-1 terms in the tensor decomposition, I need to:
            # For each term in A_form and each term in B_form, create a rank-1 term
            # that multiplies those A and B entries and contributes to the C entries.
            
            # But wait - this would give |A_form| * |B_form| rank-1 terms per block mult,
            # which is not what Strassen does. Strassen reduces the number of multiplications,
            # but each multiplication is still of the form (linear combo) * (linear combo).
            
            # In the tensor decomposition, each such multiplication corresponds to ONE rank-1 term
            # where U has the coefficients of the A linear form, V has the coefficients of the B
            # linear form, and W has the coefficients of the C contribution.
            
            # So for each block mult M, I create one rank-1 term (or more precisely, the
            # Strassen 2x2*2x2 decomposition gives 7 rank-1 terms for the 2x2*2x2 subproblem).
            
            # Let me reconsider. The 4x4*4x4 Strassen uses 7 block multiplications, each of
            # which is a 2x2*2x2 multiplication. Each 2x2*2x2 multiplication has rank 7 (Strassen).
            # So total rank = 7 * 7 = 49.
            
            # To construct the 49 rank-1 terms, I need to:
            # For each of the 7 block mults, expand the 2x2*2x2 Strassen decomposition.
            
            # Let me do this by iterating over each block mult and each Strassen 2x2 term.
            
            U_terms = []  # list of (row, coeff)
            V_terms = []  # list of (row, coeff)
            W_terms = []  # list of (row, coeff)
            
            for block_name, A_form, B_form, C_contrib in block_mults:
                # For each block mult, I need to figure out the 2x2*2x2 subproblem.
                # A_form is a linear combo of A blocks, B_form is a linear combo of B blocks.
                # The product is a 2x2*2x2 multiplication where:
                # A_sub = sum of A blocks with coefficients
                # B_sub = sum of B blocks with coefficients
                # C_sub = sum of C blocks with coefficients
                
                # But actually, each A block is 2x2, so A_sub is a 2x2 matrix.
                # Similarly for B_sub and C_sub.
                
                # The 2x2*2x2 Strassen decomposition gives 7 rank-1 terms.
                # For each Strassen term, I have:
                # u_form: linear combo of A_sub entries
                # v_form: linear combo of B_sub entries
                # w_form: linear combo of C_sub entries
                
                # I need to expand these into full 4x4 indices.
                
                # Let me build A_sub, B_sub, C_sub as 2x2 matrices with symbolic entries.
                # A_sub[ia, ja] = sum over A blocks in A_form: coeff * A_block[ia, ja]
                # But A_block[ia, ja] = A_full[a_rows[ia], a_cols[ja]]
                
                # So A_sub[ia, ja] = sum over (block, coeff) in A_form: 
                #   coeff * A_full[block_A_rows[block][ia], block_A_cols[block][ja]]
                
                # Similarly for B_sub and C_sub.
                
                # Now for each Strassen 2x2 term (u_form, v_form, w_form):
                # u_form is a dict mapping (ia, ja) -> coeff
                # v_form is a dict mapping (jb, kb) -> coeff
                # w_form is a dict mapping (kc, ic) -> coeff
                
                # The rank-1 term has:
                # U: for each (ia, ja, coeff_u) in u_form, and each (block, coeff_A) in A_form:
                #   U row = block_A_rows[block][ia] * m + block_A_cols[block][ja]
                #   coeff = coeff_u * coeff_A
                # V: for each (jb, kb, coeff_v) in v_form, and each (block, coeff_B) in B_form:
                #   V row = block_B_rows[block][jb] * p + block_B_cols[block][kb]
                #   coeff = coeff_v * coeff_B
                # W: for each (kc, ic, coeff_w) in w_form, and each (block, coeff_C) in C_contrib:
                #   W row = block_C_rows[block][kc] * n + block_C_cols[block][ic]
                #   coeff = coeff_w * coeff_C
                
                # Let me implement this.
                
                for u_form, v_form, w_form in strassen_terms_2x2:
                    # Build U coefficients
                    U_coeff = {}
                    for (ia, ja), cu in u_form.items():
                        for block_A, ca in A_form.items():
                            row = block_A_rows[block_A][ia] * m + block_A_cols[block_A][ja]
                            U_coeff[row] = U_coeff.get(row, 0) + cu * ca
                    
                    V_coeff = {}
                    for (jb, kb), cv in v_form.items():
                        for block_B, cb in B_form.items():
                            row = block_B_rows[block_B][jb] * p + block_B_cols[block_B][kb]
                            V_coeff[row] = V_coeff.get(row, 0) + cv * cb
                    
                    W_coeff = {}
                    for (kc, ic), cw in w_form.items():
                        for block_C, cc in C_contrib.items():
                            row = block_C_rows[block_C][kc] * n + block_C_cols[block_C][ic]
                            W_coeff[row] = W_coeff.get(row, 0) + cw * cc
                    
                    U_terms.append(U_coeff)
                    V_terms.append(V_coeff)
                    W_terms.append(W_coeff)
            
            # Now build U, V, W matrices
            rank = len(U_terms)
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for r in range(rank):
                for row, coeff in U_terms[r].items():
                    U[row][r] = coeff
                for row, coeff in V_terms[r].items():
                    V[row][r] = coeff
                for row, coeff in W_terms[r].items():
                    W[row][r] = coeff
            
            return U, V, W, rank
        
        U_s, V_s, W_s, rank_s = strassen_4x4_4x4()
        return U_s, V_s, W_s, rank_s
    
    # Build the full 4x4*4x5 decomposition
    def build_full_decomposition():
        """Build 4x4*4x5 decomposition using Strassen for 4x4*4x4 + schoolbook for 4x4*4x1."""
        U_s, V_s, W_s, rank_s = strassen_4x4_4x4()
        
        # Add schoolbook terms for the last column of B (k=4)
        # For k=4, we need: sum_j A[i,j] * B[j,4] = C[i,4]
        # This is 4*4 = 16 terms (one for each i,j)
        
        rank_total = rank_s + n * m  # 49 + 16 = 65
        
        U = [row[:] for row in U_s]  # copy
        V = [row[:] for row in V_s]
        W = [row[:] for row in W_s]
        
        # Extend U, V, W with new columns
        for row in U:
            row.extend([0] * n * m)
        for row in V:
            row.extend([0] * n * m)
        for row in W:
            row.extend([0] * n * m)
        
        # Add schoolbook terms for k=4
        col = rank_s
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1  # B[j, 4]
                W[4 * n + i][col] = 1  # C[4, i] -> W row = 4*n + i
                col += 1
        
        return U, V, W, rank_total
    
    # Build and verify
    U, V, W, rank = build_full_decomposition()
    
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
# EVOLVE-BLOCK-END
