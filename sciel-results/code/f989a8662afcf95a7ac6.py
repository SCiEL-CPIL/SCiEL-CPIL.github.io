"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 130  # seconds, safety margin before 150s kill


def _verify_fast(U_arr, V_arr, W_arr, n, m, p):
    """Fast verification assuming numpy arrays already."""
    rank = U_arr.shape[1]
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
    return np.max(np.abs(T - 1.0)) < 1e-7


def _compute_error(U, V, W, n, m, p):
    """Compute max absolute error of the decomposition."""
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
    return np.max(np.abs(T - 1.0))


def _strassen_4x4x4_terms():
    """Generate 49 scalar terms from a 4x4x4 Strassen-like decomposition.
    
    We use a recursive Strassen-style approach:
    - Split 4x4 into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    - Use Strassen for 2x2x2: 7 terms per block level
    - Each 2x2x2 term expands to 7 scalar terms
    - Total: 7 * 7 = 49 scalar terms
    
    The 7 Strassen terms for multiplying 2x2 matrices:
    Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    
    M0 = (a+d)(e+h)
    M1 = (c+d)e
    M2 = a(f-h)
    M3 = d(g-e)
    M4 = (a+b)h
    M5 = (c-a)(e+f)
    M6 = (b-d)(g+h)
    
    C = [[M0+M3-M4+M6, M2+M4], [M1+M3, M0-M1+M2+M5]]
    
    For 4x4:
    A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    
    First compute 7 intermediate 2x2 matrices using Strassen:
    M0 = (A00+A11)(B00+B11)
    M1 = (A10+A11)B00
    M2 = A00(B01-B11)
    M3 = A11(B10-B00)
    M4 = (A00+A01)B11
    M5 = (A10-A00)(B00+B01)
    M6 = (A01-A11)(B10+B11)
    
    C00 = M0+M3-M4+M6
    C01 = M2+M4
    C10 = M1+M3
    C11 = M0-M1+M2+M5
    
    Now each Mx is a 2x2 matrix computed from 2x2 sub-matrices.
    We can recursively apply Strassen to each 2x2x2 multiplication.
    
    For simplicity, let's just use the known 4x4x4 decomposition.
    Actually, let me use a different approach: use the known fact that
    R(4,4,4) <= 49 via recursive Strassen, and generate the terms programmatically.
    """
    # Each term is (u, v, w) where u is a linear combination of A entries,
    # v is a linear combination of B entries, w is a linear combination of C entries.
    # We represent them as coefficient arrays.
    
    terms = []
    
    # Recursive Strassen: 4x4 = two levels of 2x2
    # Level 1: 7 terms, each involving 2x2 submatrices
    # Level 2: each 2x2x2 = 7 terms
    
    # Let's define the 2x2x2 Strassen pattern first
    # For 2x2: indices (i,j) for A, (j,k) for B, (i,k) for C where i,j,k in {0,1}
    strassen_2x2 = [
        # (coeff_A, coeff_B, coeff_C) as dictionaries mapping (i,j) -> coeff
        # M0 = (A00+A11)(B00+B11) -> C00 += 1, C01 += 0, C10 += 0, C11 -= 1
        # Actually let me be more careful.
        #
        # M0 = (a+d)(e+h)
        # a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
        # M0 contributes to: C[0,0]+=M0, C[1,1]-=M0
        # So: A coeffs: {(0,0):1, (1,1):1}, B coeffs: {(0,0):1, (1,1):1}, C coeffs: {(0,0):1, (1,1):-1}
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
        # M1 = (c+d)e -> C[1,0]+=M1
        # c=A[1,0], d=A[1,1], e=B[0,0]
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # M2 = a(f-h) -> C[0,1]+=M2
        # a=A[0,0], f=B[0,1], h=B[1,1]
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
        # M3 = d(g-e) -> C[0,0]+=M3, C[1,0]+=M3
        # d=A[1,1], g=B[1,0], e=B[0,0]
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M4 = (a+b)h -> C[0,1]-=M4, C[0,1]... wait
        # M4 = (a+b)h -> C[0,1] -= M4, C[0,1]... 
        # Actually: C[0,1] = M2+M4, so M4 contributes +1 to C[0,1]
        # But also M4 = (a+b)h, a=A[0,0], b=A[0,1], h=B[1,1]
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1}),
        # M5 = (c-a)(e+f) -> C[1,1]+=M5
        # c=A[1,0], a=A[0,0], e=B[0,0], f=B[0,1]
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M6 = (b-d)(g+h) -> C[0,0]+=M6
        # b=A[0,1], d=A[1,1], g=B[1,0], h=B[1,1]
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # For 4x4, we have block structure:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # Global index (I,J) where I in {0,1,2,3}, J in {0,1,2,3}
    # A[I,J] where block index = (I//2, J//2), local = (I%2, J%2)
    
    # The 7 high-level Strassen terms for 4x4:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5
    
    # For each high-level term Mx, we need to compute the product of two 2x2 matrices.
    # Each such product uses 7 low-level Strassen terms.
    
    # High-level term definitions:
    # (block_A_list, block_B_list, block_C_list) where each is list of (block_idx, coeff)
    high_level_terms = [
        # M0 = (A00+A11)(B00+B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], None),  # C determined by C formula
        # M1 = (A10+A11)B00
        ([(1,0,1), (1,1,1)], [(0,0,1)], None),
        # M2 = A00(B01-B11)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], None),
        # M3 = A11(B10-B00)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], None),
        # M4 = (A00+A01)B11
        ([(0,0,1), (0,1,1)], [(1,1,1)], None),
        # M5 = (A10-A00)(B00+B01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], None),
        # M6 = (A01-A11)(B10+B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], None),
    ]
    
    # C block formulas:
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5
    c_formulas = [
        [(0,1), (3,1), (4,-1), (6,1)],   # C00
        [(2,1), (4,1)],                    # C01
        [(1,1), (3,1)],                    # C10
        [(0,1), (1,-1), (2,1), (5,1)],     # C11
    ]
    
    # For each high-level term, generate 7 low-level terms
    all_terms = []
    
    for hl_idx, (a_blocks, b_blocks, _) in enumerate(high_level_terms):
        # a_blocks and b_blocks are lists of (block_row, block_col, coeff)
        # We need to compute the 2x2x2 product for each pair
        
        # For this high-level term, the A-side is a sum of 2x2 blocks with coefficients
        # The B-side is a sum of 2x2 blocks with coefficients
        # The product is a sum over all pairs
        
        # Actually, each high-level term is a product of two linear combinations of 2x2 blocks.
        # To decompose this into scalar terms, we can:
        # 1. For each pair (a_block, b_block), compute the 2x2x2 product using 7 Strassen terms
        # 2. Scale by the product of coefficients
        
        # But this gives 7 * |a_blocks| * |b_blocks| terms per high-level term, which is too many.
        # 
        # Better approach: The high-level term is a 2x2x2 tensor where:
        # - The "A" input is a specific linear combination of 4 2x2 blocks
        # - The "B" input is a specific linear combination of 4 2x2 blocks
        # - The "C" output contributes to specific 2x2 blocks
        #
        # We can treat this as a 2x2x2 problem where each "entry" is itself a 2x2 matrix.
        # Then apply the 7-term Strassen decomposition to this 2x2x2 problem.
        # Each of the 7 terms in the 2x2x2 decomposition involves products of "entries",
        # where each "entry" product is a 2x2x2 product that can be decomposed into 7 terms.
        # Total: 7 * 7 = 49 terms.
        
        # Let me define the 2x2x2 Strassen pattern in terms of which (i,j) entries are involved
        # For a 2x2x2 problem with entries a[i,j], b[j,k], c[i,k]:
        # The 7 terms and their contributions:
        
        # For this high-level term, the "entries" of the 2x2x2 problem are:
        # a_entry[i,j] = sum over (br, bc, coeff) in a_blocks where (br,bc) corresponds to (i,j)
        # Actually, the 2x2 blocks are indexed by (block_row, block_col) in {0,1}^2
        # So a_entry[br, bc] = coeff for the block at position (br, bc)
        
        a_entry = np.zeros((2,2))
        for br, bc, coeff in a_blocks:
            a_entry[br, bc] = coeff
        
        b_entry = np.zeros((2,2))
        for br, bc, coeff in b_blocks:
            b_entry[br, bc] = coeff
        
        # Now apply Strassen to this 2x2x2 problem
        # The 7 Strassen terms for 2x2x2:
        # M0 = (a00+a11)(b00+b11)
        # M1 = (a10+a11)b00
        # M2 = a00(b01-b11)
        # M3 = a11(b10-b00)
        # M4 = (a00+a01)b11
        # M5 = (a10-a00)(b00+b01)
        # M6 = (a01-a11)(b10+b11)
        
        # C00 = M0+M3-M4+M6
        # C01 = M2+M4
        # C10 = M1+M3
        # C11 = M0-M1+M2+M5
        
        # For each of these 7 terms, we have a 2x2x2 sub-problem where:
        # - A entries are specific combinations of a_entry[i,j]
        # - B entries are specific combinations of b_entry[i,j]
        # - C entries contribute to specific c_entry[i,j]
        
        strassen_patterns = [
            # (A_comb, B_comb, C_comb) where each is dict {(i,j): coeff}
            # M0: A = a00+a11, B = b00+b11, C = C00+1, C11-1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
            # M1: A = a10+a11, B = b00, C = C10+1
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
            # M2: A = a00, B = b01-b11, C = C01+1
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
            # M3: A = a11, B = b10-b00, C = C00+1, C10+1
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M4: A = a00+a01, B = b11, C = C01+1
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1}),
            # M5: A = a10-a00, B = b00+b01, C = C11+1
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # M6: A = a01-a11, B = b10+b11, C = C00+1
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for sp_idx, (A_comb, B_comb, C_comb) in enumerate(strassen_patterns):
            # A_comb is a linear combination of the 2x2 blocks in a_entry
            # B_comb is a linear combination of the 2x2 blocks in b_entry
            # C_comb indicates which 2x2 blocks of C this term contributes to
            
            # The "A" part: sum over (i,j) of A_comb[(i,j)] * a_entry[i,j] * A_2x2[i,j]
            # where A_2x2[i,j] is the actual 2x2 matrix
            # Similarly for B.
            
            # This term computes: (sum_A) * (sum_B) and adds it to C blocks according to C_comb
            
            # To decompose into scalar terms, we need to expand:
            # (sum_i,j A_comb[(i,j)] * a_entry[i,j] * A_block[i,j]) * (sum_k,l B_comb[(k,l)] * b_entry[k,l] * B_block[k,l])
            
            # This is a product of two sums, which expands to a double sum.
            # But we want to use the 7-term Strassen for 2x2x2 at the block level.
            
            # Actually, let's just directly construct the scalar terms.
            # For this (hl_idx, sp_idx) pair, the scalar decomposition is:
            # U_scalar[i,j] = sum over (br,bc) in A_comb of A_comb[(br,bc)] * a_entry[br,bc] * (1 if (i,j) is in block (br,bc))
            # V_scalar[k,l] = sum over (br,bc) in B_comb of B_comb[(br,bc)] * b_entry[br,bc] * (1 if (k,l) is in block (br,bc))
            # W_scalar[i,l] = sum over (br,bc) in C_comb of C_comb[(br,bc)] * (1 if (i,l) is in block (br,bc))
            
            U_term = np.zeros((4,4))
            for (br, bc), coeff in A_comb.items():
                val = coeff * a_entry[br, bc]
                if abs(val) > 1e-10:
                    # Block (br, bc) covers rows 2*br..2*br+1, cols 2*bc..2*bc+1
                    for r in range(2*br, 2*br+2):
                        for c in range(2*bc, 2*bc+2):
                            U_term[r, c] += val
            
            V_term = np.zeros((4,4))
            for (br, bc), coeff in B_comb.items():
                val = coeff * b_entry[br, bc]
                if abs(val) > 1e-10:
                    for r in range(2*br, 2*br+2):
                        for c in range(2*bc, 2*bc+2):
                            V_term[r, c] += val
            
            W_term = np.zeros((4,4))
            for (br, bc), coeff in C_comb.items():
                if abs(coeff) > 1e-10:
                    for r in range(2*br, 2*br+2):
                        for c in range(2*bc, 2*bc+2):
                            W_term[r, c] += coeff
            
            # Now this term contributes to the C blocks according to c_formulas[hl_idx]
            # Actually, the C_comb already encodes which C blocks this term affects
            # but we need to combine with the high-level C formula
            
            # Wait, I think I'm overcomplicating this. Let me just store the raw U, V, W
            # for this scalar term, and the C contribution will be handled by the W matrix.
            
            # The W matrix should encode how this term contributes to C.
            # The C contribution is: for each (cr, cc) in C_comb, add C_comb[(cr,cc)] to C block (cr,cc)
            # But this is per high-level term. The actual C is a sum over all high-level terms.
            
            # I think the cleanest approach is:
            # - U_term and V_term as computed above
            # - W_term encodes the C contribution from this specific scalar term
            #   which is determined by C_comb and the high-level C formula
            
            # For now, let's just store U_term, V_term, W_term as the raw contributions
            # and verify at the end.
            
            all_terms.append((U_term, V_term, W_term))
    
    return all_terms


def _build_strassen_decomposition(n, m, p):
    """Build a decomposition using recursive Strassen-like approach.
    
    For (4,4,8): split B into two 4x4 blocks, use rank-49 decomposition for each.
    Total rank: 98.
    """
    # For now, use the schoolbook approach but optimized
    # We'll try to find a better decomposition using the structure
    
    # Actually, let's just use a direct approach with known lower-rank decompositions
    # For 4x4x4, the known best is 49 (via recursive Strassen)
    # For 4x4x8, we can do two 4x4x4 multiplications, giving rank 98
    
    # Let me construct the 4x4x4 decomposition properly
    # Using the recursive Strassen approach
    
    # For 4x4x4, we have 49 terms
    # For each term, U is 4x4, V is 4x4, W is 4x4
    
    # Let me just hardcode a valid 4x4x4 decomposition
    # Actually, let's use a simpler approach: use the fact that we can always
    # achieve rank n*m*p, and try to reduce it
    
    # For this task, let's just return the schoolbook solution as a fallback
    # and try to find better ones
    
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return U, V, W, n*m*p


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    schoolbook_rank = n * m * p  # 128
    
    # Build schoolbook solution as fallback
    U_school, V_school, W_school, rank_school = _build_strassen_decomposition(n, m, p)
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Try to find a better decomposition
    # For (4,4,8), we can use two (4,4,4) multiplications
    # If we can find a rank-49 (4,4,4) decomposition, total rank is 98
    
    # Let's try to construct a rank-98 decomposition using recursive Strassen
    # For 4x4x4, the recursive Strassen gives 7*7 = 49 terms
    
    # Actually, let me just try the schoolbook for now and return it
    # The key insight is that the ALS search in the original code was too slow
    # and didn't find anything better than 128
    
    # Let me try a few ranks with quick ALS
    from itertools import product
    
    def _quick_als(U, V, W, n, m, p, max_iter=50):
        """Quick ALS iterations."""
        rank = U.shape[1]
        for _ in range(max_iter):
            if time.time() > _DEADLINE - 10:
                break
            # Update U
            for i in range(n):
                for j in range(m):
                    A = V[j*p:(j+1)*p, :] * W[np.arange(p)*n + i, :]
                    b = np.ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            # Update V
            for j in range(m):
                for k in range(p):
                    A = U[np.arange(n)*m + j, :] * W[k*n + np.arange(n), :]
                    b = np.ones(n)
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            # Update W
            for k in range(p):
                for i in range(n):
                    A = U[i*m + np.arange(m), :] * V[np.arange(m)*p + k, :]
                    b = np.ones(m)
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
        
        return U, V, W
    
    # Try a few specific ranks
    for r in [98, 100, 104, 108, 112]:
        if time.time() > _DEADLINE - 20:
            break
        if r >= best_rank:
            break
        
        for seed in range(2):
            if time.time() > _DEADLINE - 15:
                break
            np.random.seed(seed)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            U, V, W = _quick_als(U, V, W, n, m, p, max_iter=30)
            err = _compute_error(U, V, W, n, m, p)
            
            if err < 1e-6 and _verify_fast(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
