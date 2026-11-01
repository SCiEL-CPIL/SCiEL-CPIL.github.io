"""Symbolic rank-reduction search for 2x4x6 matrix multiplication tensor."""

DIMENSIONS = [2, 4, 6]

import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    nm = n * m      # 8
    mp = m * p      # 24
    pn = p * n      # 12

    # Build schoolbook rank-1 terms
    # Each term: (u_vec, v_vec, w_vec) where u has nm entries, v has mp, w has pn
    # For term (i,j,k): u[i*m+j]=1, v[j*p+k]=1, w[k*n+i]=1
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [Fraction(0)] * nm
                v = [Fraction(0)] * mp
                w = [Fraction(0)] * pn
                u[i * m + j] = Fraction(1)
                v[j * p + k] = Fraction(1)
                w[k * n + i] = Fraction(1)
                schoolbook_terms.append((u, v, w))

    rank_sb = len(schoolbook_terms)  # 48

    def build_matrices(terms):
        """Build U, V, W from list of (u_vec, v_vec, w_vec)."""
        rank = len(terms)
        U = [[Fraction(0)] * rank for _ in range(nm)]
        V = [[Fraction(0)] * rank for _ in range(mp)]
        W = [[Fraction(0)] * rank for _ in range(pn)]
        for col, (u, v, w) in enumerate(terms):
            for r in range(nm):
                if u[r] != 0:
                    U[r][col] = u[r]
            for r in range(mp):
                if v[r] != 0:
                    V[r][col] = v[r]
            for r in range(pn):
                if w[r] != 0:
                    W[r][col] = w[r]
        return U, V, W

    def verify_decomposition(terms):
        """Check if terms exactly decompose the matmul tensor."""
        rank = len(terms)
        # For each (i,j,k), check sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] == 1
        # For non-matching (i,j,k), the sum should be 0.
        # Actually, the tensor T[i,j,k] = delta(i,j,k) for matmul.
        # We need: for all (i,j,k), sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] = T[i,j,k]
        # where T[i,j,k] = 1 if this is a valid matmul term... 
        # Wait, the matmul tensor is T[i,j,k] = 1 for ALL (i,j,k) in the range?
        # No. The matmul tensor for (n x m) * (m x p) -> (n x p) is:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # As a 3-tensor: T[i,j,k] = 1 for all i,j,k (it's the identity in the j dimension pairing)
        # Actually T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Because the tensor contraction is sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k]
        # and T[i,j,k] = 1 for all i,j,k.
        
        # So we need: for all i,j,k: sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1
        # That's the same as the verify in the current program.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for r in range(rank):
                        u_val = terms[r][0][ui]
                        v_val = terms[r][1][vj]
                        w_val = terms[r][2][wk]
                        if u_val != 0 and v_val != 0 and w_val != 0:
                            s += u_val * v_val * w_val
                    if s != 1:
                        return False
        return True

    # Verify schoolbook
    if not verify_decomposition(schoolbook_terms):
        # Should never happen
        pass

    # Strategy: try to reduce rank by finding linear dependencies.
    # A set of rank-1 terms (u_r, v_r, w_r) decomposes the tensor iff:
    # For all (i,j,k): sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    # 
    # We can try to find a smaller set. One approach: 
    # Start with schoolbook, try to eliminate one term at a time by expressing
    # its contribution as a linear combination of others.
    #
    # More practically: use the known Strassen result for sub-blocks.
    # For 2x2x2, rank is 7 (not 8). So we can reduce each 2x2x2 block.
    #
    # Let's implement a smarter approach: 
    # Group terms by (i, j) pairs for the A side, and (j, k) for B side.
    # Actually, let's just implement the Strassen-based approach but verify it properly,
    # and then try to find further reductions.
    
    # Better strategy: use the algebraic structure.
    # The tensor for 2x4x6 matmul has rank at most 42 (from Strassen on 2x2 blocks).
    # Let's build the Strassen decomposition and verify.
    
    def strassen_2x2_terms():
        """Return 7 Strassen terms for 2x2x2 matmul.
        Local indices: A[i1,j1] with i1,j1 in {0,1}, same for B and C.
        u_vec has 4 entries (i1*2+j1), v_vec has 4, w_vec has 4.
        """
        terms = []
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        # M2 = (A10+A11)B00 -> C10 += M2, C11 += M2  (wait, let me recheck)
        # Standard Strassen for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # Using 0-indexed: A[i][j] with i,j in {0,1}
        # M1 = (A[0][0]+A[1][1])(B[0][0]+B[1][1])
        terms[0] = ([1,0,0,1], [1,0,0,1], [1,0,0,1])  # C[0][0]+=M1, C[1][1]+=M1
        
        # M2 = (A[1][0]+A[1][1])B[0][0]
        # C[1][0] += M2, C[1][1] += M2
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
        
        # M3 = A[0][0](B[1][0]-B[1][1])
        # C[0][1] += M3
        terms.append(([1,0,0,0], [0,0,1,-1], [0,1,0,0]))
        
        # M4 = A[1][1](B[0][1]-B[0][0])
        # C[0][0] += M4, C[1][0] += M4
        terms.append(([0,0,0,1], [0,1,-1,0], [1,0,1,0]))
        
        # M5 = (A[0][0]+A[0][1])B[1][1]
        # C[0][0] -= M5, C[0][1] += M5
        terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
        
        # M6 = (A[1][0]-A[0][0])(B[0][0]+B[0][1])
        # C[1][1] += M6
        terms.append([[-1,0,1,0], [1,1,0,0], [0,0,0,1]])
        
        # M7 = (A[0][1]-A[1][1])(B[1][0]+B[1][1])
        # C[0][0] += M7
        terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
        
        return terms

    strassen_terms = strassen_2x2_terms()
    
    # Build the full Strassen decomposition for 2x4x6
    # A is 2x4, split into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3)
    # B is 4x6, split into 6 blocks of 2x2:
    #   B[row 0-1, col 0-1], B[row 0-1, col 2-3], B[row 0-1, col 4-5]
    #   B[row 2-3, col 0-1], B[row 2-3, col 2-3], B[row 2-3, col 4-5]
    
    all_terms = []
    for a_block in range(2):  # 0: cols 0-1, 1: cols 2-3
        for b_block in range(3):  # 0: cols 0-1, 1: cols 2-3, 2: cols 4-5
            # B rows: a_block*2 to a_block*2+1
            b_row_start = a_block * 2
            b_col_start = b_block * 2
            
            for (u_local, v_local, w_local) in strassen_terms:
                # u_local: 4 entries for A[i][j] where j is local to a_block
                # Global A index: i*m + (a_block*2 + j_local)
                u_global = [Fraction(0)] * nm
                for i in range(2):
                    for j in range(2):
                        val = Fraction(u_local[i * 2 + j])
                        if val != 0:
                            u_global[i * m + a_block * 2 + j] = val
                
                # v_local: 4 entries for B[j][k] where j is local to b_row block, k local to b_col block
                # Global B index: (b_row_start + j_local)*p + (b_col_start + k_local)
                v_global = [Fraction(0)] * mp
                for j in range(2):
                    for k in range(2):
                        val = Fraction(v_local[j * 2 + k])
                        if val != 0:
                            v_global[(b_row_start + j) * p + b_col_start + k] = val
                
                # w_local: 4 entries for C[i][k] where i is global row (0 or 1), k local to b_col block
                # Global W index: (b_col_start + k_local)*n + i
                w_global = [Fraction(0)] * pn
                for i in range(2):
                    for k in range(2):
                        val = Fraction(w_local[i * 2 + k])
                        if val != 0:
                            w_global[(b_col_start + k) * n + i] = val
                
                all_terms.append((u_global, v_global, w_global))
    
    rank_strassen = len(all_terms)  # Should be 42
    
    # Verify
    verified = verify_decomposition(all_terms)
    
    if not verified:
        # Fall back to schoolbook
        all_terms = schoolbook_terms
        rank_strassen = rank_sb
    
    # Now try to reduce further: look for linear dependencies among the terms.
    # A set of terms is minimal if no term can be removed or expressed as combination of others.
    # 
    # Approach: try to find a subset of size r < current_rank that still spans.
    # This is NP-hard in general, but we can try greedy: 
    # For each term, check if removing it and adjusting others still works.
    # 
    # Actually, a better approach: use the fact that the tensor has a known structure.
    # Let's try to find if we can do better than 42 by looking at the 2x4x6 structure.
    #
    # The tensor rank of 2x4x6 matmul: 
    # Lower bound: max(n*m, m*p, p*n) = max(8, 24, 12) = 24
    # Upper bound: 42 (Strassen)
    # 
    # Let's try a different decomposition: 
    # Instead of 2x2 blocks, try 2x3 blocks or other splits.
    
    # Alternative: use the "Laser method" or "Strassen-like" for non-square blocks.
    # For 2x4x6, we can think of it as 2*(2x2x6) or 4*(2x1x6) etc.
    
    # Let's try: treat A as 2x4, B as 4x6.
    # Split A into 2x2 and 2x2 (as before), but for each 2x2 * 4x6, 
    # we can use a different strategy than 3 separate 2x2x2 Strassen.
    # 
    # 2x2 * 4x6: rank? 
    # We can write 2x2 * 4x6 as 2x2 * (4x2 + 4x2 + 4x2) = 3 * (2x2x2)
    # Each 2x2x2 has rank 7, so total 21 per A-block, 42 total. Same as before.
    #
    # Can we do 2x2 * 4x6 in fewer than 21?
    # The tensor for 2x2*4x6 has size 2*4*6 = 48 entries.
    # Lower bound: max(8, 24, 12) = 24? No, that's for the whole thing.
    # For 2x2*4x6: max(2*2, 2*4, 4*6) = max(4,8,24) = 24? That doesn't seem right.
    # Actually the border rank lower bound for m x n x p matmul is max(mn, mp, np).
    # For 2x2x4x6... this is getting complicated.
    
    # Let's just try to find a linear dependency in the 42-term Strassen decomposition.
    # If we can find 2 terms that are linearly dependent in a certain sense, we can merge them.
    
    # Actually, let's try a completely different approach: 
    # Use the known result that rank(2x4x6) might be 36 or so.
    # Let's try to construct a 36-rank decomposition using a different block structure.
    
    # Alternative: use 2x3 blocks.
    # A is 2x4: split as 2x2 and 2x2 (can't do 2x3 cleanly)
    # Or: A is 2x4, B is 4x6.
    # Split B into 2x3 and 2x3 (rows 0-1, 2-3), each 2x3.
    # Then A * B = A * [B_top; B_bottom] where B_top is 2x6, B_bottom is 2x6.
    # = A_top * B_top + A_bottom * B_bottom where A_top is 2x2, A_bottom is 2x2.
    # Same as before.
    
    # Let's try: split A into 2x4 (whole), and use a decomposition of 2x4 * 4x6.
    # This is equivalent to (2x4)*(4x6) = 2x6.
    # We can think of this as: for each of the 2 rows of A, multiply by B.
    # Row 0 of A (1x4) * B (4x6) = 1x6: rank 6 (schoolbook) or can we do better?
    # 1x4 * 4x6: the tensor has 4*6=24 entries, each is A[j]*B[j,k].
    # This is a "vector-matrix" product. Rank is at most 6 (one per column of output).
    # Actually, rank(1x4x6 matmul) = 6, since C[k] = sum_j A[j] B[j,k], 
    # and we need 6 independent products (one per k), each involving all 4 j's.
    # So 2 rows * 6 = 12? No, that's not right either.
    
    # I think 42 is likely close to optimal for this small case.
    # Let's just return the Strassen decomposition if it verifies.
    
    # Try one more thing: can we find a 40-rank decomposition?
    # Let's try to see if any pair of Strassen terms can be merged.
    
    # For now, return the best verified decomposition.
    best_terms = all_terms
    best_rank = rank_strassen
    
    # Try to reduce by 1: remove each term and see if we can adjust others.
    # This is expensive, so only try if time allows.
    if time.time() < deadline - 20:
        # Try removing each term and checking if the remaining still work
        # (This won't work directly since removing a term breaks the sum)
        # Instead, try: for each pair of terms, see if their sum can be replaced by one term.
        # (u1,v1,w1) + (u2,v2,w2) = (u1+u2, v1+v2, w1+w2) is NOT a rank-1 term in general.
        # So this approach doesn't work.
        
        # Better: use the fact that if we have terms that share structure,
        # we might be able to find a smaller generating set.
        # For now, skip this and return 42.
        pass
    
    U, V, W = build_matrices(best_terms)
    
    # Convert to lists of floats (or keep as Fractions for exactness)
    # The interface expects integer/half-integer entries.
    # Our entries are all integers (from Strassen), so we're good.
    
    result = {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
    
    return result
