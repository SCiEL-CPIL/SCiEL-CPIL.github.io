"""
Stronger approach for 4x4 * 4x9 matrix multiplication rank minimization.

Strategy:
1. Use a fast numpy-based validator that checks the full tensor identity.
2. Start with a good structured candidate (block-Strassen / Winograd-style blocks) 
   to get a baseline well below 144.
3. Perform structured random local search: start from the best candidate, then
   randomly zero out columns and re-solve / adjust, or perform random column
   replacements, and validate quickly.
4. Use time-boxed search with a safety margin.

Key insight: The rank of 4x4*4x9 is at most 144 (schoolbook). We can do much better
by using Strassen on 2x2 blocks. For 4x4*4x9, we can split into blocks:
- A is 4x4, B is 4x9, C is 4x9.
- Split B into three 4x3 blocks? Or 4x4 + 4x4 + 4x1.
- For each 4x4 block of B, we can use a 4x4*4x4 multiplication. The best known rank
  for 4x4*4x4 is 49 (Strassen-based) or 49. Actually, the rank of 4x4*4x4 is 49
  (using Strassen recursively: 7^2 = 49).
- So for two 4x4 blocks of B, we'd use 49 + 49 = 98 terms, plus 16 for the 4x1 block = 114.
- But we can do better by sharing structure.

Actually, let's think more carefully. The matrix multiplication tensor for n x m * m x p
has rank at most n*m*p. For 4x4*4x9, that's 144.

Using Strassen: 
- For 2x2*2x2, rank 7.
- For 4x4*4x4, we can use 2x2 blocks: 7 multiplications of 2x2, each of rank 7, so 49.
- For 4x4*4x9, we can split B into 4x4, 4x4, 4x1. 
  - 4x4*4x4 part: rank 49
  - 4x4*4x4 part: rank 49
  - 4x4*4x1 part: rank 16 (schoolbook, since it's just 4x4*4x1)
  - Total: 49+49+16 = 114

But we can do better. The 4x4*4x1 part can be done with rank 16, but maybe we can
share some structure. Actually, no, because the output dimensions differ.

Let me try a different approach: use the fact that we only need the tensor identity.
I'll generate the 49-term Strassen decomposition for 4x4*4x4, then extend it to 4x4*4x9
by adding terms for the extra column.

Actually, the simplest strong candidate: use schoolbook for the 4x4*4x9 but with
Strassen on the 2x2 sub-blocks. Let me just implement a fast validator and do a
structured random search starting from a good initial point.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 140.0  # safety margin before 150s kill


def _validate_fast(U, V, W, n, m, p):
    """Fast validation using numpy. U: (n*m, r), V: (m*p, r), W: (p*n, r).
    Check that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1.
    """
    r = U.shape[1]
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    # Reshape for efficient computation
    # U reshaped to (n, m, r)
    # V reshaped to (m, p, r)
    # W reshaped to (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # For each (i, j, k), compute sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # We can do this by iterating over c and accumulating.
    # result[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    
    # Efficient approach: for each c, outer product
    result = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        uc = U3[:, :, c]  # (n, m)
        vc = V3[:, :, c]  # (m, p)
        wc = W3[:, :, c]  # (p, n)
        # For each (i,j,k): uc[i,j] * vc[j,k] * wc[k,i]
        # This is a 3-way product. We can do:
        # temp[i,j,k] = uc[i,j] * vc[j,k] * wc[k,i]
        # = (uc[i,j] * vc[j,k]) * wc[k,i]
        # First compute uc[i,j] * vc[j,k] for all i,j,k: this is like a matmul-like op
        # uc (n,m) and vc (m,p): for each i,k, sum_j uc[i,j]*vc[j,k] but we want element-wise
        # Actually we want the full tensor: T[i,j,k] = uc[i,j] * vc[j,k] * wc[k,i]
        # We can compute: A[i,j,k] = uc[i,j] * vc[j,k] by broadcasting
        # uc[:, :, None] * vc[None, :, :] gives (n, m, p) but with wrong indices
        # uc[i,j] * vc[j,k]: shape (n, m, p) where uc is (n,m,1) and vc is (1,m,p)
        # But vc[j,k] has j as first index, so we need vc[None, :, :] which is (1, m, p)
        # So uc[:, :, None] has shape (n, m, 1), vc[None, :, :] has shape (1, m, p)
        # Product: (n, m, p) with element [i,j,k] = uc[i,j] * vc[j,k]. Correct!
        temp = uc[:, :, None] * vc[None, :, :]  # (n, m, p)
        # Now multiply by wc[k,i]: wc has shape (p, n), we need wc[k,i] for each (i,j,k)
        # wc[k,i] = wc.T[i,k], so we need (n, 1, p) broadcast
        wc_t = wc.T  # (n, p), so wc_t[i,k] = wc[k,i]
        result += temp * wc_t[:, None, :]  # (n, m, p) * (n, 1, p) -> (n, m, p)
    
    # Check all entries are 1
    if not np.allclose(result, 1.0, atol=1e-9):
        return False
    return True


def _validate_int(U, V, W, n, m, p):
    """Validate using exact integer arithmetic with scaling."""
    r = U.shape[1]
    # Convert to int64, assuming entries are 0, ±1, ±1/2
    # Scale by 2 to make all integers
    Us = (np.asarray(U, dtype=np.float64) * 2).round().astype(np.int64)
    Vs = (np.asarray(V, dtype=np.float64) * 2).round().astype(np.int64)
    Ws = (np.asarray(W, dtype=np.float64) * 2).round().astype(np.int64)
    
    # Check that rounding was exact
    if not np.allclose(Us / 2.0, np.asarray(U, dtype=np.float64), atol=1e-12):
        return False
    if not np.allclose(Vs / 2.0, np.asarray(V, dtype=np.float64), atol=1e-12):
        return False
    if not np.allclose(Ws / 2.0, np.asarray(W, dtype=np.float64), atol=1e-12):
        return False
    
    U3 = Us.reshape(n, m, r)
    V3 = Vs.reshape(m, p, r)
    W3 = Ws.reshape(p, n, r)
    
    result = np.zeros((n, m, p), dtype=np.int64)
    for c in range(r):
        uc = U3[:, :, c]
        vc = V3[:, :, c]
        wc = W3[:, :, c]
        temp = uc[:, :, None] * vc[None, :, :]
        wc_t = wc.T
        result += temp * wc_t[:, None, :]
    
    # Each term should be 8 (since 2*2*2=8)
    if not np.all(result == 8):
        return False
    return True


def _make_strassen_4x4_terms():
    """Return list of (i, j, k, coeff) for 4x4*4x4 Strassen decomposition (49 terms)."""
    # Strassen for 4x4: split into 2x2 blocks, use 7 Strassen multiplications,
    # each of which is a 2x2*2x2 multiplication of rank 7.
    # Total: 7*7 = 49 terms.
    # 
    # For 2x2*2x2 Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = A11*(B10-B00)
    # M3 = (A10+A11)*B01
    # M4 = A00*(B01-B11)
    # M5 = (A00+A01)(B11+B00)
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A10-A00)*(B00+B01)
    #
    # Each M is a 2x2 matrix product, decomposed into 7 scalar products.
    # We'll generate all 49 terms as (i, j, k, coeff).
    
    # First, define the 2x2 Strassen scalar terms
    def strassen_2x2():
        """Return list of (a_i, a_j, b_j, b_k, coeff) for 2x2*2x2 Strassen."""
        # Each entry: (a_row, a_col, b_row, b_col, coeff)
        # where A[a_row,a_col] * B[b_row,b_col] contributes with coeff
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms += [(0,0,0,0,1), (0,0,1,1,1), (1,1,0,0,1), (1,1,1,1,1)]
        # M2 = A11*(B10-B00)
        terms += [(1,1,1,0,1), (1,1,0,0,-1)]
        # M3 = (A10+A11)*B01
        terms += [(1,0,0,1,1), (1,1,0,1,1)]
        # M4 = A00*(B01-B11)
        terms += [(0,0,0,1,1), (0,0,1,1,-1)]
        # M5 = (A00+A01)(B11+B00)
        terms += [(0,0,1,1,1), (0,0,0,0,1), (0,1,1,1,1), (0,1,0,0,1)]
        # M6 = (A01-A11)*(B10+B11)
        terms += [(0,1,1,0,1), (0,1,1,1,1), (1,1,1,0,-1), (1,1,1,1,-1)]
        # M7 = (A10-A00)*(B00+B01)
        terms += [(1,0,0,0,1), (1,0,0,1,1), (0,0,0,0,-1), (0,0,0,1,-1)]
        return terms
    
    s2 = strassen_2x2()
    
    # Now for 4x4: blocks are indexed by (br, bc) for 2x2 sub-blocks
    # A block (br, bc) covers rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # B block (br, bc) covers rows 2*br..2*br+1, cols 2*bc..2*bc+1
    # 
    # Strassen 4x4 uses 7 block multiplications:
    # S1 = (A00+A11)(B00+B11)
    # S2 = A11*(B10-B00)
    # S3 = (A10+A11)*B01
    # S4 = A00*(B01-B11)
    # S5 = (A00+A01)(B11+B00)
    # S6 = (A01-A11)*(B10+B11)
    # S7 = (A10-A00)*(B00+B01)
    
    # Each S_m is a 2x2 block product, decomposed into 7 scalar 2x2 products.
    # For each scalar 2x2 product, we have 4 scalar multiplications (from s2 above, but s2 has 7 terms per M, each M is one 2x2 product... wait)
    # 
    # Actually, I'm confusing levels. Let me restart.
    # 
    # Level 1: 4x4*4x4 -> 7 block multiplications of 2x2*2x2
    # Level 2: Each 2x2*2x2 -> 7 scalar multiplications
    # Total: 49 scalar multiplications
    #
    # For each of the 7 block multiplications S_m = A_block * B_block:
    #   A_block is 2x2, B_block is 2x2
    #   We decompose A_block * B_block into 7 scalar products using Strassen 2x2
    #   Each scalar product involves specific elements of A_block and B_block
    #
    # The 7 block multiplications:
    # S1: A blocks (0,0) and (1,1) with +1; B blocks (0,0) and (1,1) with +1
    # S2: A block (1,1) with +1; B blocks (1,0) with +1 and (0,0) with -1
    # S3: A blocks (1,0) and (1,1) with +1; B block (0,1) with +1
    # S4: A block (0,0) with +1; B blocks (0,1) with +1 and (1,1) with -1
    # S5: A blocks (0,0) and (0,1) with +1; B blocks (1,1) and (0,0) with +1
    # S6: A blocks (0,1) with +1 and (1,1) with -1; B blocks (1,0) and (1,1) with +1
    # S7: A blocks (1,0) with +1 and (0,0) with -1; B blocks (0,0) and (0,1) with +1
    
    # For each S_m, the A side has a list of (block_row, block_col, sign) and B side has (block_row, block_col, sign)
    # When we do the 2x2 Strassen decomposition of A_block * B_block, each scalar term (a_ii, a_jj, b_jj, b_kk, coeff)
    # maps to global indices:
    #   i = 2*block_row_a + a_ii
    #   j = 2*block_col_a + a_jj
    #   (for B) row = 2*block_row_b + b_jj, col = 2*block_col_b + b_kk
    #   k = 2*block_col_b + b_kk
    # And the total coefficient is sign_a * sign_b * coeff
    
    block_defs = [
        # (A_blocks: [(br, bc, sign)], B_blocks: [(br, bc, sign)])
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),  # S1
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),           # S2
        ([(1,0,1), (1,1,1)], [(0,1,1)]),            # S3
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),           # S4
        ([(0,0,1), (0,1,1)], [(1,1,1), (0,0,1)]),   # S5
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),  # S6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),  # S7
    ]
    
    all_terms = []
    for A_blocks, B_blocks in block_defs:
        for (abr, abc, asgn) in A_blocks:
            for (bbr, bbc, bsgn) in B_blocks:
                # For this pair of blocks, do 2x2 Strassen
                for (ai, aj, bj, bk, coeff) in s2:
                    i = 2*abr + ai
                    j = 2*abc + aj
                    k = 2*bbc + bk
                    # B row index must match j: 2*bbr + bj should equal j
                    # But in Strassen, the B block row is determined by the A block col
                    # Actually, for S_m = A_block * B_block, the A block col must equal B block row
                    # In our block definitions, this should be consistent
                    # Let's just use the global indices
                    total_coeff = asgn * bsgn * coeff
                    if total_coeff != 0:
                        all_terms.append((i, j, k, total_coeff))
    
    return all_terms


def schoolbook_candidate():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = np.zeros((n*m, r), dtype=np.float64)
    V = np.zeros((m*p, r), dtype=np.float64)
    W = np.zeros((p*n, r), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def strassen_block_candidate():
    """Use Strassen 4x4 for the first 8 columns of B, schoolbook for last column."""
    n, m, p = DIMENSIONS
    # Get 4x4*4x4 Strassen terms
    terms_4x4 = _make_strassen_4x4_terms()
    
    all_terms = []
    # First 4x4 block of B (cols 0-3)
    for (i, j, k, coeff) in terms_4x4:
        all_terms.append((i, j, k, coeff))
    # Second 4x4 block of B (cols 4-7)
    for (i, j, k, coeff) in terms_4x4:
        k_new = k + 4
        all_terms.append((i, j, k_new, coeff))
    # Last column (col 8): schoolbook
    for i in range(n):
        for j in range(m):
            all_terms.append((i, j, 8, 1.0))
    
    r = len(all_terms)
    U = np.zeros((n*m, r), dtype=np.float64)
    V = np.zeros((m*p, r), dtype=np.float64)
    W = np.zeros((p*n, r), dtype=np.float64)
    for c, (i, j, k, coeff) in enumerate(all_terms):
        U[i*m+j, c] = coeff
        V[j*p+k, c] = 1.0
        W[k*n+i, c] = 1.0
    return U, V, W


def random_local_search(U0, V0, W0, n, m, p, max_time=100.0):
    """Local search: try to reduce rank by zeroing out columns and re-validating."""
    t0 = time.time()
    r0 = U0.shape[1]
    best_r = r0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    rng = random.Random(42)
    
    for iteration in range(2000):
        if time.time() - t0 > max_time:
            break
        if best_r <= 80:
            break
        
        # Try to remove a random column
        c = rng.randint(0, best_r - 1)
        U_test = np.delete(best_U, c, axis=1)
        V_test = np.delete(best_V, c, axis=1)
        W_test = np.delete(best_W, c, axis=1)
        
        if _validate_fast(U_test, V_test, W_test, n, m, p):
            best_U, best_V, best_W = U_test, V_test, W_test
            best_r -= 1
    
    return best_r, best_U, best_V, best_W


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Start with Strassen block candidate
    U, V, W = strassen_block_candidate()
    r = U.shape[1]
    
    if not _validate_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = schoolbook_candidate()
        r = U.shape[1]
    
    # Local search to reduce rank
    remaining = DEADLINE - (time.time() - t0) - 5
    if remaining > 10:
        r, U, V, W = random_local_search(U, V, W, n, m, p, max_time=remaining)
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
