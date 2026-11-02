import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def _strassen_222_entries(n, m, p, i0, j0, k0, U, V, W, col):
    """Add a 2x2x2 Strassen block starting at (i0, j0, k0)."""
    def uid(i, j): return i * m + j
    def vid(j, k): return j * p + k
    def wid(k, i): return k * n + i

    # M1 = (A00+A11)(B00+B11)
    c = col + 0
    U[uid(i0,j0)][c] += 1; U[uid(i0+1,j0+1)][c] += 1
    V[vid(j0,k0)][c] += 1; V[vid(j0+1,k0+1)][c] += 1
    W[wid(k0,i0)][c] += 1; W[wid(k0+1,i0+1)][c] += 1

    # M2 = (A10+A11)B00
    c = col + 1
    U[uid(i0+1,j0)][c] += 1; U[uid(i0+1,j0+1)][c] += 1
    V[vid(j0,k0)][c] += 1
    W[wid(k0,i0+1)][c] += 1; W[wid(k0+1,i0+1)][c] += -1

    # M3 = A00(B01-B11)
    c = col + 2
    U[uid(i0,j0)][c] += 1
    V[vid(j0,k0+1)][c] += 1; V[vid(j0+1,k0+1)][c] += -1
    W[wid(k0+1,i0)][c] += 1; W[wid(k0+1,i0+1)][c] += 1

    # M4 = A11(B10-B00)
    c = col + 3
    U[uid(i0+1,j0+1)][c] += 1
    V[vid(j0+1,k0)][c] += 1; V[vid(j0,k0)][c] += -1
    W[wid(k0,i0)][c] += 1; W[wid(k0,i0+1)][c] += 1

    # M5 = (A00+A01)B11
    c = col + 4
    U[uid(i0,j0)][c] += 1; U[uid(i0,j0+1)][c] += 1
    V[vid(j0+1,k0+1)][c] += 1
    W[wid(k0,i0)][c] += -1; W[wid(k0+1,i0)][c] += 1

    # M6 = (A10-A00)(B00+B01)
    c = col + 5
    U[uid(i0+1,j0)][c] += 1; U[uid(i0,j0)][c] += -1
    V[vid(j0,k0)][c] += 1; V[vid(j0,k0+1)][c] += 1
    W[wid(k0+1,i0+1)][c] += 1

    # M7 = (A01-A11)(B10+B11)
    c = col + 6
    U[uid(i0,j0+1)][c] += 1; U[uid(i0+1,j0+1)][c] += -1
    V[vid(j0+1,k0)][c] += 1; V[vid(j0+1,k0+1)][c] += 1
    W[wid(k0,i0)][c] += 1

def _strassen_2x22x2x2_entries(n, m, p, i0, j0, k0, U, V, W, col):
    """Add a (2,2)x(2,2) block: A is 2x2, B is 2x2x2 (two output planes).
    i indices: i0, i0+1
    j indices: j0, j0+1
    k indices: k0, k0+1, k0+2 (three planes)
    Uses 2 Strassen 2x2x2 blocks sharing the same A factor.
    Total: 14 rank terms.
    """
    _strassen_222_entries(n, m, p, i0, j0, k0, U, V, W, col)
    _strassen_222_entries(n, m, p, i0, j0, k0+1, U, V, W, col)

def _verify(U, V, W, n, m, p):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    return np.allclose(T, target, atol=1e-9)

def construct():
    start = time.time()
    deadline = start + 130.0
    n, m, p = DIMENSIONS  # 3, 5, 5

    # Strategy: Use 2x2x2 Strassen blocks where beneficial.
    # Grid: i in {0,1}, j in {0..4}, k in {0..4}
    # Non-overlapping 2x2x2 blocks on the 2x5x5 subgrid (i=0,1):
    # Option A: 4 blocks at (0,0,0),(0,2,0),(0,0,2),(0,2,2) -- covers i=0,1; j=0-3; k=0-3
    #   Leaves: i=2 (all 25), i=0,1 j=4 (5 terms), i=0,1 j=0-3 k=4 (8 terms) = 38 schoolbook
    #   Rank = 28 + 38 = 66... wait that's not right.
    #   4 blocks * 7 = 28, remaining = 75 - 32 = 43, total = 71. That matches current.
    
    # Let me try: 2x2x2 blocks that cover i=0,1; all j pairs; all k pairs
    # But j has 5 elements, k has 5 elements. We can't tile 5x5 with 2x2.
    # Best packing: 2x2 blocks on j: {0,1},{2,3} leaves j=4. On k: {0,1},{2,3} leaves k=4.
    # 4 blocks: (i=0-1, j=0-1, k=0-1), (i=0-1, j=2-3, k=0-1), (i=0-1, j=0-1, k=2-3), (i=0-1, j=2-3, k=2-3)
    # Covers: 4*8 = 32 triples out of 2*25 = 50 in the i=0,1 subgrid.
    # Remaining in i=0,1 subgrid: 50 - 32 = 18 terms.
    # i=2 row: 25 terms.
    # Total schoolbook: 18 + 25 = 43.
    # Total rank: 28 + 43 = 71. Same as before.

    # Can we do better? Try 3x2x2 or other asymmetric blocks?
    # A 3x2x2 block would need a rank-? algorithm. Standard approach doesn't help directly.
    
    # Alternative: use the 2x5x5 block (i=0,1, all j, all k) with a custom algorithm.
    # The 2x5x5 multiplication tensor has 50 entries.
    # We can split it: A is 2x5, B is 5x5, C is 2x5.
    # Split B into 5 diagonal columns... no that's schoolbook.
    # Split: A = [A_top(1x5), A_bot(1x5)], then C_top = A_top * B, C_bot = A_bot * B.
    # Each 1x5x5 is rank 5 (schoolbook, irreducible for row vector times matrix).
    # So 10 rank terms for the 2x5x5 block, plus 5 for i=2. Total 15? No, that's just 50 schoolbook terms = 50.
    # Wait, 1x5*5x5 = 1x5 requires 5 multiplications. Two such = 10. Plus 5 for i=2. Total 15. But that's for 50 products... no.
    # 2x5 * 5x5 = 2x5. That's 10 output entries. Each output entry is a dot product of length 5.
    # Schoolbook: 10*5 = 50 scalar mults. But as a tensor, rank <= 50.
    # Actually the 1x5x5 tensor (row vec * matrix = row vec) has rank exactly 5 (it's the identity in the j-k part).
    # So 2 copies: rank 10. Plus i=2: rank 5. Total 15.
    # Wait no. The full 3x5*5x5 tensor decomposes as:
    # (i=0 part) + (i=1 part) + (i=2 part)
    # i=0 part: 1x5*5x5 -> rank 5
    # i=1 part: 1x5*5x5 -> rank 5  
    # i=2 part: 1x5*5x5 -> rank 5
    # Total: 15? No, that's wrong. The 1x5x5 tensor T_{j,k} = delta_{j,k} has rank 5.
    # But the 2x5x5 tensor for i=0,1 with all j,k is:
    # U part: for output (i,j), coefficient of A[i,j'] and B[j',k]
    # For i=0: sum over j' A[0,j'] B[j',k] = C[0,k]. Tensor: sum_{j,k} A[0,j] B[j,k] C[0,k]
    # This has rank 5 (for fixed i=0, it's the 5x5 identity tensor).
    # Similarly for i=1, rank 5.
    # For i=2, rank 5.
    # Total rank: 15.
    
    # Wait, but schoolbook gives 75 for 3x5x5. Let me recheck.
    # 3x5 * 5x5 = 3x5. 15 output entries. Each is a sum of 5 products. Total 75 scalar products.
    # Tensor rank: for the full 3x5x5 -> 3x5x5 tensor, each output (i,k) depends on A[i,j] and B[j,k].
    # The tensor is T_{i,j; j',k; k',i'} where T = 1 if i=i' and j=j' and k=k'.
    # So T = sum_{i=0}^{2} sum_{j=0}^{4} sum_{k=0}^{4} A_{i,j} B_{j,k} C_{k,i}
    # That's 75 rank-1 terms. Rank <= 75.
    # But can we do better? The tensor is actually sum_i (sum_j A_{i,j} B_{j,0} C_{0,i} + ... + sum_j A_{i,j} B_{j,4} C_{4,i})
    # For each fixed i, the part involving row i of A and row i of C (which is column i of output) is:
    # sum_{j,k} A_{i,j} B_{j,k} C_{k,i}
    # This is the 5x5 multiplication tensor for the i-th row, which has rank 5.
    # Since the three i-parts are orthogonal (different i indices in A and C), total rank = 3*5 = 15.
    
    # So rank 15 should be possible! Let me verify this is correct.
    # Actually wait - the tensor is:
    # T[a,b,c] = 1 if a//m == c%m and a%m == b//p and b%p == c//n
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1
    # For fixed i: T[i*m+j, j*p+k, k*n+i] for j in 0..4, k in 0..4
    # This is sum_{j=0}^{4} (A[i*m+j]) * (B[j*p: (j+1)*p] as a vector in k) * (C[k*n+i])
    # = sum_{j=0}^{4} sum_{k=0}^{4} A[i,j] B[j,k] C[k,i]
    # The tensor for fixed i is: sum_{j,k} A[i,j] B[j,k] C[k,i]
    # As a trilinear form on (A_part, B_part, C_part) where A_part has indices (i,j), B_part (j,k), C_part (k,i):
    # This equals sum_{j=0}^{4} A[i,j] * (sum_{k=0}^{4} B[j,k] * C[k,i])
    # The inner sum is (B @ C_col_i)[j] where C_col_i is column i of C viewed as a vector.
    # The rank of this tensor (as a function of the vector variables) is exactly 5:
    # It's sum_{j=0}^{4} A_{i,j} (B_{j,0} C_{0,i} + B_{j,1} C_{1,i} + ... + B_{j,4} C_{4,i})
    # = sum_{k=0}^{4} (sum_j A_{i,j} B_{j,k}) C_{k,i}
    # As a tensor in the free variables, for fixed i, it's:
    # sum_{k=0}^{4} C_{k,i} * (A_i^T B)_k
    # The bilinear form A_i^T B is a product, so each k gives a rank-1 term: C_{k,i} * (dot(A_i, B_col_k))
    # But dot(A_i, B_col_k) is not a single variable - it's a product of 5 pairs.
    # So we need 5 more multiplications per k? No...
    
    # Let me think more carefully. The rank-1 decomposition requires terms of the form:
    # (linear in A) * (linear in B) * (linear in C)
    # For the i=0 part: we need to express sum_{j=0}^{4} sum_{k=0}^{4} A[0,j] B[j,k] C[k,0]
    # as sum of rank-1 terms.
    # Each term is (u_0^T A)(v_0^T B)(w_0^T C) where u,v,w are vectors.
    # The i=0 part of the tensor involves only A[0,j], B[j,k], C[k,0].
    # This is the 5x5 "matrix multiplication tensor" restricted to row 0 of output.
    # Specifically, it's the tensor T^{(i=0)}[j,k] = sum over terms where the A part picks A[0,j], B part picks B[j,k], C part picks C[k,0].
    # This is just the 5x5 identity matrix tensor! rank = 5.
    # Decomposition: for k=0..4: A[0,k] * B[k,k] * C[k,0]. That's 5 terms.
    # But wait, that only covers the diagonal. We need the full product A_row0 * B = C_row0.
    # C_row0[k] = sum_j A[0,j] B[j,k].
    # The tensor for this is: sum_j sum_k A[0,j] B[j,k] C[k,0] with C being the OUTPUT, so C[k,0] is the k-th entry of row 0 of output.
    # As a trilinear form: it's sum_{j,k} A[0,j] B[j,k] C_out[0,k].
    # This is NOT the identity tensor - it's the full matrix multiplication for a single row.
    # The rank of the n x 1 x n matrix multiplication tensor (row vector times n x n matrix = row vector) is n.
    # Proof: C_out[k] = sum_j A[j] B[j,k]. Tensor: sum_{j,k} A[j] B[j,k] C_out[k].
    # Rank-1 terms: for each j, we have A[j] * (sum_k B[j,k] C_out[k]). But sum_k B[j,k] C_out[k] is not linear in B and C_out separately.
    # Hmm, actually: sum_{j,k} A[j] B[j,k] C[k] where k is the output index.
    # This can be written as sum_{j=0}^{4} A[j] * (B[j,:] dot C_vec)
    # where B[j,:] is row j of B, C_vec = (C[0],...,C[4]).
    # B[j,:] dot C_vec = sum_k B[j,k] C[k], which is linear in B and linear in C.
    # So each term A[j] * (sum_k B[j,k] C[k]) is a product of three linear forms:
    #   f_A = e_j^T A  (picks A[j])
    #   f_B = sum_k B[j,k] * (coeff from f_C) -- no, this doesn't factor.
    
    # I'm confusing myself. Let me use the standard result: the tensor rank of the (1 x n) x (n x n) -> (1 x n) matrix multiplication is n.
    # The decomposition is: for k = 0, ..., n-1:
    #   Term k: A_part = sum_j A[j] * (coefficient), B_part = B[:,k] (column k), C_part = C[k]
    #   More precisely: (sum_j c_{j,k} A[j]) * (sum_j d_{j,k} B[j,k']) ... 
    # 
    # Actually the standard decomposition of the n x 1 x n tensor (row * square = row) with rank n:
    # C_k = sum_j A_j B_{j,k} for k=0..n-1
    # Tensor T = sum_{j,k} A_j B_{j,k} C_k
    # Rank-1 decomposition with rank n: T = sum_{k=0}^{n-1} (A_{part_k}) (B_{part_k}) (C_{part_k})
    # where A_{part_k} = A_{pi_k} (some permutation), B_{part_k} = B_{pi_k, k} ... 
    # 
    # No, the correct simple decomposition: T = sum_{k=0}^{n-1} A_{sigma(k)} * B_{sigma(k), k} * C_k? That only works if n=1.
    
    # I think I'm overcomplicating. The full 3x5x5 tensor has rank at most 75 (schoolbook).
    # The current solution achieves 71 using Strassen. Let me just try to find a better block decomposition.
    
    # Let me try: use 2x2x2 Strassen on a different set of blocks, or use larger blocks.
    # What about a 2x3x3 block? Does that help?
    # A 2x3x3 multiplication: A is 2x3, B is 3x3, C is 2x3. 18 output entries, 54 schoolbook.
    # Can we do better than 54? 2x3x3 tensor... split A into two rows: each 1x3x3 has rank 3. Total 6? No, that's not how it works.
    # 1x3x3 (row vec * 3x3 = row vec): tensor rank is 3. Two of them: 6.
    # Wait, that would give rank 6 for 2x3x3?? That seems too good.
    # 1x3 * 3x3 = 1x3. Tensor: sum_{j,k} A[j] B[j,k] C[k]. This has 9 entries in the j,k space but the "multiplication" is:
    # C[k] = sum_j A[j] B[j,k]
    # The tensor T_{j,k} (with A indexed by j, B by (j,k), C by k) is:
    # T = sum_{j=0}^{2} sum_{k=0}^{2} A[j] B[j,k] C[k]
    # = A[0](B[0,0]C[0] + B[0,1]C[1] + B[0,2]C[2]) + A[1](B[1,0]C[0] + B[1,1]C[1] + B[1,2]C[2]) + A[2](B[2,0]C[0] + B[2,1]C[1] + B[2,2]C[2])
    # Each "A[j] * (sum_k B[j,k] C[k])" is NOT a rank-1 term because sum_k B[j,k] C[k] is a product, not a linear form.
    # For it to be rank 1, we need the B-part and C-part to be separate linear forms.
    # So T = sum_{j,k} (A[j]) * (B[j,k]) * (C[k]) - each (j,k) pair gives a rank-1 term.
    # Total 9 rank-1 terms for the 1x3x3 tensor. So rank = 9 = 3*3.
    # I was wrong. The rank of the (1xn) x (nxm) -> (1xm) tensor is n*m, not min(n,m).
    # It's just schoolbook and can't be improved because each output entry C[k] depends on all j.
    
    # OK so my rank-15 hypothesis was wrong. The rank is indeed 75 for schoolbook, and we can save with Strassen-like tricks.
    # Current: 71. Let me see if I can do 70 or lower.
    
    # The 2x5x5 sub-tensor (i=0,1): rank?
    # 4 Strassen 2x2x2 blocks + 18 schoolbook = 28 + 18 = 46.
    # Can we do the 2x5x5 in less than 46?
    # Alternative: use 5 2x2x2 blocks that overlap in A but not in (j,k)?
    # No, Strassen blocks must be non-overlapping in all three dimensions for the simple construction.
    
    # What if I use a 2x5x5 algorithm based on splitting differently?
    # Split j into {0,1,2,3,4}, use 5 "column" operations... that's schoolbook = 50.
    # Use 2x2 Strassen on 4 of the 5 j-pairs... can't, 5 is odd.
    
    # Let me try a different approach: maybe use the 3x2x2 and 3x3x3 blocks.
    # 3x2x2 block: A is 3x2, B is 2x2, C is 3x2. 12 outputs, 24 schoolbook.
    # Strassen doesn't directly apply to 3x2x2.
    # But we can do: C = A * B. Split A into rows. Each row: 1x2*2x2 = 1x2, rank 4.
    # 3 rows: 12. Same as schoolbook. No improvement.
    
    # What about 3x3x3? 27 outputs, 81 schoolbook.
    # Strassen: 7 + schoolbook for remaining? 3x3 can be done with 1 Strassen 2x2x2 (covering 8 entries) + 5 schoolbook = 12.
    # Or two overlapping Strassen? No, must be non-overlapping.
    # One 2x2x2 covers entries (i,j,k) for i in {0,1}, j in {0,1}, k in {0,1}. 8 entries.
    # Remaining: 27 - 8 = 19. Total: 7 + 19 = 26.
    # Can we do 3x3x3 better? The 3x3x3 multiplication tensor has known rank 23 (by Bini et al.) or maybe less.
    # But implementing an optimal 3x3x3 decomposition is complex.
    
    # Let me just go with the 71-rank solution but verify it's correct.
    # Actually, let me try yet another partition: what if I use the i=2 row with Strassen too?
    # i=2 has 1x5*5x5 = 1x5. 25 entries. No Strassen benefit (1x5 can't use 2x2x2).
    
    # I'll stick with rank 71 but write clean code.
    
    # Build the decomposition:
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    
    total_rank = 4 * 7 + (75 - 4 * 8)  # 28 + 43 = 71
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i0, j0, k0 in blocks:
        _strassen_222_entries(n, m, p, i0, j0, k0, U, V, W, col)
        col += 7
    
    # Mark covered triples
    covered = [[[[False]*p for _ in range(m)] for _ in range(n)] for _ in range(0)]
    covered = [[[[False]*p for _ in range(m)] for _ in range(n)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    ii, jj, kk = i0+di, j0+dj, k0+dk
                    if ii < n and jj < m and kk < p:
                        covered[ii][jj][kk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Verify
    if time.time() < deadline - 10.0:
        if not _verify(U, V, W, n, m, p):
            # Fallback
            rank_school = n * m * p
            U_s = [[0] * rank_school for _ in range(n*m)]
            V_s = [[0] * rank_school for _ in range(m*p)]
            W_s = [[0] * rank_school for _ in range(p*n)]
            c2 = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i*m+j][c2] = 1
                        V_s[j*p+k][c2] = 1
                        W_s[k*n+i][c2] = 1
                        c2 += 1
            return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
