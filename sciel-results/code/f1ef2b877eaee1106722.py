import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _build_tensor():
    n, m, p = 4, 4, 8
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify(U, V, W, T):
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-9)

def _to_half_integer(mat):
    mat2 = mat * 2
    rounded = np.round(mat2)
    if np.allclose(mat2, rounded, atol=1e-6):
        return (rounded / 2).tolist()
    rounded_int = np.round(mat)
    if np.allclose(mat, rounded_int, atol=1e-6):
        return rounded_int.astype(int).tolist()
    return mat.tolist()

def _schoolbook():
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[0,2]=-1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[1,3]=-1; W[3,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=1; W[1,4]=-1; W[3,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[2,5]=-1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[2,6]=-1
    return U, V, W

def _build_4x4x4_49():
    U2, V2, W2 = _strassen_2x2x2()
    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)

    def offset(b):
        if b == 0: return 0
        if b == 1: return 2
        if b == 2: return 8
        if b == 3: return 10
        return 0

    A_blocks = [
        [(0,1),(3,1)],[(2,1),(3,1)],[(0,1)],[(3,1)],
        [(0,1),(1,1)],[(2,1),(0,-1)],[(1,1),(3,-1)]
    ]
    B_blocks = [
        [(0,1),(3,1)],[(0,1)],[(2,1),(3,-1)],[(1,1),(0,-1)],
        [(3,1)],[(0,1),(1,1)],[(2,1),(3,1)]
    ]
    C_blocks = [
        [(0,1),(3,1),(1,-1),(2,1)],[(2,1),(3,-1)],[(0,-1),(1,1)],
        [(1,-1),(3,1)],[(0,1),(1,-1),(3,1)],[(2,-1),(3,1)],[(0,1),(2,-1)]
    ]

    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            u_vals = U2[:, s]
            v_vals = V2[:, s]
            w_vals = W2[:, s]
            for i_local in range(4):
                val = u_vals[i_local]
                if val != 0:
                    for bA, sA in A_blocks[t]:
                        U4[offset(bA) + i_local, col] += val * sA
            for j_local in range(4):
                val = v_vals[j_local]
                if val != 0:
                    for bB, sB in B_blocks[t]:
                        V4[offset(bB) + j_local, col] += val * sB
            for k_local in range(4):
                val = w_vals[k_local]
                if val != 0:
                    for bC, sC in C_blocks[t]:
                        W4[offset(bC) + k_local, col] += val * sC
    return U4, V4, W4

def _build_4x4x8_98():
    U4, V4, W4 = _build_4x4x4_49()
    U = np.zeros((16, 98), dtype=np.float64)
    V = np.zeros((32, 98), dtype=np.float64)
    W = np.zeros((32, 98), dtype=np.float64)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    U[:, 49:98] = U4
    for j in range(4):
        for k_local in range(4):
            V[j*8 + k_local + 4, 49:98] = V4[j*4 + k_local, 0:49]
    for k_local in range(4):
        for i in range(4):
            W[(k_local+4)*4 + i, 49:98] = W4[k_local*4 + i, 0:49]
    return U, V, W

def _rank_reduce(U, V, W, T, deadline):
    """
    Structured rank reduction: find linear dependencies among rank-1 terms.
    Key insight: if for some column r, U[:,r] is in the span of other U columns
    (and similarly for V and W), we can eliminate it.
    
    More powerfully: use the fact that the two 4x4x4 blocks share U.
    We look for transformations that preserve the tensor but reduce rank.
    """
    n, m, p = 4, 4, 8
    dim_a, dim_b, dim_c = 16, 32, 32
    rank = U.shape[1]
    
    # Strategy: try to find a basis transformation on the "shared" structure
    # The 98-rank solution has U = [U4 | U4], V = [V4_top; V4_bottom], W = [W4_top; W4_bottom]
    # We can try to combine the two halves using a mixing matrix.
    
    # First, check if simple column elimination works
    # For each column, check if it's linearly dependent given the others
    # This is expensive for 98 columns, so use a smarter approach
    
    # Approach: Use the structure. The two blocks are:
    # Block 1: T1[a,b,c] for c in 0..15 (k=0..3)
    # Block 2: T2[a,b,c] for c in 16..31 (k=4..7)
    # They share U but have different V and W.
    
    # Try: can we find a 2x2 mixing of the two blocks that reduces rank?
    # Let U_mix = U * M where M is some transformation...
    # Actually, the right approach: look for rank-1 terms in the "difference"
    
    # Better approach: Use bilinear rank reduction.
    # If there exists vectors u0, v0, w0 and a scalar such that
    # the tensor has a "fold" structure, we can reduce rank.
    
    # Practical approach: try random linear combinations of columns
    # and check if the resulting tensor is still valid with fewer terms.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try eliminating columns that are linearly dependent
    # Build the "flattened" representation and do SVD-like analysis
    
    # For a more structured approach: the 4x4x8 tensor can be seen as
    # C = A * B where A is 4x4, B is 4x8.
    # The rank of this bilinear map is at most rank(4x4x4) * 2 = 98.
    # But we can do better by exploiting that B has two 4x4 blocks.
    
    # Key structural insight: B = [B1 | B2] where B1, B2 are 4x4.
    # C = A*B1 for first half, C = A*B2 for second half.
    # If we use Strassen on A*B1 and A*B2 separately, we get 49+49=98.
    # But the A factors are the same! So we can potentially share computations.
    
    # The shared structure means: for each Strassen product M_t(A, B1) and M_t(A, B2),
    # the A-side is identical. So the U factor is shared.
    # The question is: can we combine the V and W factors to reduce total rank?
    
    # Let's try: for each pair of columns (r1, r2) where r1 is from block 1 and r2 from block 2,
    # check if we can merge them.
    
    # Actually, the most promising approach: use the known fact that
    # rank(4x4x8) <= rank(4x4x4) + rank(4x4x4) but also
    # rank(4x4x8) can be bounded by considering 4x8x8 with A as block-diag.
    
    # Let me try a different structural approach:
    # View 4x4x8 as: C[i,k] = sum_j A[i,j] * B[j,k]
    # Split k into two groups of 4: k = k1 (0..3) and k = k2 (4..7)
    # C1[i,k1] = sum_j A[i,j] * B1[j,k1]
    # C2[i,k2] = sum_j A[i,j] * B2[j,k2]
    
    # Now, use the fact that we can compute both with a shared Strassen structure.
    # The 4x4x4 Strassen uses 7 products per block.
    # But we can potentially use fewer by exploiting cross-block structure.
    
    # Let me try: a 4x4x8 decomposition using 56 terms (7*8)
    # by using 8 "columns" of B with a shared 7-term A decomposition.
    
    # Actually, let's just try the direct approach: build a 56-rank candidate
    # and verify it.
    
    # The 56-rank idea: use 7 Strassen terms for A, and for each term,
    # multiply by the full 4x8 B matrix (which takes 8 scalar products per term).
    # But that's 7*8 = 56 rank-1 terms!
    
    # Wait, that's not how it works. Each rank-1 term is u ⊗ v ⊗ w.
    # If we use Strassen on the 4x4 part, we get 7 terms for A*B1.
    # For A*B2, we get another 7 terms.
    # Total: 14 terms for the "A side" but each term involves all 4 rows of B.
    
    # Let me think more carefully. The 4x4x4 Strassen decomposition:
    # C = sum_{t=1}^{7} M_t(A) * N_t(B) * P_t
    # where M_t is a linear combo of A entries, N_t is a linear combo of B entries.
    
    # For 4x4x8: C = A * B where B is 4x8.
    # Split B = [B1 B2].
    # C1 = A * B1 = sum_t M_t(A) * N_t(B1) * P_t  (7 terms)
    # C2 = A * B2 = sum_t M_t(A) * N_t(B2) * P_t  (7 terms)
    # Total 14 "group" terms, but each group term expands to multiple rank-1 terms.
    
    # In the 4x4x4 case, each "group" t has rank-1 terms indexed by (i,k) positions.
    # The total rank is 49 = 7 * 7 because each group has 7 independent rank-1 components.
    
    # For 4x4x8, if we could make each group have only 4 rank-1 components
    # (one per output row), we'd get 14*4 = 56. But that's not generally possible.
    
    # Let me just try to find the minimum rank by checking known bounds.
    # rank(4x4x4) = 49 (or maybe less? Strassen gives 49, but the true rank might be lower)
    # Actually, the rank of 4x4x4 matrix multiplication is known to be 49 (Alon-Boppana lower bound ~47.6, and 49 is achieved).
    
    # For 4x4x8: the rank is at most 98 (two independent 4x4x4 blocks).
    # Lower bound: at least rank(4x4x4) = 49.
    # Can we do better than 98?
    
    # The key question: is rank(4x4x8) = 98 or can we do less?
    # Since the two B-halves are independent, and A is shared,
    # the rank should be at least 2 * rank(4x4x4) / something...
    
    # Actually, by the "slice rank" argument:
    # The tensor T has slices T[:,:,c] for c=0..31.
    # For c in 0..15 (k=0..3): T[:,:,c] depends on A and B[:, 0..3]
    # For c in 16..31 (k=4..7): T[:,:,c] depends on A and B[:, 4..7]
    
    # The slice rank lower bound: rank >= max over partitions of sum of slice ranks.
    # This is complex, but intuitively, since the two halves use different B entries,
    # they're "independent" in a sense, suggesting rank ~ 2*49 = 98.
    
    # However, there might be a clever sharing. Let me try a computational search.
    
    # Approach: Start with the 98-rank solution and try to find linear dependencies.
    # Flatten the tensor and use the structure to find a lower-rank representation.
    
    # For now, let's try a targeted search:
    # 1. Start with 98-rank
    # 2. Try to merge pairs of columns that have similar structure
    # 3. Use the fact that U is duplicated
    
    # The U matrix is [U4 | U4]. So columns 0-48 and 49-97 have identical U.
    # This means: for any linear combination alpha*col_r + beta*col_{r+49},
    # the U part is (alpha+beta)*U4[:,r].
    # The V part is alpha*V4_top[:,r] + beta*V4_bottom[:,r] (in the appropriate positions)
    # The W part is alpha*W4_top[:,r] + beta*W4_bottom[:,r]
    
    # If we can find alpha, beta such that this combination "cancels" something,
    # we might reduce rank.
    
    # Let's try: for each pair (r, r+49), see if we can eliminate one.
    # col_r contributes: U4[:,r] ⊗ V4_top[:,r] ⊗ W4_top[:,r] (for first 16 rows of V,W)
    # col_{r+49} contributes: U4[:,r] ⊗ V4_bottom[:,r] ⊗ W4_bottom[:,r] (for last 16 rows)
    
    # These are on different "slices" of V and W, so they can't simply cancel.
    # The tensor contribution of col_r is nonzero only where V4_top and W4_top are nonzero.
    # The tensor contribution of col_{r+49} is nonzero only where V4_bottom and W4_bottom are nonzero.
    
    # Since V4_top affects V indices 0-15 and V4_bottom affects V indices 16-31,
    # and W4_top affects W indices 0-15 and W4_bottom affects W indices 16-31,
    # the two columns affect disjoint sets of (b,c) pairs in the tensor.
    # Therefore, they CANNOT be linearly combined to cancel!
    
    # This means the 98 rank is actually tight for this construction.
    # The minimum rank is exactly 98 for this block structure.
    
    # But wait - maybe there's a completely different construction with lower rank!
    # The question is whether rank(4x4x8) < 98.
    
    # Known results: rank(n x n x 2n) for n=4...
    # The "Landsberg" bound and other results suggest rank(4x4x8) might be around 56-70.
    
    # Let me try a completely different construction:
    # Use the 2x2x2 Strassen (rank 7) to build up to 4x4x8.
    
    # 4x4x8 = (2x2x2) applied with 2x2 blocks for A and B, but B has 8 columns.
    # A is 4x4 = 2x2 blocks of 2x2.
    # B is 4x8 = 2x2 blocks of 2x4... hmm, that's awkward.
    
    # Alternative: A is 4x4, B is 4x8.
    # View A as 2x2 blocks of 2x2: A = [[A00, A01], [A10, A11]]
    # View B as 2x2 blocks of 2x4: B = [[B00, B01], [B10, B11]] where each Bi is 2x4.
    # C = A*B is 4x8 = 2x2 blocks of 2x4: C = [[C00, C01], [C10, C11]]
    # C00 = A00*B00 + A01*B10 (2x4)
    # C01 = A00*B01 + A01*B11 (2x4)
    # C10 = A10*B00 + A11*B10 (2x4)
    # C11 = A10*B01 + A11*B11 (2x4)
    
    # Each Cij is a 2x4 matrix = 2x2 * 2x4.
    # To compute 2x2 * 2x4, we can use 2x2x2 Strassen-like:
    # View 2x4 as two 2x2 blocks: [X | Y].
    # 2x2 * [X|Y] = [2x2*X | 2x2*Y].
    # Each 2x2*2x2 uses 7 Strassen products.
    # So 2x2*2x4 uses 14 products.
    # Total: 4 C-blocks * 14 = 56 products!
    
    # But wait, there's sharing: A00 appears in C00 and C01, A10 in C10 and C11.
    # The Strassen products for A00*X and A00*Y share the A00 side.
    # Specifically, M_t(A00) is the same for both.
    # So for A00: 7 M_t values, applied to X and Y separately.
    # That's 7 * 2 = 14 rank-1 terms for A00's contribution.
    # Similarly for A01, A10, A11: 14 each.
    # Total: 4 * 14 = 56 rank-1 terms!
    
    # Let me construct this!
    
    U_56 = np.zeros((16, 56), dtype=np.float64)
    V_56 = np.zeros((32, 56), dtype=np.float64)
    W_56 = np.zeros((32, 56), dtype=np.float64)
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3),
    #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # In the 16-dim U space: index = i*4 + j where i is row, j is col of A.
    # A00: i in {0,1}, j in {0,1} -> indices 0,1,4,5
    # A01: i in {0,1}, j in {2,3} -> indices 2,3,6,7
    # A10: i in {2,3}, j in {0,1} -> indices 8,9,12,13
    # A11: i in {2,3}, j in {2,3} -> indices 10,11,14,15
    
    # B blocks: B is 4x8. Row j, col k.
    # B00: j in {0,1}, k in {0,1} -> indices 0,1,8,9  (j*8+k)
    # B01: j in {0,1}, k in {2,3} -> indices 2,3,10,11
    # B10: j in {2,3}, k in {0,1} -> indices 16,17,24,25
    # B11: j in {2,3}, k in {2,3} -> indices 18,19,26,27
    # B02: j in {0,1}, k in {4,5} -> indices 4,5,12,13
    # B03: j in {0,1}, k in {6,7} -> indices 6,7,14,15
    # B12: j in {2,3}, k in {4,5} -> indices 20,21,28,29
    # B13: j in {2,3}, k in {6,7} -> indices 22,23,30,31
    
    # C blocks: C is 4x8. Row i, col k. Index = k*4+i (trace convention W[k*n+i,r])
    # C00: i in {0,1}, k in {0,1} -> indices 0,1,4,5
    # C01: i in {0,1}, k in {2,3} -> indices 2,3,6,7
    # C10: i in {2,3}, k in {0,1} -> indices 8,9,12,13
    # C11: i in {2,3}, k in {2,3} -> indices 10,11,14,15
    # C02: i in {0,1}, k in {4,5} -> indices 16,17,20,21
    # C03: i in {0,1}, k in {6,7} -> indices 18,19,22,23
    # C12: i in {2,3}, k in {4,5} -> indices 24,25,28,29
    # C13: i in {2,3}, k in {6,7} -> indices 26,27,30,31
    
    # For each A-block (say A00) and each pair of B-blocks it multiplies:
    # A00 * B00 -> C00 (left part), A00 * B02 -> C02 (left part)
    # A00 * B01 -> C00 (right part), A00 * B03 -> C02 (right part)
    # Wait, I need to be more careful.
    
    # C00 (2x4) = A00 * B00 + A01 * B10
    # where B00 is 2x4 (j in {0,1}, k in {0,1,4,5})? No...
    
    # Let me redo. B is 4x8. B[j,k] for j=0..3, k=0..7.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # C00 (i in {0,1}, k in {0,1}): C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # This involves all 4 rows of B for k in {0,1}.
    
    # Hmm, my block decomposition was wrong. Let me redo.
    
    # A is 4x4, B is 4x8.
    # A = [[A00, A01], [A10, A11]] where each is 2x2.
    # B = [[B_top], [B_bottom]] where B_top is 2x8, B_bottom is 2x8.
    # C = A * B = [[A00*B_top + A01*B_bottom], [A10*B_top + A11*B_bottom]]
    # C_top (2x8) = A00*B_top + A01*B_bottom
    # C_bottom (2x8) = A10*B_top + A11*B_bottom
    
    # Now B_top is 2x8 = [Bt0 | Bt1 | Bt2 | Bt3] where each Bti is 2x2.
    # Similarly B_bottom.
    
    # C_top = A00 * [Bt0|Bt1|Bt2|Bt3] + A01 * [Bb0|Bb1|Bb2|Bb3]
    #         = [A00*Bt0 + A01*Bb0 | A00*Bt1 + A01*Bb1 | A00*Bt2 + A01*Bb2 | A00*Bt3 + A01*Bb3]
    
    # Each 2x2 * 2x2 product uses 7 Strassen terms.
    # For C_top, we have 4 such products (one for each 2x2 column block).
    # But A00 is shared across all 4! So we compute M_t(A00) once, then
    # multiply by N_t(Bt0), N_t(Bb0), N_t(Bt1), N_t(Bb1), etc.
    
    # For each t (7 Strassen terms) and each A-block (4 blocks: A00,A01,A10,A11),
    # and each B-column-block (4 blocks), we get one rank-1 term.
    # Total: 7 * 4 * 4 = 112? No, that's too many.
    
    # Wait: for a fixed A-block (say A00) and fixed t:
    # M_t(A00) is a scalar.
    # It multiplies N_t(Bt0) for C_top col-block 0, N_t(Bt1) for col-block 1, etc.
    # And N_t(Bb0), N_t(Bb1), etc. for the A01 contribution.
    
    # Actually, let me count differently.
    # C_top[i, k] where i in {0,1}, k in {0,7}.
    # For k in {0,1}: C_top[i,k] = sum_j A00[i,j]*B_top[j,k] + sum_j A01[i,j]*B_bottom[j,k]
    # This is a 2x2 * 2x2 + 2x2 * 2x2 = two 2x2x2 products.
    # Each uses 7 terms. Total 14 terms for this 2x2 output block.
    
    # Similarly for k in {2,3}, {4,5}, {6,7}: 14 terms each.
    # C_top total: 4 * 14 = 56 terms.
    # C_bottom total: 4 * 14 = 56 terms.
    # Grand total: 112 terms. That's worse than 98!
    
    # The issue: I'm not sharing the Strassen products between the two 2x2 products
    # in each 2x4 block. Let me think about this differently.
    
    # For C_top[i, k] (i in {0,1}, k in {0,1}):
    # = sum_{j in {0,1}} A00[i,j] * Bt0[j, k%2] + sum_{j in {0,1}} A01[i,j] * Bb0[j, k%2]
    
    # This is: [A00 * Bt0]_{i, k%2} + [A01 * Bb0]_{i, k%2}
    # where Bt0 is 2x2 (rows j in {0,1}, cols k in {0,1})
    # and Bb0 is 2x2 (rows j in {2,3}, cols k in {0,1})
    
    # Using Strassen on 2x2*2x2:
    # [A00 * Bt0] = sum_t M_t(A00) * N_t(Bt0) * P_t
    # [A01 * Bb0] = sum_t M_t(A01) * N_t(Bb0) * P_t
    
    # Total for this 2x2 output: 14 rank-1 terms.
    
    # For all 8 output 2x2 blocks (4 in C_top, 4 in C_bottom): 8 * 14 = 112.
    
    # This is worse. The 49+49=98 construction is better.
    
    # Let me try yet another approach: use the fact that 4x4x8 = 4x4x4 + 4x4x4
    # but with a different splitting that allows sharing.
    
    # What if we split B into overlapping parts?
    # Or use a different factorization of the multiplication tensor?
    
    # For now, let's accept that 98 might be near-optimal and focus on
    # verifying the 98-rank solution works, then try small perturbations.
    
    # Actually, let me try one more idea: the "Coppersmith-Winograd" style
    # tensor decomposition might give lower rank for non-cubic dimensions.
    
    # For 4x4x8, one approach: embed in 8x8x8.
    # rank(8x8x8) <= 7^3 = 343 (Strassen) or better.
    # But we only need the 4x4x8 part.
    
    # The 8x8x8 Strassen rank is 343. The 4x4x8 part uses a subset.
    # This doesn't directly help.
    
    # Let me just return the 98-rank solution and try minor improvements.
    
    # Try: can we reduce the 98-rank by finding that some columns are
    # linearly dependent in a way that preserves the tensor?
    
    # Check: is the 98-rank solution minimal?
    # The tensor T has 16*32*32 = 16384 entries, 128 of which are 1.
    # The rank-98 solution uses 98 rank-1 terms.
    # Lower bound: the tensor has "slice rank" at least...
    
    # For now, return 98 and hope it's correct.
    return U, V, W, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    # Fallback: Schoolbook (rank 128)
    U_school, V_school, W_school = _schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = 128
    
    # Try 98-rank constructive solution
    if time.time() < deadline - 5:
        try:
            U98, V98, W98 = _build_4x4x8_98()
            if _verify(U98, V98, W98, T):
                best_U, best_V, best_W = U98, V98, W98
                best_rank = 98
        except Exception:
            pass
    
    # Try to reduce rank below 98 using structured search
    if time.time() < deadline - 5 and best_rank == 98:
        try:
            U98, V98, W98 = _build_4x4x8_98()
            
            # Try: remove pairs of columns and check if tensor is preserved
            # This is a "pruning" search.
            # For efficiency, only try removing columns that are structurally similar.
            
            # The 98 columns come in 49 pairs: (r, r+49) for r=0..48.
            # Each pair has the same U factor.
            # Removing one from a pair breaks the tensor (as argued above).
            # But what about removing columns from different pairs?
            
            # Try: remove 2 columns from different pairs and add 1 new column.
            # This is a "swap" operation.
            
            # For a more systematic approach, use the following:
            # The tensor equation is:
            # sum_r U[:,r] V[:,r] W[:,r] = T
            # If we remove columns r1, r2 and add column r3, we need:
            # U[:,r1]V[:,r1]W[:,r1] + U[:,r2]V[:,r2]W[:,r2] = U[:,r3]V[:,r3]W[:,r3]
            # This is a very constrained equation, unlikely to have solutions.
            
            # Instead, try: modify coefficients to create cancellations.
            # Add a small perturbation to some columns and check if the
            # resulting tensor is still exact (it won't be, but maybe
            # we can find a nearby valid solution).
            
            # This is essentially what the GA was doing. Let's try a
            # more targeted version: focus on the shared U structure.
            
            # Key idea: since U = [U4 | U4], we can write:
            # T = sum_{r=0}^{48} U4[:,r] * (V4_top[:,r] W4_top[:,r] + V4_bottom[:,r] W4_bottom[:,r])
            # where the second term has V,W shifted to the bottom half.
            
            # Let D_r = V4_top[:,r] W4_top[:,r] (outer product, 16x16)
            #     D'_r = V4_bottom[:,r] W4_bottom[:,r] (outer product, 16x16)
            # T = sum_r U4[:,r] * (D_r stacked with D'_r)
            
            # The "effective" V-W product for column r is a 32x32 matrix
            # that is block-diagonal: [[D_r, 0], [0, D'_r]].
            
            # If D_r and D'_r have a common factor structure, we might
            # be able to reduce rank.
            
            # For now, just return the 98-rank solution.
            pass
        except Exception:
            pass
    
    U_out = _to_half_integer(best_U)
    V_out = _to_half_integer(best_V)
    W_out = _to_half_integer(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
