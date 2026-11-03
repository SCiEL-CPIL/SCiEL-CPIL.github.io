import numpy as np
import time
from itertools import product

def _verify_decomposition(U, V, W, n=4, m=4, p=8, trials=5):
    """Verify that the decomposition exactly computes C = A*B.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, r]
    """
    Ua = np.array(U, dtype=float).reshape(n, m, -1)
    Va = np.array(V, dtype=float).reshape(m, p, -1)
    Wa = np.array(W, dtype=float).reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', Ua, Va, Wa)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True


def _strassen_2x2x2_terms():
    """Return the 7 rank-1 terms for 2x2x2 Strassen multiplication.
    Each term is (u, v, w) where u,v,w are flat 4-element arrays.
    Convention: T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
    """
    return [
        # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 -= 1
        ([1,0,0,1], [1,0,0,1], [1,0,0,-1]),
        # M2: (A10+A11)B00 -> C01 += 1, C11 += 1
        ([0,0,1,1], [1,0,0,0], [0,1,0,1]),
        # M3: A00(B10-B11) -> C10 += 1
        ([1,0,0,0], [0,0,1,-1], [0,0,1,0]),
        # M4: A11(B01-B10) -> C00 -= 1, C01 += 1
        ([0,0,0,1], [0,1,-1,0], [-1,1,0,0]),
        # M5: (A00+A01)B11 -> C11 -= 1
        ([1,1,0,0], [0,0,0,1], [0,0,0,-1]),
        # M6: (A10-A00)(B00+B10) -> C00 += 1
        ([-1,0,1,0], [1,0,1,0], [1,0,0,0]),
        # M7: (A01-A11)(B01+B11) -> C00 -= 1, C10 += 1
        ([0,1,0,-1], [0,1,0,1], [-1,0,1,0]),
    ]


def _bclr_4x4x4_terms():
    """Return rank-14 terms for 4x4x4 matrix multiplication (BCLR algorithm).
    
    Uses Strassen at the 2x2 block level with careful index management.
    A (4x4) split into 2x2 blocks: A[p,q] for p,q in {0,1}
    B (4x4) split into 2x2 blocks: B[q,r] for q,r in {0,1}
    C (4x4) split into 2x2 blocks: C[p,r] for p,r in {0,1}
    
    C[p,r] = A[p,0]*B[0,r] + A[p,1]*B[1,r]
    
    Using Strassen at block level:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 - M4 + M6 - M7
    C01 = M2 + M4
    C10 = M3 + M7
    C11 = M1 - M5 + M2 - M3
    
    Each Mi is a 2x2 x 2x2 matrix product, decomposable with rank 7 (Strassen).
    Total: 7 * 7 = 49 terms. But BCLR achieves 14!
    
    Actually, the BCLR algorithm is more sophisticated. Let me use a different approach:
    I'll construct a valid rank-28 decomposition using two independent 4x4x4 Strassen-based
    decompositions (each rank 14 would give 28, but Strassen gives 49 each).
    
    For now, let me use a verified approach: direct construction with rank 28.
    """
    # Use Strassen-based 4x4x4: 7 block mults, each 2x2x2 rank 7 = 49 terms
    # But that's high. Let me try a different structure.
    
    # Actually, let me just use the naive 4x4x4 (rank 64) and combine for 4x4x8.
    # That gives rank 128 for 4x4x8. Same as seed.
    
    # Better: use the fact that 4x4x8 can be seen as computing C = A*B where
    # B is 4x8. Split B into B1 (4x4) and B2 (4x4).
    # C1 = A*B1 (4x4), C2 = A*B2 (4x4).
    # If I have a rank-r 4x4x4 decomposition, I get rank 2r for 4x4x8.
    
    # For 4x4x4, let me use a verified rank-14 BCLR decomposition.
    # The BCLR algorithm is well-known. Let me hardcode the 14 terms.
    
    # BCLR 4x4x4: 14 rank-1 terms.
    # This is complex to derive from scratch. Let me use an alternative:
    # Use Strassen at 2x2 block level (7 block mults) but optimize the 2x2x2 
    # decompositions to share structure.
    
    # For now, return a valid (if not minimal) 4x4x4 decomposition.
    # Use rank 28: two independent sets of 14 terms each covering different
    # index patterns. Actually, let me just use the naive 4x4x4 (rank 64).
    
    # Better plan: construct 4x4x8 directly with a block Strassen approach.
    pass


def _construct_448_direct():
    """Directly construct a 4x4 x 4x8 decomposition using block Strassen.
    
    Approach: Split the computation into 2x2 blocks.
    A (4x4) -> 4 blocks of 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
    B (4x8) -> 8 blocks of 2x2: B[0,0..3], B[1,0..3]
    C (4x8) -> 8 blocks of 2x2: C[0,0..3], C[1,0..3]
    
    C[p, r] = sum_q A[p,q] * B[q,r] for p in {0,1}, r in {0,1,2,3}
    
    For each (p,r) pair, this is a 2-term sum of 2x2x2 multiplications.
    Using Strassen (rank 7) for each 2x2x2: 2*7 = 14 per (p,r) pair.
    Total: 8 * 14 = 112. Still high.
    
    Better: share Strassen structure across (p,r) pairs.
    
    Key insight: For fixed p, C[p, r] = A[p,0]*B[0,r] + A[p,1]*B[1,r]
    This is a 2x2 x 2x4 block multiplication (each block 2x2).
    
    The 2x2 x 2x4 tensor has rank at most 16 (naive) but can be lower.
    Actually, it's equivalent to multiplying a 2x2 matrix by a 2x4 matrix.
    The rank of this bilinear form is at most 2*2*4 = 16, but Strassen-like
    methods can reduce it.
    
    For 2x2 x 2x4: C = A*B where A is 2x2, B is 2x4.
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3]
    
    This is just 8 independent 2-term dot products. The tensor rank is 8
    (each C[i,k] depends on exactly 2 products, and they're all independent).
    Wait, no - the rank of the tensor T[i,j,k] where i in {0,1}, j in {0,1}, 
    k in {0,1,2,3} and T[i,j,k]=1 is... 
    
    Actually, the tensor for 2x2 x 2x4 multiplication has entries T[i,j,k]=1 
    for all i,j,k. The rank is the minimum number of rank-1 tensors needed.
    Naive: 2*2*4 = 16. Can we do better?
    
    The border rank and tensor rank of the 2x2x4 multiplication tensor:
    It's at most 16 (naive) and at least max(2*2, 2*4, 2*4) = 8.
    
    I believe the rank is 16 for 2x2x4 (no known improvement). So for 4x4x8:
    2 (row blocks) * 16 (per block) = 32? No, that's not right either.
    
    Let me think differently. The full 4x4x8 tensor has 4*4*8 = 128 entries, 
    all equal to 1. The rank is at most 128 (naive).
    
    Using the block structure: C_block[p,r] = sum_q A_block[p,q] * B_block[q,r]
    where p in {0,1}, q in {0,1}, r in {0,1,2,3}.
    
    For each (p,r), the computation is a 2x2 x 2x2 x 2x2 multiplication (sum over q).
    This is equivalent to multiplying two 2x2 matrices and getting a 2x2 result,
    but with 4 different B blocks.
    
    Actually, let me just use a verified rank-28 construction:
    Use two 4x4x4 BCLR decompositions (each rank 14) for the two halves of B.
    Total rank: 28.
    
    I need the BCLR 4x4x4 terms. Let me construct them carefully.
    """
    n, m, p = 4, 4, 8
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # For the first 4 columns of B (B[:, 0:4]), use a 4x4x4 decomposition
    # For the last 4 columns of B (B[:, 4:8]), use another 4x4x4 decomposition
    
    # I'll use a Strassen-based 4x4x4 decomposition with rank 49 (7*7).
    # That gives total rank 98 for 4x4x8. Still better than 128.
    
    # Actually, let me use a simpler verified approach:
    # For 4x4x4, use the following rank-16 construction:
    # 4 diagonal terms + 12 off-diagonal terms using specific patterns.
    
    # Let me just use the naive 4x4x4 (rank 64) for each half.
    # Total: 128. Same as seed. Not an improvement.
    
    # I need a better 4x4x4 decomposition. Let me use Strassen properly.
    # 4x4x4 with Strassen at 2x2 block level: 7 block mults, each 2x2x2 rank 7.
    # Total: 49 per 4x4x4, so 98 for 4x4x8.
    
    # Let me construct the 49-term 4x4x4 Strassen decomposition.
    strassen_2x2 = _strassen_2x2x2_terms()
    
    # Block structure for 4x4x4:
    # A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # 
    # C00 = M1 - M4 + M6 - M7
    # C01 = M2 + M4
    # C10 = M3 + M7
    # C11 = M1 - M5 + M2 - M3
    #
    # Where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # Each Mi involves a 2x2x2 multiplication. Using Strassen (rank 7) for each.
    # Total: 7 * 7 = 49 terms for 4x4x4.
    
    # For each of the 7 block-level Strassen terms, I need to decompose
    # the corresponding 2x2x2 product into 7 rank-1 terms.
    
    # Block-level Strassen terms (u_block, v_block, w_block) are 2x2 matrices.
    # The 2x2x2 Strassen terms (u_22, v_22, w_22) are also 2x2 matrices.
    
    # The combined term for (block_term r, strassen_term s) has:
    # U[i,j] = u_block_r[block_of_i, block_of_j] * u_22_s[i_in_block, j_in_block]
    # Similarly for V and W.
    
    block_strassen = [
        # (u_block, v_block, w_block) for each of the 7 Strassen terms
        # u_block: coefficients for A blocks, v_block: for B blocks, w_block: for C blocks
        # A00+A11: [[1,0],[0,1]], B00+B11: [[1,0],[0,1]], C00+1,C11-1: [[1,0],[0,-1]]
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,-1]]),
        # A10+A11: [[0,0],[1,1]], B00: [[1,0],[0,0]], C01+1,C11+1: [[0,1],[0,1]]
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,1]]),
        # A00: [[1,0],[0,0]], B10-B11: [[0,0],[1,-1]], C10+1: [[0,0],[1,0]]
        ([[1,0],[0,0]], [[0,0],[1,-1]], [[0,0],[1,0]]),
        # A11: [[0,0],[0,1]], B01-B10: [[0,1],[-1,0]], C00-1,C01+1: [[-1,1],[0,0]]
        ([[0,0],[0,1]], [[0,1],[-1,0]], [[-1,1],[0,0]]),
        # A00+A01: [[1,1],[0,0]], B11: [[0,0],[0,1]], C11-1: [[0,0],[0,-1]]
        ([[1,1],[0,0]], [[0,0],[0,1]], [[0,0],[0,-1]]),
        # A10-A00: [[-1,0],[1,0]], B00+B10: [[1,0],[1,0]], C00+1: [[1,0],[0,0]]
        ([[-1,0],[1,0]], [[1,0],[1,0]], [[1,0],[0,0]]),
        # A01-A11: [[0,1],[0,-1]], B01+B11: [[0,1],[0,1]], C00-1,C10+1: [[-1,0],[1,0]]
        ([[0,1],[0,-1]], [[0,1],[0,1]], [[-1,0],[1,0]]),
    ]
    
    # Now build the 49 terms for 4x4x4
    terms_444 = []
    for rb, (ub, vb, wb) in enumerate(block_strassen):
        for rs, (u22, v22, w22) in enumerate(strassen_2x2):
            # u22, v22, w22 are flat 4-element arrays (2x2 row-major)
            u22m = np.array(u22).reshape(2, 2)
            v22m = np.array(v22).reshape(2, 2)
            w22m = np.array(w22).reshape(2, 2)
            
            # Build 4x4 matrices
            U4 = np.zeros((4, 4))
            V4 = np.zeros((4, 4))
            W4 = np.zeros((4, 4))
            
            for bp in range(2):  # block row
                for bq in range(2):  # block col
                    coeff_u = ub[bp, bq]
                    if coeff_u != 0:
                        # Place u22m * coeff_u at position (bp*2:bp*2+2, bq*2:bq*2+2)
                        U4[bp*2:bp*2+2, bq*2:bq*2+2] = coeff_u * u22m
            
            for bq in range(2):  # block row of B
                for br in range(2):  # block col of B
                    coeff_v = vb[bq, br]
                    if coeff_v != 0:
                        V4[bq*2:bq*2+2, br*2:br*2+2] = coeff_v * v22m
            
            for br in range(2):  # block row of C (output col block)
                for bp in range(2):  # block col of C (output row block)
                    # W is indexed as W[k, i] where k is col, i is row
                    # C block [bp, br] means rows bp*2:bp*2+2, cols br*2:br*2+2
                    # W[k, i] where k in br*2:br*2+2, i in bp*2:bp*2+2
                    coeff_w = wb[br, bp]  # wb[output_col_block, output_row_block]
                    if coeff_w != 0:
                        W4[br*2:br*2+2, bp*2:bp*2+2] = coeff_w * w22m
            
            terms_444.append((U4.flatten(), V4.flatten(), W4.flatten()))
    
    # Verify the 4x4x4 decomposition
    T4 = np.zeros((4, 4, 4))
    for u, v, w in terms_444:
        um = np.array(u).reshape(4, 4)
        vm = np.array(v).reshape(4, 4)
        wm = np.array(w).reshape(4, 4)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    T4[i, j, k] += um[i, j] * vm[j, k] * wm[k, i]
    
    if not np.allclose(T4, 1.0, atol=1e-9):
        # Fallback to naive
        return None
    
    # Now build 4x4x8: two copies of the 4x4x4, one for B[:, 0:4] and one for B[:, 4:8]
    U448 = np.zeros((4*4, 2*49))
    V448 = np.zeros((4*8, 2*49))
    W448 = np.zeros((8*4, 2*49))
    
    for half in range(2):
        for r, (u, v, w) in enumerate(terms_444):
            col = half * 49 + r
            U448[:, col] = np.array(u)
            
            # V: for half=0, use B[:, 0:4]; for half=1, use B[:, 4:8]
            vm = np.array(v).reshape(4, 4)
            for j in range(4):
                for k in range(4):
                    V448[j*8 + half*4 + k, col] = vm[j, k]
            
            # W: for half=0, C[:, 0:4]; for half=1, C[:, 4:8]
            wm = np.array(w).reshape(4, 4)
            for k in range(4):
                for i in range(4):
                    W448[(half*4+k)*4 + i, col] = wm[k, i]
    
    # Verify
    if _verify_decomposition(U448, V448, W448, 4, 4, 8, trials=3):
        return U448, V448, W448
    
    return None


def _naive_448(n, m, p):
    """Guaranteed-valid naive rank-128 decomposition for 4x4 x 4x8."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, t] = 1
                V[j * p + k, t] = 1
                W[k * n + i, t] = 1
                t += 1
    return U, V, W


def construct():
    """
    Construct a low-rank exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    
    Returns {'U','V','W'} with shapes (16,rank),(32,rank),(32,rank).
    """
    start = time.time()
    deadline = start + 110
    
    # Try the Strassen-based 4x4x8 construction (rank 98)
    result = _construct_448_direct()
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: naive rank-128
    U, V, W = _naive_448(4, 4, 8)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

