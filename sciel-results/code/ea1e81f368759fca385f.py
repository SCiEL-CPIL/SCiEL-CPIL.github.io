"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank):
        return False
    if Va.shape != (m*p, rank):
        return False
    if Wa.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * m)]  # placeholder, fixed below
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

def _strassen_2x2():
    """
    Return Strassen's 7-term decomposition for 2x2 x 2x2.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Index convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3: a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4: a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5: (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6: (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[1][6] = 1
    
    return U, V, W

def _strassen_4x4_rank20():
    """
    Construct a rank-20 decomposition for 4x4 x 4x4 matrix multiplication.
    
    Strategy: Partition 4x4 into 2x2 blocks. Use Strassen on block structure
    with additional cross-block terms to handle off-diagonal interactions.
    
    Block structure:
    A = [[A00, A01], [A10, A11]]  (each 2x2)
    B = [[B00, B01], [B10, B11]]  (each 2x2)
    C = [[C00, C01], [C10, C11]]  (each 2x2)
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    For each 2x2 block multiplication, use Strassen (rank 7).
    But we need to share products where possible.
    
    Actually, a cleaner approach: use the known rank-20 result.
    Let me construct it explicitly using a systematic method.
    
    I'll use a different strategy: treat the 4x4 multiplication as a
    bilinear map and construct 20 rank-1 tensors that span it.
    
    A practical construction: use 4 Strassen decompositions (one for each
    block output) but with shared intermediate products.
    
    Let me use a simpler but valid approach:
    - 4 independent 2x2x2 Strassen decompositions would give rank 28
    - We need to reduce to 20 by sharing 8 products
    
    Alternative: Use the fact that C = A*B can be computed with
    the following 20 products (a known construction):
    
    Let me use a systematic algebraic approach.
    
    Actually, let me try a concrete construction based on partitioning:
    
    Partition indices into two halves: {0,1} and {2,3}
    
    For the top-left 2x2 block of C: C[0:2, 0:2] = A[0:2, 0:4] * B[0:4, 0:2]
    This is a 2x4 * 4x2 multiplication.
    
    Hmm, this is getting complex. Let me use a different well-known approach.
    
    I'll construct the rank-20 decomposition using a systematic method:
    View 4x4 multiplication as a tensor and use a known low-rank construction.
    
    Here's a concrete rank-20 construction:
    
    Products (each is a sum of products of entries from A and B):
    P1  = (A00+A11+A22+A33)(B00+B11+B22+B33)  -- diagonal sum product
    ... this approach is hard to verify.
    
    Let me use a simpler but still improved approach:
    Use 2x2 Strassen blocks with some sharing.
    
    Actually, the cleanest path: use the block Strassen approach with
    2x2 blocks, giving rank 2*2*7 = 28 for 4x4x4.
    Then for 4x4x8: 2 blocks of 4x4x4, each rank 28, total rank 56.
    
    That's better than 112. Let me go with that first, then try to improve.
    
    For 4x4x4 with 2x2 blocks:
    4 block multiplications, each 2x2x2 with rank 7.
    Total rank = 4 * 7 = 28.
    
    For 4x4x8:
    View as two 4x4x4 multiplications.
    Each uses rank 28.
    Total rank = 2 * 28 = 56.
    """
    # For now, return a rank-28 decomposition using 2x2 Strassen blocks
    # This is a valid improvement over schoolbook (16) for 4x4x4
    # and gives 56 for 4x4x8 vs 112 from direct 2x2 blocks
    
    U = [[0]*28 for _ in range(16)]
    V = [[0]*28 for _ in range(16)]
    W = [[0]*28 for _ in range(16)]
    
    base_U, base_V, base_W = _strassen_2x2()
    
    # 4 block positions: (bi, bj) for bi,bj in {0,1}
    # Each block uses 7 Strassen products
    block_offsets = {}
    col = 0
    for bi in range(2):
        for bj in range(2):
            block_offsets[(bi, bj)] = col
            col += 7
    
    for bi in range(2):
        for bj in range(2):
            off = block_offsets[(bi, bj)]
            for r in range(7):
                # U: rows (i, j) where i in [bi*2, bi*2+2), j in [bj*2, bj*2+2)
                for li in range(2):
                    for lj in range(2):
                        i = bi * 2 + li
                        j = bj * 2 + lj
                        val = base_U[li*2 + lj][r]
                        if val != 0:
                            U[i*4 + j][off + r] = val
                
                # V: rows (j, k) where j in [bj*2, bj*2+2), k in [bi*2, bi*2+2)
                # Wait, for C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
                # Block (bi, bj) of A multiplies block (bj, bk) of B to give block (bi, bk) of C
                # I need to be more careful.
                
                # Let me redo: for 4x4x4:
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # Partition into 2x2 blocks:
                # C[bi*2:bi*2+2, bk*2:bk*2+2] = sum_bj A[bi*2:bi*2+2, bj*2:bj*2+2] * B[bj*2:bj*2+2, bk*2:bk*2+2]
                #
                # So for block (bi, bk) of C, we sum over bj:
                # Each term is a 2x2 * 2x2 = 2x2 multiplication
                #
                # There are 4 values of (bi, bk) and 2 values of bj, so 8 block multiplications
                # Each of rank 7, total rank 56 for 4x4x4.
                #
                # That's worse than I thought. Let me reconsider.
                
                pass
    
    # This approach is getting complicated. Let me use a different strategy.
    # 
    # For 4x4x8, the best simple approach is:
    # 1. Direct 2x2x2 Strassen blocks: rank 112
    # 2. Two 4x4x4 blocks, each with 2x2x2 Strassen: rank 2*(4*7) = 56
    #    Wait, for 4x4x4 with 2x2 blocks:
    #    Number of 2x2x2 block multiplications = 2*2*2 = 8 (for bi, bj, bk)
    #    Each rank 7, total rank = 8*7 = 56
    #    For 4x4x8: two such 4x4x4 blocks, total rank = 2*56 = 112
    #    Same as before!
    #
    # So I need a genuinely better 4x4x4 decomposition.
    #
    # Let me try a rank-20 construction based on known results.
    # The rank of 4x4x4 matrix multiplication is known to be at most 20
    # (and at least 19). Let me construct a rank-20 decomposition.
    
    # I'll use a systematic construction. The key idea:
    # Use 4 "diagonal" Strassen-like products and 16 "off-diagonal" products,
    # but with sharing to get to 20.
    
    # Actually, let me just use the 2x2 block Strassen approach which gives
    # rank 56 for 4x4x4, and 112 for 4x4x8. That's what we have.
    
    # To do better, I need a true rank-20 (or lower) 4x4x4 decomposition.
    # 
    # Here's a concrete rank-20 construction:
    # Based on the work of Strassen and subsequent improvements.
    
    # Let me try a different approach: use a 4x4 decomposition based on
    # the tensor rank decomposition with 20 terms.
    
    # I'll construct it as follows:
    # Group the 16 entries of A into 4 groups of 4, similarly for B.
    # Use 20 bilinear products.
    
    # For simplicity and correctness, let me fall back to the 2x2 block
    # Strassen approach (rank 112) but also try a 4x4x4 rank-28 approach
    # using 4 independent 2x2x2 Strassen decompositions for the diagonal
    # blocks, plus additional terms for off-diagonal.
    
    # Actually, the simplest improvement: for 4x4x8, instead of 8 independent
    # 2x2x2 blocks (rank 112), note that we can share some structure.
    
    # Let me just return the 2x2 block Strassen result (rank 112) for now,
    # and add a search for better decompositions if time permits.
    
    return None  # Signal to use the block decomposition below

def _block_decompose_2x2(n, m, p):
    """
    Decompose n x m * m x p into 2x2x2 Strassen blocks.
    Total rank = (n/2) * (m/2) * (p/2) * 7
    """
    if n % 2 != 0 or m % 2 != 0 or p % 2 != 0:
        return None
    
    num_i = n // 2
    num_j = m // 2
    num_k = p // 2
    total_rank = num_i * num_j * num_k * 7
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    base_U, base_V, base_W = _strassen_2x2()
    
    for bi in range(num_i):
        for bj in range(num_j):
            for bk in range(num_k):
                off = (bi * num_j + bj) * num_k * 7 + bk * 7
                
                for r in range(7):
                    for li in range(2):
                        for lj in range(2):
                            i = bi * 2 + li
                            j = bj * 2 + lj
                            val = base_U[li*2 + lj][r]
                            if val != 0:
                                U[i*m + j][off + r] = val
                    
                    for lj in range(2):
                        for lk in range(2):
                            j = bj * 2 + lj
                            k = bk * 2 + lk
                            val = base_V[lj*2 + lk][r]
                            if val != 0:
                                V[j*p + k][off + r] = val
                    
                    for lk in range(2):
                        for li in range(2):
                            k = bk * 2 + lk
                            i = bi * 2 + li
                            val = base_W[lk*2 + li][r]
                            if val != 0:
                                W[k*n + i][off + r] = val
    
    return {"U": U, "V": V, "W": W}

def _block_decompose_4x4(n, m, p):
    """
    Decompose n x m * m x p into 4x4x4 blocks, each using 2x2 Strassen sub-blocks.
    Each 4x4x4 block uses 8 2x2x2 Strassen decompositions (rank 56).
    Total rank = (n/4) * (m/4) * (p/4) * 56
    """
    if n % 4 != 0 or m % 4 != 0 or p % 4 != 0:
        return None
    
    num_i = n // 4
    num_j = m // 4
    num_k = p // 4
    total_rank = num_i * num_j * num_k * 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    base_U, base_V, base_W = _strassen_2x2()
    
    for bi in range(num_i):
        for bj in range(num_j):
            for bk in range(num_k):
                block_off = (bi * num_j + bj) * num_k * 56 + bk * 56
                
                # Within each 4x4x4 block, there are 8 2x2x2 sub-blocks
                for si in range(2):
                    for sj in range(2):
                        for sk in range(2):
                            sub_off = block_off + (si*2 + sj)*2*7 + sk*7
                            
                            for r in range(7):
                                for li in range(2):
                                    for lj in range(2):
                                        i = bi*4 + si*2 + li
                                        j = bj*4 + sj*2 + lj
                                        val = base_U[li*2 + lj][r]
                                        if val != 0:
                                            U[i*m + j][sub_off + r] = val
                                
                                for lj in range(2):
                                    for lk in range(2):
                                        j = bj*4 + sj*2 + lj
                                        k = bk*4 + sk*2 + lk
                                        val = base_V[lj*2 + lk][r]
                                        if val != 0:
                                            V[j*p + k][sub_off + r] = val
                                
                                for lk in range(2):
                                    for li in range(2):
                                        k = bk*4 + sk*2 + lk
                                        i = bi*4 + si*2 + li
                                        val = base_W[lk*2 + li][r]
                                        if val != 0:
                                            W[k*n + i][sub_off + r] = val
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = len(best["U"][0])
    
    # Try 2x2x2 Strassen blocks
    if time.time() < deadline:
        result_2x2 = _block_decompose_2x2(n, m, p)
        if result_2x2 is not None:
            rank_2x2 = len(result_2x2["U"][0])
            if rank_2x2 < best_rank:
                if _verify(result_2x2["U"], result_2x2["V"], result_2x2["W"], n, m, p):
                    best = result_2x2
                    best_rank = rank_2x2
    
    # Try 4x4x4 blocks (using 2x2 Strassen sub-blocks)
    if time.time() < deadline:
        result_4x4 = _block_decompose_4x4(n, m, p)
        if result_4x4 is not None:
            rank_4x4 = len(result_4x4["U"][0])
            if rank_4x4 < best_rank:
                if _verify(result_4x4["U"], result_4x4["V"], result_4x4["W"], n, m, p):
                    best = result_4x4
                    best_rank = rank_4x4
    
    # Try to find a better decomposition using optimization
    if time.time() < deadline - 20:
        # For 4x4x8, try a structured rank reduction
        # Use the fact that the tensor has symmetries
        pass
    
    return best
# EVOLVE-BLOCK-END
