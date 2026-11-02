"""Block-structured low-rank decomposition for 4x5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # 5s safety margin before 150s kill

def _strassen_4x4x4_terms():
    """
    Generate 49 rank-1 terms for 4x4x4 matrix multiplication using Strassen's algorithm.
    A(4x4), B(4x4), C(4x4) where C = A*B.
    U indices: (i,j) -> i*4+j, i,j in 0..3
    V indices: (j,k) -> j*4+k, j,k in 0..3
    W indices: (k,i) -> k*4+i, k,i in 0..3
    """
    terms = []
    # Split into 2x2 blocks:
    # A = [[A00,A01],[A10,A11]] where Aab is 2x2
    # B = [[B00,B01],[B10,B11]]
    # C = [[C00,C01],[C10,C11]]
    # Strassen: 7 products M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2x2 multiplication, which we decompose with Winograd (rank 7)
    winograd_222 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [-1,0,1,0], [-1,1,1,0]),
        ([1,1,0,0], [0,0,1,0], [1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,-1,1]),
        ([0,1,0,-1], [0,1,0,1], [-1,1,1,0]),
    ]
    
    # Helper: for a 2x2x2 product with coefficient pattern, generate 7 terms
    # A_block indices: (bi, bj) where bi,bj in 0..1 -> flat bi*2+bj
    # B_block indices: (bj, bk) where bj,bk in 0..1 -> flat bj*2+bk
    # C_block indices: (bk, bi) where bk,bi in 0..1 -> flat bk*2+bi
    
    def add_strassen_222(a_coeffs, b_coeffs, c_coeffs, scale=1):
        """Add 7 Winograd terms for a 2x2x2 product with given coefficient patterns.
        a_coeffs: 4x4 matrix (2x2 blocks of 2x2) for A side
        b_coeffs: 4x4 matrix for B side  
        c_coeffs: 4x4 matrix for C side
        Actually, let's use a different approach: specify which 2x2 sub-blocks are involved.
        """
        pass
    
    # Better approach: directly enumerate the 49 terms
    # For each of the 7 Strassen products, we have 7 Winograd terms = 49 total
    
    # Define the 7 Strassen combinations
    # Each combination specifies: A_linear_combo, B_linear_combo, C_linear_combo
    # A_linear_combo: 4x4 matrix showing which A elements are used (with signs)
    # B_linear_combo: 4x4 matrix showing which B elements are used
    # C_linear_combo: 4x4 matrix showing which C elements receive the product (with signs)
    
    strassen_combos = [
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        # A: A00 and A11 contribute with +1
        # B: B00 and B11 contribute with +1
        # C: C00 += M1, C11 += M1
        ({(0,0):1,(0,1):0,(1,0):0,(1,1):1}, {(0,0):1,(0,1):0,(1,0):0,(1,1):1}, {(0,0):1,(0,1):0,(1,0):0,(1,1):1}),
        # M2 = (A10+A11)B00 -> C10 += 1, C11 += -1
        ({(0,0):0,(0,1):0,(1,0):1,(1,1):1}, {(0,0):1,(0,1):0,(1,0):0,(1,1):0}, {(0,0):0,(0,1):0,(1,0):1,(1,1):-1}),
        # M3 = A00(B10-B11) -> C01 += 1, C11 += 1
        ({(0,0):1,(0,1):0,(1,0):0,(1,1):0}, {(0,0):0,(0,1):0,(1,0):1,(1,1):-1}, {(0,0):0,(0,1):1,(1,0):0,(1,1):1}),
        # M4 = A11(B01-B10) -> C00 += -1, C01 += 1
        ({(0,0):0,(0,1):0,(1,0):0,(1,1):1}, {(0,0):0,(0,1):1,(1,0):-1,(1,1):0}, {(0,0):-1,(0,1):1,(1,0):0,(1,1):0}),
        # M5 = (A00+A01)B11 -> C00 += 1, C10 += 1
        ({(0,0):1,(0,1):1,(1,0):0,(1,1):0}, {(0,0):0,(0,1):0,(1,0):0,(1,1):1}, {(0,0):1,(0,1):0,(1,0):1,(1,1):0}),
        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        ({(0,0):-1,(0,1):0,(1,0):1,(1,1):0}, {(0,0):1,(0,1):1,(1,0):0,(1,1):0}, {(0,0):0,(0,1):0,(1,0):0,(1,1):1}),
        # M7 = (A01-A11)(B10+B11) -> C00 += 1
        ({(0,0):0,(0,1):1,(1,0):0,(1,1):-1}, {(0,0):0,(0,1):0,(1,0):1,(1,1):1}, {(0,0):1,(0,1):0,(1,0):0,(1,1):0}),
    ]
    
    for a_combo, b_combo, c_combo in strassen_combos:
        # For this Strassen product, we need to multiply two 2x2 "super-matrices"
        # where each entry is a 2x2 block
        # The Winograd 2x2x2 operates on the 4 scalar elements of each 2x2 block
        # But here, the "2x2" is in the block sense, so we need to be careful
        
        # Actually, let me think of it differently. The 4x4x4 Strassen decomposition
        # can be seen as: we have 8 2x2 sub-blocks in A, 8 in B, 8 in C.
        # Each Strassen product M_i is a sum of products of 2x2 blocks.
        # Each such 2x2 block product can be done with 7 multiplications (Winograd).
        # But the blocks are shared, so we need to combine them.
        
        # This is getting complicated. Let me use a direct approach:
        # Generate all 49 terms by brute force using the known 4x4x4 Strassen decomposition.
        
        pass
    
    # Alternative: use the fact that 4x4x4 Strassen = 7 products of 2x2x2,
    # but each "2x2x2" here means multiplying two 2x2 matrices to get a 2x2 result.
    # So we have 7 independent 2x2x2 multiplications, each with rank 7, total 49.
    # But they're not independent - they share A and B blocks.
    
    # Let me just return None for now and use the block method instead.
    return None

def remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W, "rank": rank}

def _verify(U, V, W, rank):
    """Verify exact decomposition using random integer matrices."""
    import random
    n, m, p = DIMENSIONS
    for _ in range(5):
        if remaining() <= 0:
            return False
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    for c in range(rank):
                        u_val = U[i * m + j][c]
                        if u_val == 0:
                            continue
                        acc += u_val * W[k * n + i][c] * V[j * p + k][c]
                if acc != C[i][k]:
                    return False
    return True

def _winograd_2x2x2():
    """Returns 7 rank-1 terms for 2x2x2 matmul."""
    # Standard Winograd F^T(2,2,2) with 7 multiplications
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C00 = p1 - p4 + p5 - p7
    # C01 = p3 + p4 + p5 + p7
    # C10 = p2 + p4 - p6 + p7
    # C11 = p1 - p2 + p6 - p7
    terms = [
        # (U_vec, V_vec, W_vec) where U_vec is flat [a,b,c,d], V_vec is [e,f,g,h], W_vec is [C00,C01,C10,C11]
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),    # p1 = (a+d)(e+h)
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # p2 = (c+d)e
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),   # p3 = a(f-h)
        ([0,0,0,1], [-1,0,1,0], [-1,1,1,0]),  # p4 = d(g-e)
        ([1,1,0,0], [0,0,1,0], [1,1,0,0]),    # p5 = (a+b)g
        ([-1,0,1,0], [1,1,0,0], [0,0,-1,1]),  # p6 = (c-a)(e+f)
        ([0,1,0,-1], [0,1,0,1], [-1,1,1,0]),  # p7 = (b-d)(h+f)
    ]
    return terms

def _strassen_4x4x4():
    """
    Generate 49 rank-1 terms for 4x4x4 matrix multiplication using Strassen's algorithm.
    U indices: (i,j) -> i*4+j, i,j in 0..3
    V indices: (j,k) -> j*4+k, j,k in 0..3  
    W indices: (k,i) -> k*4+i, k,i in 0..3
    """
    winograd_222 = _winograd_2x2x2()
    
    # Strassen's 7 products for 4x4x4:
    # Split A into 4 2x2 blocks: A00(top-left), A01(top-right), A10(bottom-left), A11(bottom-right)
    # Similarly for B and C.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    terms = []
    
    # For each Strassen product, we have a 2x2x2 multiplication where:
    # The "A" side is a linear combination of 2x2 blocks
    # The "B" side is a linear combination of 2x2 blocks
    # The "C" side contributes to specific 2x2 blocks of the output
    
    # We'll enumerate by: for each Strassen combo, for each Winograd term,
    # expand the block-level operations to scalar-level
    
    # Block positions in 4x4:
    # Block (bi,bj) covers rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # Within a block, element (a,b) is at global position (2*bi+a, 2*bj+b)
    
    def block_elem(block_i, block_j, local_i, local_j):
        """Get global (i,j) index for element in block (block_i,block_j) at local (local_i,local_j)."""
        return (2*block_i + local_i, 2*block_j + local_j)
    
    # For each of the 7 Strassen products, define:
    # a_blocks: list of (block_i, block_j, coeff) for A
    # b_blocks: list of (block_i, block_j, coeff) for B  
    # c_blocks: list of (block_i, block_j, coeff) for C
    
    strassen_defs = [
        # M1 = (A00+A11)(B00+B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)B00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = A00(B10-B11)
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = A11(B01-B10)
        ([(1,1,1)], [(0,1,1),(1,0,-1)], [(0,0,-1),(0,1,1)]),
        # M5 = (A00+A01)B11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(1,0,1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    for a_blocks, b_blocks, c_blocks in strassen_defs:
        # For this Strassen product, we need to compute:
        # sum_{(ai,aj,ca) in a_blocks} sum_{(bi,bj,cb) in b_blocks} ca*cb * A[ai][aj] * B[bi][bj]
        # and add it to sum_{(ci,cj,cc) in c_blocks} cc * C[ci][cj]
        
        # This is a 2x2x2 multiplication where:
        # A_side has contributions from multiple 2x2 blocks
        # B_side has contributions from multiple 2x2 blocks
        # C_side receives contributions to multiple 2x2 blocks
        
        # The key insight: this is equivalent to multiplying two "super" 2x2 matrices
        # where each entry is itself a 2x2 block. We can use Winograd on the block level,
        # but that would give 7 * 7 = 49 terms per Strassen product, which is too many.
        
        # Actually, no. The standard Strassen for 4x4x4 uses 49 total multiplications,
        # not 49 per product. Let me reconsider.
        
        # The correct interpretation: Strassen's algorithm for n=4 recursively applies
        # Strassen for n=2. So we have 7 "multiplications" of 2x2 matrices, each of
        # which can be done with 7 scalar multiplications (Winograd), giving 49 total.
        # But the 2x2 matrices being multiplied are linear combinations of the original
        # 2x2 blocks, so we need to track how the scalar multiplications map back.
        
        # Let me use a different approach: directly construct the 49 terms by
        # considering that each Strassen "product" M_i is a 2x2 matrix product,
        # and each such product uses 7 Winograd terms.
        
        # For M_i = X * Y where X and Y are 2x2 matrices (each entry is a 2x2 block),
        # the Winograd decomposition gives 7 terms. Each term involves products of
        # specific entries of X and Y, which are linear combinations of original blocks.
        
        # This is getting very complex. Let me use a simpler approach:
        # Just use 2x2x2 Winograd blocks to tile the 4x4x4, giving 8 blocks * 7 = 56 terms.
        # That's worse than 49, but let's see if we can do better with a hybrid.
        
        pass
    
    # For now, let's just return None and use the block tiling approach.
    # The 4x4x4 tiled with 2x2x2 Winograd gives 8*7 = 56 terms.
    # Combined with the rest: 56 + 28 + 24 + 8 = 116, which is worse than 108.
    
    return None

def _embed_block(terms, i_base, j_base, k_base, i_size, j_size, k_size, n, m, p):
    """Embed a block's terms into full-sized vectors."""
    full_terms = []
    for (u, v, w) in terms:
        ufull = [0] * (n * m)
        vfull = [0] * (m * p)
        wfull = [0] * (p * n)
        
        for a in range(i_size):
            for b in range(j_size):
                idx = (i_base + a) * m + (j_base + b)
                ufull[idx] = u[a * j_size + b]
        
        for b in range(j_size):
            for c in range(k_size):
                idx = (j_base + b) * p + (k_base + c)
                vfull[idx] = v[b * k_size + c]
        
        for c in range(k_size):
            for a in range(i_size):
                idx = (k_base + c) * n + (i_base + a)
                wfull[idx] = w[c * i_size + a]
        
        full_terms.append((ufull, vfull, wfull))
    return full_terms

def _build_from_terms(terms_U, terms_V, terms_W, n, m, p):
    """Build U, V, W matrices from lists of term vectors."""
    rank = len(terms_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(zip(terms_U, terms_V, terms_W)):
        for i in range(n * m):
            U[i][c] = u[i]
        for j in range(m * p):
            V[j][c] = v[j]
        for k in range(p * n):
            W[k][c] = w[k]
    
    return {"U": U, "V": V, "W": W, "rank": rank}

def _strategy_optimal_222():
    """
    Use 2x2x2 Winograd blocks to cover i:0-3, j:0-3, k:0-5,
    then schoolbook for j=4 (all i,k).
    
    Blocks: 2 (i) x 2 (j) x 3 (k) = 12 blocks, each rank 7 = 84 terms
    Schoolbook for j=4: 4 * 6 = 24 terms
    Total: 84 + 24 = 108
    """
    n, m, p = DIMENSIONS
    winograd_222 = _winograd_2x2x2()
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    # 2x2x2 Winograd blocks for i:0-3, j:0-3, k:0-5
    for ib in range(2):  # i blocks: 0-1, 2-3
        for jb in range(2):  # j blocks: 0-1, 2-3
            for kb in range(3):  # k blocks: 0-1, 2-3, 4-5
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 2 * kb
                
                embedded = _embed_block(winograd_222, i_base, j_base, k_base, 2, 2, 2, n, m, p)
                terms_U.extend([t[0] for t in embedded])
                terms_V.extend([t[1] for t in embedded])
                terms_W.extend([t[2] for t in embedded])
    
    # Schoolbook for j=4 (all i, k)
    for i in range(n):
        for k in range(p):
            ufull = [0] * (n * m)
            vfull = [0] * (m * p)
            wfull = [0] * (p * n)
            ufull[i * m + 4] = 1
            vfull[4 * p + k] = 1
            wfull[k * n + i] = 1
            terms_U.append(ufull)
            terms_V.append(vfull)
            terms_W.append(wfull)
    
    return _build_from_terms(terms_U, terms_V, terms_W, n, m, p)

def _strategy_strassen_4x4x4():
    """
    Use 2x2x2 Winograd blocks for i:0-3, j:0-3, k:0-3,
    then schoolbook for remainder (j=4 all i,k and k=4,5 all i,j).
    
    Blocks: 2 (i) x 2 (j) x 2 (k) = 8 blocks, each rank 7 = 56 terms
    Schoolbook remainder: 
      j=4: 4*6 = 24
      k=4,5 with j=0-3: 4*4*2 = 32
      Total remainder: 56
    Total: 56 + 56 = 112
    """
    n, m, p = DIMENSIONS
    winograd_222 = _winograd_2x2x2()
    
    terms_U = []
    terms_V = []
    terms_W = []
    covered = set()
    
    # 2x2x2 Winograd blocks for i:0-3, j:0-3, k:0-3
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 2 * kb
                
                embedded = _embed_block(winograd_222, i_base, j_base, k_base, 2, 2, 2, n, m, p)
                terms_U.extend([t[0] for t in embedded])
                terms_V.extend([t[1] for t in embedded])
                terms_W.extend([t[2] for t in embedded])
                
                for a in range(2):
                    for b in range(2):
                        for c in range(2):
                            covered.add((i_base + a, j_base + b, k_base + c))
    
    # Schoolbook for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                ufull = [0] * (n * m)
                vfull = [0] * (m * p)
                wfull = [0] * (p * n)
                ufull[i * m + j] = 1
                vfull[j * p + k] = 1
                wfull[k * n + i] = 1
                terms_U.append(ufull)
                terms_V.append(vfull)
                terms_W.append(wfull)
    
    return _build_from_terms(terms_U, terms_V, terms_W, n, m, p)

def _strategy_hybrid():
    """
    Hybrid strategy: Use 2x2x2 Winograd for 4x4x6 sub-block + schoolbook for 4x1x6.
    This should give rank 84 + 24 = 108.
    """
    n, m, p = DIMENSIONS
    winograd_222 = _winograd_2x2x2()
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    # 2x2x2 Winograd blocks for i:0-3, j:0-3, k:0-5
    for ib in range(2):  # i blocks: 0-1, 2-3
        for jb in range(2):  # j blocks: 0-1, 2-3
            for kb in range(3):  # k blocks: 0-1, 2-3, 4-5
                i_base = 2 * ib
                j_base = 2 * jb
                k_base = 2 * kb
                
                embedded = _embed_block(winograd_222, i_base, j_base, k_base, 2, 2, 2, n, m, p)
                terms_U.extend([t[0] for t in embedded])
                terms_V.extend([t[1] for t in embedded])
                terms_W.extend([t[2] for t in embedded])
    
    # Schoolbook for j=4 (all i, k)
    for i in range(n):
        for k in range(p):
            ufull = [0] * (n * m)
            vfull = [0] * (m * p)
            wfull = [0] * (p * n)
            ufull[i * m + 4] = 1
            vfull[4 * p + k] = 1
            wfull[k * n + i] = 1
            terms_U.append(ufull)
            terms_V.append(vfull)
            terms_W.append(wfull)
    
    return _build_from_terms(terms_U, terms_V, terms_W, n, m, p)

def construct():
    n, m, p = DIMENSIONS
    
    candidates = []
    
    # Strategy 1: Hybrid 2x2x2 Winograd + schoolbook (rank 108)
    if remaining() > 10:
        try:
            candidates.append(_strategy_hybrid())
        except Exception as e:
            pass
    
    # Strategy 2: Original optimal_222 (should be same as hybrid, rank 108)
    if remaining() > 5:
        try:
            candidates.append(_strategy_optimal_222())
        except Exception as e:
            pass
    
    # Strategy 3: Schoolbook (rank 120)
    candidates.append(_schoolbook())
    
    # Verify and pick best
    best = None
    for cand in candidates:
        if remaining() <= 0:
            break
        rank = cand["rank"]
        U, V, W = cand["U"], cand["V"], cand["W"]
        if _verify(U, V, W, rank):
            if best is None or rank < best["rank"]:
                best = cand
    
    if best is None:
        best = candidates[-1]  # schoolbook
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
