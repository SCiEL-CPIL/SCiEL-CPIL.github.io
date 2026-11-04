"""Matrix multiplication 4x4x7 via recursive Strassen decomposition."""
import time

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (U, V, W) where each is a dict mapping (i,j) -> coeff.
    U: A entries, V: B entries, W: C entries.
    Convention: C[i,j] = sum_t U[t][i,j] * V[t][i,j] * W[t][i,j]... 
    Actually: C[i,k] = sum_j A[i,j]*B[j,k].
    In tensor form: T[i,j,k] = A[i,j]*B[j,k]
    Rank-1 term: U[i,j] * V[j,k] * W[i,k] such that sum_t U[t][i,j]*V[t][j,k]*W[t][i,k] = T[i,j,k]
    
    Strassen's 7 terms for 2x2x2:
    m1 = (A00+A11)*(B00+B11)
    m2 = (A10+A11)*B00
    m3 = A00*(B10-B11)
    m4 = A11*(B01-B10)
    m5 = (A00+A01)*B11
    m6 = (A10-A00)*(B00+B10)
    m7 = (A01-A11)*(B01+B11)
    
    C00 = m1 + m4 - m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m2 + m3 + m6
    
    Let me encode each term as U (A side), V (B side), W (C side):
    """
    terms = []
    
    # m1 = (A00+A11)*(B00+B11)
    # U: A00=1, A11=1; V: B00=1, B11=1; W: C00=1, C11=1
    terms.append({
        'U': {(0,0):1, (1,1):1},
        'V': {(0,0):1, (1,1):1},
        'W': {(0,0):1, (1,1):1}
    })
    
    # m2 = (A10+A11)*B00
    # U: A10=1, A11=1; V: B00=1; W: C10=1, C11=-1
    terms.append({
        'U': {(1,0):1, (1,1):1},
        'V': {(0,0):1},
        'W': {(1,0):1, (1,1):-1}
    })
    
    # m3 = A00*(B10-B11)
    # U: A00=1; V: B10=1, B11=-1; W: C01=1, C11=1
    terms.append({
        'U': {(0,0):1},
        'V': {(1,0):1, (1,1):-1},
        'W': {(0,1):1, (1,1):1}
    })
    
    # m4 = A11*(B01-B10)
    # U: A11=1; V: B01=1, B10=-1; W: C00=1, C10=1
    terms.append({
        'U': {(1,1):1},
        'V': {(0,1):1, (1,0):-1},
        'W': {(0,0):1, (1,0):1}
    })
    
    # m5 = (A00+A01)*B11
    # U: A00=1, A01=1; V: B11=1; W: C00=-1, C01=1
    terms.append({
        'U': {(0,0):1, (0,1):1},
        'V': {(1,1):1},
        'W': {(0,0):-1, (0,1):1}
    })
    
    # m6 = (A10-A00)*(B00+B10)
    # U: A10=1, A00=-1; V: B00=1, B10=1; W: C11=1
    terms.append({
        'U': {(1,0):1, (0,0):-1},
        'V': {(0,0):1, (1,0):1},
        'W': {(1,1):1}
    })
    
    # m7 = (A01-A11)*(B01+B11)
    # U: A01=1, A11=-1; V: B01=1, B11=1; W: C00=1
    terms.append({
        'U': {(0,1):1, (1,1):-1},
        'V': {(0,1):1, (1,1):1},
        'W': {(0,0):1}
    })
    
    return terms


def _compose_2x4x4():
    """
    Compute 4x4 = A(2x2 blocks) * B(2x2 blocks) using block Strassen.
    A is 4x4, B is 4x4, C is 4x4.
    Split into 2x2 blocks of size 2x2.
    Block indices: A00, A01, A10, A11 each 2x2.
    Same for B and C.
    
    Block Strassen:
    m1 = (A00+A11)*(B00+B11)  -- 2x2*2x2
    m2 = (A10+A11)*B00
    m3 = A00*(B10-B11)
    m4 = A11*(B01-B10)
    m5 = (A00+A01)*B11
    m6 = (A10-A00)*(B00+B10)
    m7 = (A01-A11)*(B01+B11)
    
    C00 = m1 + m4 - m5 + m7
    C01 = m3 + m5
    C10 = m2 + m4
    C11 = m1 - m2 + m3 + m6
    
    Each mt is a 2x2 matrix product, which we decompose with 2x2x2 Strassen (7 terms).
    Total: 7*7 = 49 rank-1 terms.
    
    We need to return terms as dicts with entries in the full 4x4 space.
    A indexed by (i,j) with i,j in 0..3. Block (bi,bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    """
    # First get the 7 block-level terms
    block_terms = _strassen_2x2x2()  # Each has U, V, W as dicts with 2x2 block indices
    
    # For each block term, we need to compose with 2x2x2 Strassen at the inner level.
    # A block term has U_block (coefficients on A blocks), V_block (coefficients on B blocks), W_block (coefficients on C blocks).
    # Each block is 2x2, so we need to expand each 2x2 matrix product.
    
    all_terms = []
    for bt in block_terms:
        U_block = bt['U']  # dict: (bi,bj) -> coeff
        V_block = bt['V']  # dict: (bi,bj) -> coeff
        W_block = bt['W']  # dict: (ci,cj) -> coeff
        
        # For each combination of A block and B block in this term,
        # we have a 2x2*2x2 product. Decompose with 2x2x2 Strassen.
        for (ai, aj), ua in U_block.items():
            for (bi, bj), vb in V_block.items():
                # The 2x2 product: A_block(ai,aj) * B_block(bi,bj)
                # Inner indices: a in 0..1 (row within A block), b in 0..1 (col within A block / row within B block), c in 0..1 (col within B block)
                # A global index: (2*ai + a, 2*aj + b)
                # B global index: (2*bi + b, 2*bj + c)
                # C global index: (2*ai + a, 2*bj + c)
                
                for st in _strassen_2x2x2():
                    U_inner = st['U']  # (a,b) -> coeff
                    V_inner = st['V']  # (b,c) -> coeff
                    W_inner = st['W']  # (a,c) -> coeff
                    
                    U_full = {}
                    V_full = {}
                    W_full = {}
                    
                    for (a, b), ua_inner in U_inner.items():
                        key = (2*ai + a, 2*aj + b)
                        U_full[key] = U_full.get(key, 0) + ua * ua_inner
                    
                    for (b, c), vb_inner in V_inner.items():
                        key = (2*bi + b, 2*bj + c)
                        V_full[key] = V_full.get(key, 0) + vb * vb_inner
                    
                    for (a, c), wc in W_inner.items():
                        key = (2*ai + a, 2*bj + c)
                        W_full[key] = W_full.get(key, 0) + wc  # W_block coeff will be applied later
                    
                    # Apply W_block coefficient: the C output goes to block (ci,cj) with coefficient from W_block
                    # But wait, the W_block in the block-level term tells us which C block this contributes to.
                    # Actually, the block-level W_block maps C block indices to coefficients.
                    # The inner W gives the contribution within that C block.
                    # We need to sum over all (ci,cj) in W_block.
                    
                    # Hmm, I think I'm conflating things. Let me reconsider.
                    
                    # The block-level term says: contribution to C = W_block[(ci,cj)] * (2x2 product result)
                    # But the 2x2 product result is a 2x2 matrix, and W_block[(ci,cj)] is a scalar.
                    # So for each C block (ci,cj) with coefficient wc_block, the inner W entries get multiplied by wc_block.
                    
                    # Let me redo this:
                    U_full = {}
                    V_full = {}
                    W_full = {}
                    
                    for (a, b), ua_inner in U_inner.items():
                        key = (2*ai + a, 2*aj + b)
                        U_full[key] = U_full.get(key, 0) + ua * ua_inner
                    
                    for (b, c), vb_inner in V_inner.items():
                        key = (2*bi + b, 2*bj + c)
                        V_full[key] = V_full.get(key, 0) + vb * vb_inner
                    
                    for (ci, cj), wc_block in W_block.items():
                        for (a, c), wc_inner in W_inner.items():
                            key = (2*ci + a, 2*cj + c)
                            W_full[key] = W_full.get(key, 0) + wc_block * wc_inner
                    
                    all_terms.append({'U': U_full, 'V': V_full, 'W': W_full})
    
    return all_terms


def _schoolbook_4x4x3():
    """
    Schoolbook decomposition of 4x4 * 4x3 = 4x3.
    Rank = 4*4*3 = 48.
    T[i,j,k] = A[i,j]*B[j,k]
    Rank-1 term for each (i,j,k): U[i,j]=1, V[j,k]=1, W[i,k]=1
    """
    terms = []
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U = {(i,j): 1}
                V = {(j,k): 1}
                W = {(i,k): 1}
                terms.append({'U': U, 'V': V, 'W': W})
    return terms


def construct():
    """
    Construct the tensor decomposition for 4x4 * 4x7.
    
    Strategy: Split the 4x7 B matrix into B_left (4x4) and B_right (4x3).
    C = A * B = A * B_left + A * B_right (4x4 + 4x3)
    
    Use Strassen (rank 49) for 4x4x4 and schoolbook (rank 48) for 4x4x3.
    Total rank = 49 + 48 = 97.
    
    Output format:
    U: (n*m, rank) = (16, rank) -- A is 4x4, flattened row-major
    V: (m*p, rank) = (28, rank) -- B is 4x7, flattened row-major  
    W: (p*n, rank) = (28, rank) -- C is 7x4, but wait...
    
    Actually, let me re-read the convention: W[k*n+i, r]
    C has dimensions p x n = 7 x 4? No, C = A(4x4) * B(4x7) = 4x7.
    So C is 4x7, with C[i,k] where i in 0..3, k in 0..6.
    W[k*n+i, r] means W is indexed by (k, i) -> flattened to k*n+i.
    n=4, so W[k*4+i, r]. W has shape (p*n, rank) = (28, rank).
    
    U[i*m+j, r] where i in 0..3, j in 0..3. U has shape (16, rank).
    V[j*p+k, r] where j in 0..3, k in 0..6. V has shape (28, rank).
    
    The tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    where T[i,j,k] = A[i,j]*B[j,k] (the matrix multiplication tensor).
    
    For our decomposition, each rank-1 term has:
    U entries on A[i,j], V entries on B[j,k], W entries on C[i,k] (but stored as W[k*n+i]).
    """
    n, m, p = 4, 4, 7
    
    # Get terms for 4x4x4 part (columns 0-3 of B)
    terms_444 = _compose_2x4x4()
    # Get terms for 4x4x3 part (columns 4-6 of B)
    terms_443 = _schoolbook_4x4x3()
    
    # Now combine. The 444 terms use B indices j in 0..3, k in 0..3.
    # The 443 terms use B indices j in 0..3, k in 0..2 (for the 4x3 subproblem).
    # But in the full 4x7 B, the right part is at columns 4,5,6.
    # So for the 443 terms, B[j, k_local] maps to B[j, k_local+4].
    # And C[i, k_local] maps to C[i, k_local+4].
    
    all_terms = []
    
    # Add 444 terms
    for t in terms_444:
        U = t['U']  # (i,j) -> coeff, i,j in 0..3
        V = t['V']  # (j,k) -> coeff, j in 0..3, k in 0..3
        W = t['W']  # (i,k) -> coeff, i in 0..3, k in 0..3
        all_terms.append((U, V, W))
    
    # Add 443 terms, shifting B and C indices by 4
    for t in terms_443:
        U = t['U']  # (i,j) -> coeff
        V_orig = t['V']  # (j,k) -> coeff, k in 0..2
        W_orig = t['W']  # (i,k) -> coeff, k in 0..2
        
        V = {}
        for (j, k), coeff in V_orig.items():
            V[(j, k+4)] = coeff
        
        W = {}
        for (i, k), coeff in W_orig.items():
            W[(i, k+4)] = coeff
        
        all_terms.append((U, V, W))
    
    rank = len(all_terms)
    
    # Build U, V, W matrices
    U_mat = [[0]*rank for _ in range(n*m)]
    V_mat = [[0]*rank for _ in range(m*p)]
    W_mat = [[0]*rank for _ in range(p*n)]
    
    for col, (U_dict, V_dict, W_dict) in enumerate(all_terms):
        for (i, j), coeff in U_dict.items():
            U_mat[i*m+j][col] += coeff
        
        for (j, k), coeff in V_dict.items():
            V_mat[j*p+k][col] += coeff
        
        for (i, k), coeff in W_dict.items():
            # W stored as W[k*n+i]
            W_mat[k*n+i][col] += coeff
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
