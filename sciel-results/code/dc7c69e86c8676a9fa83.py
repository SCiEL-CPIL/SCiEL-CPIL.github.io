"""Matrix multiplication 2x2 x 2x16 via shared Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the Strassen decomposition of 2x2 matrix multiplication.
    The 7 rank-1 terms have coefficients in {-1, 0, 1}.
    
    For C = A * B where A is 2x2 and B is 2x2:
    A = [[a, b], [c, d]]
    B = [[e, f], [g, h]]
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen's 7 products:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification of the tensor decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    result = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(result, 1.0, atol=tol)


def _find_2x2_2x4_decomposition():
    """
    Find a low-rank decomposition for 2x2 * 2x4 matrix multiplication.
    Uses the structure that A is only 2x2, so we can share A-side terms.
    
    We want to compute C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,1,2,3}
    
    Key idea: Use Strassen-like combinations on the A side that are shared
    across all 4 columns, then multiply with B-side combinations.
    
    Let's try: define 7 Strassen-like A combinations and 4 B combinations per
    "virtual 2x2 block", but share the A part.
    
    Actually, a cleaner approach: the tensor for 2x2 * 2x4 has dimensions
    (2,2) x (2,4) x (4,2) = 4 x 8 x 8.
    
    We can write this as a sum of rank-1 terms. Let's try to find a
    decomposition with rank < 14 (which is 7*2, the naive Strassen tiling).
    
    Approach: For each column k of B, we need C[:,k] = A * B[:,k].
    This is a 2-vector = 2x2 * 2-vector product.
    
    We can use the fact that for a 2x2 * 2x1 (vector) multiplication,
    we need rank at least 4 (schoolbook) but Strassen doesn't directly apply.
    
    Better: tile the 4 columns as two 2-column blocks and use Strassen on each,
    but share the A-side linear combinations.
    
    For block 1 (cols 0,1): M1..M7 with A-combos and B-combos for cols 0,1
    For block 2 (cols 2,3): M1..M7 with SAME A-combos and B-combos for cols 2,3
    
    If we share the A-side, we need:
    - 7 A-combinations (shared)
    - For each block: 7 B-combinations and 7 W-combinations
    
    But the rank is the number of scalar multiplications. If we share A-combos,
    we still need 7 multiplications per block (since each M = A_combo * B_combo).
    So total rank = 7 + 7 = 14? No, that's not how it works.
    
    Actually, each rank-1 term is U_r * V_r * W_r. The U part is fixed per term.
    If two terms share the same U vector, they're still separate rank-1 terms
    unless V and W also align.
    
    Let me try a direct search for a low-rank decomposition using the
    structure of the problem.
    
    Alternative: Use the 4x4 matrix multiplication Strassen (rank 19) applied
    to a 4x4 * 4x4 where we embed our problem. But that gives rank 19 for 4x4*4x4,
    which when restricted might give something for 2x2*2x4.
    
    Let me just try: treat 2x2 * 2x4 as part of a 4x4 * 4x4 problem.
    Embed A in top-left 2x2 of a 4x4, B in top-left 2x4 of a 4x4 (pad with zeros).
    Then use rank-19 Strassen for 4x4*4x4. The rank-19 decomposition has
    terms that involve all 4x4 elements, but many will be zero due to padding.
    
    This is getting complex. Let me try a simpler approach:
    
    For 2x2 * 2x4, the tensor T[i,j,k,l] = delta_{j,k} where i in {0,1}, j in {0,1},
    k in {0,1}, l in {0,1,2,3}. Wait, let me re-index.
    
    T[A_idx, B_idx, C_idx] where:
    A_idx = (i,j) for A[i,j], i in {0,1}, j in {0,1}
    B_idx = (j',k) for B[j',k], j' in {0,1}, k in {0,1,2,3}
    C_idx = (l,i') for C[i',l], i' in {0,1}, l in {0,1,2,3}
    
    T[(i,j), (j',k), (l,i')] = delta_{j,j'} * delta_{i,i'}
    
    This is the standard matrix multiplication tensor restricted.
    
    Let me try to find a decomposition by solving a linear system.
    For rank r, we need to find U (4 x r), V (8 x r), W (8 x r) such that
    sum_c U[i,j,c] * V[j',k,c] * W[l,i',c] = delta_{j,j'} * delta_{i,i'}
    
    This is a tensor decomposition problem. Let me try small ranks.
    """
    # For now, let's just use 2 Strassen blocks (rank 14) as the 2x2*2x4 core
    # and see if we can do better.
    
    # Actually, let me try a direct approach: 
    # The 2x2 * 2x4 problem can be seen as computing 8 outputs from 4 A-entries
    # and 8 B-entries.
    
    # Try rank 12: maybe we can save 2 terms by sharing.
    
    # Let me just construct the 2x2*2x4 via 2 Strassen blocks for now
    # (rank 14) and focus on the overall structure.
    
    U_core, V_core, W_core = _strassen_2x2()
    
    # For 2x2 * 2x4: two Strassen blocks for columns (0,1) and (2,3)
    rank_24 = 14
    U_24 = np.zeros((4, rank_24), dtype=np.int64)
    V_24 = np.zeros((8, rank_24), dtype=np.int64)  # 2 rows * 4 cols
    W_24 = np.zeros((8, rank_24), dtype=np.int64)  # 4 cols * 2 rows
    
    for block in range(2):
        col_base = block * 2
        for r in range(7):
            U_24[:, block*7 + r] = U_core[:, r]
            # V: row j*4 + k
            V_24[0*4 + col_base + 0, block*7 + r] = V_core[0, r]
            V_24[0*4 + col_base + 1, block*7 + r] = V_core[1, r]
            V_24[1*4 + col_base + 0, block*7 + r] = V_core[2, r]
            V_24[1*4 + col_base + 1, block*7 + r] = V_core[3, r]
            # W: row k*2 + i
            W_24[(col_base + 0)*2 + 0, block*7 + r] = W_core[0, r]
            W_24[(col_base + 1)*2 + 0, block*7 + r] = W_core[1, r]
            W_24[(col_base + 0)*2 + 1, block*7 + r] = W_core[2, r]
            W_24[(col_base + 1)*2 + 1, block*7 + r] = W_core[3, r]
    
    return U_24, V_24, W_24, rank_24


def _search_better_2x4_rank(max_rank=13, time_limit=30):
    """
    Try to find a decomposition of 2x2 * 2x4 with rank < 14.
    Uses random initialization + iterative refinement.
    """
    start_time = time.time()
    
    n, m = 2, 2
    p = 4  # 2x4 means p=4 columns
    
    # Target tensor: T[i,j, j',k, l,i'] = delta_{j,j'} * delta_{i,i'}
    # In our indexing: U is (4, r), V is (8, r), W is (8, r)
    # U index: i*m + j = i*2 + j
    # V index: j'*p + k = j'*4 + k
    # W index: l*n + i' = l*2 + i'
    
    def build_target():
        """Build the target tensor as a 3-way array (4, 8, 8)."""
        T = np.zeros((4, 8, 8), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                for j2 in range(2):
                    for k in range(4):
                        for l in range(4):
                            for i2 in range(2):
                                if j == j2 and i == i2:
                                    T[i*2+j, j2*4+k, l*2+i2] = 1.0
        return T
    
    T_target = build_target()
    
    def compute_tensor(U, V, W):
        """Compute the tensor from factor matrices."""
        r = U.shape[1]
        # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        return np.einsum('ar,br,cr->abc', U, V, W)
    
    def error(U, V, W):
        T_comp = compute_tensor(U, V, W)
        return np.linalg.norm(T_comp - T_target)
    
    best_rank = None
    best_factors = None
    
    for r in range(8, min(max_rank + 1, 14)):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations
        for trial in range(200):
            if time.time() - start_time > time_limit:
                break
            
            np.random.seed(42 + r * 1000 + trial)
            U = np.random.randn(4, r) * 0.5
            V = np.random.randn(8, r) * 0.5
            W = np.random.randn(8, r) * 0.5
            
            # Iterative refinement (ALS-style)
            for iteration in range(200):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U: for each a, U[a,:] should satisfy
                # U[a,:] * (V[:,r] * W[:,r])_b,c = T_target[a,b,c]
                # This is a linear system in U[a,:]
                
                # For fixed V, W: T_target[a] = U[a] @ (V * W)^T reshaped
                # V*W element-wise: (V[:,r] * W[:,r]) has shape (8, r)
                # Actually: T_comp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed V,W: T_comp[a] = sum_r U[a,r] * (V[:,r] * W[:,r]).reshape(8,8)
                # So: T_target[a].reshape(64) = U[a] @ M_a where M_a[b*8+c, r] = V[b,r]*W[c,r]
                
                M_VW = np.einsum('br,cr->bc r', V, W)  # (8, 8, r)
                M_VW_flat = M_VW.reshape(64, r)
                
                # Solve for U: T_target[a].reshape(64) = U[a] @ M_VW_flat
                for a in range(4):
                    rhs = T_target[a].reshape(64)
                    # Least squares: U[a] = rhs @ M_VW_flat^+ 
                    try:
                        U[a] = np.linalg.lstsq(M_VW_flat, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Update V similarly
                M_UW = np.einsum('ar,cr->ac r', U, W)  # (4, 8, r)
                M_UW_flat = M_UW.reshape(32, r)
                for b in range(8):
                    rhs = T_target[:, b, :].reshape(32)
                    try:
                        V[b] = np.linalg.lstsq(M_UW_flat, rhs, rcond=None)[0]
                    except:
                        pass
                
                # Update W similarly
                M_UV = np.einsum('ar,br->ab r', U, V)  # (4, 8, r)
                M_UV_flat = M_UV.reshape(32, r)
                for c in range(8):
                    rhs = T_target[:, :, c].reshape(32)
                    try:
                        W[c] = np.linalg.lstsq(M_UV_flat, rhs, rcond=None)[0]
                    except:
                        pass
                
                if error(U, V, W) < 1e-6:
                    break
            
            err = error(U, V, W)
            if err < 1e-6:
                # Found a valid decomposition at rank r
                best_rank = r
                best_factors = (U.copy(), V.copy(), W.copy())
                break
        
        if best_rank is not None:
            break
    
    if best_factors is not None:
        U, V, W = best_factors
        # Round to nearest integers or half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify after rounding
        if error(U_r, V_r, W_r) < 1e-6:
            # Convert to integer representation (multiply by 2)
            U_int = np.round(U_r * 2).astype(np.int64)
            V_int = np.round(V_r * 2).astype(np.int64)
            W_int = np.round(W_r * 2).astype(np.int64)
            return U_int, V_int, W_int, best_rank, True
    
    return None, None, None, None, False


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    Uses Strassen's rank-7 decomposition for the 2x2 core, applied
    to column blocks. Attempts to find a lower-rank 2x2*2x4 core
    decomposition, falling back to 2 Strassen blocks (rank 14) if not found.
    
    Total rank = 4 * (rank of 2x2*2x4 core)
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Try to find a better 2x2*2x4 decomposition
    U_24, V_24, W_24, rank_24, found = _search_better_2x4_rank(max_rank=13, time_limit=40)
    
    if found and time.time() < deadline:
        # Use the found 2x2*2x4 core, tiled 4 times for 2x16
        num_blocks = p // 4  # 4 blocks of 4 columns
        total_rank = rank_24 * num_blocks
        
        U = np.zeros((n * m, total_rank), dtype=np.int64)
        V = np.zeros((m * p, total_rank), dtype=np.int64)
        W = np.zeros((p * n, total_rank), dtype=np.int64)
        
        for block in range(num_blocks):
            col_offset = block * 4
            rank_offset = block * rank_24
            
            for r in range(rank_24):
                U[:, rank_offset + r] = U_24[:, r]
                
                # V: row j*p + k, for block, local col is col_offset + local_k
                for j in range(2):
                    for local_k in range(4):
                        V[j * p + col_offset + local_k, rank_offset + r] = V_24[j * 4 + local_k, r]
                
                # W: row k*n + i, for block, local col is col_offset + local_k
                for local_k in range(4):
                    for i in range(2):
                        W[(col_offset + local_k) * n + i, rank_offset + r] = W_24[local_k * 2 + i, r]
        
        # Verify
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: standard Strassen tiling (rank 56)
    U_core, V_core, W_core = _strassen_2x2()
    
    num_pairs = p // 2  # 8 pairs
    rank = 7 * num_pairs  # 56
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for t in range(num_pairs):
        col_offset = t * 7
        
        for r in range(7):
            U[:, col_offset + r] = U_core[:, r]
            V[0 * p + 2 * t, col_offset + r] = V_core[0, r]
            V[0 * p + 2 * t + 1, col_offset + r] = V_core[1, r]
            V[1 * p + 2 * t, col_offset + r] = V_core[2, r]
            V[1 * p + 2 * t + 1, col_offset + r] = V_core[3, r]
            W[(2 * t) * n + 0, col_offset + r] = W_core[0, r]
            W[(2 * t + 1) * n + 0, col_offset + r] = W_core[1, r]
            W[(2 * t) * n + 1, col_offset + r] = W_core[2, r]
            W[(2 * t + 1) * n + 1, col_offset + r] = W_core[3, r]
    
    # Verify the decomposition
    if not _verify_decomposition_fast(U, V, W, n, m, p):
        # Fallback: schoolbook decomposition (rank = n*m*p = 64)
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=np.int64)
        V_fb = np.zeros((m * p, rank_fb), dtype=np.int64)
        W_fb = np.zeros((p * n, rank_fb), dtype=np.int64)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, column] = 1
                    V_fb[j * p + k, column] = 1
                    W_fb[k * n + i, column] = 1
                    column += 1
        U, V, W = U_fb, V_fb, W_fb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
