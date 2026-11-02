"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Test with random matrices
    for trial in range(20):
        A = np.random.randint(-3, 4, size=(n, m)).astype(float)
        B = np.random.randint(-3, 4, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _find_2x2x3_rank10():
    """
    Find a rank-10 decomposition for 2x2x3 matrix multiplication.
    Returns (U, V, W) with shapes (4, 10), (6, 10), (6, 10) or None.
    
    The tensor M[i,j,k] for 2x2*2x3 has entries:
    M[i,j,k] = 1 if this combination contributes to C[i,k] = sum_j A[i,j]*B[j,k]
    
    Specifically:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    """
    n, m, p = 2, 2, 3
    # U indices: i*m + j, i in 0..1, j in 0..1 -> 4 rows
    # V indices: j*p + k, j in 0..1, k in 0..2 -> 6 rows
    # W indices: k*n + i, k in 0..2, i in 0..1 -> 6 rows
    
    # The tensor has 12 non-zero entries (one for each of the 12 products)
    # We need rank-10 decomposition.
    
    # Strategy: Start from schoolbook rank-12 and try to merge two terms.
    # Or use a known construction.
    
    # Let me try a systematic search with half-integer coefficients.
    # We'll use a gradient-free search or try known patterns.
    
    # Known approach: For 2x2x3, one can use 10 multiplications.
    # Let me construct one based on combining Strassen-like ideas.
    
    # Alternative: use the fact that we can share some terms across the 3 columns.
    
    # Let me try to build it explicitly.
    # Think of it as: we have 4 A-values, 6 B-values, and 6 C-values.
    
    # Let me try a construction:
    # For columns 0,1: use Strassen (7 terms)
    # For column 2: need to add 3 more terms (since Strassen doesn't cover it)
    # But we can modify the Strassen terms to partially cover column 2.
    
    # Actually, let me just do a numerical search.
    
    deadline = time.time() + 120
    
    # Initialize with schoolbook (rank 12)
    U0 = np.zeros((4, 12))
    V0 = np.zeros((6, 12))
    W0 = np.zeros((6, 12))
    
    # Schoolbook terms:
    # Term 0: A00 * B00 -> C00
    U0[0, 0] = 1; V0[0, 0] = 1; W0[0, 0] = 1
    # Term 1: A01 * B10 -> C00
    U0[1, 1] = 1; V0[3, 1] = 1; W0[0, 1] = 1
    # Term 2: A00 * B01 -> C01
    U0[0, 2] = 1; V0[1, 2] = 1; W0[2, 2] = 1
    # Term 3: A01 * B11 -> C01
    U0[1, 3] = 1; V0[4, 3] = 1; W0[2, 3] = 1
    # Term 4: A00 * B02 -> C02
    U0[0, 4] = 1; V0[2, 4] = 1; W0[4, 4] = 1
    # Term 5: A01 * B12 -> C02
    U0[1, 5] = 1; V0[5, 5] = 1; W0[4, 5] = 1
    # Term 6: A10 * B00 -> C10
    U0[2, 6] = 1; V0[0, 6] = 1; W0[1, 6] = 1
    # Term 7: A11 * B10 -> C10
    U0[3, 7] = 1; V0[3, 7] = 1; W0[1, 7] = 1
    # Term 8: A10 * B01 -> C11
    U0[2, 8] = 1; V0[1, 8] = 1; W0[3, 8] = 1
    # Term 9: A11 * B11 -> C11
    U0[3, 9] = 1; V0[4, 9] = 1; W0[3, 9] = 1
    # Term 10: A10 * B02 -> C12
    U0[2, 10] = 1; V0[2, 10] = 1; W0[5, 10] = 1
    # Term 11: A11 * B12 -> C12
    U0[3, 11] = 1; V0[5, 11] = 1; W0[5, 11] = 1
    
    # Now try to find a rank-10 by merging two pairs of terms.
    # We'll try to find two terms that can be combined.
    
    # For each pair of terms (a, b), check if we can replace them with one term.
    # This is a complex search, so let me try a different approach.
    
    # Use a simple local search: start from rank 12, randomly perturb,
    # and try to project onto rank 10.
    
    # Actually, let me try a known result. The rank of 2x2x3 matrix multiplication
    # is 10. Let me construct it.
    
    # One known construction:
    # Use 7 Strassen terms for columns 0,1 and add 3 terms for column 2,
    # but modify to save 2 terms.
    
    # Let me try a direct construction.
    
    # After some thought, here's a rank-10 construction:
    # We'll use the following 10 products:
    
    U10 = np.zeros((4, 10))
    V10 = np.zeros((6, 10))
    W10 = np.zeros((6, 10))
    
    # I'll use a numerical optimization approach.
    # Start with random initialization and minimize the error.
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for trial in range(200):
        if time.time() > deadline - 30:
            break
        
        # Random initialization with half-integer values
        scale = 2.0
        U_rand = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], size=(4, 10))
        V_rand = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], size=(6, 10))
        W_rand = np.random.choice([-1.5, -1, -0.5, 0.5, 1, 1.5], size=(6, 10))
        
        # Compute error
        error = 0.0
        for i in range(2):
            for k in range(3):
                target = 1.0  # Each C[i,k] gets contribution from exactly one product per j
                # Actually, we need the full tensor
                pass
        
        # Let me compute the tensor error properly
        # M[i,j,k] should be 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
        # The decomposition gives: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = M[i,j,k]
        
        # For 2x2x3, M has 12 non-zero entries (all 1s) at positions:
        # (i,j,k) where the product A[i,j]*B[j,k] appears in C[i,k]
        # That's all 2*2*3 = 12 combinations, each with value 1.
        
        tensor_computed = np.zeros((2, 2, 3))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = 0.0
                    for r in range(10):
                        s += U_rand[i*2+j, r] * V_rand[j*3+k, r] * W_rand[k*2+i, r]
                    tensor_computed[i, j, k] = s
        
        # Target tensor: all ones
        target = np.ones((2, 2, 3))
        error = np.sum((tensor_computed - target) ** 2)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
            if error < 1e-6:
                break
    
    if best_error < 1e-6:
        # Round to half-integers
        U_final = np.round(best_U * 2) / 2
        V_final = np.round(best_V * 2) / 2
        W_final = np.round(best_W * 2) / 2
        
        # Verify
        tensor_check = np.zeros((2, 2, 3))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = 0.0
                    for r in range(10):
                        s += U_final[i*2+j, r] * V_final[j*3+k, r] * W_final[k*2+i, r]
                    tensor_check[i, j, k] = s
        
        if np.allclose(tensor_check, np.ones((2, 2, 3)), atol=1e-9):
            return U_final.tolist(), V_final.tolist(), W_final.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 150
    
    # Try to find 2x2x3 rank-10 decomposition
    block_decomp = _find_2x2x3_rank10()
    
    if block_decomp is not None:
        U_block, V_block, W_block = block_decomp
        block_rank = 10
        num_blocks = p // 3  # 5 blocks
        rank = num_blocks * block_rank
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for b in range(num_blocks):
            k0 = 3 * b
            col_offset = b * block_rank
            
            for r in range(block_rank):
                # U: same for each block
                for i in range(n * m):
                    U[i][col_offset + r] += U_block[i][r]
                
                # V: shift k indices
                for j in range(m):
                    for k in range(3):
                        v_idx = j * p + (k0 + k)
                        v_block_idx = j * 3 + k
                        V[v_idx][col_offset + r] += V_block[v_block_idx][r]
                
                # W: shift k indices
                for k in range(3):
                    for i in range(n):
                        w_idx = (k0 + k) * n + i
                        w_block_idx = k * n + i
                        W[w_idx][col_offset + r] += W_block[w_block_idx][r]
        
        # Verify
        if time.time() < deadline - 10:
            if not _verify_decomposition(U, V, W, n, m, p):
                # Fallback to Strassen
                return _construct_strassen_53(n, m, p)
        
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to Strassen rank 53
        return _construct_strassen_53(n, m, p)

def _construct_strassen_53(n, m, p):
    """Fallback: Strassen blocks + schoolbook, rank 53."""
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, col):
        for idx, val in u_coeffs:
            U[idx][col] += val
        for idx, val in v_coeffs:
            V[idx][col] += val
        for idx, val in w_coeffs:
            W[idx][col] += val

    col = 0
    for b in range(7):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        add_term(
            [(0, 1), (3, 1)],
            [(k0, 1), (p + k1, 1)],
            [(k0 * n + 0, 1), (k1 * n + 1, 1)],
            col
        )
        col += 1
        
        add_term(
            [(2, 1), (3, 1)],
            [(k0, 1)],
            [(k0 * n + 1, 1), (k1 * n + 1, -1)],
            col
        )
        col += 1
        
        add_term(
            [(0, 1)],
            [(k1, 1), (p + k1, -1)],
            [(k1 * n + 0, 1), (k1 * n + 1, 1)],
            col
        )
        col += 1
        
        add_term(
            [(3, 1)],
            [(p + k0, 1), (k0, -1)],
            [(k0 * n + 0, 1), (k0 * n + 1, 1)],
            col
        )
        col += 1
        
        add_term(
            [(0, 1), (1, 1)],
            [(p + k1, 1)],
            [(k0 * n + 0, -1), (k1 * n + 0, 1)],
            col
        )
        col += 1
        
        add_term(
            [(2, 1), (0, -1)],
            [(k0, 1), (k1, 1)],
            [(k1 * n + 1, 1)],
            col
        )
        col += 1
        
        add_term(
            [(1, 1), (3, -1)],
            [(p + k0, 1), (p + k1, 1)],
            [(k0 * n + 0, 1)],
            col
        )
        col += 1
        
    k_last = 14
    add_term([(0, 1)], [(k_last, 1)], [(k_last * n + 0, 1)], col)
    col += 1
    add_term([(1, 1)], [(p + k_last, 1)], [(k_last * n + 0, 1)], col)
    col += 1
    add_term([(2, 1)], [(k_last, 1)], [(k_last * n + 1, 1)], col)
    col += 1
    add_term([(3, 1)], [(p + k_last, 1)], [(k_last * n + 1, 1)], col)
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
