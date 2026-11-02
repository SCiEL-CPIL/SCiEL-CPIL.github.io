"""Matrix multiplication 4,4,6 - rank-minimal exact tensor decomposition search."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(Ua[i*m+j, :] * Va[j*p+k, :], Wa[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_decomposition():
    """Return U, V, W for 2x2 x 2x2 -> 2x2 using Strassen (rank 7).
    
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen formulas for C = A*B where A,B,C are 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each Strassen product M_r, we need to decompose it as sum over sub-ranks
    # But actually, each M_r is a single scalar product of two linear combinations,
    # so in the tensor decomposition, each M_r contributes one rank-1 term.
    
    # Let's define the linear combinations:
    # M1: A-part: A11+A22 = indices (0,0) and (1,1); B-part: B11+B22 = (0,0) and (1,1)
    # M2: A-part: A21+A22 = (1,0) and (1,1); B-part: B11 = (0,0)
    # M3: A-part: A11 = (0,0); B-part: B12-B22 = (0,1) and (1,1)
    # M4: A-part: A22 = (1,1); B-part: B21-B11 = (1,0) and (0,0)
    # M5: A-part: A11+A12 = (0,0) and (0,1); B-part: B22 = (1,1)
    # M6: A-part: A21-A11 = (1,0) and (0,0); B-part: B11+B12 = (0,0) and (0,1)
    # M7: A-part: A12-A22 = (0,1) and (1,1); B-part: B21+B22 = (1,0) and (1,1)
    
    # And the output contributions:
    # C11 gets: +M1, +M4, -M5, +M7
    # C12 gets: +M3, +M5
    # C21 gets: +M2, +M4
    # C22 gets: +M1, -M2, +M3, +M6
    
    # Define helper to set up a Strassen term
    def set_term(r, a_coeffs, b_coeffs, c_coeffs):
        """a_coeffs: dict {(i,j): coeff} for A part
           b_coeffs: dict {(j,k): coeff} for B part
           c_coeffs: dict {(k,i): coeff} for C part (W index is k*n+i)
        """
        for (i, j), coeff_a in a_coeffs.items():
            U[i*m+j][r] += coeff_a
        for (j, k), coeff_b in b_coeffs.items():
            V[j*p+k][r] += coeff_b
        for (k, i), coeff_c in c_coeffs.items():
            W[k*n+i][r] += coeff_c
    
    # M1
    set_term(0, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
    # Wait, C11 is at position (i=0,k=0) which is W[k*n+i] = W[0*2+0] = W[0]
    # C12 is at position (i=0,k=1) which is W[1*2+0] = W[2]
    # C21 is at position (i=1,k=0) which is W[0*2+1] = W[1]
    # C22 is at position (i=1,k=1) which is W[1*2+1] = W[3]
    
    # Let me redo with correct W indexing: W[k*n + i]
    # C[i,k] is stored at W[k*n + i]
    
    # M1: C11 += M1, C22 += M1
    # C11: i=0, k=0 -> W[0*2+0] = W[0]
    # C22: i=1, k=1 -> W[1*2+1] = W[3]
    set_term(0, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
    
    # M2: C21 += M2, C22 -= M2
    # C21: i=1, k=0 -> W[0*2+1] = W[1]
    # C22: i=1, k=1 -> W[1*2+1] = W[3]
    set_term(1, {(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1})
    
    # M3: C12 += M3, C22 += M3
    # C12: i=0, k=1 -> W[1*2+0] = W[2]
    # C22: i=1, k=1 -> W[1*2+1] = W[3]
    set_term(2, {(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    
    # M4: C11 += M4, C21 += M4
    # C11: i=0, k=0 -> W[0]
    # C21: i=1, k=0 -> W[1]
    set_term(3, {(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1})
    
    # M5: C12 += M5, C11 -= M5
    # C12: i=0, k=1 -> W[2]
    # C11: i=0, k=0 -> W[0]
    set_term(4, {(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1, (0,0): -1})
    
    # M6: C22 += M6
    set_term(5, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1})
    
    # M7: C11 += M7
    set_term(6, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    
    return U, V, W

def _block_strassen_4x4x4x6():
    """Construct a rank-42 decomposition using 2x2 block Strassen.
    
    Treat 4x4 as 2x2 blocks of 2x2, 4x6 as 2x3 blocks of 2x2.
    Block multiplication: [A11 A12; A21 A22] * [B11 B12 B13; B21 B22 B23]
    = [C11 C12 C13; C21 C22 C23]
    
    Each Cij = A11*Bij + A12*B2j (for j=1,2,3) or similar.
    We can use Strassen-like tricks at the block level.
    
    Actually, let's just do the direct block approach:
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C13 = A11*B13 + A12*B23
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    C23 = A21*B13 + A22*B23
    
    Each product like A11*B11 is a 2x2 x 2x2 -> 2x2 multiplication with rank 7.
    There are 6 such products, so total rank = 6 * 7 = 42.
    """
    n, m, p = 4, 4, 6
    rank = 42
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Block indices:
    # A is 4x4, split into 2x2 blocks:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    #
    # B is 4x6, split into 2x2 blocks:
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B13: rows 0-1, cols 4-5
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    # B23: rows 2-3, cols 4-5
    #
    # C is 4x6, split into 2x2 blocks:
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C13: rows 0-1, cols 4-5
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    # C23: rows 2-3, cols 4-5
    
    # For each block product, we use Strassen decomposition
    # Block product (ai, aj, bk, bk_offset) where:
    # - ai: block row index of A (0 or 1)
    # - aj: block col index of A (0 or 1)  
    # - bk: block row index of B (0 or 1)
    # - bk_offset: block col index of B (0, 1, or 2)
    
    def add_block_product(U, V, W, base_rank, a_block_row, a_block_col, b_block_row, b_block_col):
        """Add a 2x2 x 2x2 -> 2x2 Strassen decomposition starting at base_rank."""
        # Get the Strassen decomposition
        U_s, V_s, W_s = _strassen_2x2_decomposition()
        
        for r in range(7):
            col = base_rank + r
            # Map 2x2 indices to 4x4/4x6 indices
            # A block (a_block_row, a_block_col): rows a_block_row*2 to a_block_row*2+1, cols a_block_col*2 to a_block_col*2+1
            # B block (b_block_row, b_block_col): rows b_block_row*2 to b_block_row*2+1, cols b_block_col*2 to b_block_col*2+1
            # C block (a_block_row, b_block_col): rows a_block_row*2 to a_block_row*2+1, cols b_block_col*2 to b_block_col*2+1
            
            a_row_base = a_block_row * 2
            a_col_base = a_block_col * 2
            b_row_base = b_block_row * 2
            b_col_base = b_block_col * 2
            c_row_base = a_block_row * 2
            c_col_base = b_block_col * 2
            
            # U: (n*m, rank) -> index [row, col] where row = i*m+j for A[i,j]
            # In 2x2 Strassen, U[i*2+j, r] maps to U[(a_row_base+i)*4 + (a_col_base+j), col]
            for i in range(2):
                for j in range(2):
                    U[(a_row_base+i)*4 + (a_col_base+j)][col] += U_s[i*2+j][r]
            
            # V: (m*p, rank) -> index [row, col] where row = j*p+k for B[j,k]
            # In 2x2 Strassen, V[j*2+k, r] maps to V[(b_row_base+j)*6 + (b_col_base+k), col]
            for j in range(2):
                for k in range(2):
                    V[(b_row_base+j)*6 + (b_col_base+k)][col] += V_s[j*2+k][r]
            
            # W: (p*n, rank) -> index [row, col] where row = k*n+i for C[i,k]
            # In 2x2 Strassen, W[k*2+i, r] maps to W[(c_col_base+k)*4 + (c_row_base+i), col]
            for k in range(2):
                for i in range(2):
                    W[(c_col_base+k)*4 + (c_row_base+i)][col] += W_s[k*2+i][r]
    
    # C11 = A11*B11 + A12*B21
    add_block_product(U, V, W, 0, 0, 0, 0, 0)   # A11*B11
    add_block_product(U, V, W, 7, 0, 1, 1, 0)   # A12*B21
    
    # C12 = A11*B12 + A12*B22
    add_block_product(U, V, W, 14, 0, 0, 0, 1)  # A11*B12
    add_block_product(U, V, W, 21, 0, 1, 1, 1)  # A12*B22
    
    # C13 = A11*B13 + A12*B23
    add_block_product(U, V, W, 28, 0, 0, 0, 2)  # A11*B13
    add_block_product(U, V, W, 35, 0, 1, 1, 2)  # A12*B23
    
    # C21 = A21*B11 + A22*B21
    add_block_product(U, V, W, 42, 1, 0, 0, 0)  # A21*B11
    add_block_product(U, V, W, 49, 1, 1, 1, 0)  # A22*B21
    
    # C22 = A21*B12 + A22*B22
    add_block_product(U, V, W, 56, 1, 0, 0, 1)  # A21*B12
    add_block_product(U, V, W, 63, 1, 1, 1, 1)  # A22*B22
    
    # C23 = A21*B13 + A22*B23
    add_block_product(U, V, W, 70, 1, 0, 0, 2)  # A21*B13
    add_block_product(U, V, W, 77, 1, 1, 1, 2)  # A22*B23
    
    # We have 12 block products * 7 = 84 rank terms, but we only allocated 42
    # Wait, I made an error. Let me recalculate.
    # We have 6 C blocks, each is sum of 2 products, so 12 products total, each rank 7 = 84
    # But we can combine: for each C block, we only need 7 terms total (Strassen on the block level)
    
    # Actually, let me redo this properly. The block-level multiplication is:
    # [A11 A12] * [B11 B12 B13] = [A11*B11+A12*B21, A11*B12+A12*B22, A11*B13+A12*B23]
    # [A21 A22]   [B21 B22 B23]   [A21*B11+A22*B21, A21*B12+A22*B22, A21*B13+A22*B23]
    
    # Each entry like A11*B11+A12*B21 is a sum of two 2x2x2->2 products.
    # We can't directly apply Strassen to the sum. We need to compute each product separately.
    # So total rank = 12 * 7 = 84.
    
    # Let me fix the rank allocation
    rank = 84
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    base = 0
    # C11
    add_block_product(U, V, W, base, 0, 0, 0, 0); base += 7
    add_block_product(U, V, W, base, 0, 1, 1, 0); base += 7
    # C12
    add_block_product(U, V, W, base, 0, 0, 0, 1); base += 7
    add_block_product(U, V, W, base, 0, 1, 1, 1); base += 7
    # C13
    add_block_product(U, V, W, base, 0, 0, 0, 2); base += 7
    add_block_product(U, V, W, base, 0, 1, 1, 2); base += 7
    # C21
    add_block_product(U, V, W, base, 1, 0, 0, 0); base += 7
    add_block_product(U, V, W, base, 1, 1, 1, 0); base += 7
    # C22
    add_block_product(U, V, W, base, 1, 0, 0, 1); base += 7
    add_block_product(U, V, W, base, 1, 1, 1, 1); base += 7
    # C23
    add_block_product(U, V, W, base, 1, 0, 0, 2); base += 7
    add_block_product(U, V, W, base, 1, 1, 1, 2); base += 7
    
    return U, V, W, rank

def _als_polish(U, V, W, rank, n_iters=50, tol=1e-14):
    """Polish a near-optimal decomposition using ALS."""
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for it in range(n_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((rank, p))
                for r in range(rank):
                    M[r, :] = Va[j*p:(j+1)*p, r] * Wa[:, i]
                    # Actually, Wa[k*n+i] for k in range(p)
                    M[r, :] = [Va[j*p+k, r] * Wa[k*n+i, r] for k in range(p)]
                try:
                    Ua[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((rank, n))
                for r in range(rank):
                    M[r, :] = [Ua[i*m+j, r] * Wa[k*n+i, r] for i in range(n)]
                try:
                    Va[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((rank, m))
                for r in range(rank):
                    M[r, :] = [Ua[i*m+j, r] * Va[j*p+k, r] for j in range(m)]
                try:
                    Wa[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pred = np.dot(Ua[i*m+j, :], Va[j*p+k, :] * Wa[k*n+i, :])
                    err += (pred - 1.0)**2
        if np.sqrt(err) < tol:
            break
    
    # Convert back to list of lists
    U_out = Ua.tolist()
    V_out = Va.tolist()
    W_out = Wa.tolist()
    return U_out, V_out, W_out

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with block Strassen construction (rank 84)
    U_block, V_block, W_block, rank_block = _block_strassen_4x4x4x6()
    
    # Verify
    if not _verify_decomposition(U_block, V_block, W_block, n, m, p, tol=1e-8):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0]*rank_school for _ in range(n*m)]
        V_s = [[0]*rank_school for _ in range(m*p)]
        W_s = [[0]*rank_school for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col] = 1
                    V_s[j*p+k][col] = 1
                    W_s[k*n+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    best_result = {"U": U_block, "V": V_block, "W": W_block}
    best_rank = rank_block
    
    # Try to polish and reduce rank
    # First, polish the block decomposition
    if time.time() < deadline - 30:
        U_polished, V_polished, W_polished = _als_polish(U_block, V_block, W_block, rank_block, n_iters=30)
        if _verify_decomposition(U_polished, V_polished, W_polished, n, m, p, tol=1e-10):
            best_result = {"U": U_polished, "V": V_polished, "W": W_polished}
    
    # Try random search for lower ranks
    for rank in range(40, 84):
        if time.time() > deadline - 20:
            break
        if rank >= best_rank:
            continue
        
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            
            # Generate random candidate with integer/half-integer values
            rng = np.random.RandomState(trial * 1000 + rank)
            U_rand = rng.randint(-2, 3, size=(n*m, rank)) * 0.5
            V_rand = rng.randint(-2, 3, size=(m*p, rank)) * 0.5
            W_rand = rng.randint(-2, 3, size=(p*n, rank)) * 0.5
            
            # Quick check
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_rand[i*m+j, :] * V_rand[j*p+k, :], W_rand[k*n+i, :])
                        if abs(val - 1.0) > 0.1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Polish
                U_p, V_p, W_p = _als_polish(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), rank, n_iters=50)
                if _verify_decomposition(U_p, V_p, W_p, n, m, p, tol=1e-10):
                    best_result = {"U": U_p, "V": V_p, "W": W_p}
                    best_rank = rank
                    break
    
    return best_result
# EVOLVE-BLOCK-END
