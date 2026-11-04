"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need T[i,j,k] = 1 for all i,j,k (since A_{i,j} B_{j,k} contributes to C_{i,k})
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _build_strassen_2x2_2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # p1 = (a00+a11)(b00+b11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # p2 = (a10+a11)b00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # p3 = a00(b01-b11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1
    
    # p4 = a11(b10-b00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = -1
    W[2,3] = 1
    
    # p5 = (a00+a01)b11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = 1
    W[2,4] = 1
    
    # p6 = (a10-a00)(b00+b01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = -1
    
    # p7 = (a01-a11)(b10+b11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = -1
    
    return U, V, W

def _build_strassen_4x4_4x4():
    """
    Build the rank-28 Strassen-based decomposition for 4x4 matrix multiplication.
    Uses 4 sub-blocks of 2x2, each with rank 7.
    Returns U (16x28), V (16x28), W (16x28)
    """
    n, m, p = 4, 4, 4
    U_sub, V_sub, W_sub = _build_strassen_2x2_2x2()
    
    U = np.zeros((n*m, 28))
    V = np.zeros((m*p, 28))
    W = np.zeros((p*n, 28))
    
    # 4 sub-blocks: (block_row, block_col)
    # For 4x4 * 4x4:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each Cij is sum of two 2x2*2x2 multiplications
    # But Strassen works on the whole 2x2 block multiplication
    # 
    # Actually, for 4x4*4x4, we partition into 2x2 blocks and apply Strassen
    # to each pair. But the standard Strassen recursion gives:
    # 4x4 mult = 7 * (2x2 mult) = 7 * 7 * (1x1 mult) = 49 mults
    # 
    # Wait, that's not right. Strassen for 4x4:
    # Partition 4x4 into four 2x2 blocks.
    # The Strassen algorithm for multiplying two 4x4 matrices using 2x2 sub-blocks:
    # 
    # M1 = (A00+A11)(B00+B11)  -- 2x2 * 2x2
    # M2 = (A10+A11)B00         -- 2x2 * 2x2
    # M3 = A00(B01-B11)         -- 2x2 * 2x2
    # M4 = A11(B10-B00)         -- 2x2 * 2x2
    # M5 = (A00+A01)B11         -- 2x2 * 2x2
    # M6 = (A10-A00)(B00+B01)   -- 2x2 * 2x2
    # M7 = (A01-A11)(B10+B11)   -- 2x2 * 2x2
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 - M4
    #
    # Each Mi is a 2x2 * 2x2 multiplication, which we decompose with rank 7.
    # So total rank = 7 * 7 = 49.
    
    # Let me construct this properly.
    # For each Strassen term Mi, we need to express it as a product of
    # linear combinations of A blocks and B blocks.
    
    # This is getting complex. Let me use a different approach:
    # Directly construct the 4x4*4x4 decomposition using the Kronecker product
    # of the 2x2 Strassen decomposition with itself.
    
    # Actually, the simplest correct approach: the rank of 4x4*4x4 via Strassen
    # is 7^2 = 49. Let me verify this is correct and build it.
    
    # Alternative: just use schoolbook for 4x4*4x4 which is rank 64.
    # 49 < 64, so Strassen is better, but let me make sure I can construct it.
    
    # Let me use a cleaner construction. The 4x4*4x4 tensor can be decomposed
    # by viewing it as a 2x2 block matrix multiplication where each block is 2x2.
    # The block-level Strassen gives 7 block multiplications, each of which is
    # a 2x2*2x2 multiplication with rank 7. Total: 49.
    
    # For each of the 7 Strassen block multiplications, I need to map the
    # 2x2*2x2 rank-7 decomposition into the appropriate positions.
    
    # Let me define the Strassen coefficients for the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mi = (sum of A blocks) * (sum of B blocks)
    # For the tensor decomposition, each Mi contributes to specific C blocks.
    
    # This is quite involved. Let me just return schoolbook for now and
    # focus on the overall 4x5x6 strategy.
    
    # For the 4x5x6 problem, let me try a block decomposition approach.
    return None

def _build_block_decomposition(n, m, p):
    """
    Try to build a block-based decomposition for n x m * m x p.
    Partition A into blocks and B into corresponding blocks,
    use Strassen where beneficial, schoolbook elsewhere.
    """
    n, m, p = 4, 5, 6
    
    # Strategy: Partition A (4x5) and B (5x6) into blocks.
    # 
    # Option 1: Split along the middle dimension (j=0..4)
    # A = [A_left (4x2) | A_right (4x3)]
    # B = [B_top (2x6); B_bottom (3x6)]
    # C = A_left * B_top + A_right * B_bottom
    # 
    # A_left * B_top: 4x2 * 2x6, schoolbook rank = 4*2*6 = 48
    # A_right * B_bottom: 4x3 * 3x6, schoolbook rank = 4*3*6 = 72
    # Total: 120 (same as schoolbook, no gain)
    
    # Option 2: Use Strassen on sub-blocks
    # Split A into two 2x5 blocks: A1 (2x5), A2 (2x5)
    # Split B into two 5x3 blocks: B1 (5x3), B2 (5x3)
    # C1 = A1*B1 (2x3), C2 = A1*B2 (2x3), C3 = A2*B1 (2x3), C4 = A2*B2 (2x3)
    # Each 2x3*... wait, A1 is 2x5, B1 is 5x3, so A1*B1 is 2x3.
    # Schoolbook rank for 2x5*5x3 = 2*5*3 = 30 each, total 120. No gain.
    
    # Option 3: Try to use the fact that we can share computations
    # This requires a more sophisticated approach.
    
    # For now, let me try an ALS-based search for a lower-rank decomposition.
    return None

def _als_search(n, m, p, target_rank, max_iters=100, tol=1e-10, seed=42, deadline=None):
    """
    Use ALS (Alternating Least Squares) to search for a rank-`target_rank`
    decomposition of the matrix multiplication tensor.
    """
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    np.random.seed(seed)
    
    U = np.random.randn(nm, target_rank)
    V = np.random.randn(mp, target_rank)
    W = np.random.randn(pn, target_rank)
    
    def verify(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for t in range(r):
            uv = np.outer(U[:,t], V[:,t])
            T_recon += np.einsum('ab,c->abc', uv, W[:,t])
        return np.max(np.abs(T_recon - T))
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Fix V, W: solve for U
        # For each a: sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c]
        # X[t, (b,c)] = V[b,t] * W[c,t]
        X = np.einsum('bt,ct->bct', V, W).reshape(target_rank, mp*pn).T
        for a in range(nm):
            U[a, :] = np.linalg.lstsq(X, T[a, :, :].reshape(-1), rcond=None)[0]
        
        # Fix U, W: solve for V
        X = np.einsum('at,ct->act', U, W).reshape(target_rank, nm*pn).T
        for b in range(mp):
            V[b, :] = np.linalg.lstsq(X, T[:, b, :].reshape(-1), rcond=None)[0]
        
        # Fix U, V: solve for W
        X = np.einsum('at,bt->abt', U, V).reshape(target_rank, nm*mp).T
        for c in range(pn):
            W[c, :] = np.linalg.lstsq(X, T[:, :, c].reshape(-1), rcond=None)[0]
        
        err = verify(U, V, W)
        if err < tol:
            break
    
    err = verify(U, V, W)
    if err < tol:
        return U, V, W, True
    return U, V, W, False

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    nm, mp, pn = n*m, m*p, p*n  # 20, 30, 24
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    # Start with schoolbook decomposition
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_decomp = (U_sb.copy(), V_sb.copy(), W_sb.copy())
    
    # Try ALS search for lower ranks
    # The theoretical minimum is at least max(n*p, m*p, n*m) = max(24, 30, 20) = 30
    # but realistically, for 4x5x6, we expect something between 30 and 120.
    
    # Try a range of target ranks
    target_ranks = [40, 50, 60, 70, 80, 90, 100, 110]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            U, V, W, success = _als_search(
                n, m, p, target_rank, 
                max_iters=150, tol=1e-10, 
                seed=42 + attempt,
                deadline=deadline - 15
            )
            
            if success and target_rank < best_rank:
                best_rank = target_rank
                best_decomp = (U.copy(), V.copy(), W.copy())
                break  # Found valid decomposition for this rank, try lower
    
    # Return the best decomposition found
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
