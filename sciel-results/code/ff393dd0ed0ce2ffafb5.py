"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

def _snap_to_half(x):
    return round(2 * x) / 2.0

def _snap_numpy(arr):
    return np.round(2 * arr) / 2.0

def _verify_numpy(Ua, Va, Wa, n, m, p, atol=1e-9):
    if Ua.shape != (n*m, Ua.shape[1]) or Va.shape != (m*p, Va.shape[1]) or Wa.shape != (p*n, Wa.shape[1]):
        return False
    if Ua.shape[1] != Va.shape[1] or Ua.shape[1] != Wa.shape[1]:
        return False
    rank = Ua.shape[1]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, np.ones((n, m, p)), atol=atol)

def _verify_fast(U, V, W, n, m, p):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-9):
            return False
    return _verify_numpy(Ua, Va, Wa, n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication using Strassen (rank 7).
    U: (2*2, 7) = (4, 7)
    V: (2*2, 7) = (4, 7)
    W: (2*2, 7) = (4, 7)
    
    Strassen formulas:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U factors for A blocks: A11, A12, A21, A22
    # V factors for B blocks: B11, B12, B21, B22
    # W factors for C blocks: C11, C12, C21, C22
    
    # Each of the 7 Strassen products has U and V coefficients
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U: rows correspond to A11, A12, A21, A22 (indices 0,1,2,3)
    # V: rows correspond to B11, B12, B21, B22 (indices 0,1,2,3)
    # W: rows correspond to C11, C12, C21, C22 (indices 0,1,2,3)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: U=[1,0,0,1], V=[1,0,0,1]
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1  # C11 += M1
    
    # M2: U=[0,0,1,1], V=[1,0,0,0]
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1  # C21 += M2
    
    # M3: U=[1,0,0,0], V=[0,0,1,-1]
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[1,2] = 1  # C12 += M3
    W[3,2] = 1  # C22 += M3
    
    # M4: U=[0,0,0,1], V=[-1,1,0,0]
    U[3,3] = 1
    V[0,3] = -1; V[1,3] = 1
    W[0,3] = 1  # C11 += M4
    W[2,3] = 1  # C21 += M4
    
    # M5: U=[1,1,0,0], V=[0,0,0,1]
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1  # C11 -= M5
    
    # M6: U=[-1,0,1,0], V=[1,1,0,0]
    U[0,5] = -1; U[2,5] = 1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1  # C22 += M6
    
    # M7: U=[0,1,0,-1], V=[0,0,1,1]
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1  # C11 += M7
    
    return U, V, W

def _strassen_block_decomposition(n, m, p):
    """
    Use block structure with Strassen for 3x5x6.
    
    Partition:
    - A is 3x5: split rows as [2,1], cols as [2,3]
    - B is 5x6: split rows as [2,3], cols as [2,4]
    - C is 3x6: split rows as [2,1], cols as [2,4]
    
    Actually, let's use a simpler approach:
    Split into 2x2 blocks where possible, using Strassen for 2x2x2 subproblems.
    
    For 3x5x6:
    - Split A (3x5) into blocks: top-left 2x2, top-right 2x3, bottom-left 1x2, bottom-right 1x3
    - Split B (5x6) into blocks: top-left 2x2, top-right 2x4, bottom-left 3x2, bottom-right 3x4
    - Split C (3x6) into blocks: top-left 2x2, top-right 2x4, bottom-left 1x2, bottom-right 1x4
    
    This is complex. Let me use a different approach:
    Use the fact that we can decompose into independent subproblems.
    
    Actually, the simplest improvement: use Strassen for the 2x2x2 corner, 
    and schoolbook for the rest.
    """
    # Let's use a cleaner block decomposition
    # Split dimensions: n=3 -> [2,1], m=5 -> [2,3], p=6 -> [2,4]
    
    n1, n2 = 2, 1
    m1, m2 = 2, 3
    p1, p2 = 2, 4
    
    # Subproblems:
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2]  -> 2x2x2 Strassen (rank 7)
    # C[0:2, 2:6] = A[0:2, 0:2] * B[0:2, 2:6]  -> 2x2x4 schoolbook (rank 16)
    # C[0:2, 2:6] += A[0:2, 2:5] * B[2:5, 2:6]  -> 2x3x4 schoolbook (rank 24)
    # C[2:3, 0:2] = A[2:3, 0:2] * B[0:2, 0:2]  -> 1x2x2 schoolbook (rank 4)
    # C[2:3, 0:2] += A[2:3, 2:5] * B[2:5, 0:2]  -> 1x3x2 schoolbook (rank 6)
    # C[2:3, 2:6] = A[2:3, 0:2] * B[0:2, 2:6]  -> 1x2x4 schoolbook (rank 8)
    # C[2:3, 2:6] += A[2:3, 2:5] * B[2:5, 2:6]  -> 1x3x4 schoolbook (rank 12)
    
    # Total rank: 7 + 16 + 24 + 4 + 6 + 8 + 12 = 77
    
    # But wait, I need to be more careful about how the blocks overlap.
    # Let me use a simpler approach: just replace the 2x2x2 corner with Strassen.
    
    # Start with schoolbook rank 90
    # Replace the 2x2x2 subproblem at (rows 0:2, cols 0:2) with Strassen
    # This saves 8 - 7 = 1 rank, giving 89.
    
    # That's not great. Let me think differently.
    
    # Better approach: partition into non-overlapping blocks
    # n=3: blocks [2,1]
    # m=5: blocks [2,3]  
    # p=6: blocks [2,4]
    
    # Subproblem (i_block, j_block, k_block) has size n_i x m_j x p_k
    # There are 2*2*2 = 8 subproblems:
    # (0,0,0): 2x2x2 -> Strassen rank 7
    # (0,0,1): 2x2x4 -> schoolbook rank 16
    # (0,1,0): 2x3x2 -> schoolbook rank 12
    # (0,1,1): 2x3x4 -> schoolbook rank 24
    # (1,0,0): 1x2x2 -> schoolbook rank 4
    # (1,0,1): 1x2x4 -> schoolbook rank 8
    # (1,1,0): 1x3x2 -> schoolbook rank 6
    # (1,1,1): 1x3x4 -> schoolbook rank 12
    
    # Total: 7+16+12+24+4+8+6+12 = 89
    
    # Hmm, that's only 89, not 84. The inspiration said ~84.
    # Maybe they used a different partition or a more clever combination.
    
    # Let me try: n=3 -> [1,2], m=5 -> [3,2], p=6 -> [4,2]
    # (0,0,0): 1x3x4 -> 12
    # (0,0,1): 1x3x2 -> 6
    # (0,1,0): 1x2x4 -> 8
    # (0,1,1): 1x2x2 -> 4
    # (1,0,0): 2x3x4 -> 24
    # (1,0,1): 2x3x2 -> 12
    # (1,1,0): 2x2x4 -> 16
    # (1,1,1): 2x2x2 -> 7 (Strassen)
    # Total: 12+6+8+4+24+12+16+7 = 89
    
    # Same. Let me try to use Strassen more aggressively.
    # What if I use Strassen on multiple 2x2x2 blocks?
    
    # n=3 -> [2,1], m=5 -> [2,2,1], p=6 -> [2,2,2]
    # This gives 2*3*3 = 18 subproblems, most of which are 2x2x2.
    # But the 1x* and *x1 blocks use schoolbook.
    
    # Actually, let me just implement the block approach with Strassen for 2x2x2 blocks.
    
    total_rank = 0
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    for ni, mi, pi in zip(n_parts, m_parts, p_parts):
        if ni == 2 and mi == 2 and pi == 2:
            # Use Strassen
            Us, Vs, Ws = _strassen_2x2x2()
            U_blocks.append(Us)
            V_blocks.append(Vs)
            W_blocks.append(Ws)
            total_rank += 7
        else:
            # Use schoolbook
            r = ni * mi * pi
            U_blocks.append(np.zeros((ni*mi, r)))
            V_blocks.append(np.zeros((mi*pi, r)))
            W_blocks.append(np.zeros((pi*ni, r)))
            total_rank += r
    
    # Now assemble the full U, V, W
    # U has rows indexed by (i, j) where i in 0..n-1, j in 0..m-1
    # V has rows indexed by (j, k) where j in 0..m-1, k in 0..p-1
    # W has rows indexed by (k, i) where k in 0..p-1, i in 0..n-1
    
    n, m, p = 3, 5, 6
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    col_offset = 0
    n_off = 0
    m_off = 0
    p_off = 0
    
    for ni, mi, pi in zip(n_parts, m_parts, p_parts):
        block_rank = ni * mi * pi if not (ni == 2 and mi == 2 and pi == 2) else 7
        U_block = U_blocks.pop(0)
        V_block = V_blocks.pop(0)
        W_block = W_blocks.pop(0)
        
        # Place U_block into U_full
        # U_block has shape (ni*mi, block_rank)
        # Rows correspond to i in [n_off, n_off+ni), j in [m_off, m_off+mi)
        for local_i in range(ni):
            for local_j in range(mi):
                global_i = n_off + local_i
                global_j = m_off + local_j
                U_full[global_i*m + global_j, col_offset:col_offset+block_rank] = U_block[local_i*mi + local_j]
        
        # Place V_block into V_full
        # V_block has shape (mi*pi, block_rank)
        # Rows correspond to j in [m_off, m_off+mi), k in [p_off, p_off+pi)
        for local_j in range(mi):
            for local_k in range(pi):
                global_j = m_off + local_j
                global_k = p_off + local_k
                V_full[global_j*p + global_k, col_offset:col_offset+block_rank] = V_block[local_j*pi + local_k]
        
        # Place W_block into W_full
        # W_block has shape (pi*ni, block_rank)
        # Rows correspond to k in [p_off, p_off+pi), i in [n_off, n_off+ni)
        for local_k in range(pi):
            for local_i in range(ni):
                global_k = p_off + local_k
                global_i = n_off + local_i
                W_full[global_k*n + global_i, col_offset:col_offset+block_rank] = W_block[local_k*ni + local_i]
        
        col_offset += block_rank
        n_off += ni
        m_off += mi
        p_off += pi
    
    return U_full, V_full, W_full

def _alt_solve(U, V, W, n, m, p, max_iter=50, tol=1e-12):
    """Alternating linear solve to polish a near-solution."""
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    rank = Ua.shape[1]
    
    for it in range(max_iter):
        if _remaining() < 1.0:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                V_j = Va[j*p:(j+1)*p, :]
                W_i = Wa[i::n, :]
                M = V_j * W_i
                try:
                    Ua[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                U_j = Ua[j::m, :]
                W_k = Wa[k::p, :]
                M = U_j * W_k
                try:
                    Va[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                U_i = Ua[i*m:(i+1)*m, :]
                V_k = Va[k::p, :]
                M = U_i * V_k
                try:
                    Wa[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < tol:
            break
    
    return Ua, Va, Wa

def _try_rank(n, m, p, rank, max_seeds=20, time_per_seed=2.0):
    """Try to find a valid decomposition of given rank."""
    if _remaining() < 3.0:
        return None
    
    Ua = np.zeros((n*m, rank))
    Va = np.zeros((m*p, rank))
    Wa = np.zeros((p*n, rank))
    
    for seed in range(max_seeds):
        if _remaining() < 3.0:
            break
        
        rng = random.Random(seed * 7919 + rank * 104729)
        
        # Initialize with random values
        Ua = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)], dtype=float)
        Va = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)], dtype=float)
        Wa = np.array([[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)], dtype=float)
        
        # Run alternating solve
        Ua, Va, Wa = _alt_solve(Ua, Va, Wa, n, m, p, max_iter=30, tol=1e-10)
        
        # Check error
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.max(np.abs(T - 1.0))
        
        if error < 1e-8:
            # Try to snap to half-integers
            Ua_s = _snap_numpy(Ua)
            Va_s = _snap_numpy(Va)
            Wa_s = _snap_numpy(Wa)
            
            if _verify_numpy(Ua_s, Va_s, Wa_s, n, m, p, atol=1e-9):
                return Ua_s, Va_s, Wa_s
            
            # Try without snapping
            if _verify_numpy(Ua, Va, Wa, n, m, p, atol=1e-9):
                return Ua, Va, Wa
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    # Try Strassen block decomposition first
    try:
        U_str, V_str, W_str = _strassen_block_decomposition(n, m, p)
        if _verify_numpy(U_str, V_str, W_str, n, m, p, atol=1e-9):
            best = {"U": U_str.tolist(), "V": V_str.tolist(), "W": W_str.tolist()}
            best_rank = U_str.shape[1]
    except:
        pass
    
    min_rank = max(n*m, m*p, n*p)
    
    # Try ranks from min_rank upward, but be smart about time budgeting
    for rank in range(min_rank, best_rank + 1):
        if _remaining() < 10.0:
            break
        
        result = _try_rank(n, m, p, rank, max_seeds=15, time_per_seed=2.0)
        if result is not None:
            Ua, Va, Wa = result
            if _verify_numpy(Ua, Va, Wa, n, m, p):
                best = {"U": Ua.tolist(), "V": Va.tolist(), "W": Wa.tolist()}
                best_rank = rank
                break
    
    # Final validation
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        best = schoolbook
    
    return best
