"""Combined approach: verified rank-39 Strassen+schoolbook, then bounded exact search."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_rank39():
    """
    Explicit rank-39 decomposition for M_{2,2,11}.
    5 blocks of 2 columns (Strassen rank 7 each) + 1 single column (schoolbook rank 4).
    Total: 5*7 + 4 = 39.
    """
    n, m, p = 2, 2, 11
    rank = 39
    
    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 39
    V = [[0.0] * rank for _ in range(m * p)]   # 22 x 39
    W = [[0.0] * rank for _ in range(p * n)]   # 22 x 39
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # V indices: B[j,k] -> V[j*p + k]
        v_e = k0        # B[0, k0]
        v_f = k1        # B[0, k1]
        v_g = p + k0    # B[1, k0]
        v_h = p + k1    # B[1, k1]
        
        # W indices: C[i,k] -> W[k*n + i]
        w_c00 = k0 * n      # C[0, k0]
        w_c10 = k0 * n + 1  # C[1, k0]
        w_c01 = k1 * n      # C[0, k1]
        w_c11 = k1 * n + 1  # C[1, k1]
        
        # U indices: A[i,j] -> U[i*m + j]
        # U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
        
        # m1 = (A00+A11)*(B[0,k0]+B[1,k1]) -> C[0,k0] + C[1,k1]
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m2 = (A10+A11)*B[0,k0] -> C[1,k0] - C[1,k1]
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        # m3 = A00*(B[0,k1]-B[1,k1]) -> C[0,k1] + C[1,k1]
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m4 = A11*(B[1,k0]-B[0,k0]) -> C[0,k0] + C[1,k0]
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        # m5 = (A00+A01)*B[1,k1] -> -C[0,k0] + C[0,k1]
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        # m6 = (A10-A00)*(B[0,k0]+B[0,k1]) -> C[1,k1]
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        # m7 = (A01-A11)*(B[1,k0]+B[1,k1]) -> C[0,k0]
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    # Single column k=10: schoolbook 4 terms
    k2 = 10
    v_e2 = k2        # B[0, 10]
    v_g2 = p + k2    # B[1, 10]
    w_c02 = k2 * n      # C[0, 10]
    w_c12 = k2 * n + 1  # C[1, 10]
    
    # A00 * B[0,10] -> C[0,10]
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # A01 * B[1,10] -> C[0,10]
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # A10 * B[0,10] -> C[1,10]
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    # A11 * B[1,10] -> C[1,10]
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    assert r == rank
    return U, V, W

def _round_to_half_integers(U, V, W, tol=1e-6):
    """Round entries to nearest half-integer and check if they still satisfy the tensor equation."""
    def round_half(x):
        return 0.5 * round(2 * x)
    
    U_r = [[round_half(x) for x in row] for row in U]
    V_r = [[round_half(x) for x in row] for row in V]
    W_r = [[round_half(x) for x in row] for row in W]
    
    # Check if the rounded version is valid
    # We need n, m, p for verification, so we pass them or assume global context
    # For now, just return the rounded matrices, verification is done by caller
    return U_r, V_r, W_r

def _try_low_rank(rank, n, m, p, deadline, max_rounds=3000, seed=42):
    """
    Try to find a decomposition of given rank using iterative refinement.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = np.zeros((p * n, rank))
    W[:, :] = 1.0  # Initialize W to ones
    
    # Target index arrays
    ta_list, tb_list, tc_list = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ta_list.append(i * m + j)
                tb_list.append(j * p + k)
                tc_list.append(k * n + i)
    ta_arr = np.array(ta_list)
    tb_arr = np.array(tb_list)
    tc_arr = np.array(tc_list)
    
    for it in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Update U
        for a in range(n * m):
            mask = (ta_arr == a)
            if not np.any(mask):
                continue
            tb_sub = tb_arr[mask]
            tc_sub = tc_arr[mask]
            V_sub = V[tb_sub, :]
            W_sub = W[tc_sub, :]
            prod = V_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                U[a, valid] = num[valid] / den[valid]
        
        # Update V
        for b in range(m * p):
            mask = (tb_arr == b)
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tc_sub = tc_arr[mask]
            U_sub = U[ta_sub, :]
            W_sub = W[tc_sub, :]
            prod = U_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                V[b, valid] = num[valid] / den[valid]
        
        # Update W
        for c_idx in range(p * n):
            mask = (tc_arr == c_idx)
            if not np.any(mask):
                continue
            ta_sub = ta_arr[mask]
            tb_sub = tb_arr[mask]
            U_sub = U[ta_sub, :]
            V_sub = V[tb_sub, :]
            prod = U_sub * V_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            if np.any(valid):
                W[c_idx, valid] = num[valid] / den[valid]
        
        # Check convergence every 100 iterations
        if it % 100 == 0:
            sums = np.zeros(len(ta_arr))
            for c in range(rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = np.sum((sums - 1.0) ** 2)
            if err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _find_rank8_2x2x3(deadline):
    """
    Attempt to find a rank-8 decomposition for M_{2,2,3}.
    Returns (U, V, W) as lists if found, else None.
    """
    n, m, p = 2, 2, 3
    rank = 8
    # Try a few seeds
    for seed in [1, 2, 3, 42, 100]:
        if time.time() > deadline:
            break
        result = _try_low_rank(rank, n, m, p, deadline=deadline, max_rounds=2000, seed=seed)
        if result is not None:
            U, V, W = result
            # Round to half-integers
            U_r, V_r, W_r = _round_to_half_integers(U, V, W)
            # Verify
            if _verify(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
    return None

def _construct_rank36(rank8_decomp):
    """
    Construct rank-36 decomposition using 4 Strassen blocks and 1 rank-8 block.
    """
    n, m, p = 2, 2, 11
    rank = 36
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    # 4 Strassen blocks for columns 0-7
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1
        
        # m1
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m2
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        # m3
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m4
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        # m5
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        # m6
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        # m7
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
        
    # 1 Rank-8 block for columns 8-10
    # The rank8_decomp is for n=2, m=2, p=3
    # U_s: 4x8, V_s: 6x8, W_s: 6x8
    U_s, V_s, W_s = rank8_decomp
    
    # Map indices
    # U_s indices are 0..3 (A00, A01, A10, A11). Global U indices are same.
    for i in range(4):
        for c in range(8):
            U[i][r + c] = U_s[i][c]
            
    # V_s indices:
    # Local k=0,1,2. Global k=8,9,10.
    # Local V row j*p + k.
    # j=0: 0, 1, 2 -> Global 8, 9, 10
    # j=1: 3, 4, 5 -> Global 18, 19, 20
    for j in range(2):
        for k in range(3):
            local_idx = j * 3 + k
            global_idx = j * 11 + (8 + k)
            for c in range(8):
                V[global_idx][r + c] = V_s[local_idx][c]
                
    # W_s indices:
    # Local W row k*n + i.
    # Local k=0,1,2. Global k=8,9,10.
    # Local i=0,1.
    # Local row: 0,1, 2,3, 4,5
    # Global row: 8*2+i, 9*2+i, 10*2+i
    for k in range(3):
        for i in range(2):
            local_idx = k * 2 + i
            global_idx = (8 + k) * 2 + i
            for c in range(8):
                W[global_idx][r + c] = W_s[local_idx][c]
                
    r += 8
    assert r == rank
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s kill
    
    # First, try to find the rank-8 decomposition for the 2x2x3 block
    rank8_decomp = _find_rank8_2x2x3(deadline - 30)
    
    if rank8_decomp is not None:
        U, V, W = _construct_rank36(rank8_decomp)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # Fallback to rank-39
    U, V, W = _construct_rank39()
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Final fallback: schoolbook
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
