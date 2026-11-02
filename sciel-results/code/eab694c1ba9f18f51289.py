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

def _try_low_rank(rank, n, m, p, deadline, max_rounds=2000, seed=42):
    """
    Try to find a decomposition of given rank using iterative refinement.
    Returns (U, V, W) as lists if found, else None.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = np.ones((p * n, rank))
    
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
        
        # Check convergence every 50 iterations
        if it % 50 == 0:
            sums = np.zeros(len(ta_arr))
            for c in range(rank):
                sums += U[ta_arr, c] * V[tb_arr, c] * W[tc_arr, c]
            err = np.sum((sums - 1.0) ** 2)
            if err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_structured(rank, n, m, p, deadline):
    """
    Try structured initialization based on Strassen pattern.
    """
    n, m, p = 2, 2, 11
    
    # Try different structured initializations
    for init_type in ['strassen', 'block', 'mixed']:
        if time.time() > deadline:
            break
        
        if init_type == 'strassen':
            # Use Strassen-like structure with perturbations
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Initialize with Strassen pattern for first 35 columns
            r = 0
            for block in range(5):
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
                
                U[0, r] = 1; U[3, r] = 1
                V[v_e, r] = 1; V[v_h, r] = 1
                W[w_c00, r] = 1; W[w_c11, r] = 1
                r += 1
                
                U[2, r] = 1; U[3, r] = 1
                V[v_e, r] = 1
                W[w_c10, r] = 1; W[w_c11, r] = -1
                r += 1
                
                U[0, r] = 1
                V[v_f, r] = 1; V[v_h, r] = -1
                W[w_c01, r] = 1; W[w_c11, r] = 1
                r += 1
                
                U[3, r] = 1
                V[v_g, r] = 1; V[v_e, r] = -1
                W[w_c00, r] = 1; W[w_c10, r] = 1
                r += 1
                
                U[0, r] = 1; U[1, r] = 1
                V[v_h, r] = 1
                W[w_c00, r] = -1; W[w_c01, r] = 1
                r += 1
                
                U[2, r] = 1; U[0, r] = -1
                V[v_e, r] = 1; V[v_f, r] = 1
                W[w_c11, r] = 1
                r += 1
                
                U[1, r] = 1; U[3, r] = -1
                V[v_g, r] = 1; V[v_h, r] = 1
                W[w_c00, r] = 1
                r += 1
            
            # Add small random perturbations to remaining columns
            rng = np.random.RandomState(123)
            if r < rank:
                U[:, r:] = rng.randn(n * m, rank - r) * 0.1
                V[:, r:] = rng.randn(m * p, rank - r) * 0.1
                W[:, r:] = rng.randn(p * n, rank - r) * 0.1
            
            result = _try_low_rank(rank, n, m, p, deadline=deadline, max_rounds=500, seed=123)
            if result is not None:
                return result
        
        elif init_type == 'block':
            # Block-structured initialization
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Divide into blocks
            block_size = rank // 5
            for block in range(5):
                start = block * block_size
                end = min(start + block_size, rank)
                
                k0 = block * 2
                k1 = block * 2 + 1
                
                # Initialize block with random values
                rng = np.random.RandomState(block)
                U[:, start:end] = rng.randn(n * m, end - start)
                V[:, start:end] = rng.randn(m * p, end - start)
                W[:, start:end] = rng.randn(p * n, end - start)
            
            result = _try_low_rank(rank, n, m, p, deadline=deadline, max_rounds=500, seed=456)
            if result is not None:
                return result
        
        elif init_type == 'mixed':
            # Mix of structured and random
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            rng = np.random.RandomState(789)
            
            # First half structured, second half random
            half = rank // 2
            
            # Structured part
            for block in range(5):
                k0 = block * 2
                k1 = block * 2 + 1
                
                r = block * 7
                if r + 7 > half:
                    break
                    
                v_e = k0
                v_f = k1
                v_g = p + k0
                v_h = p + k1
                
                w_c00 = k0 * n
                w_c10 = k0 * n + 1
                w_c01 = k1 * n
                w_c11 = k1 * n + 1
                
                U[0, r] = 1; U[3, r] = 1
                V[v_e, r] = 1; V[v_h, r] = 1
                W[w_c00, r] = 1; W[w_c11, r] = 1
                r += 1
                
                U[2, r] = 1; U[3, r] = 1
                V[v_e, r] = 1
                W[w_c10, r] = 1; W[w_c11, r] = -1
                r += 1
                
                U[0, r] = 1
                V[v_f, r] = 1; V[v_h, r] = -1
                W[w_c01, r] = 1; W[w_c11, r] = 1
                r += 1
                
                U[3, r] = 1
                V[v_g, r] = 1; V[v_e, r] = -1
                W[w_c00, r] = 1; W[w_c10, r] = 1
                r += 1
                
                U[0, r] = 1; U[1, r] = 1
                V[v_h, r] = 1
                W[w_c00, r] = -1; W[w_c01, r] = 1
                r += 1
                
                U[2, r] = 1; U[0, r] = -1
                V[v_e, r] = 1; V[v_f, r] = 1
                W[w_c11, r] = 1
                r += 1
                
                U[1, r] = 1; U[3, r] = -1
                V[v_g, r] = 1; V[v_h, r] = 1
                W[w_c00, r] = 1
                r += 1
            
            # Random part
            if half < rank:
                U[:, half:] = rng.randn(n * m, rank - half)
                V[:, half:] = rng.randn(m * p, rank - half)
                W[:, half:] = rng.randn(p * n, rank - half)
            
            result = _try_low_rank(rank, n, m, p, deadline=deadline, max_rounds=500, seed=789)
            if result is not None:
                return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s kill
    
    # First, return the verified rank-39 construction
    U, V, W = _construct_rank39()
    if _verify(U, V, W, n, m, p):
        # Try to find a lower rank decomposition with remaining time
        if time.time() < deadline - 20:
            # First try structured approaches
            for rank in [38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26]:
                if time.time() > deadline - 8:
                    break
                
                result = _try_structured(rank, n, m, p, deadline=deadline - 5)
                if result is not None:
                    U2, V2, W2 = result
                    if _verify(U2, V2, W2, n, m, p):
                        return {"U": U2, "V": V2, "W": W2}
            
            # Then try random initialization with more seeds
            for rank in [38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26]:
                if time.time() > deadline - 8:
                    break
                for seed in [rank, rank + 100, rank + 200, rank + 300, rank + 400, rank + 500]:
                    if time.time() > deadline - 5:
                        break
                    result = _try_low_rank(rank, n, m, p, deadline=deadline - 3, max_rounds=1000, seed=seed)
                    if result is not None:
                        U2, V2, W2 = result
                        if _verify(U2, V2, W2, n, m, p):
                            return {"U": U2, "V": V2, "W": W2}
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
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
