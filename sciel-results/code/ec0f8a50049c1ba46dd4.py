"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Build the schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 54
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    def to_half_integer(x):
        """Round to nearest 0.5"""
        return np.round(2.0 * x) / 2.0
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Check if decomposition is exact"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_strict(U, V, W, n, m, p):
        """Check if decomposition is exact with strict integer/half-integer check"""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return verify(U, V, W, n, m, p, tol=1e-12)
    
    # Strategy: Use Kronecker product structure to decompose 3x9 into three 3x3 blocks.
    # For each block b in [0,3), the 3x3 sub-multiplication has rank at most 5 (Strassen-like).
    # Total rank = 3 * 5 = 15 if we can find rank-5 decompositions for each block.
    # But the blocks share the same left matrix A (2x3), so we need to be careful.
    
    # Actually, let's think more carefully:
    # A is 2x3, B is 3x9. We can write B as [B0, B1, B2] where each Bj is 3x3.
    # Then C = A*B = [A*B0, A*B1, A*B2].
    # Each A*Bj is a 2x3 matrix.
    # The tensor for A*Bj involves U[i*3+j], V[j*3+k], W[k*2+i] for k in [0,3) for block j.
    # But W is indexed by [k*n+i] where k ranges over all 9 columns of B, i.e., k in [0,9).
    # So for block j, the W indices are [(j*3+k)*2 + i] for k in [0,3), i in [0,2).
    
    # Let's try a structured approach: assume the decomposition has a block-diagonal structure
    # where each rank term corresponds to one of the three 3x3 blocks.
    # For block j, we use rank r_j terms. Total rank = sum r_j.
    # If we can find rank-5 for each 3x3, total = 15.
    
    # Let's try to construct such a decomposition.
    # For a 3x3 matrix multiplication, we need:
    #   sum_{r in block} U[i,j,r] * V[j,k,r] * W[k,i,r] = delta (1 if valid, but here all valid)
    # Actually for 3x3 * 3x3 -> 3x3, the tensor is T[i,j,k] = 1 for all i,j,k in [0,3).
    # Wait, no. The matrix multiplication tensor for (n x m) * (m x p) is:
    #   T[a, b, c] = 1 where a indexes (i,j), b indexes (j,k), c indexes (k,i)
    # For 3x3 * 3x3: a in [0,9), b in [0,9), c in [0,9)
    # T[a,b,c] = 1 iff a = i*3+j, b = j*3+k, c = k*3+i for some i,j,k in [0,3)
    
    # Let's try a different approach: use known low-rank decompositions.
    # For 3x3 matrix multiplication, the minimum rank is not known to be 5.
    # The best known upper bound is 19 (for 3x3), but schoolbook is 27.
    # Actually, I think the best known is around 19-20 for 3x3.
    
    # Let me reconsider. Let's just try to find any rank < 54 decomposition.
    # Use a structured ALS approach with Kronecker-inspired initialization.
    
    def try_rank_als(r, seed, time_limit):
        """Try to find a rank-r decomposition using structured ALS"""
        start = time.time()
        np.random.seed(seed)
        
        # Initialize with structure: group columns into blocks of size ~r/3
        block_size = max(1, r // 3)
        
        if r >= rank_schoolbook:
            U = U_sb[:, :r].copy()
            V = V_sb[:, :r].copy()
            W = W_sb[:, :r].copy()
        else:
            # Structured random initialization
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            for b in range(3):
                start_col = b * block_size
                end_col = min((b + 1) * block_size, r)
                if start_col >= r:
                    break
                # For block b, randomize the relevant parts
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, start_col:end_col] = np.random.randn(end_col - start_col) * 0.5
                for j in range(m):
                    for k in range(3):
                        V[j * p + b * 3 + k, start_col:end_col] = np.random.randn(end_col - start_col) * 0.5
                for k in range(3):
                    for i in range(n):
                        W[(b * 3 + k) * n + i, start_col:end_col] = np.random.randn(end_col - start_col) * 0.5
        
        # ALS iterations with adaptive tolerance
        for iteration in range(100):
            if time.time() - start > time_limit:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=1e-10)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=1e-10)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=1e-10)[0]
                    except:
                        pass
        
        # High-precision polish pass
        for iteration in range(100):
            if time.time() - start > time_limit:
                break
            
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=1e-15)[0]
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=1e-15)[0]
                    except:
                        pass
            
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=1e-15)[0]
                    except:
                        pass
        
        # Round to half-integers and verify
        U_h = to_half_integer(U)
        V_h = to_half_integer(V)
        W_h = to_half_integer(W)
        
        if verify_strict(U_h, V_h, W_h, n, m, p):
            return (U_h, V_h, W_h)
        
        if verify(U, V, W, n, m, p, tol=1e-10):
            U_close = to_half_integer(U)
            V_close = to_half_integer(V)
            W_close = to_half_integer(W)
            if verify_strict(U_close, V_close, W_close, n, m, p):
                return (U_close, V_close, W_close)
        
        return None
    
    # Try ranks from 53 down to 20
    # Use Kronecker-inspired initialization for lower ranks
    for r in range(53, 19, -1):
        if time.time() > deadline - 15:
            break
        max_seeds = 15 if r <= 30 else 8
        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                break
            remaining = deadline - time.time()
            time_per_try = min(5.0, remaining / (max_seeds - seed + 1))
            result = try_rank_als(r, seed, time_limit=time_per_try)
            if result is not None:
                U_h, V_h, W_h = result
                return {
                    "U": U_h.tolist(),
                    "V": V_h.tolist(),
                    "W": W_h.tolist()
                }
    
    # Fall back to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
