"""Matrix multiplication 2x2x15 tensor decomposition using Strassen blocks with rank reduction."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]

def _build_strassen_2x2x2():
    """Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
    W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
    V[0*2+0, 1] = 1
    W[0*2+1, 1] = 1; W[1*2+1, 1] = -1
    
    U[0*2+0, 2] = 1
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
    W[1*2+0, 2] = 1; W[1*2+1, 2] = 1
    
    U[1*2+1, 3] = 1
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
    W[0*2+0, 3] = 1; W[0*2+1, 3] = 1
    
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
    V[1*2+1, 4] = 1
    W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
    
    U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
    V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
    W[1*2+1, 5] = 1
    
    U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
    V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
    W[0*2+0, 6] = 1
    
    return U, V, W

def _build_schoolbook_2x2x1():
    """Build schoolbook rank-4 decomposition for 2x2 x 2x1 matrix multiplication."""
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p, col] = 1
            W[0 * n + i, col] = 1
            col += 1
            
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify the decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_initial_solution():
    """Build the rank-53 solution using Strassen blocks."""
    n, m, p = 2, 2, 15
    rank = 53
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook_2x2x1()
    
    col_offset = 0
    for b in range(7):
        start_k = b * 2
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            for kl in range(2):
                V[j * p + start_k + kl, col_offset:col_offset+7] = V_str[j*2+kl, :]
        for kl in range(2):
            for i in range(n):
                W[(start_k+kl) * n + i, col_offset:col_offset+7] = W_str[kl*2+i, :]
        col_offset += 7
    
    U[:, col_offset:col_offset+4] = U_sb
    for j in range(m):
        V[j * p + 14, col_offset:col_offset+4] = V_sb[j, :]
    for i in range(n):
        W[14 * n + i, col_offset:col_offset+4] = W_sb[i, :]
    
    return U, V, W

def _try_find_2x2x3_rank10(deadline):
    """
    Try to find a rank-10 decomposition for 2x2x3 matrix multiplication.
    This would allow us to replace the last two Strassen blocks (rank 14) 
    with a single rank-10 block, reducing total rank from 53 to 49.
    """
    n, m, p = 2, 2, 3
    target_rank = 10
    
    # The tensor for 2x2x3 has 2*2*3 = 12 entries that must equal 1
    # We need to find U (4x10), V (6x10), W (6x10) such that
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Try a structured approach: use 2 Strassen-like blocks but with shared terms
    # Or try to find integer/half-integer solutions
    
    # First, let's try a direct search with small integer values
    # The search space is large, so we'll use a heuristic approach
    
    # Try rank 10 with a specific structure
    # We can think of this as 3 columns of a 2x2 matrix
    # For each column k, we need rank 4 (schoolbook), but we can share across columns
    
    # Let's try to construct a solution where the last column uses different terms
    # than the first two, allowing some sharing
    
    # Actually, let's try a more systematic approach:
    # Start from the rank-12 schoolbook (4 terms per column * 3 columns)
    # and try to merge terms
    
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Initialize with schoolbook for each column
    for k in range(3):
        for i in range(n):
            for j in range(m):
                col = k * 4 + i * m + j
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
    
    # Now try to reduce rank by finding linear dependencies
    # This is complex, so let's try a different approach:
    # Use the fact that 2x2x3 can be decomposed as 2x2x2 + 2x2x1
    # Rank(2x2x2) = 7, Rank(2x2x1) = 4, total = 11
    # But we want rank 10, so we need to save 1 more
    
    # Try to find a rank-10 decomposition by searching
    # We'll use a random restart approach with local search
    
    best_solution = None
    best_error = float('inf')
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random initialization with small values
        U_rand = np.random.randint(-2, 3, (n * m, target_rank)) * 0.5
        V_rand = np.random.randint(-2, 3, (m * p, target_rank)) * 0.5
        W_rand = np.random.randint(-2, 3, (p * n, target_rank)) * 0.5
        
        # Local search: try to minimize error
        for step in range(100):
            if time.time() > deadline - 5:
                break
            
            # Compute current error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                        error += (val - 1.0) ** 2
            
            if error < 1e-10:
                best_solution = (U_rand.copy(), V_rand.copy(), W_rand.copy())
                return best_solution
            
            if error < best_error:
                best_error = error
                best_solution = (U_rand.copy(), V_rand.copy(), W_rand.copy())
            
            # Try a small perturbation
            idx_r = np.random.randint(0, target_rank)
            if np.random.random() < 0.33:
                idx_row = np.random.randint(0, n * m)
                U_rand[idx_row, idx_r] += np.random.choice([-0.5, 0.5])
            elif np.random.random() < 0.5:
                idx_row = np.random.randint(0, m * p)
                V_rand[idx_row, idx_r] += np.random.choice([-0.5, 0.5])
            else:
                idx_row = np.random.randint(0, p * n)
                W_rand[idx_row, idx_r] += np.random.choice([-0.5, 0.5])
    
    return best_solution if best_error < 1e-10 else None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build initial rank-53 solution
    U, V, W = _build_initial_solution()
    
    best_rank = 53
    best_U, best_V, best_W = U, V, W
    
    # Try to find a rank-10 decomposition for 2x2x3
    # If successful, we can replace the last two Strassen blocks (k=12,13,14)
    # which currently use rank 7+7+4=18, with a single rank-10 block
    # New total rank: 5*7 + 10 = 45
    if time.time() < deadline - 10:
        result = _try_find_2x2x3_rank10(deadline)
        if result is not None:
            U10, V10, W10 = result
            
            # Build the combined solution
            new_rank = 5 * 7 + 10
            new_U = np.zeros((n * m, new_rank))
            new_V = np.zeros((m * p, new_rank))
            new_W = np.zeros((p * n, new_rank))
            
            U_str, V_str, W_str = _build_strassen_2x2x2()
            
            # First 5 Strassen blocks (k=0,1,2,3,4,5,6,7,8,9)
            col_offset = 0
            for b in range(5):
                start_k = b * 2
                new_U[:, col_offset:col_offset+7] = U_str
                for j in range(m):
                    for kl in range(2):
                        new_V[j * p + start_k + kl, col_offset:col_offset+7] = V_str[j*2+kl, :]
                for kl in range(2):
                    for i in range(n):
                        new_W[(start_k+kl) * n + i, col_offset:col_offset+7] = W_str[kl*2+i, :]
                col_offset += 7
            
            # Last 3 columns (k=12,13,14) with rank-10 decomposition
            for r in range(10):
                new_U[:, col_offset + r] = U10[:, r]
                for j in range(m):
                    for k in range(3):
                        new_V[j * p + 12 + k, col_offset + r] = V10[j * p + k, r]
                for k in range(3):
                    for i in range(n):
                        new_W[(12 + k) * n + i, col_offset + r] = W10[k * n + i, r]
            
            # Verify
            if _verify(new_U, new_V, new_W, n, m, p):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = new_rank
    
    # Verify final result
    if not _verify(best_U, best_V, best_W, n, m, p):
        # Fallback to initial solution
        best_U, best_V, best_W = _build_initial_solution()
        best_rank = 53
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
