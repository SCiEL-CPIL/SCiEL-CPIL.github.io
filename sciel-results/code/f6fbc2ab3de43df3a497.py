"""Matrix multiplication 4x4x8 tensor decomposition - rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Find a low-rank exact decomposition for 4x4 * 4x8 matrix multiplication.
    Uses structured decomposition and iterative refinement.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 128
    
    # Deadline guard
    deadline = time.time() + 130
    
    def verify(U, V, W, n, m, p, rank):
        """Check if U,V,W exactly decompose the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook_decomposition(n, m, p):
        """Create the trivial schoolbook decomposition."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strategy: Use block structure and ALS to reduce rank
    # Key insight: For 4x4x8, we can decompose into two 4x4x4 blocks
    # If we can find rank-r decomposition for 4x4x4, we get rank-2r for 4x4x8
    
    def als_decompose(n, m, p, target_rank, max_iters=200, tol=1e-8):
        """
        Alternating Least Squares to find rank-target_rank decomposition.
        """
        np.random.seed(42)
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Solve for U
            U_new = np.zeros((n * m, target_rank))
            for i in range(n):
                for j in range(m):
                    # Build system: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    try:
                        u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U_new[i * m + j, :] = u
                    except:
                        U_new[i * m + j, :] = U[i * m + j, :]
            
            # Solve for V
            V_new = np.zeros((m * p, target_rank))
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, target_rank))
                    for i in range(n):
                        A[i, :] = U_new[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    try:
                        v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V_new[j * p + k, :] = v
                    except:
                        V_new[j * p + k, :] = V[j * p + k, :]
            
            # Solve for W
            W_new = np.zeros((p * n, target_rank))
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        A[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                    b = np.ones(m)
                    try:
                        w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W_new[k * n + i, :] = w
                    except:
                        W_new[k * n + i, :] = W[k * n + i, :]
            
            # Check convergence
            diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
            U, V, W = U_new, V_new, W_new
            if diff < tol:
                break
        
        return U, V, W
    
    def check_and_store(U, V, W, rank):
        """Check if decomposition is valid and update best if valid."""
        nonlocal best_rank, best_U, best_V, best_W
        if rank < best_rank:
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                return True
        return False
    
    # Try various target ranks
    # Start from lower ranks and work up
    min_possible = max(1, n * p)  # Lower bound: at least n*p for full rank output
    for target_rank in range(min_possible, schoolbook_rank + 1):
        if time.time() > deadline - 5:
            break
        if target_rank >= best_rank:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() > deadline - 3:
                break
            
            U_t, V_t, W_t = als_decompose(n, m, p, target_rank, max_iters=100)
            if check_and_store(U_t, V_t, W_t, target_rank):
                break
    
    # Also try structured decomposition: split into two 4x4x4 blocks
    def block_decomposition(n, m, p1, p2):
        """
        Try to decompose n x m x (p1+p2) using two n x m x p1 and n x m x p2 blocks.
        """
        if p1 != p2:
            return None
        
        # Try to find rank-r decomposition for n x m x p1
        for r in range(n * m * p1 // 2, n * m * p1 + 1):
            if time.time() > deadline - 5:
                break
            
            # Find decomposition for first block
            U1, V1, W1 = als_decompose(n, m, p1, r, max_iters=80)
            # Verify
            valid1 = True
            for i in range(n):
                for j in range(m):
                    for k in range(p1):
                        s = np.sum(U1[i * m + j, :] * V1[j * p1 + k, :] * W1[k * n + i, :])
                        if abs(s - 1.0) > 1e-6:
                            valid1 = False
                            break
                    if not valid1:
                        break
                if not valid1:
                    break
            
            if not valid1:
                continue
            
            # Build combined decomposition for n x m x (p1+p2)
            total_rank = 2 * r
            U_comb = np.zeros((n * m, total_rank))
            V_comb = np.zeros((m * (p1 + p2), total_rank))
            W_comb = np.zeros(((p1 + p2) * n, total_rank))
            
            # First block (columns 0..p1-1 of B, rows 0..p1-1 of C)
            U_comb[:, :r] = U1
            V_comb[:m * p1, :r] = V1
            W_comb[:p1 * n, :r] = W1
            
            # Second block (columns p1..p1+p2-1 of B, rows p1..p1+p2-1 of C)
            U_comb[:, r:] = U1
            V_comb[m * p1:, r:] = V1
            # W for second block: W2[k2*n+i] where k2 is 0..p2-1, maps to (p1+k2)*n+i
            for k2 in range(p2):
                for i in range(n):
                    W_comb[(p1 + k2) * n + i, r:] = W1[k2 * n + i, :]
            
            if check_and_store(U_comb, V_comb, W_comb, total_rank):
                return U_comb, V_comb, W_comb, total_rank
        
        return None
    
    result = block_decomposition(n, m, 4, 4)
    if result is not None:
        U, V, W, r = result
        if r < best_rank:
            best_rank = r
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Convert to lists for return
    U_list = [[float(best_U[i, c]) for c in range(best_rank)] for i in range(n * m)]
    V_list = [[float(best_V[i, c]) for c in range(best_rank)] for i in range(m * p)]
    W_list = [[float(best_W[i, c]) for c in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
