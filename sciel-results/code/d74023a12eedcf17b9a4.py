import random
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def validate_exact(U, V, W, n, m, p):
    """Validate with exact fraction arithmetic."""
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(r):
                    val += Fraction(U[idx_u][c]) * Fraction(V[idx_v][c]) * Fraction(W[idx_w][c])
                if val != Fraction(1, 1):
                    return False
    return True

def validate_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float validation."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = W.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def build_strassen_blocks(n, m, p):
    """
    Build a decomposition using Strassen's algorithm for 2x2x2 blocks
    where possible, and schoolbook for the rest.
    This gives a rank lower than n*m*p.
    """
    terms = []
    
    def add_strassen(i_start, j_start, k_start):
        # Adds 7 terms for 2x2x2 block at (i_start, j_start, k_start)
        # i in {i_start, i_start+1}
        # j in {j_start, j_start+1}
        # k in {k_start, k_start+1}
        
        # M1
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M2
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        
        # M3
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M4
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        
        # M5
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        
        # M6
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M7
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))

    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))

    # Decomposition plan for 2x5x5:
    # We have i in {0,1}, j in {0,1,2,3,4}, k in {0,1,2,3,4}
    
    # j=0,1:
    #   k=0,1: Strassen(0,0,0)
    add_strassen(0, 0, 0)
    #   k=2,3: Strassen(0,0,2)
    add_strassen(0, 0, 2)
    #   k=4: Schoolbook for i in {0,1}, j in {0,1}, k={4}
    add_schoolbook(0, 2, 0, 2, 4, 5)
    
    # j=2,3:
    #   k=0,1: Strassen(0,2,0)
    add_strassen(0, 2, 0)
    #   k=2,3: Strassen(0,2,2)
    add_strassen(0, 2, 2)
    #   k=4: Schoolbook for i in {0,1}, j in {2,3}, k={4}
    add_schoolbook(0, 2, 2, 4, 4, 5)
    
    # j=4:
    #   k=0,1: Schoolbook for i in {0,1}, j={4}, k in {0,1}
    add_schoolbook(0, 2, 4, 5, 0, 2)
    #   k=2,3: Schoolbook for i in {0,1}, j={4}, k in {2,3}
    add_schoolbook(0, 2, 4, 5, 2, 4)
    #   k=4: Schoolbook for i in {0,1}, j={4}, k={4}
    add_schoolbook(0, 2, 4, 5, 4, 5)

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
            
    return U, V, W

def tensor_error(U, V, W, n, m, p):
    """Calculate L2 error of the tensor decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = W.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                error += (val - 1.0) ** 2
    return error

def try_merge_terms(U, V, W, n, m, p, deadline, max_attempts=100):
    """
    Try to merge terms to reduce rank.
    Two terms can be merged if there exists a linear combination
    that still satisfies the constraints.
    
    For simplicity, we try random linear combinations of pairs of terms
    and check if the resulting tensor still has rank 1 structure.
    """
    rank = len(W[0])
    if rank <= 1:
        return U, V, W
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
            
        # Pick two random terms to try to merge
        c1, c2 = random.sample(range(best_rank), 2)
        
        # Try to find a linear combination that eliminates one term
        # This is a heuristic approach
        
        U_new = [row[:] for row in best_U]
        V_new = [row[:] for row in best_V]
        W_new = [row[:] for row in best_W]
        
        # Remove term c2
        for i in range(n * m):
            U_new[i].pop(c2)
        for j in range(m * p):
            V_new[j].pop(c2)
        for k in range(p * n):
            W_new[k].pop(c2)
        
        new_rank = best_rank - 1
        
        # Check if the remaining terms still satisfy the constraints
        # with the merged term
        # This is a simplified check - in reality we need to verify
        # that the tensor can be decomposed with new_rank terms
        
        # For now, just check if the current terms still work
        # (this won't actually merge, but it's a placeholder)
        
        # Actually, let's try a different approach:
        # Check if we can find a linear dependency among the terms
        
        # Build the matrix of term coefficients
        # For each (i,j,k) that should equal 1, we have:
        # sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1
        
        # This is hard to check directly, so let's use a different heuristic
        
    return best_U, best_V, best_W

def try_optimize_rank(n, m, p, rank, deadline, max_iter=1000, lr=0.05):
    """
    Try to find a decomposition of given rank using gradient descent.
    Returns (U, V, W) as lists if found, else None.
    """
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    # Initialize with some sparsity
    mask_u = np.random.rand(n * m, rank) < 0.3
    mask_v = np.random.rand(m * p, rank) < 0.3
    mask_w = np.random.rand(p * n, rank) < 0.3
    U *= mask_u
    V *= mask_v
    W *= mask_w
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Calculate error
        err = tensor_error(U, V, W, n, m, p)
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-6:
            # Try to snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            
            if validate_exact(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
            
        # Gradient step (approximate)
        step = 0.1
        U += np.random.randn(n * m, rank) * step
        V += np.random.randn(m * p, rank) * step
        W += np.random.randn(p * n, rank) * step
        
        # Soft thresholding to encourage sparsity
        U[np.abs(U) < 0.1] = 0
        V[np.abs(V) < 0.1] = 0
        W[np.abs(W) < 0.1] = 0

    # Final check on best found
    if best_error < 1e-4:
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if validate_exact(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
            
    return None

def build_rank13_slice(m, p, seed=42, deadline=None):
    """
    Build a rank-13 decomposition for a 1xm x mxp matrix multiplication tensor.
    The tensor has entries T[j,k] = delta_{j,k} for j in 0..m-1, k in 0..p-1.
    We use numerical optimization to find a rank-13 decomposition.
    
    Returns (U, V, W) as lists where:
    - U is m x 13
    - V is (m*p) x 13  
    - W is p x 13
    """
    np.random.seed(seed)
    rank = 13
    U = np.random.randn(m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p, rank) * 0.5
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    max_iter = 3000
    lr = 0.05
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Compute error and gradients
        error = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for j in range(m):
            for k in range(p):
                target = 1.0 if j == k else 0.0
                val = np.dot(U[j, :], V[j * p + k, :] * W[k, :])
                err = val - target
                error += err * err
                
                # Gradients
                dU[j, :] += 2 * err * (V[j * p + k, :] * W[k, :])
                dV[j * p + k, :] += 2 * err * (U[j, :] * W[k, :])
                dW[k, :] += 2 * err * (U[j, :] * V[j * p + k, :])
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-12:
            break
        
        # Gradient descent step with adaptive learning rate
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Occasionally perturb if stuck
        if it > 0 and it % 300 == 0 and error > 1e-6:
            U = best_U + np.random.randn(m, rank) * 0.05
            V = best_V + np.random.randn(m * p, rank) * 0.05
            W = best_W + np.random.randn(p, rank) * 0.05
    
    # Check if we found a good solution
    if best_error < 1e-8:
        # Try to snap to rational values
        U_snap = np.round(best_U * 1000) / 1000
        V_snap = np.round(best_V * 1000) / 1000
        W_snap = np.round(best_W * 1000) / 1000
        
        # Verify
        valid = True
        for j in range(m):
            for k in range(p):
                target = 1.0 if j == k else 0.0
                val = np.dot(U_snap[j, :], V_snap[j * p + k, :] * W_snap[k, :])
                if abs(val - target) > 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
    
    return None

def build_combined_rank26(n, m, p, deadline=None):
    """
    Build a rank-26 decomposition for 2x5x5 by combining two rank-13 
    decompositions for the 1x5x5 slices (i=0 and i=1).
    
    For the full 2x5x5 tensor:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{j,k} for all i,j,k
    
    We split into two independent problems:
    - i=0: sum_{c=0}^{12} U[j, c] * V[j*p+k, c] * W[k, c] = delta_{j,k}
    - i=1: sum_{c=13}^{25} U[m+j, c] * V[j*p+k, c] * W[k*n+1, c] = delta_{j,k}
    
    Each sub-problem is a 1x5x5 tensor decomposition of rank 13.
    """
    # Get rank-13 decomposition for slice i=0
    result0 = build_rank13_slice(m, p, seed=42, deadline=deadline)
    if result0 is None:
        return None
    
    U0, V0, W0 = result0
    
    # Get rank-13 decomposition for slice i=1
    result1 = build_rank13_slice(m, p, seed=43, deadline=deadline)
    if result1 is None:
        return None
    
    U1, V1, W1 = result1
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Slice i=0: terms c=0..12
    for c in range(13):
        for j in range(m):
            U[j][c] = U0[j][c]
        for j in range(m):
            for k in range(p):
                V[j * p + k][c] = V0[j * p + k][c]
        for k in range(p):
            W[k * n + 0][c] = W0[k][c]
    
    # Slice i=1: terms c=13..25
    for c in range(13):
        c_full = 13 + c
        for j in range(m):
            U[m + j][c_full] = U1[j][c]
        for j in range(m):
            for k in range(p):
                V[j * p + k][c_full] = V1[j * p + k][c]
        for k in range(p):
            W[k * n + 1][c_full] = W1[k][c]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    # Try the rank-26 approach first
    result26 = build_combined_rank26(n, m, p, deadline=deadline)
    if result26 is not None:
        U26, V26, W26 = result26
        if validate_float(U26, V26, W26, n, m, p):
            # Verify with exact arithmetic
            if validate_exact(U26, V26, W26, n, m, p):
                return {"U": U26, "V": V26, "W": W26}
    
    # Fall back to Strassen-based decomposition
    U, V, W = build_strassen_blocks(n, m, p)
    best_result = (U, V, W)
    best_rank = len(W[0])
    
    # Verify the Strassen decomposition is valid
    if not validate_float(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = schoolbook_decomposition(n, m, p)
        best_result = (U, V, W)
        best_rank = len(W[0])
    
    # Try to find lower ranks
    candidate_ranks = [30, 28, 25, 20, 15]
    
    for rank in candidate_ranks:
        if rank >= best_rank:
            continue
        if time.time() > deadline - 20:
            break
            
        # Try multiple random starts
        found = False
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
                
            result = try_optimize_rank(n, m, p, rank, deadline, max_iter=100)
            if result:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = rank
                found = True
                break
        
        if found:
            pass

    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
