DIMENSIONS = [2, 2, 4]
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _trivial_solution(n, m, p):
    """Return the trivial rank n*m*p solution."""
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

def _strassen_2x2_block(n, m, p, k_offset, U, V, W, col_offset):
    """
    Add a Strassen rank-7 decomposition for a 2x2 x 2x2 block.
    The block multiplies A (2x2) by B_block (2x2) to produce C_block (2x2).
    k_offset: the offset in the k-index (0 to p-1) for this block.
    """
    # Strassen's 7 multiplications for 2x2 matrix multiplication:
    # M1 = (A00 - A11) * (B10 + B11)
    # M2 = (A00 + A01) * B11
    # M3 = A00 * (B00 - B10)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A11) * B00
    # M6 = (A01 - A11) * (B10 + B11)
    # M7 = (A00 - A01) * B00
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3
    # C10 = M6 + M5
    # C11 = M1 - M2 + M4 + M3
    #
    # In tensor form: C[i,k] = sum_{j} A[i,j] * B[j,k]
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    #
    # For the block at k_offset, the k indices are k_offset and k_offset+1.
    # The i indices are 0 and 1 (n=2).
    # The j indices are 0 and 1 (m=2).
    
    # For each of the 7 Strassen terms, we have a rank-1 contribution.
    # Term r contributes to C[i, k] via:
    #   U[i, j, r] * V[j, k_block, r] * W[k_block, i, r]
    # where k_block is the local k index (0 or 1), and the global k is k_offset + k_block.
    
    # Let's define the 7 terms as (coeff_A, coeff_B, coeff_C) for each (i, j, k) combination.
    # Actually, it's easier to just enumerate the 7 rank-1 tensors.
    
    # Term 1: (A00 - A11) * (B10 + B11)
    # This contributes to C via:
    # C00 += M1, C11 += M1
    # In tensor form:
    # For i=0, j=0: U[0,0,r] = 1, V[0, k, r] = ... 
    # Let me think differently.
    
    # The product M1 = (A00 - A11)(B10 + B11) contributes to:
    # C00 gets +M1, C11 gets +M1
    # So: C[0,0] += A[0,0]*B[1,0] + A[0,0]*B[1,1] - A[1,1]*B[1,0] - A[1,1]*B[1,1]
    # In our tensor: we need U[i,j,r]*V[j,k,r]*W[k,i,r] to produce these terms.
    
    # Let's define the rank-1 factors for each term.
    # Term 1: M1 = (A00 - A11)(B10 + B11)
    # U part: for i=0, j=0: +1; for i=1, j=1: -1
    # V part: for j=1, k=0: +1; for j=1, k=1: +1
    # W part: for k=0, i=0: +1; for k=1, i=1: +1
    
    # Let me just hardcode the 7 terms with their U, V, W vectors.
    # U is indexed by (i, j), V by (j, k), W by (k, i)
    
    terms = [
        # (U_vec, V_vec, W_vec) where each is a dict mapping (i,j), (j,k), (k,i) to coefficient
        # Term 1: (A00-A11)(B10+B11)
        ({(0,0): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # Term 2: (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1}),
        # Term 3: A00(B00-B10)
        ({(0,0): 1}, {(0,0): 1, (1,0): -1}, {(0,0): 1}),
        # Term 4: A11(B01-B00)
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(1,1): 1}),
        # Term 5: (A00+A11)B00
        ({(0,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 1}),
        # Term 6: (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,1): 1}),
        # Term 7: (A00-A01)B00
        ({(0,0): 1, (0,1): -1}, {(0,0): 1}, {(0,0): 1}),
    ]
    
    for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
        col = col_offset + t_idx
        for (i, j), val in u_dict.items():
            U[i*m + j, col] = val
        for (j, k_local), val in v_dict.items():
            k_global = k_offset + k_local
            V[j*p + k_global, col] = val
        for (k_local, i), val in w_dict.items():
            k_global = k_offset + k_local
            W[k_global*n + i, col] = val

def _strassen_solution(n, m, p):
    """
    Construct a rank-14 solution for 2x2 x 2x4 matrix multiplication
    using two Strassen rank-7 blocks.
    """
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 0: k in {0, 1}
    _strassen_2x2_block(n, m, p, 0, U, V, W, 0)
    # Block 1: k in {2, 3}
    _strassen_2x2_block(n, m, p, 2, U, V, W, 7)
    
    return {"U": U, "V": V, "W": W}

def _search_rank(n, m, p, target_rank, max_iters=5000, seed=42, time_limit=None, start_time=None):
    """Try to find a decomposition of the given rank using random search + local refinement."""
    if start_time is None:
        start_time = time.time()
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, target_rank)
    V = rng.randn(m*p, target_rank)
    W = rng.randn(p*n, target_rank)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Compute loss: sum over all (i,j,k) of (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(target_rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err = s - 1.0
                    loss += err * err
                    for r in range(target_rank):
                        dU = err * 2 * V[j*p+k, r] * W[k*n+i, r]
                        dV = err * 2 * U[i*m+j, r] * W[k*n+i, r]
                        dW = err * 2 * U[i*m+j, r] * V[j*p+k, r]
                        grad_U[i*m+j, r] += dU
                        grad_V[j*p+k, r] += dV
                        grad_W[k*n+i, r] += dW
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
        
        # Gradient step with adaptive learning rate
        lr = 0.01
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Periodic random restarts within the iteration
        if it % 500 == 0 and it > 0:
            # Small perturbation
            U += 0.01 * rng.randn(*U.shape)
            V += 0.01 * rng.randn(*V.shape)
            W += 0.01 * rng.randn(*W.shape)
    
    if best_loss < 1e-10:
        # Convert to list format
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        # Try to snap to rational values (half-integers)
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    mat[i][j] = round(mat[i][j] * 2) / 2
        # Verify after snapping
        if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
            return U_list, V_list, W_list
        else:
            # Return unsnapped if snapped fails
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Leave margin from 150s limit
    
    # First, try the structured rank-14 solution (two Strassen blocks)
    sol = _strassen_solution(n, m, p)
    if _verify_decomposition(sol["U"], sol["V"], sol["W"], n, m, p, tol=1e-8):
        return sol
    
    # If Strassen verification fails (shouldn't happen), try search for rank 13, 12, ...
    for target_rank in range(13, 8, -1):
        if time.time() > deadline - 10:
            break
        result = _search_rank(n, m, p, target_rank, max_iters=2000, seed=target_rank*1000,
                              time_limit=deadline - 10, start_time=start_time)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to trivial solution
    return _trivial_solution(n, m, p)
