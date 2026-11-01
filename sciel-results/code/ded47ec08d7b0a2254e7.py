import time
import numpy as np
import itertools

DIMENSIONS = [2, 4, 8]

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=20, tol=1e-8):
    """Fast verification using vectorized numpy operations."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    if U_mat.shape != (n*m, rank) or V_mat.shape != (m*p, rank) or W_mat.shape != (p*n, rank):
        return False

    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        U_reshaped = U_mat.reshape(n, m, rank)
        V_reshaped = V_mat.reshape(m, p, rank)
        W_reshaped = W_mat.reshape(p, n, rank)

        factor_A = np.einsum('ij,ijc->ic', A, U_reshaped)
        factor_B = np.einsum('jk,jkc->kc', B, V_reshaped)
        C_computed = np.einsum('ic,kc,kic->ik', factor_A, factor_B, W_reshaped)

        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _try_rank_r(r, n=2, m=4, p=8, time_limit=10):
    """
    Attempt to find a rank-r decomposition.
    Strategy:
    We know that for small matrices, decompositions often involve entries in {0, 1, -1, 0.5, -0.5}.
    The tensor equation is: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
    
    We can fix W (or part of it) and try to solve for U and V.
    However, U and V are also non-linear relative to each other.
    
    A more robust approach for small ranks is to use the fact that the decomposition is a 3-tensor.
    We can iterate over possible structures for W.
    Since W has shape (p*n, r) = (16, r), and entries are likely small integers/halves,
    we can sample or search for W.
    
    Alternatively, we can use the "Strassen-like" structure but search for the specific 
    integer coefficients.
    
    Let's try a direct search for small r (e.g., r < 56) using random restarts with 
    integer/half-integer constraints on the factors.
    
    Actually, a better heuristic for this specific "exploration" task is to check if 
    a known lower bound or specific small decomposition exists.
    For 2x4x8, the schoolbook is 64. Strassen is 56.
    Is there a rank < 56?
    The asymptotic rank of 2x2x2 is 7.
    For 2x4x8, it's effectively 8 blocks of 2x2x2 if we don't share.
    But we can share.
    
    Let's try to construct a decomposition using a "tensor ring" or "MPS" approach 
    with bond dimension d. The rank would be roughly n*m*p/d or similar?
    No, for matrix mult, the rank is related to the bond dimension of the ring.
    
    Let's try a simple random search with integer/half-integer entries for small r.
    We will fix r and try to find U, V, W.
    
    To make it tractable, we will fix the "gauge" or structure.
    For example, we can set U to be a random sparse matrix, and try to solve for V and W.
    But that's still hard.
    
    Let's stick to the optimization approach but with a much tighter, smarter initialization 
    and a different objective: minimize the number of non-zero entries (sparsity) 
    while satisfying the equation? No, that's NP-hard.
    
    Let's try the ILP idea:
    Fix r.
    We need U (8, r), V (32, r), W (16, r).
    Total variables: 8r + 32r + 16r = 56r.
    Equations: 64.
    
    If we assume entries are in {0, 1, -1}, the search space is 3^(56r). Too big.
    
    However, we can exploit the structure.
    C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T[i,j,k] = 1.
    
    Let's try to find a decomposition of rank r by solving a linear system 
    for a subset of variables.
    
    Suppose we fix U and V. Then for each (i,k), we have:
    sum_c (U[i,:,c] * V[:,k,c]) W[k,i,:] = 1 (vector of 1s of length m? No, scalar 1).
    Wait, the equation is scalar for each (i,j,k).
    sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    
    If we fix U and V, this is a linear system for W?
    For a fixed (i,k), the unknowns are W[k,i,0..r-1].
    The equations are for j=0..m-1:
    sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    This is a system of m equations in r unknowns.
    If r >= m, we can solve this in least squares.
    If the solution W has integer/half-integer entries (within tolerance), we have a valid decomposition.
    
    So the algorithm is:
    1. Pick r.
    2. Randomly initialize U and V with small integers/halves (or sparse).
    3. For each (i,k), solve the linear system for W[k,i,:].
    4. Check if W entries are close to half-integers.
    5. If yes, verify.
    
    This is a "fix U, V, solve W" strategy.
    """
    start_time = time.time()
    n, m, p = 2, 4, 8
    
    # We will try to find a solution.
    # To increase chances, we can use the Strassen solution as a starting point 
    # and perturb it, or just random restarts.
    
    # Let's try random restarts with integer/half-integer initialization for U and V.
    # To make it faster, we can restrict U and V to be sparse.
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize U and V with random sparse integers {-1, 0, 1}
        # Density 0.3
        U = np.zeros((n, m, r))
        V = np.zeros((m, p, r))
        
        for i in range(n):
            for j in range(m):
                for c in range(r):
                    if np.random.rand() < 0.3:
                        U[i, j, c] = np.random.choice([-1, 1])
                        
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    if np.random.rand() < 0.3:
                        V[j, k, c] = np.random.choice([-1, 1])
                        
        # Solve for W
        W = np.zeros((p, n, r))
        valid = True
        for i in range(n):
            for k in range(p):
                # A_ik (m, r)
                A_ik = U[i, :, :] * V[:, k, :]
                b = np.ones(m)
                
                try:
                    # Solve A_ik @ w = b
                    # Use lstsq
                    w, residuals, rank, sv = np.linalg.lstsq(A_ik, b, rcond=None)
                    W[k, i, :] = w
                except:
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            continue
            
        # Check if W is close to half-integers
        W_half = np.round(W * 2) / 2
        if np.max(np.abs(W - W_half)) < 1e-5:
            # Candidate found! Verify
            U_out = U.reshape(n*m, r).tolist()
            V_out = V.reshape(m*p, r).tolist()
            W_out = W_half.reshape(p*n, r).tolist()
            
            if _verify_fast(U_out, V_out, W_out, n, m, p, trials=10, tol=1e-6):
                return U_out, V_out, W_out
                
    return None

def _build_strassen_2x2x2():
    """Standard Strassen decomposition for 2x2 * 2x2 -> 2x2, rank 7."""
    U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, 1]
    ], dtype=float)
    V = np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U.tolist(), V.tolist(), W.tolist()

def _build_block_decomposition(n, m, p):
    """
    Build a block-Strassen decomposition.
    """
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_s, V_s, W_s = _build_strassen_2x2x2()
    U_s = np.array(U_s)
    V_s = np.array(V_s)
    W_s = np.array(W_s)
    
    col_offset = 0
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[r][col] = U_s[f][r]
            for r in range(4):
                j = r // 2
                k = 2 * block + r % 2
                V[j * 8 + k][col] = V_s[f][r]
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W[k * 2 + i][col] = W_s[r][f]
        col_offset += 7
    
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[2 + r][col] = U_s[f][r]
            for r in range(4):
                j = 2 + r // 2
                k = 2 * block + r % 2
                V[j * 8 + k][col] = V_s[f][r]
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W[k * 2 + i][col] = W_s[r][f]
        col_offset += 7
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_linear_reduction(U, V, W, n, m, p, start_time, deadline):
    """
    Placeholder for the old reduction method.
    """
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Build the standard Strassen decomposition (Rank 56)
    U_56, V_56, W_56 = _build_block_decomposition(n, m, p)
    
    # Verify Rank 56
    if not _verify_fast(U_56, V_56, W_56, n, m, p, trials=5, tol=1e-9):
        # Fallback to schoolbook if Strassen fails (shouldn't happen)
        schoolbook_rank = n * m * p
        U_sb = np.zeros((n*m, schoolbook_rank))
        V_sb = np.zeros((m*p, schoolbook_rank))
        W_sb = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    best_U = U_56
    best_V = V_56
    best_W = W_56
    best_rank = 56
    
    # 2. Try to find a lower rank using the new "Fix U,V solve W" strategy.
    # We will try ranks from 55 down to 30.
    # To save time, we will limit the time per rank.
    
    for r in range(55, 29, -1):
        if time.time() > deadline - 10:
            break
            
        # Allocate time for this rank
        time_per_rank = min(5, deadline - 10 - time.time())
        if time_per_rank < 1:
            break
            
        result = _try_rank_r(r, n, m, p, time_limit=time_per_rank)
        
        if result is not None:
            U_opt, V_opt, W_opt = result
            best_U = U_opt
            best_V = V_opt
            best_W = W_opt
            best_rank = r
            # If we find a very low rank, we can break or continue.
            # Let's continue to see if we can go lower, but be careful with time.
            # If we found rank 40, maybe we can find 30.
            # But the search space grows exponentially with r? No, it shrinks?
            # Actually, lower rank is harder to find with random search.
            # So if we find 50, 49 might be harder.
            # Let's just try a few more steps if time permits.
            
    return {"U": best_U, "V": best_V, "W": best_W}
