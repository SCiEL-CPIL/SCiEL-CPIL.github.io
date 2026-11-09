import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin within 120s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def verify_decomposition(U, V, W, T_target, tol=1e-9):
        """Verify that sum_r U[:,r] V[:,r] W[:,r] == T_target"""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_recon, T_target, atol=tol, rtol=0):
            return True
        return False
    
    # Schoolbook decomposition (rank 50)
    U_school = np.zeros((n*m, schoolbook_rank), dtype=float)
    V_school = np.zeros((m*p, schoolbook_rank), dtype=float)
    W_school = np.zeros((p*n, schoolbook_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = schoolbook_rank
    
    # Verify schoolbook
    if not verify_decomposition(best_U, best_V, best_W, T):
        # Should never happen, but return anyway
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try structured lower-rank decompositions
    # Strategy: Use the fact that M_{2,5,5} can be decomposed using
    # Kronecker product structure or block decomposition
    
    # Approach 1: View as two M_{1,5,5} subproblems
    # For each row i of A, compute C[i,:] = A[i,:] * B
    # M_{1,5,5} has rank at most 25 (schoolbook), so total rank at most 50
    # But we might do better with shared structure
    
    # Approach 2: Try rank 25 (the lower bound) with structured patterns
    # The tensor T[i*5+j, j*5+k, k*2+i] = 1
    
    # Let's try a few structured patterns for rank 25
    # Pattern: use all combinations of certain indices
    
    # For rank 25, we can try: for each (j, k) pair, create a term
    # U[i*5+j] = 1 for all i, V[j*5+k] = 1, W[k*2+i] = 1 for all i
    # This gives contribution: for a=i*5+j, b=j*5+k, c=k*2+i': 1*1*1 = 1
    # But we want it to be 1 only when i' = i, so this doesn't work directly
    
    # Alternative: use rank 50 but verify it's correct, then try random searches for improvement
    
    # Random search for lower rank
    rng = np.random.RandomState(42)
    
    for target_rank in range(25, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        max_attempts = 100
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Use sparse random initialization
            U = rng.randint(-2, 3, (n*m, target_rank)).astype(float)
            V = rng.randint(-2, 3, (m*p, target_rank)).astype(float)
            W = rng.randint(-2, 3, (p*n, target_rank)).astype(float)
            
            # Make them sparse (set many entries to 0)
            U[np.abs(U) < 1] = 0
            V[np.abs(V) < 1] = 0
            W[np.abs(W) < 1] = 0
            
            if verify_decomposition(U, V, W, T):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            break
    
    # Also try a specific structured approach for rank 25
    # Using the identity: M_{2,5,5} can potentially be decomposed with rank 25
    # if we use a clever basis change
    
    # Try rank 25 with a specific pattern based on modular arithmetic
    if best_rank > 25 and time.time() < deadline - 10:
        # Pattern: U[i*5+j, r] = 1 if r == (j*2 + i) % 25
        # This is just one guess among many
        U_test = np.zeros((n*m, 25), dtype=float)
        V_test = np.zeros((m*p, 25), dtype=float)
        W_test = np.zeros((p*n, 25), dtype=float)
        
        for i in range(n):
            for j in range(m):
                r = (j * 2 + i) % 25
                U_test[i*5+j, r] = 1.0
        
        for j in range(m):
            for k in range(p):
                r = (j * 2 + k) % 25
                V_test[j*5+k, r] = 1.0
        
        for k in range(p):
            for i in range(n):
                r = (k * 2 + i) % 25
                W_test[k*2+i, r] = 1.0
        
        if verify_decomposition(U_test, V_test, W_test, T):
            best_rank = 25
            best_U = U_test
            best_V = V_test
            best_W = W_test
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
