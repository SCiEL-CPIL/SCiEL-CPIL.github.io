"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x4 matrix multiplication.
    
    Returns dict with 'U', 'V', 'W' matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    The decomposition satisfies: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    and for all other index combinations, the sum equals 0.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 6, 12, 8
    
    # The tensor T[i,j,k] = 1 if (i,j,k) is a valid multiplication index, else 0
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right. Let me reconsider.
    
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # A rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # (using the trace convention W[k*n+i, r])
    
    # So we need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is the schoolbook decomposition with rank n*m*p = 24.
    
    # We want to find a lower rank decomposition.
    # The known lower bound for 2x3x4 is not trivially known to be less than 24.
    # Let's try to find if rank < 24 is possible.
    
    # Approach: Use integer programming / search for small rank decompositions.
    # For small ranks, we can enumerate or use optimization.
    
    # Let's start with the schoolbook solution as a baseline.
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Improved search strategy:
    # 1. Target rank 20 (known minimal for 2x3x4).
    # 2. Randomly generate sparse integer U and V.
    # 3. Solve the linear system for W.
    # 4. Check if W is close to integer/half-integer.
    # 5. If yes, verify and return.
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify if U, V, W form an exact decomposition."""
        T = np.zeros((n, m, p))
        r = U.shape[1]
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

    def try_rank_search(r, max_trials=2000):
        """Search for a rank-r decomposition by fixing U, V and solving for W."""
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(int(time.time() * 1000) % 10000)
        
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
                
            # Generate sparse random U and V with integer entries
            # Use -1, 0, 1 to keep numbers small
            U = np.zeros((nm, r), dtype=int)
            V = np.zeros((mp, r), dtype=int)
            
            # Sparsity: ~30% non-zero
            for i in range(nm):
                for c in range(r):
                    if np.random.rand() < 0.3:
                        U[i, c] = np.random.choice([-1, 1])
            
            for j in range(mp):
                for c in range(r):
                    if np.random.rand() < 0.3:
                        V[j, c] = np.random.choice([-1, 1])
            
            # Construct linear system A W_flat = b
            # Variables: W[k*n+i, c] for k in [0,p), i in [0,n), c in [0,r)
            # Total variables: p*n*r
            # Equations: n*m*p
            # For each (i,j,k), equation is:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            A = np.zeros((n*m*p, p*n*r))
            b = np.ones(n*m*p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Coefficients for W[k*n+i, c]
                        # W index in flat vector: (k*n + i) * r + c
                        w_base = (k*n + i) * r
                        for c in range(r):
                            coeff = U[i*m+j, c] * V[j*p+k, c]
                            if coeff != 0:
                                A[eq_idx, w_base + c] = coeff
                        eq_idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            
            W = W_flat.reshape(pn, r)
            
            # Check if W is close to integers or half-integers
            # Try rounding to nearest integer
            W_int = np.round(W)
            if np.allclose(W, W_int, atol=1e-6):
                if verify_decomposition(U, V, W_int, n, m, p):
                    return (U, V, W_int)
            
            # Try rounding to nearest half-integer
            W_half = np.round(W * 2) / 2
            if np.allclose(W, W_half, atol=1e-6):
                if verify_decomposition(U, V, W_half, n, m, p):
                    return (U, V, W_half)
        
        return None
    
    # Try ranks from 20 up to 23 (20 is the known minimum)
    # We start at 20 to try to find the optimal solution first.
    for r in range(20, rank_schoolbook):
        if time.time() > deadline - 10:
            break
            
        # Increase trials for lower ranks as they are harder
        trials = 5000 if r == 20 else 2000
        result = try_rank_search(r, trials)
        if result is not None:
            U_f, V_f, W_f = result
            best_rank = r
            best_U, best_V, best_W = U_f, V_f, W_f
            break  # We found a solution, stop
    
    # If no lower rank found, use schoolbook
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
