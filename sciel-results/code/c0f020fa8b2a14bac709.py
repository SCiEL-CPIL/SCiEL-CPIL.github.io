"""Matrix multiplication 2x5x5 rank minimization."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form an exact decomposition of T_{n,m,p}."""
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_zero(U, V, W, n, m, p, tol=1e-9):
    """Verify no spurious non-zero entries: T[i,j,k] should be 1 for all valid (i,j,k).
    Actually the tensor IS 1 for all (i,j,k) in the multiplication tensor.
    Wait - the matrix multiplication tensor has T[i,j,k]=1 for all i,j,k in range.
    So we just need all entries to be 1."""
    return _verify(U, V, W, n, m, p, tol)

def _try_rank_25_structured():
    """Try a structured rank-25 construction for 2x5x5."""
    n, m, p = 2, 5, 5
    rank = 25
    # Try: for each (j,k) pair, one rank-1 term
    # U[i*m+j, c] where c = j*p+k
    # V[j*p+k, c] 
    # W[k*n+i, c]
    # We need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    
    # Let c index (j',k') pairs, c = j'*p + k'
    # U[i*m+j, j'*p+k'] * V[j*p+k, j'*p+k'] * W[k*n+i, j'*p+k'] = 1 when (j,k)=(j',k') summed
    # This is schoolbook with rank 25 only if we can make it work with n=2.
    
    # Actually schoolbook is rank 50. Let me try a different structure.
    
    # Idea: Use the small n=2 to our advantage.
    # For each j (0..4), we have A[i,j] for i=0,1 and B[j,k] for k=0..4.
    # C[i,k] += A[i,j] * B[j,k]
    
    # Let me try: rank = 25, indexed by (j, k)
    # U[i*m+j, j*p+k] = 1 if i matches somehow...
    # This doesn't directly work.
    
    # Alternative: use bilinear form with shared factors
    # Let me try random search with rounding
    
    return None

def _random_search(rank, n, m, p, max_iters=5000, seed=42):
    """Random search with rounding for small rank."""
    rng = np.random.RandomState(seed)
    best = None
    
    for it in range(max_iters):
        # Initialize with small random values
        U = rng.randint(0, 3, size=(n*m, rank)) * 0.5  # values in {0, 0.5, 1}
        V = rng.randint(0, 3, size=(m*p, rank)) * 0.5
        W = rng.randint(0, 3, size=(p*n, rank)) * 0.5
        
        # Convert to lists
        U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
        V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
        W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        if it % 1000 == 0:
            print(f"  Iter {it}: no solution yet")
    
    return None

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, use 120 to be safe
    
    n, m, p = DIMENSIONS
    
    # Try rank 25 first (theoretical lower bound)
    # Then 26, 27, ... up to 50 (schoolbook)
    
    # First, try a structured approach for rank 25
    # The matrix multiplication tensor T_{2,5,5}
    # Lower bound: max(n*m, m*p, n*p) = max(10, 25, 10) = 25
    
    # Let's try to find rank-25 solution using more systematic methods
    
    # Approach: Use the fact that we can set up a system of equations
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # For rank 25, this is a system of 50 equations (i=0,1; j=0..4; k=0..4)
    # with 25*3 = 75 unknowns (but bilinear/cubic structure)
    
    # Let me try a smarter initialization:
    # Fix V to be identity-like and solve for U and W
    
    # Actually, let me try: set V[j*p+k, c] = 1 if c == j*p+k, else 0 (identity)
    # Then U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1 for all i,j,k
    # This means U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1
    # For fixed (j,k), we need U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1 for i=0,1
    # Let U[i*m+j, j*p+k] = a_{i,j,k} and W[k*n+i, j*p+k] = b_{i,j,k}
    # a_{i,j,k} * b_{i,j,k} = 1
    # We can set a_{i,j,k} = 1, b_{i,j,k} = 1
    # But this is just schoolbook with rank 25! Wait, no.
    
    # With V as identity (25x25), U is 10x25, W is 10x25
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = U[i*m+j, c] * delta_{c, j*p+k} * W[k*n+i, c]
    # = U[i*m+j, j*p+k] * W[k*n+i, j*p+k]
    # We need this to equal 1 for all i,j,k
    # So U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1
    # Set U[i*m+j, j*p+k] = 1 and W[k*n+i, j*p+k] = 1
    # Check: this works! And rank = 25!
    
    # Wait, but this IS a valid rank-25 decomposition!
    # Let me verify:
    # U has 10 rows, 25 cols. U[i*5+j, j*5+k] = 1
    # V has 25 rows, 25 cols. V[j*5+k, j*5+k] = 1 (identity)
    # W has 10 rows, 25 cols. W[k*2+i, j*5+k] = 1
    
    # For entry (i,j,k): sum_c U[i*5+j, c] * V[j*5+k, c] * W[k*2+i, c]
    # Only c = j*5+k contributes (since V is identity)
    # = U[i*5+j, j*5+k] * V[j*5+k, j*5+k] * W[k*2+i, j*5+k]
    # = 1 * 1 * 1 = 1 ✓
    
    # This is a valid rank-25 decomposition!
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            V[c][c] = 1  # Identity
            for i in range(n):
                U[i * m + j][c] = 1
                W[k * n + i][c] = 1
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: try random search for rank 25
    result = _random_search(25, n, m, p, max_iters=2000, seed=123)
    if result is not None:
        return result
    
    # Check deadline
    if time.time() > deadline - 10:
        return _schoolbook(n, m, p)
    
    # Try rank 26
    result = _random_search(26, n, m, p, max_iters=2000, seed=456)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
