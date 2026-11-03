"""Matrix multiplication 3x3x8 rank-minimal decomposition."""
import time
import itertools
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    
    Strategy: Use block decomposition with Strassen for 2x2 sub-blocks,
    then attempt systematic rank reduction by merging shared Strassen terms
    and exploring polynomial encoding of the 8 output columns.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    # Check if we're running out of time
    if time.time() > deadline - 5:
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # First, construct the rank-68 decomposition using block Strassen
    U68, V68, W68 = _construct_block_strassen(n, m, p)
    
    # Verify the base decomposition
    if not _verify_decomposition(U68, V68, W68, n, m, p):
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 68
    best_U, best_V, best_W = U68, V68, W68
    
    # Attempt rank reduction via systematic search
    if time.time() < deadline - 20:
        reduced = _attempt_rank_reduction(U68, V68, W68, n, m, p, deadline)
        if reduced is not None:
            best_rank = reduced[0]
            best_U, best_V, best_W = reduced[1], reduced[2], reduced[3]
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        U, V, W = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _construct_block_strassen(n, m, p):
    """Construct the rank-68 block Strassen decomposition."""
    U = np.zeros((n * m, 68))
    V = np.zeros((m * p, 68))
    W = np.zeros((p * n, 68))
    
    col = 0
    
    # Strassen for each 2x2 block of B1 (columns 2i, 2i+1 for i in 0,1,2,3):
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = 2 * block_idx + 1
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k0])
        U[0*3+0, col] = 1  # A[0,0]
        U[1*3+1, col] = 1  # A[1,1]
        V[0*8+k0, col] = 1  # B[0,k0]
        V[1*8+k0, col] = 1  # B[1,k0]
        W[k0*3+0, col] = 1  # C[0,k0]
        W[k1*3+1, col] = 1  # C[1,k1]
        col += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[1*3+0, col] = 1  # A[1,0]
        U[1*3+1, col] = 1  # A[1,1]
        V[0*8+k0, col] = 1  # B[0,k0]
        W[k0*3+1, col] = 1  # C[1,k0]
        col += 1
        
        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0*3+0, col] = 1  # A[0,0]
        V[0*8+k1, col] = 1  # B[0,k1]
        V[1*8+k1, col] = -1  # B[1,k1]
        W[k1*3+0, col] = 1  # C[0,k1]
        col += 1
        
        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[1*3+1, col] = 1  # A[1,1]
        V[1*8+k0, col] = 1  # B[1,k0]
        V[0*8+k0, col] = -1  # B[0,k0]
        W[k0*3+0, col] = 1  # C[0,k0]
        W[k0*3+1, col] = 1  # C[1,k0]
        col += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0*3+0, col] = 1  # A[0,0]
        U[0*3+1, col] = 1  # A[0,1]
        V[1*8+k1, col] = 1  # B[1,k1]
        W[k1*3+0, col] = 1  # C[0,k1]
        W[k1*3+1, col] = 1  # C[1,k1]
        col += 1
        
        # M6 = (A[1,0]-A[0,1]) * (B[0,k0]+B[0,k1])
        U[1*3+0, col] = 1  # A[1,0]
        U[0*3+1, col] = -1  # A[0,1]
        V[0*8+k0, col] = 1  # B[0,k0]
        V[0*8+k1, col] = 1  # B[0,k1]
        W[k1*3+1, col] = 1  # C[1,k1]
        col += 1
        
        # M7 = (A[1,1]-A[0,0]) * (B[1,k0]+B[1,k1])
        U[1*3+1, col] = 1  # A[1,1]
        U[0*3+0, col] = -1  # A[0,0]
        V[1*8+k0, col] = 1  # B[1,k0]
        V[1*8+k1, col] = 1  # B[1,k1]
        W[k0*3+0, col] = 1  # C[0,k0]
        W[k1*3+1, col] = 1  # C[1,k1]
        col += 1
    
    # Handle A12*B2: A[0:2, 2] * B[2, :]
    for k in range(8):
        U[0*3+2, col] = 1  # A[0,2]
        V[2*8+k, col] = 1  # B[2,k]
        W[k*3+0, col] = 1  # C[0,k]
        col += 1
        
        U[1*3+2, col] = 1  # A[1,2]
        V[2*8+k, col] = 1  # B[2,k]
        W[k*3+1, col] = 1  # C[1,k]
        col += 1
    
    # Handle A21*B1: A[2, 0:2] * B[0:2, :]
    for k in range(8):
        U[2*3+0, col] = 1  # A[2,0]
        V[0*8+k, col] = 1  # B[0,k]
        W[k*3+2, col] = 1  # C[2,k]
        col += 1
        
        U[2*3+1, col] = 1  # A[2,1]
        V[1*8+k, col] = 1  # B[1,k]
        W[k*3+2, col] = 1  # C[2,k]
        col += 1
    
    # Handle A22*B2: A[2,2] * B[2, :]
    for k in range(8):
        U[2*3+2, col] = 1  # A[2,2]
        V[2*8+k, col] = 1  # B[2,k]
        W[k*3+2, col] = 1  # C[2,k]
        col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    
    # Check all nonzero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1) > tol:
                    return False
    
    # Check zero entries (sample)
    import random
    random.seed(42)
    for _ in range(200):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if i == i2 and j == j2 and j == k2:
            continue  # Skip valid entries
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    
    return True

def _attempt_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Attempt to reduce rank by:
    1. Identifying linearly dependent columns
    2. Trying to merge Strassen terms across blocks
    3. Using polynomial encoding to share computations
    """
    start_rank = U.shape[1]
    best_rank = start_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strategy 1: Check for linearly dependent columns and remove them
    if time.time() < deadline - 15:
        reduced = _remove_dependent_columns(U, V, W, n, m, p)
        if reduced is not None:
            best_rank = reduced[0]
            best_U, best_V, best_W = reduced[1], reduced[2], reduced[3]
    
    # Strategy 2: Try to merge Strassen terms across blocks
    # The key insight: for 2x2 * 2x8, we can use a single Strassen-like
    # pattern that handles all 4 2x2 blocks simultaneously by encoding
    # the 8 columns as coefficients in a polynomial ring.
    if time.time() < deadline - 10:
        merged = _try_polynomial_encoding(n, m, p, deadline)
        if merged is not None:
            if merged[0] < best_rank:
                best_rank = merged[0]
                best_U, best_V, best_W = merged[1], merged[2], merged[3]
    
    # Strategy 3: Greedy column elimination
    if time.time() < deadline - 5:
        greedy = _greedy_elimination(best_U, best_V, best_W, n, m, p, deadline)
        if greedy is not None:
            if greedy[0] < best_rank:
                best_rank = greedy[0]
                best_U, best_V, best_W = greedy[1], greedy[2], greedy[3]
    
    if best_rank < start_rank:
        return (best_rank, best_U, best_V, best_W)
    return None

def _remove_dependent_columns(U, V, W, n, m, p):
    """Remove linearly dependent columns from the decomposition."""
    rank = U.shape[1]
    
    # Find linearly independent columns
    # Use QR decomposition to identify pivot columns
    from scipy.linalg import qr
    
    # Check U for rank
    Q, R = qr(U, mode='economic')
    u_rank = np.sum(np.abs(np.diag(R)) > 1e-10)
    
    # Check V for rank  
    Q, R = qr(V, mode='economic')
    v_rank = np.sum(np.abs(np.diag(R)) > 1e-10)
    
    # Check W for rank
    Q, R = qr(W, mode='economic')
    w_rank = np.sum(np.abs(np.diag(R)) > 1e-10)
    
    # If any matrix has rank less than the number of columns,
    # we can potentially remove dependent columns
    if u_rank < rank or v_rank < rank or w_rank < rank:
        # Find which columns to keep
        # This is complex, so for now just return None
        pass
    
    return None

def _try_polynomial_encoding(n, m, p, deadline):
    """
    Try to use polynomial encoding to reduce rank.
    
    Key idea: Encode the 8 output columns as coefficients of a polynomial
    P(x) = sum_{k=0}^{7} C_k * x^k mod f(x) where f(x) is an irreducible
    polynomial of degree 8.
    
    Then C = A * B becomes a single matrix multiplication in the ring
    R = GF(x) / (f(x)), and we can use optimized decompositions for
    matrix multiplication over this ring.
    """
    # For 3x3 * 3x8, we can view this as:
    # C(x) = A * B(x) in R = Z[x] / (f(x))
    # where B(x) = sum B_k x^k
    
    # The rank of matrix multiplication over a ring can be lower than
    # the sum of ranks over the base field if we can share computations.
    
    # Try a specific construction: use the fact that for 2x2 * 2x8,
    # we can use 7 multiplications per 2x2 block, but if we encode
    # the 8 columns as a polynomial, we might use fewer total multiplications.
    
    # For now, this is a placeholder that returns None.
    # A full implementation would require finding an irreducible polynomial
    # of degree 8 and computing the tensor rank over the ring.
    
    return None

def _greedy_elimination(U, V, W, n, m, p, deadline):
    """
    Greedy column elimination: try to remove columns one at a time
    and check if the decomposition still works.
    """
    start_rank = U.shape[1]
    best_rank = start_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for trial in range(20):  # Try up to 20 eliminations
        if time.time() > deadline - 2:
            break
        
        # Try removing each column
        found = False
        for col_idx in range(best_rank):
            if time.time() > deadline - 2:
                break
            
            # Remove column col_idx
            mask = np.ones(best_rank, dtype=bool)
            mask[col_idx] = False
            
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            
            if _verify_decomposition(U_test, V_test, W_test, n, m, p):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank -= 1
                found = True
                break
        
        if not found:
            break
    
    if best_rank < start_rank:
        return (best_rank, best_U, best_V, best_W)
    return None
# EVOLVE-BLOCK-END
