"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3 rows of A, m=4 cols of A/rows of B, p=4 cols of B
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (12, rank) with W[k*n+i]
    
    deadline = time.time() + 140  # safety margin
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    
    # Try to find lower rank
    for rank in range(n*m*p - 1, 0, -1):
        if time.time() > deadline:
            break
        result = try_rank(n, m, p, rank)
        if result is not None:
            best_result = result
            break
    
    return best_result

def try_rank(n, m, p, rank, max_iters=2000, tol=1e-10):
    """Try to find a rank decomposition for n x m times m x p matrix mult."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor equation: for all i,j,k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For all (i',j',k') where not (i'=i, j'=j, k'=k): sum should be 0
    
    np.random.seed(42)
    
    # Build the tensor T with indices (i,j,k) -> 1, and we want:
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # This is a trilinear system. Use ALS: fix two, solve for third.
    
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Precompute index mappings
    # For each (i,j,k), the equation is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    best_residual = np.inf
    
    for it in range(max_iters):
        # Update U: for each (i,j), solve for U[i*m+j, :]
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        # This is: U[i*m+j,:] @ (V[j*p+k,:] * W[k*n+i,:])^T = 1 for all k
        # Let's define for fixed (i,j): we need sum over k of ... 
        # Actually, for fixed (i,j), the equations are over k:
        # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for each k
        
        # So for fixed (i,j), we have p equations in rank unknowns (U[i*m+j,:])
        # A_{k,c} = V[j*p+k,c] * W[k*n+i,c]
        # U[i*m+j,:] @ A^T = ones(p)
        
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve U[i*m+j,:] @ A^T = ones(p)
                # i.e., A @ U[i*m+j,:]^T = ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve for V[j*p+k, :]
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :]
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute residual
        residual = compute_residual(n, m, p, U, V, W)
        if residual < best_residual:
            best_residual = residual
        if residual < tol:
            # Try to verify and return
            return verify_and_format(n, m, p, U, V, W)
    
    return None

def compute_residual(n, m, p, U, V, W):
    """Compute the tensor residual."""
    res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                res += (s - 1)**2
    return res

def verify_and_format(n, m, p, U, V, W):
    """Verify the decomposition and format output."""
    # Check that the tensor equation holds
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1) > 1e-8:
                    return None
    # Also check that off-diagonal terms are zero
    # Actually, the above check is sufficient if we're only checking the "on-diagonal" terms
    # But we need to ensure the full tensor matches
    # The full tensor has T[i,j,k] = 1 for all i,j,k, and 0 otherwise
    # Wait, no. Let me reconsider.
    
    # The tensor T has entries T[(i,j),(j',k),(i',k')] = δ_{i,i'}δ_{j,j'}δ_{k,k'}
    # So we need: sum_c U[i,j,c] * V[j',k,c] * W[i',k',c] = δ_{i,i'}δ_{j,j'}δ_{k,k'}
    
    # Let's verify this fully
    for i in range(n):
        for j in range(m):
            for jp in range(m):  # j'
                for k in range(p):
                    for kp in range(p):  # k'
                        for ip in range(n):  # i'
                            s = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :])
                            expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            if abs(s - expected) > 1e-8:
                                return None
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
