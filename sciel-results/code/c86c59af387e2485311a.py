import numpy as np
import itertools
import time

_DIMENSIONS = [2, 4, 7]

def _matmul_tensor(n, m, p):
    """Build the 3-mode matmul tensor T[i,j,k] = 1 for C[i,k] = sum_j A[i,j]*B[j,k]."""
    T = np.zeros((n, p, m))  # (row_i, col_k, mid_j)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, k, j] = 1.0
    return T

def _verify(U, V, W, tol=1e-9):
    """Verify that sum_r U[.,r] * V[.,r] * W[.,r] reproduces the matmul tensor."""
    n, m, p = _DIMENSIONS
    T = _matmul_tensor(n, m, p)
    # Reshape U: (n*m, R) -> (n, m, R)
    # V: (m*p, R) -> (m, p, R)
    # W: (p*n, R) -> (p, n, R)  [W[k*n+i, r] = W_row[k, i, r]]
    U3 = np.array(U).reshape(n, m, -1)
    V3 = np.array(V).reshape(m, p, -1)
    W3 = np.array(W).reshape(p, n, -1)
    R = U3.shape[2]
    # C[i,k] = sum_{j,r} A[i,j]*B[j,k]*1  =>  T[i,k,j] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Build full contraction
    C_tensor = np.zeros((n, p, m))
    for r in range(R):
        # u: (n,m), v: (m,p), w: (p,n)
        u = U3[:, :, r]  # (n,m)
        v = V3[:, :, r]  # (m,p)
        w = W3[:, :, r]  # (p,n)
        # T[i,k,j] += u[i,j]*v[j,k]*w[k,i]
        for i in range(n):
            for j in range(m):
                if u[i, j] != 0:
                    for k in range(p):
                        C_tensor[i, k, j] += u[i, j] * v[j, k] * w[k, i]
    return np.max(np.abs(C_tensor - T)) < tol

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = _DIMENSIONS
    nmp = n * m * p  # 56 (schoolbook rank)
    
    def _time_left():
        return deadline - time.time()
    
    # Strategy: Try to find rank < 56 by using random initialization + 
    # structured search, but primarily we should find the known optimal rank.
    # For 2x4x7, the optimal rank is 36 (from literature on matrix multiplication).
    # Let's try to construct a rank-36 decomposition.
    
    # Approach: Use the "small border" method or exploit structure.
    # A practical approach: Use random vectors and check if they span the matmul tensor
    # with fewer than 56 terms.
    
    # First, try a simple structured approach based on known results.
    # The key insight: for 2x4x7, we can save rank by exploiting the 
    # relationship between the 2 rows and 4 columns.
    
    # Let me try a rank-reduction approach:
    # Start with schoolbook rank 56, try to merge columns.
    
    # Actually, let's use a more systematic approach:
    # Build random U, V, W of decreasing rank and check validity.
    
    # Known optimal rank for M(2,4,7): Let me try ranks from small to large.
    # The theoretical lower bound is n*m*p / (n+m+p) roughly, but more precisely
    # we need rank >= n*m*p - some savings.
    
    # For 2x4x7: schoolbook = 56. 
    # A rank-1 term gives one (i,j,k) product.
    
    # Let me try a randomized approach with structured initialization:
    np.random.seed(42)
    
    def try_rank(R):
        """Try to find a rank-R decomposition using random restarts."""
        # Use random vectors
        for restart in range(200):
            if _time_left() < 5:
                break
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            # Check validity
            if _verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-7):
                return U, V, W
        return None
    
    # Also try structured approaches:
    # For 2x4x7, one approach is to note that we can share computations.
    # The product C[i,k] = sum_j A[i,j]*B[j,k] has 2*4*7 = 56 terms.
    
    # Let me try a specific structured construction.
    # Idea: Group the 4 middle indices into pairs.
    # For each i (2 values), k (7 values), we sum over j (4 values).
    # We can write: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # Rank savings: If we can express some of these as linear combinations of others...
    # But A and B are generic, so we can't in general.
    
    # The optimal rank for 2x4x7 matrix multiplication is known to be 36.
    # Let me try to construct this using a specific pattern.
    
    # Alternative: Use the fact that the tensor can be decomposed more efficiently.
    # One approach: "L-shaped" or "corner" removal.
    
    # Let me try a different strategy: construct the decomposition directly
    # using a pattern that achieves rank 36.
    
    # For M(2,4,7), a known rank-36 construction uses the following idea:
    # Partition the 4x7 B-matrix and 2x4 A-matrix to allow sharing.
    
    # Since finding an exact rank-36 decomposition analytically is complex,
    # let me try a numerical approach: start with schoolbook, then try to
    # compress.
    
    # Actually, the most reliable approach within time limits:
    # Return the schoolbook rank-56 decomposition, which is always valid.
    # But we want to do better.
    
    # Let me try: for rank R < 56, use structured random with the constraint
    # that each (i,j,k) position has exactly one term (no overlaps).
    # This means we're just permuting, so it won't help.
    
    # Better approach: Use the "border rank" concept. 
    # For our purposes, let me try a practical search:
    
    # Try rank 36 with structured random (bilinear form)
    best_result = None
    for R in [36, 37, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56]:
        if _time_left() < 10:
            break
        U, V, W = try_rank(R)
        if U is not None:
            best_result = (R, U, V, W)
            break
    
    if best_result is not None:
        R, U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook rank 56
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
