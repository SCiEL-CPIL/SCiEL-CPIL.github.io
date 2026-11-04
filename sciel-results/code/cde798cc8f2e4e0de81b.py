"""Matrix multiplication 3x4x6 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Build the multiplication tensor T[i,j,k] which should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # But actually the tensor for matrix multiplication is: T[i,j,k] = delta[i,j,k] where
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, let me re-read the task.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The convention is W[k*n+i, r]
    # So the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] should be 1 if we're decomposing the identity-like tensor for multiplication.
    # Actually for matrix multiplication, the tensor is T[i,j,k] = 1 for all valid (i,j,k)
    # No wait. The multiplication tensor T has T[i,j,k] = 1 if the indices are compatible.
    # For multiplying A (n x m) by B (m x p), result C (n x p):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has components T[i,j,k] such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the identity tensor in the appropriate sense)
    # Actually T[i,j,k] = delta_{i,j,k} doesn't make sense. Let me think again.
    # The tensor is T in V* ⊗ V* ⊗ V* where the bilinear form is <T, A ⊗ B> = trace(A B)
    # No. The tensor T satisfies: T(A,B) = A*B where T is a trilinear map.
    # T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # Because C[i,k] = sum_j A[i,j] * B[j,k] * 1
    # So the target tensor T[i,j,k] = 1 for all i,j,k.
    
    # Let's verify: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    return max_err < 1e-9

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 24
    pn = p * n  # 18
    
    # Start with schoolbook as fallback
    rank_sb = n * m * p  # 72
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # The known lower bound for 3x4x6 is at least max(n*m, m*p, p*n) = max(12,24,18) = 24
    # Upper bound is n*m*p = 72
    # Let's try ranks from 24 upward and see if we can find an exact decomposition
    
    # Use a randomized approach with rounding
    # For a given rank r, we need U (12 x r), V (24 x r), W (18 x r) with integer/half-integer entries
    # such that for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Strategy: Try small integer values (-2,-1,0,1,2) for entries
    # This is a constraint satisfaction problem
    
    # Let's try a systematic search for small ranks
    # Given the small sizes, we can try to build up solutions
    
    # Actually, let's try a different approach: use the known structure
    # For matrix multiplication, there are known decompositions.
    # Let's try rank = 24 (the lower bound)
    
    # For rank 24, each column c corresponds to a term.
    # U has 12 rows, V has 24 rows, W has 18 rows.
    # Each column c: U[:,c] is a vector in R^12, V[:,c] in R^24, W[:,c] in R^18
    # The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is equivalent to: the tensor T = sum_c u_c ⊗ v_c ⊗ w_c equals the all-ones tensor
    # where u_c is indexed by (i,j), v_c by (j,k), w_c by (k,i)
    
    # Let's try a greedy/iterative approach:
    # 1. Initialize with random small integers
    # 2. Use ALS-like iterations to minimize the error
    # 3. Round to nearest half-integer and verify
    
    def try_rank(r, max_iters=5000, seed=42):
        np.random.seed(seed)
        # Initialize with small random integers in {-1, 0, 1}
        U = np.random.randint(-1, 2, (nm, r)).astype(np.float64)
        V = np.random.randint(-1, 2, (mp, r)).astype(np.float64)
        W = np.random.randint(-1, 2, (pn, r)).astype(np.float64)
        
        # Make sure not all zeros in any column
        for c in range(r):
            if np.all(U[:,c] == 0) or np.all(V[:,c] == 0) or np.all(W[:,c] == 0):
                U[np.random.randint(nm), c] = 1
                V[np.random.randint(mp), c] = 1
                W[np.random.randint(pn), c] = 1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each (i,j), U[i*m+j, c] should satisfy
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is a linear system in U[i*m+j, :] for fixed (i,j)
            # For fixed (i,j): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
            # Let A_ij be a p x r matrix with A_ij[k,c] = V[j*p+k,c] * W[k*n+i,c]
            # Then A_ij @ U[i*m+j,:] = ones(p)
            
            # Similarly for V: for fixed (j,k), sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
            # And for W: for fixed (k,i), sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
            
            # Use least squares for each row
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build A: p x r
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    # Solve A @ x = b in least squares sense
                    if r <= p:
                        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    else:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                    U[i*m+j, :] = x
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    V[j*p+k, :] = x
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    W[k*n+i, :] = x
            
            # Check error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-6:
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify rounded solution
                max_err_r = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_r[i*m+j, :], V_r[j*p+k, :] * W_r[k*n+i, :])
                            err = abs(s - 1.0)
                            if err > max_err_r:
                                max_err_r = err
                
                if max_err_r < 1e-9:
                    # Convert to list of lists
                    U_list = [[float(U_r[i,c]) for c in range(r)] for i in range(nm)]
                    V_list = [[float(V_r[i,c]) for c in range(r)] for i in range(mp)]
                    W_list = [[float(W_r[i,c]) for c in range(r)] for i in range(pn)]
                    return (r, U_list, V_list, W_list)
        
        return None
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, p*n)  # 24
    
    for r in range(lower_bound, rank_sb + 1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random seeds
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            result = try_rank(r, max_iters=2000, seed=seed)
            if result is not None:
                r_found, U_f, V_f, W_f = result
                if r_found < best_rank:
                    best_rank = r_found
                    best_U, best_V, best_W = U_f, V_f, W_f
                break  # Found a solution for this rank, move to next
    
    return {"U": best_U, "V": best_V, "W": best_W}
