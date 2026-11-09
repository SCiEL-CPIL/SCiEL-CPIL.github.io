"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check if the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the full multiplication tensor from U, V, W
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 if i,j,k form a valid triple (always do), 0 otherwise
    # Actually for matrix multiplication, T[i,j,k] = 1 for all i,j,k
    # Wait, no. The tensor for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # has entries T[i,j,k,l] = delta[j,k] where C[i,l] = sum_j A[i,j] B[j,l]
    # But here we're using a different convention. Let me re-read.
    
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # The tensor being decomposed is the matrix multiplication tensor.
    # For multiplying (n x m) by (m x p) to get (n x p):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = delta[j,j] ... no that's not right.
    
    # Let me think again. The tensor T has indices (i, j, k) where:
    # i indexes rows of A (0..n-1)
    # j indexes the shared dimension (0..m-1)  
    # k indexes columns of B (0..p-1)
    # T[i,j,k] = 1 if this term appears in C, which is always the case for valid i,j,k
    # But that's just all ones, which doesn't make sense for a non-trivial decomposition.
    
    # Actually, I think the tensor is defined differently. Let me reconsider.
    # The matrix multiplication tensor T_{ijk} where:
    # i is row of C (0..n-1), k is column of C (0..p-1), j is the summation index (0..m-1)
    # T_{ijk} = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j] B[j,k])
    
    # Wait, that means the tensor is just all ones, and any rank-1 decomposition would work?
    # No, that can't be right. The tensor must be sparse in some sense.
    
    # Let me re-read the problem. "Find a rank-minimal exact tensor decomposition for multiplying 
    # a 2 by 4 matrix by a 4 by 6 matrix."
    
    # I think the tensor T is defined as: T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B 
    # contribute to C[i,k]. But that's always true.
    
    # Actually, I think I'm overcomplicating this. The standard matrix multiplication tensor 
    # T_{ijk} has T_{ijk} = 1 if and only if the indices are "compatible" in the multiplication.
    # For C = AB where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # But that's a tensor of all ones, which has rank 1! That can't be the intended problem.
    
    # I think the issue is that the tensor is not just T[i,j,k] but rather the full 4-index tensor
    # T[i,j,k,l] = delta[j,k] which represents C[i,l] = sum_{j,k} T[i,j,k,l] A[i,j] B[k,l]
    # But the problem states U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # Let me reconsider. Maybe the tensor is T[i,j,k] where:
    # i indexes (row of A, col of A) pairs -> n*m
    # j indexes (row of B, col of B) pairs -> m*p  
    # k indexes (row of C, col of C) pairs -> n*p
    # And T[i,j,k] = 1 if the product contributes.
    
    # Actually, looking at the schoolbook solution:
    # U[i*m+j][column] = 1  -> i is row of A, j is col of A (which is row of B)
    # V[j*p+k][column] = 1  -> j is row of B, k is col of B (which is col of C)
    # W[k*n+i][column] = 1  -> k is col of C, i is row of C
    # So the tensor T[i,j,k] where i in [0,n*m), j in [0,m*p), k in [0,p*n)
    # T[i,j,k] = 1 if the indices match up properly.
    
    # Specifically, if i = i_A*m + j_A, j = j_B*p + k_B, k = k_C*n + i_C
    # Then T[i,j,k] = 1 if j_A == j_B and i_A == i_C and k_B == k_C
    
    # So the tensor is: T[(i_A*m+j_A), (j_B*p+k_B), (k_C*n+i_C)] = 1 iff j_A==j_B and i_A==i_C and k_B==k_C
    
    # This is the matrix multiplication tensor!
    
    # Let's verify: for each column c, we have U[i,c], V[j,c], W[k,c]
    # The contribution to T[i,j,k] is sum_c U[i,c] * V[j,c] * W[k,c]
    # This should equal 1 if the indices are compatible, 0 otherwise.
    
    n, m, p = n, m, p
    rank = U.shape[1] if hasattr(U, 'shape') else len(U[0])
    
    # Build expected tensor
    # T[i,j,k] = 1 if (i//m == k%n) and (i%m == j//p) and (j%p == k//n)
    # Wait, let me be more careful.
    # i = i_A * m + j_A where i_A in [0,n), j_A in [0,m)
    # j = j_B * p + k_B where j_B in [0,m), k_B in [0,p)
    # k = k_C * n + i_C where k_C in [0,p), i_C in [0,n)
    # T[i,j,k] = 1 iff j_A == j_B and i_A == i_C and k_B == k_C
    
    max_err = 0.0
    for i in range(n * m):
        i_A = i // m
        j_A = i % m
        for j in range(m * p):
            j_B = j // p
            k_B = j % p
            for k in range(p * n):
                k_C = k // n
                i_C = k % n
                expected = 1.0 if (j_A == j_B and i_A == i_C and k_B == k_C) else 0.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i][c] * V[j][c] * W[k][c]
                err = abs(actual - expected)
                if err > max_err:
                    max_err = err
    
    return max_err < 1e-9

def _compute_error(U, V, W, n, m, p):
    """Compute the max absolute error of the tensor decomposition."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    max_err = 0.0
    for i in range(nm):
        i_A = i // m
        j_A = i % m
        for j in range(mp):
            j_B = j // p
            k_B = j % p
            for k in range(pn):
                k_C = k // n
                i_C = k % n
                expected = 1.0 if (j_A == j_B and i_A == i_C and k_B == k_C) else 0.0
                actual = np.dot(U[i], V[j] * W[k])
                err = abs(actual - expected)
                if err > max_err:
                    max_err = err
    return max_err

def _try_rank(rank, n, m, p, time_limit=10):
    """Try to find a decomposition of given rank using iterative refinement."""
    start_time = time.time()
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute index mappings for efficiency
    # For each i in U (index i_A*m + j_A), find compatible (j, k) pairs
    # j must have j_B == j_A, i.e., j = j_A*p + k_B
    # k must have i_C == i_A and k_C == k_B, i.e., k = k_B*n + i_A
    
    # For each j in V (index j_B*p + k_B), find compatible (i, k) pairs
    # i must have j_A == j_B, i.e., i = i_A*m + j_B
    # k must have i_C == i_A and k_C == k_B, i.e., k = k_B*n + i_A
    
    # For each k in W (index k_C*n + i_C), find compatible (i, j) pairs
    # i must have i_A == i_C and j_A == j_B, i.e., i = i_C*m + j_A
    # j must have j_B == j_A and k_B == k_C, i.e., j = j_A*p + k_C
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    np.random.seed(42)
    
    n_attempts = 10
    for attempt in range(n_attempts):
        if time.time() - start_time > time_limit:
            break
        
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Iterative refinement
        for iteration in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Update U: for each i, solve least squares
            # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For fixed i, the constraints are: for all compatible (j,k), U[i,:] dot (V[j,:] * W[k,:]) = 1
            # This is a linear system: sum_c U[i,c] * (V[j,c] * W[k,c]) = 1 for each compatible (j,k)
            # We can set up a least squares problem
            
            for i in range(nm):
                i_A = i // m
                j_A = i % m
                
                # Collect all compatible (j, k) pairs
                # j = j_A * p + k_B for k_B in range(p)
                # k = k_B * n + i_A for k_B in range(p)
                # So for each k_B in range(p): j = j_A*p + k_B, k = k_B*n + i_A
                
                if p == 0:
                    continue
                    
                # Build the linear system: A * x = b
                # A has rows for each k_B: row = V[j_A*p+k_B] * W[k_B*n+i_A]
                # b = 1 for each row
                A_rows = []
                for k_B in range(p):
                    j = j_A * p + k_B
                    k = k_B * n + i_A
                    A_rows.append(V[j] * W[k])
                
                if len(A_rows) == 0:
                    continue
                    
                A = np.array(A_rows)
                b = np.ones(len(A_rows))
                
                # Solve least squares: A @ U[i,:] ≈ b
                try:
                    U[i], residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    pass
            
            # Update V: for each j, solve least squares
            for j in range(mp):
                j_B = j // p
                k_B = j % p
                
                # Compatible (i, k) pairs:
                # i = i_A * m + j_B for i_A in range(n)
                # k = k_B * n + i_A for i_A in range(n)
                # So for each i_A in range(n): i = i_A*m + j_B, k = k_B*n + i_A
                
                if n == 0:
                    continue
                    
                A_rows = []
                for i_A in range(n):
                    i = i_A * m + j_B
                    k = k_B * n + i_A
                    A_rows.append(U[i] * W[k])
                
                if len(A_rows) == 0:
                    continue
                    
                A = np.array(A_rows)
                b = np.ones(len(A_rows))
                
                try:
                    V[j], residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    pass
            
            # Update W: for each k, solve least squares
            for k in range(pn):
                k_C = k // n
                i_C = k % n
                
                # Compatible (i, j) pairs:
                # i = i_C * m + j_A for j_A in range(m)
                # j = j_A * p + k_C for j_A in range(m)
                # So for each j_A in range(m): i = i_C*m + j_A, j = j_A*p + k_C
                
                if m == 0:
                    continue
                    
                A_rows = []
                for j_A in range(m):
                    i = i_C * m + j_A
                    j = j_A * p + k_C
                    A_rows.append(U[i] * V[j])
                
                if len(A_rows) == 0:
                    continue
                    
                A = np.array(A_rows)
                b = np.ones(len(A_rows))
                
                try:
                    W[k], residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    pass
            
            # Check error periodically
            if iteration % 5 == 0 or iteration == 99:
                err = _compute_error(U, V, W, n, m, p)
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if best_err < 1e-9:
                    break
    
    if best_err < 1e-9 and best_U is not None:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = nm * mp // p  # This is n*m*p... wait
    # Actually schoolbook rank is n*m*p = 2*4*6 = 48
    
    deadline = time.time() + 140  # 10 second safety margin
    
    # Try to find a lower-rank decomposition
    # Start from just below schoolbook and go down
    for rank in range(n*m*p - 1, 20, -1):
        if time.time() > deadline - 15:
            break
        
        result = _try_rank(rank, n, m, p, time_limit=12)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
