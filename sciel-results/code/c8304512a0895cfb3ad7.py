import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 3, 4, 7
    
    # Precompute indices
    # T_{(i,j), (j',k), (k',i')} = 1 if j==j' and i==i' else 0
    
    # We will try ranks from 83 down to 30
    # We start with a higher rank and try to reduce
    
    def solve_als(r, max_iters=50, tol=1e-8):
        # Initialize
        np.random.seed(42)
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Precompute target tensor values?
        # T is sparse.
        # For a given (i, j, j', k, k', i'), T is 1 if j==j' and i==i'.
        
        # We need to solve for U, V, W.
        # Equation: sum_l U[i*m+j, l] * V[j'*p+k, l] * W[k'*n+i, l] = delta[j, j'] * delta[i, i']
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None, None, None, float('inf')
                
            # Update U
            # For each (i, j), solve for U[(i,j), :]
            for i in range(n):
                for j in range(m):
                    # Build system A * x = b
                    # x is U[i*m+j, :] of length r
                    # A has shape (num_eq, r)
                    # b has shape (num_eq,)
                    # Eq index corresponds to (j', k, k', i')
                    num_eq = m * p * p * n
                    A = np.zeros((num_eq, r))
                    b = np.zeros(num_eq)
                    
                    eq_idx = 0
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    # T value
                                    if jp == j and ip == i:
                                        b[eq_idx] = 1.0
                                    else:
                                        b[eq_idx] = 0.0
                                    
                                    # A row: V[jp*p+k, :] * W[kp*n+i, :]
                                    # Element l: V[jp*p+k, l] * W[kp*n+i, l]
                                    A[eq_idx, :] = V[jp*p+k, :] * W[kp*n+i, :]
                                    eq_idx += 1
                                        
                    # Solve least squares
                    U[i*m+j, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    
            # Update V
            # For each (j, k), solve for V[j*p+k, :]
            for j in range(m):
                for k in range(p):
                    num_eq = n * m * p * n # i, j', k', i'
                    # Wait, the indices are (i, j', k', i')?
                    # Eq is sum_l U[i*m+j', l] * V[j*p+k, l] * W[k'*n+i, l] = delta[j', j] * delta[i, i']?
                    # No, the tensor is symmetric in a way.
                    # Original: C[i,k] = sum_j A[i,j] B[j,k]
                    # T_{(i,j), (j',k), (k',i')} = delta_{j,j'} delta_{i,i'}
                    # So if we fix V[j*p+k, :], the other indices are i, j', k', i'.
                    # The term is U[i*m+j', l] * W[k'*n+i, l].
                    # Target is delta_{j', j} delta_{i, i'}?
                    # Wait, the indices in T are (i, j) for A, (j', k) for B, (k', i') for C.
                    # The condition is j==j' and i==i'.
                    # So if we fix V[j*p+k, :], the free indices are i, j', k', i'.
                    # Target is 1 if j'==j and i==i'.
                    
                    num_eq = n * m * p * n
                    A = np.zeros((num_eq, r))
                    b = np.zeros(num_eq)
                    
                    eq_idx = 0
                    for i in range(n):
                        for jp in range(m):
                            for kp in range(p):
                                for ip in range(n):
                                    if jp == j and ip == i:
                                        b[eq_idx] = 1.0
                                    else:
                                        b[eq_idx] = 0.0
                                    
                                    A[eq_idx, :] = U[i*m+jp, :] * W[kp*n+i, :]
                                    eq_idx += 1
                                        
                    V[j*p+k, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    
            # Update W
            # For each (k, i), solve for W[k*n+i, :]
            for k in range(p):
                for i in range(n):
                    # Free indices: j, j', k', i'?
                    # Term: U[i*m+j, l] * V[j'*p+k, l] * W[k*n+i, l]
                    # Wait, W index is k*n+i.
                    # The tensor index for W is (k', i').
                    # So we fix k'=k, i'=i.
                    # Free indices: j, j', k (from V), i (from U)?
                    # No, U is (i, j), V is (j', k).
                    # If we fix W[k*n+i, :], the free indices are j, j', k.
                    # Wait, U depends on i. V depends on k.
                    # The term is U[i*m+j, l] * V[j'*p+k, l].
                    # The target is delta_{j, j'} delta_{i, i'}.
                    # Here i'=i is fixed. So target is delta_{j, j'} delta_{i, i}.
                    # So target is 1 if j==j' and i==i (always true for fixed i).
                    # So target is 1 if j==j'.
                    
                    num_eq = m * m * p
                    A = np.zeros((num_eq, r))
                    b = np.zeros(num_eq)
                    
                    eq_idx = 0
                    for j in range(m):
                        for jp in range(m):
                            for k_v in range(p):
                                if j == jp:
                                    b[eq_idx] = 1.0
                                else:
                                    b[eq_idx] = 0.0
                                    
                                A[eq_idx, :] = U[i*m+j, :] * V[jp*p+k_v, :]
                                eq_idx += 1
                                        
                    W[k*n+i, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            
            # Check convergence
            # Compute error
            # We can compute the residual for a few random points or the full tensor.
            # Full tensor is 6048 entries.
            error = 0
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    val = np.dot(U[i*m+j, :], V[jp*p+k, :] * W[kp*n+i, :])
                                    target = 1.0 if (jp == j and ip == i) else 0.0
                                    error += (val - target) ** 2
            error = np.sqrt(error)
            
            if error < tol:
                return U, V, W, error
                
        return U, V, W, error

    best_U = None
    best_V = None
    best_W = None
    best_rank = 84
    best_error = float('inf')
    
    # Try to find a lower rank
    # Start from 83 down to 30
    for r in range(83, 29, -1):
        if time.time() > deadline - 10:
            break
            
        U, V, W, error = solve_als(r, max_iters=20, tol=1e-6)
        
        if error < 1e-5:
            best_U = U
            best_V = V
            best_W = W
            best_rank = r
            best_error = error
            # If we find a very low rank, we can stop
            if r < 40:
                break
        
    # If no lower rank found, return schoolbook
    if best_U is None:
        # Schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_U = U
        best_V = V
        best_W = W
        best_rank = rank_school

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
