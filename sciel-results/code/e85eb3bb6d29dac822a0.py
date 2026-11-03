import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Schoolbook decomposition (rank = n*m*p = 63)
    def schoolbook():
        total_rank = n * m * p
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify a candidate decomposition
    def verify(U, V, W, tol=1e-9):
        """
        Check if U, V, W correctly decompose the matrix multiplication tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        
        The tensor T has T[(i,j),(j,k),(i,k)] = 1
        We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if consistent, 0 otherwise
        Actually the standard form is:
        C[i,k] = sum_j A[i,j] B[j,k]
        The tensor is T_{(i,j),(j,k),(i,k)} such that:
        sum_{(i,j),(j,k)} A[i,j] B[j,k] T_{(i,j),(j,k),(i,k)} = C[i,k]
        So T_{(i,j),(j,k),(i',k')} = 1 if i=i' and k=k', 0 otherwise.
        
        For rank decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        where u_r is indexed by (i,j), v_r by (j,k), w_r by (i,k)
        
        So: sum_r U[(i,j),r] * V[(j,k),r] * W[(i,k),r] = T_{(i,j),(j,k),(i,k)}
        = 1 if the indices are consistent (which they always are in the standard indexing)
        
        Wait, let me be more careful. The matrix multiplication tensor:
        For C = AB, C[i,k] = sum_j A[i,j] B[j,k]
        
        The tensor T is a 3-tensor with modes of size n*m, m*p, p*n
        T[a, b, c] where a indexes (i,j), b indexes (j,k), c indexes (i,k)
        
        T[(i,j), (j',k), (i',k')] = 1 if j=j' and i=i' and k=k', else 0
        
        So for the decomposition to work:
        sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
        
        Let's verify by checking all entries.
        """
        rank = U.shape[1]
        # Build the tensor from decomposition
        # T_decomp[(i,j),(j',k),(i',k')] = sum_r U[i*m+j,r] * V[j'*p+k,r] * W[i'*n+k',r]
        
        # Efficient check: for each r, the outer product contributes
        # Let's check a sample of entries or all entries
        
        # All entries: (n*m) * (m*p) * (p*n) = 9 * 21 * 21 = 3969 entries
        # That's manageable
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                a = i * m + j
                                b = jp * p + k
                                c = ip * n + kp
                                # Expected value
                                if j == jp and i == ip and k == kp:
                                    expected = 1.0
                                else:
                                    expected = 0.0
                                # Computed value
                                val = 0.0
                                for r in range(rank):
                                    val += U[a, r] * V[b, r] * W[c, r]
                                err = abs(val - expected)
                                if err > max_err:
                                    max_err = err
        return max_err < tol
    
    # Faster verification using numpy
    def verify_fast(U, V, W, tol=1e-9):
        rank = U.shape[1]
        # Use einsum or matrix operations
        # T_decomp[i,j,jp,k,ip,kp] = sum_r U[i*m+j,r] * V[jp*p+k,r] * W[ip*n+kp,r]
        
        # Reshape for efficient computation
        U_r = U  # (n*m, rank)
        V_r = V  # (m*p, rank)
        W_r = W  # (p*n, rank)
        
        # For each r, we have outer products
        # Let's compute the full tensor via sum of outer products
        # This might be memory intensive: 9*21*21 = 3969 floats, fine
        
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
        
        # Now check against expected
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                a = i * m + j
                                b = jp * p + k
                                c = ip * n + kp
                                if j == jp and i == ip and k == kp:
                                    expected = 1.0
                                else:
                                    expected = 0.0
                                err = abs(T[a, b, c] - expected)
                                if err > max_err:
                                    max_err = err
        return max_err < tol
    
    # Try to find a lower-rank decomposition using random initialization + optimization
    def try_rank_optimization(target_rank, num_restarts=5, max_iter=200):
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Alternating least squares style
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Fix V and W, solve for U
                # For each (i,j), U[i*m+j, :] should satisfy:
                # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[i'*n+k', r] = delta_{j,j'} delta_{i,i'} delta_{k,k'}
                # This is linear in U for fixed V, W
                
                # Let's set up the linear system for U
                # For each a = i*m+j, we need to find U[a, :] such that
                # sum_r U[a, r] * (V[:, r] * W[:, r])_b = T[a, b, c] for all b, c
                # Actually it's: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                # For fixed a, this is: sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
                # This is a linear system: M_a * U[a,:]^T = T_a
                # where M_a is a (m*p * p*n) x rank matrix with M_a[b,c, r] = V[b,r] * W[c,r]
                
                # This is expensive to set up. Let's use a simpler approach:
                # Gradient descent on the loss function
                
                # Compute current tensor
                T_current = np.zeros((n*m, m*p, p*n))
                for r in range(target_rank):
                    T_current += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
                # Compute loss
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                for ip in range(n):
                                    for kp in range(p):
                                        a = i * m + j
                                        b = jp * p + k
                                        c = ip * n + kp
                                        if j == jp and i == ip and k == kp:
                                            expected = 1.0
                                        else:
                                            expected = 0.0
                                        loss += (T_current[a, b, c] - expected) ** 2
                
                if loss < best_err:
                    best_err = loss
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if loss < 1e-12:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for r in range(target_rank):
                    dU[:, r] = 2 * np.einsum('abc,bc->a', 
                        T_current - _expected_tensor(), 
                        np.einsum('b,c->bc', V[:, r], W[:, r]))
                    dV[:, r] = 2 * np.einsum('abc,ac->b', 
                        T_current - _expected_tensor(), 
                        np.einsum('a,c->ac', U[:, r], W[:, r]))
                    dW[:, r] = 2 * np.einsum('abc,ab->c', 
                        T_current - _expected_tensor(), 
                        np.einsum('a,b->ab', U[:, r], V[:, r]))
                
                # Gradient step
                lr = 0.01
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
        
        if best_U is not None:
            return best_U, best_V, best_W, best_err
        return None, None, None, float('inf')
    
    def _expected_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    # First, let's try a structured approach based on the block structure
    # For 3x3x7, we can try to exploit the fact that n=m=3
    
    # Try various target ranks from low to high
    target_ranks = [40, 45, 50, 55, 60]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        # Try optimization
        U, V, W, err = try_rank_optimization(target_rank, num_restarts=3, max_iter=100)
        
        if U is not None and err < 1e-9:
            # Verify with fast method
            if verify_fast(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return schoolbook()
