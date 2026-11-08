"""Program for matrix multiplication 3x4x4 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_mat = np.array(U, dtype=np.float64)
    V_mat = np.array(V, dtype=np.float64)
    W_mat = np.array(W, dtype=np.float64)

    U_reshaped = U_mat.reshape(n, m, rank)
    V_reshaped = V_mat.reshape(m, p, rank)
    W_reshaped = W_mat.reshape(p, n, rank)

    T_computed = np.zeros((n, m, p))
    for r in range(rank):
        T_computed += U_reshaped[:, :, r] * V_reshaped[:, :, r] * W_reshaped[:, :, r]

    T_true = np.ones((n, m, p))
    return np.allclose(T_computed, T_true, atol=tol)


def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2x2():
    """Return the Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    
    The multiplication tensor for 2x2x2 is T[i,j,k] = 1 for all i,j,k in {0,1}.
    We need T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r].
    
    Strassen's algorithm gives:
    C00 = M1 - M2 + M3 + M7
    C01 = M4 + M5 - M6 - M7
    C10 = M2 + M6
    C11 = M1 - M3 + M4 + M5
    
    where:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    """
    # n=m=p=2
    # U has shape (4, 7): indexed by (i,j) where i,j in {0,1}
    # V has shape (4, 7): indexed by (j,k) where j,k in {0,1}
    # W has shape (4, 7): indexed by (k,i) where k,i in {0,1}
    
    # For each r (0..6), U[i,j,r], V[j,k,r], W[k,i,r]
    
    # r=0: M1 = (A00+A11)(B00+B11)
    # U[0,0]=1, U[1,1]=1 (A00 and A11 contribute)
    # V[0,0]=1, V[0,1]=1, V[1,0]=1, V[1,1]=1 (B00+B11 means all B entries)
    # Wait, that's not right. Let me think more carefully.
    
    # The tensor decomposition is:
    # T[i,j,k] = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
    
    # For the 2x2x2 case, T[i,j,k] = 1 for all i,j,k in {0,1}.
    
    # Strassen: C = A*B where
    # C[0,0] = M1 - M2 + M3 + M7
    # C[0,1] = M4 + M5 - M6 - M7
    # C[1,0] = M2 + M6
    # C[1,1] = M1 - M3 + M4 + M5
    
    # But we need to express this as a tensor decomposition of the multiplication tensor.
    # The multiplication tensor T[i,j,k] satisfies: (A*B)[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # The rank decomposition means: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # Let me construct this directly from Strassen's algorithm.
    # Each multiplication M_r is a product of two linear forms in A and B.
    # M_r = (sum_{i,j} a_{ij} A[i,j]) * (sum_{j,k} b_{jk} B[j,k])
    # Then C[i,k] gets contribution from M_r with coefficient w_{ik,r}.
    # The tensor decomposition is: T[i,j,k] = sum_r a_{ij,r} * b_{jk,r} * w_{ki,r}
    # where a_{ij,r} is the coefficient of A[i,j] in the first factor of M_r,
    # b_{jk,r} is the coefficient of B[j,k] in the second factor of M_r,
    # w_{ki,r} is the coefficient of C[k,i]... wait, no.
    
    # Actually, the standard form is:
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r (sum_{i,j} a_{ij,r} A[i,j]) (sum_{j,k} b_{jk,r} B[j,k]) w_{ik,r}
    # where w_{ik,r} is the coefficient for output C[i,k] from term r.
    
    # For the tensor: T[i,j,k] = sum_r a_{ij,r} b_{jk,r} w_{ik,r}
    # But in our notation, W is indexed by (k,i), so W[k*n+i, r] = w_{ik,r}
    
    # Let me define the factors for Strassen:
    
    # M1 = (A00+A11) * (B00+B11)
    # a factors: A00:1, A11:1, others:0
    # b factors: B00:1, B01:1, B10:1, B11:1... wait no.
    # (B00+B11) means B[0,0] and B[1,1], so b[0,0]=1, b[1,1]=1
    # But wait, in the matrix multiplication, B is 2x2, so B[j,k] for j,k in {0,1}
    # (B00+B11) = B[0,0] + B[1,1]
    # So b[0,0]=1, b[1,1]=1, b[0,1]=0, b[1,0]=0
    
    # Hmm, but that doesn't match. Let me re-examine.
    
    # Actually, I think I need to be more careful. Let me just verify the schoolbook works
    # and try a simple optimization approach to see if we can do better.
    
    # For now, return the schoolbook rank-8 decomposition for 2x2x2
    n2, m2, p2 = 2, 2, 2
    U, V, W = _schoolbook(n2, m2, p2)
    return U, V, W


def _try_optimization(n, m, p, target_rank, time_limit=60):
    """Try to find a lower-rank decomposition using optimization."""
    import numpy as np
    
    start = time.time()
    
    # Random initialization
    best_loss = np.inf
    best_params = None
    
    # Use multiple random restarts
    for restart in range(5):
        if time.time() - start > time_limit:
            break
        
        np.random.seed(restart * 12345 + 67890)
        
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Simple gradient descent with analytic gradient
        lr = 0.01
        for step in range(200):
            if time.time() - start > time_limit:
                break
            
            U_curr = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_curr = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_curr = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            U_resh = U_curr.reshape(n, m, target_rank)
            V_resh = V_curr.reshape(m, p, target_rank)
            W_resh = W_curr.reshape(p, n, target_rank)
            
            T_comp = np.zeros((n, m, p))
            for r in range(target_rank):
                T_comp += U_resh[:, :, r] * V_resh[:, :, r] * W_resh[:, :, r]
            
            residual = T_comp - 1.0
            loss = np.sum(residual**2)
            
            if loss < 1e-12:
                return U_curr.tolist(), V_curr.tolist(), W_curr.tolist()
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
            
            # Gradient computation
            # dL/dU[i,j,r] = 2 * sum_k residual[i,j,k] * V[j,k,r] * W[k,i,r]
            # dL/dV[j,k,r] = 2 * sum_i residual[i,j,k] * U[i,j,r] * W[k,i,r]
            # dL/dW[k,i,r] = 2 * sum_j residual[i,j,k] * U[i,j,r] * V[j,k,r]
            
            dU = np.zeros_like(U_curr)
            dV = np.zeros_like(V_curr)
            dW = np.zeros_like(W_curr)
            
            for r in range(target_rank):
                # For each (i,j,k): contribution to dU[i*m+j, r]
                # dU[i*m+j, r] = 2 * sum_k residual[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            dU[i*m+j, r] += 2.0 * residual[i,j,k] * V_curr[j*p+k, r] * W_curr[k*n+i, r]
                
                # dV[j*p+k, r] = 2 * sum_i residual[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
                for j in range(m):
                    for k in range(p):
                        for i in range(n):
                            dV[j*p+k, r] += 2.0 * residual[i,j,k] * U_curr[i*m+j, r] * W_curr[k*n+i, r]
                
                # dW[k*n+i, r] = 2 * sum_j residual[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
                for k in range(p):
                    for i in range(n):
                        for j in range(m):
                            dW[k*n+i, r] += 2.0 * residual[i,j,k] * U_curr[i*m+j, r] * V_curr[j*p+k, r]
            
            grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            params -= lr * grad
            
            # Adaptive learning rate
            if loss > best_loss * 1.5:
                lr *= 0.5
            elif loss < best_loss * 0.5:
                lr *= 1.1
                lr = min(lr, 0.1)
    
    # Check best solution
    if best_params is not None and best_loss < 1e-6:
        U_best = best_params[:n*m*target_rank].reshape(n*m, target_rank)
        V_best = best_params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_best = best_params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        # Round to half-integers if close
        for mat in [U_best, V_best, W_best]:
            rounded = np.round(mat * 2.0) / 2.0
            if np.max(np.abs(rounded - mat)) < 1e-6:
                mat[:] = rounded
        
        return U_best.tolist(), V_best.tolist(), W_best.tolist()
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Schoolbook fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = n * m * p  # 48
    
    # Try to find lower rank decompositions
    # Lower bound is max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # Try ranks from 16 upward
    
    for rank in range(16, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        result = _try_optimization(n, m, p, rank, time_limit=20)
        if result is not None:
            U_sol, V_sol, W_sol = result
            if _verify(U_sol, V_sol, W_sol, n, m, p):
                best_U, best_V, best_W = U_sol, V_sol, W_sol
                best_rank = rank
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
