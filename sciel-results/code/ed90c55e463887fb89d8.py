"""Tensor rank minimization for 3x4x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0  # safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition using iterative refinement
    # The constraint is: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Wait, that's not the right formulation.
    
    # Correct formulation: The matrix multiplication tensor satisfies:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In terms of the decomposition:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
    # No, that's still not right.
    
    # Let me think about this more carefully. The bilinear map is:
    # (A, B) -> C where C = A * B
    # A is n x m, B is m x p, C is n x p.
    
    # We can write: C = sum_r alpha_r(A) * beta_r(B) * gamma_r
    # where alpha_r: R^{n*m} -> R, beta_r: R^{m*p} -> R, gamma_r: R^{n*p} -> R
    # and gamma_r is a fixed matrix (not depending on A or B).
    
    # Actually, the standard formulation is:
    # C = sum_r (U_r^T * vec(A)) * (V_r^T * vec(B)) * W_r
    # where U_r in R^{n*m}, V_r in R^{m*p}, W_r in R^{n*p}
    # and (U_r^T * vec(A)) is a scalar, (V_r^T * vec(B)) is a scalar, and W_r is a matrix.
    
    # But the problem says W has shape (p*n, rank), so W_r is a vector, not a matrix.
    
    # I think the correct formulation is:
    # vec(C) = sum_r (U_r^T * vec(A)) * (V_r^T * vec(B)) * W_r
    # where U_r in R^{n*m}, V_r in R^{m*p}, W_r in R^{n*p}
    # and vec(C) is the vectorization of C in row-major order: vec(C)[k*n+i] = C[i,k].
    
    # So the constraint is: for all i, k:
    # vec(C)[k*n+i] = sum_j A[i,j] * B[j,k] = sum_r (sum_{j'} U[i*m+j', r] * A[i,j']) * (sum_{j''} V[j''*p+k, r] * B[j'',k]) * W[k*n+i, r]
    
    # This must hold for all A, B. So we need:
    # For all i, j, k, r: the coefficient of A[i,j] * B[j,k] in the sum must equal 1.
    # That is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    
    # Yes! That's the correct constraint. So we need to find U (n*m x R), V (m*p x R), W (p*n x R)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # This is the same as before. Let's try to solve it numerically.
    
    def try_rank(R, max_iter=2000, tol=1e-9):
        """Try to find a rank-R decomposition."""
        np.random.seed(123)
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        
        lr = 0.005
        best_err = float('inf')
        
        for it in range(max_iter):
            # Compute error: E[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1
            # This is expensive to compute directly. Let's use einsum.
            
            # E[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We can compute this as: for each r, the outer product of U[:,r], V[:,r], W[:,r]
            # but with appropriate index mappings.
            
            # Actually, let's just compute it directly for small sizes.
            E = np.zeros((n, m, p))
            for r in range(R):
                Ur = U[:, r]  # (n*m,)
                Vr = V[:, r]  # (m*p,)
                Wr = W[:, r]  # (p*n,)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            E[i, j, k] += Ur[i * m + j] * Vr[j * p + k] * Wr[k * n + i]
            
            err = np.max(np.abs(E - 1.0))
            if err < best_err:
                best_err = err
            
            if err < tol:
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                E_check = np.zeros((n, m, p))
                for r in range(R):
                    Ur = U_rounded[:, r]
                    Vr = V_rounded[:, r]
                    Wr = W_rounded[:, r]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                E_check[i, j, k] += Ur[i * m + j] * Vr[j * p + k] * Wr[k * n + i]
                
                if np.max(np.abs(E_check - 1.0)) < 1e-8:
                    return U_rounded, V_rounded, W_rounded
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        dE = E[i, j, k] - 1.0
                        for r in range(R):
                            grad_U[i * m + j, r] += dE * V[j * p + k, r] * W[k * n + i, r]
                            grad_V[j * p + k, r] += dE * U[i * m + j, r] * W[k * n + i, r]
                            grad_W[k * n + i, r] += dE * U[i * m + j, r] * V[j * p + k, r]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if time.time() - start_time > deadline - 5:
                break
        
        return None
    
    # Try ranks from 55 down to 50
    for R in range(55, 45, -1):
        if time.time() - start_time > deadline - 10:
            break
        result = try_rank(R, max_iter=1000)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U.T]
            V_list = [[float(x) for x in row] for row in V.T]
            W_list = [[float(x) for x in row] for row in W.T]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()
