"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import sys
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for c in range(rank):
        u = U[:, c]
        v = V[:, c]
        w = W[:, c]
        # Check that the rank-1 tensor u⊗v⊗w contributes correctly
        # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix mult, T[i,j,k] should be 1 if k == i*m+j and j == k*p+i... 
        # Actually, the tensor for matrix multiplication is:
        # T[i,j,k] = delta_{i, j} * delta_{k, i} ... no.
        # The multiplication tensor is T[i,j,k] = 1 if i=j=k... no.
        # For C = A*B where A is n x m, B is m x p, C is n x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 for all i,j,k (it's the identity tensor for the sum)
        # Wait, no. The tensor decomposition means:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c] * A[i,j] * B[j,k])
        # Actually, the standard form is:
        # C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
        # This equals (A*B)[i,k] = sum_j A[i,j] B[j,k]
        # So we need: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = delta_{j,j} = 1 for all i,j,k
        # Wait, that's not right either. Let me think again.
        # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
        # T[i,j,k] = 1 for all i,j,k? No.
        # Actually, the matrix multiplication tensor is defined such that:
        # (A*B)[i,k] = sum_{j} A[i,j] * B[j,k] = sum_{j,c} A[i,j] * B[j,k] * T[i,j,k]
        # where T[i,j,k] = 1 for all i,j,k.
        # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        # Let's verify this.
        if np.max(np.abs(sum(u[i] * v[j] * w[k] for i in range(n) for j in range(m) for k in range(p)))) > tol:
            return False
    # Actually, let me just check the full tensor equality
    T = np.zeros((n, m, p))
    for c in range(rank):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        # The contribution to T[i,j,k] from factor c is U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # This is not a simple outer product in the (i,j,k) space.
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)

def _try_rank(rank, n, m, p, time_limit=30):
    start_time = time.time()
    best_result = None
    best_loss = float('inf')
    
    for trial in range(5):
        if time.time() - start_time > time_limit:
            break
        rng = np.random.RandomState(trial * 12345 + rank)
        U0 = rng.randn(n * m, rank)
        V0 = rng.randn(m * p, rank)
        W0 = rng.randn(p * n, rank)
        
        def loss(params):
            if time.time() - start_time > time_limit:
                return float('inf')
            U = params[:n*m*rank].reshape(n, m, rank)
            V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m, p, rank)
            W = params[n*m*rank + m*p*rank:].reshape(p, n, rank)
            T = np.zeros((n, m, p))
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i,j,k] += U[i,j,c] * V[j,k,c] * W[k,i,c]
            return np.sum((T - 1.0)**2)
        
        def grad(params):
            if time.time() - start_time > time_limit:
                return np.zeros_like(params)
            U = params[:n*m*rank].reshape(n, m, rank)
            V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m, p, rank)
            W = params[n*m*rank + m*p*rank:].reshape(p, n, rank)
            T = np.zeros((n, m, p))
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i,j,k] += U[i,j,c] * V[j,k,c] * W[k,i,c]
            error = T - 1.0
            gU = np.zeros((n, m, rank))
            gV = np.zeros((m, p, rank))
            gW = np.zeros((p, n, rank))
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            gU[i,j,c] = error[i,j,k] * V[j,k,c] * W[k,i,c]
                            gV[j,k,c] = error[i,j,k] * U[i,j,c] * W[k,i,c]
                            gW[k,i,c] = error[i,j,k] * U[i,j,c] * V[j,k,c]
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        try:
            result = minimize(loss, x0, jac=grad, method='L-BFGS-B', 
                             options={'maxiter': 100, 'maxfun': 500, 'maxls': 20})
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
        except Exception:
            pass
    
    if best_result is None or best_loss > 1e-6:
        return None
    
    params = best_result.x
    U = params[:n*m*rank].reshape(n, m, rank)
    V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m, p, rank)
    W = params[n*m*rank + m*p*rank:].reshape(p, n, rank)
    
    U_list = [[U[i,j,c] for c in range(rank)] for i in range(n) for j in range(m)]
    V_list = [[V[j,k,c] for c in range(rank)] for j in range(m) for k in range(p)]
    W_list = [[W[k,i,c] for c in range(rank)] for k in range(p) for i in range(n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
