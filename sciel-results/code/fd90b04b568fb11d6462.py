import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def _build_tensor():
    """Build the 3x4x5 multiplication tensor as a 3D array T[i,k,j] = 1."""
    n, m, p = DIMENSIONS  # n=3, m=4, p=5
    T = np.ones((n, p, m))  # T[i,k,j]
    return T

def _tensor_to_matrix(T, n, p, m):
    """Reshape tensor T[i,k,j] into matrix of shape (n*p, m) for matmul checking."""
    return T.reshape(n * p, m)

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, leave margin
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=5
    T = _build_tensor()  # shape (n, p, m)
    Tmat = T.reshape(n * p, m)  # shape (n*p, m) = (15, 4)
    
    # Try ranks from low to high
    best_result = None
    
    # Schoolbook as fallback
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank decompositions
    # The tensor condition: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is a 3-way tensor decomposition.
    
    # Let's try ranks from 35 to 59
    for rank in range(35, 60):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(20):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(42 + rank * 100 + trial)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Iterative refinement: alternate fixing two and solving for the third
            for it in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Fix V and W, solve for U
                # For each (i,j): U[i*m+j, c] should satisfy sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k
                # This is: U[i*m+j, :] @ (V[j*p+k,:] * W[k*n+i,:])^T ... 
                # Actually, for fixed i,j: we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # Let M_k = V[j*p+k,:] * W[k*n+i,:] (element-wise product), shape (rank,)
                # Then U[i*m+j,:] @ M_k^T = 1 for all k
                # So we solve: U[i*m+j,:] @ M^T = 1 where M is (p, rank) with M[k,:] = V[j*p+k,:]*W[k*n+i,:]
                # This is a linear system: M @ U[i*m+j,:]^T = 1 (vector of ones, length p)
                
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        # Build M: (p, rank)
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve M @ x = ones(p)
                        try:
                            x = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                        except:
                            x = U[i * m + j, :]
                        U[i * m + j, :] = x
                
                # Fix U and W, solve for V
                for j in range(m):
                    for k in range(p):
                        # Need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                        # Let M_i = U[i*m+j,:] * W[k*n+i,:], shape (rank,)
                        # V[j*p+k,:] @ M_i^T = 1 for all i
                        M = np.zeros((n, rank))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            x = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                        except:
                            x = V[j * p + k, :]
                        V[j * p + k, :] = x
                
                # Fix U and V, solve for W
                for k in range(p):
                    for i in range(n):
                        # Need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            x = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                        except:
                            x = W[k * n + i, :]
                        W[k * n + i, :] = x
                
                # Check convergence
                if it % 20 == 19:
                    # Compute error
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                                err += (val - 1.0) ** 2
                    if err < 1e-10:
                        found = True
                        break
            
            if found:
                # Try to round to integers or half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Check if rounded version is exact
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                            err += (val - 1.0) ** 2
                
                if err < 1e-10:
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    break
                else:
                    # Check if original is close enough (maybe use as-is if entries are close to half-integers)
                    # For now, accept if error is very small
                    if err < 1e-8:
                        # Try integer rounding
                        U_i = np.round(U)
                        V_i = np.round(V)
                        W_i = np.round(W)
                        err2 = 0.0
                        for i2 in range(n):
                            for j2 in range(m):
                                for k2 in range(p):
                                    val = np.dot(U_i[i2 * m + j2, :], V_i[j2 * p + k2, :] * W_i[k2 * n + i2, :])
                                    err2 += (val - 1.0) ** 2
                        if err2 < 1e-10:
                            best_result = {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
                            break
        
        if best_result is not None:
            break
    
    if best_result is None:
        best_result = schoolbook()
    
    return best_result
