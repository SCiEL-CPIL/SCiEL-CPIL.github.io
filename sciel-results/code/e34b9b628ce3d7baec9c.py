"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j'*p+k', k*n+i'] = 1 if j==j' and k==k' else 0
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if j == j2 and k == k2:
                                T[i*m+j, j2*p+k2, k*n+i2] = 1.0
    
    deadline = time.time() + 140.0
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 42
    
    # Lower bound: max(n*p, m*p, n*m) = max(14, 21, 6) = 21
    lower_bound = max(n * p, m * p, n * m)
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from lower_bound to schoolbook_rank
    for rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts for this rank
        max_restarts = 50
        if rank <= 25:
            max_restarts = 80
        elif rank <= 30:
            max_restarts = 60
        else:
            max_restarts = 40
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + rank * 1000 + restart)
            
            # Initialize with random values
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Gradient descent
            lr = 0.01
            for step in range(3000):
                if time.time() > deadline - 5:
                    break
                
                # Compute residual: R[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c] - T[i,j,k]
                # R = np.einsum('ic,jc,kc->ijk', U, V, W) - T
                R = np.einsum('ic,jc,kc->ijk', U, V, W) - T
                residual = np.sum(R * R)
                
                if residual < 1e-10:
                    found = True
                    break
                
                # Gradients
                # dR/dU[i,c] = sum_{j,k} V[j,c]*W[k,c]*R[i,j,k]
                # dR/dV[j,c] = sum_{i,k} U[i,c]*W[k,c]*R[i,j,k]
                # dR/dW[k,c] = sum_{i,j} U[i,c]*V[j,c]*R[i,j,k]
                
                dU = 2 * np.einsum('ijk,jc,kc->ic', R, V, W)
                dV = 2 * np.einsum('ijk,ic,kc->jc', R, U, W)
                dW = 2 * np.einsum('ijk,ic,jc->kc', R, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                if step % 500 == 499:
                    lr *= 0.95
            
            if found:
                best_rank = rank
                best_result = (U, V, W)
                break
        
        if found:
            break
    
    if best_result is not None:
        U, V, W = best_result
        # Convert to list of lists with integer/half-integer entries
        # Round to nearest half-integer
        U_list = [[round(U[i,c] * 2) / 2 for c in range(rank)] for i in range(nm)]
        V_list = [[round(V[i,c] * 2) / 2 for c in range(rank)] for i in range(mp)]
        W_list = [[round(W[i,c] * 2) / 2 for c in range(rank)] for i in range(pn)]
        
        # Verify the decomposition is exact with the rounded values
        U_arr = np.array(U_list)
        V_arr = np.array(V_list)
        W_arr = np.array(W_list)
        R_check = np.einsum('ic,jc,kc->ijk', U_arr, V_arr, W_arr) - T
        if np.max(np.abs(R_check)) > 1e-6:
            # Rounded version doesn't work, try without rounding but check if values are close to half-integers
            # If not, fall back to schoolbook
            return _schoolbook(n, m, p)
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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
