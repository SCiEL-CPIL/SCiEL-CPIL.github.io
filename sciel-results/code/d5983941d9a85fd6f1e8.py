"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from low to high
    for rank in range(15, n * m * p + 1):
        if time.time() > deadline - 10:
            break
        
        n_trials = 20
        found = False
        
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(12345 + trial * 7)
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize columns
            for c in range(rank):
                nu = np.linalg.norm(U[:,c])
                nv = np.linalg.norm(V[:,c])
                nw = np.linalg.norm(W[:,c])
                total = nu * nv * nw
                if total > 1e-10:
                    U[:,c] /= total**(1/3)
                    V[:,c] /= total**(1/3)
                    W[:,c] /= total**(1/3)
            
            for iteration in range(300):
                if time.time() > deadline - 10:
                    break
                
                # Compute tensor product sum: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Vectorize: for each c, outer product
                T = np.zeros((n, m, p))
                for c in range(rank):
                    # U_c is (n*m,), V_c is (m*p,), W_c is (p*n,)
                    # Reshape: U_c -> (n,m), V_c -> (m,p), W_c -> (p,n)
                    Uc = U[:,c].reshape(n, m)
                    Vc = V[:,c].reshape(m, p)
                    Wc = W[:,c].reshape(p, n)
                    # T += Uc ⊗ Vc ⊗ Wc where (Uc⊗Vc⊗Wc)[i,j,k] = Uc[i,j]*Vc[j,k]*Wc[k,i]
                    # This is: for each i,j,k: Uc[i,j] * Vc[j,k] * Wc[k,i]
                    # = np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
                    T += np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
                
                errors = T - 1.0
                max_err = np.max(np.abs(errors))
                if max_err < 1e-9:
                    found = True
                    break
                
                # Update using gradient descent
                lr = 0.005
                for c in range(rank):
                    Uc = U[:,c].reshape(n, m)
                    Vc = V[:,c].reshape(m, p)
                    Wc = W[:,c].reshape(p, n)
                    
                    # dU/dc = errors contracted with Vc, Wc
                    # dU[i,j] = sum_k errors[i,j,k] * Vc[j,k] * Wc[k,i]
                    dUc = np.einsum('ijk,jk,ki->ij', errors, Vc, Wc)
                    dVc = np.einsum('ijk,ij,ki->jk', errors, Uc, Wc)
                    dWc = np.einsum('ijk,ij,jk->ki', errors, Uc, Vc)
                    
                    U[:,c] -= lr * dUc.reshape(-1)
                    V[:,c] -= lr * dVc.reshape(-1)
                    W[:,c] -= lr * dWc.reshape(-1)
            
            if found:
                # Verify
                T_check = np.zeros((n, m, p))
                for c in range(rank):
                    Uc = U[:,c].reshape(n, m)
                    Vc = V[:,c].reshape(m, p)
                    Wc = W[:,c].reshape(p, n)
                    T_check += np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
                if np.max(np.abs(T_check - 1.0)) < 1e-8:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
        
        if found:
            break
    
    if best_U is not None:
        # Round to half-integers
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        
        # Verify
        T_check = np.zeros((n, m, p))
        for c in range(rank):
            Uc = U_r[:,c].reshape(n, m)
            Vc = V_r[:,c].reshape(m, p)
            Wc = W_r[:,c].reshape(p, n)
            T_check += np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
        
        if np.max(np.abs(T_check - 1.0)) < 1e-6:
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
