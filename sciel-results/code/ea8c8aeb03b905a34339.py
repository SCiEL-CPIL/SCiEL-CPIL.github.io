"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school
    
    # Optimize for lower ranks
    ranks_to_try = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
        if rank >= best_rank:
            continue
        
        found = False
        for trial in range(30):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(1000 + trial * 7)
            
            # Initialize
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            U_r = U.reshape(n, m, rank)
            V_r = V.reshape(m, p, rank)
            W_r = W.reshape(p, n, rank)
            
            lr = 0.05
            for step in range(3000):
                if time.time() > deadline - 10:
                    break
                
                # Compute T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
                # Vectorized: for each c, outer product contributions
                T = np.zeros((n, m, p))
                for c in range(rank):
                    # U_r[:,:,c] is (n,m), V_r[:,:,c] is (m,p), W_r[:,:,c] is (p,n)
                    # Contribution: U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
                    # This is a 3-way outer product contracted
                    T += np.einsum('ij,jk,ki->ijk', U_r[:,:,c], V_r[:,:,c], W_r[:,:,c])
                
                error = T - 1.0
                err_norm = np.sum(error**2)
                
                if err_norm < 1e-12:
                    found = True
                    break
                
                # Gradients
                dU = np.zeros_like(U_r)
                dV = np.zeros_like(V_r)
                dW = np.zeros_like(W_r)
                
                for c in range(rank):
                    # dU[i,j,c] = sum_k error[i,j,k] * V_r[j,k,c] * W_r[k,i,c]
                    dU[:,:,c] = np.einsum('ijk,jk,ki->ij', error, V_r[:,:,c], W_r[:,:,c])
                    # dV[j,k,c] = sum_i error[i,j,k] * U_r[i,j,c] * W_r[k,i,c]
                    dV[:,:,c] = np.einsum('ijk,ij,ki->jk', error, U_r[:,:,c], W_r[:,:,c])
                    # dW[k,i,c] = sum_j error[i,j,k] * U_r[i,j,c] * V_r[j,k,c]
                    dW[:,:,c] = np.einsum('ijk,ij,jk->ki', error, U_r[:,:,c], V_r[:,:,c])
                
                U -= lr * dU.reshape(n*m, rank)
                V -= lr * dV.reshape(m*p, rank)
                W -= lr * dW.reshape(p*n, rank)
                
                U_r = U.reshape(n, m, rank)
                V_r = V.reshape(m, p, rank)
                W_r = W.reshape(p, n, rank)
                
                if step % 200 == 0:
                    lr *= 0.9
                    
                if err_norm < 1e-6:
                    # Try rounding
                    for scale in [1, 2, 4, 8, 16, 32]:
                        U_rnd = np.round(U * scale) / scale
                        V_rnd = np.round(V * scale) / scale
                        W_rnd = np.round(W * scale) / scale
                        
                        T_check = np.zeros((n, m, p))
                        for c in range(rank):
                            T_check += np.einsum('ij,jk,ki->ijk', 
                                U_rnd.reshape(n,m,rank)[:,:,c], 
                                V_rnd.reshape(m,p,rank)[:,:,c], 
                                W_rnd.reshape(p,n,rank)[:,:,c])
                        
                        if np.allclose(T_check, 1.0, atol=1e-10):
                            best_result = {
                                "U": U_rnd.tolist(),
                                "V": V_rnd.tolist(),
                                "W": W_rnd.tolist()
                            }
                            best_rank = rank
                            found = True
                            break
                    
                    if found:
                        break
            
            if found:
                break
    
    return best_result
# EVOLVE-BLOCK-END
