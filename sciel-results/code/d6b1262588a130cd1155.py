"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s budget with 20s margin
    
    # Build target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Trivial solution: rank = n*m*p
    rank_trivial = n*m*p
    U = np.zeros((n*m, rank_trivial))
    V = np.zeros((m*p, rank_trivial))
    W = np.zeros((p*n, rank_trivial))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
    
    best_rank = rank_trivial
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank solutions using ALS
    for target_rank in range(20, rank_trivial):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline:
                break
            
            U_c = np.random.randn(n*m, target_rank) * 0.1
            V_c = np.random.randn(m*p, target_rank) * 0.1
            W_c = np.random.randn(p*n, target_rank) * 0.1
            
            for it in range(30):
                if time.time() > deadline:
                    break
                
                # Solve for W: UV_kr @ W[:,gamma] = T[:,:,gamma].ravel()
                UV_kr = np.einsum('ia,jb->ijab', U_c, V_c).reshape(n*m*m*p, target_rank)
                for gamma in range(p*n):
                    rhs = T[:, :, gamma].ravel()
                    W_c[:, gamma] = np.linalg.lstsq(UV_kr, rhs, rcond=None)[0]
                
                # Solve for V: UW_kr @ V[:,beta] = T[:,beta,:].ravel()
                UW_kr = np.einsum('ia,ka->ik a', U_c, W_c).reshape(n*m*p*n, target_rank)
                for beta in range(m*p):
                    rhs = T[:, beta, :].ravel()
                    V_c[:, beta] = np.linalg.lstsq(UW_kr, rhs, rcond=None)[0]
                
                # Solve for U: VW_kr @ U[:,alpha] = T[alpha,:,:].ravel()
                VW_kr = np.einsum('jb,ka->jkab', V_c, W_c).reshape(m*p*p*n, target_rank)
                for alpha in range(n*m):
                    rhs = T[alpha, :, :].ravel()
                    U_c[:, alpha] = np.linalg.lstsq(VW_kr, rhs, rcond=None)[0]
                
                # Check error
                recon = np.einsum('ia,ja,ka->ijk', U_c, V_c, W_c)
                error = np.sum((recon - T)**2)
                
                if error < 1e-10:
                    found = True
                    best_rank = target_rank
                    best_U, best_V, best_W = U_c, V_c, W_c
                    break
                elif error > 1e2:
                    break
            
            if found:
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
