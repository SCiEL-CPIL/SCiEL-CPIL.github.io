"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = 2, 4, 6
    
    # Try to find rank-7 decomposition of 2x2x3 tensor
    # T in V(4) x W(6) x Z(6), T[i,j,k]=1 for all valid i,j,k
    T_flat = np.zeros(4*6*6)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                idx = (i*2 + j) * 36 + (j*3 + k) * 6 + (i*3 + k)
                T_flat[idx] = 1.0
    
    rank_sub = 7
    np.random.seed(42)
    
    best_loss = float('inf')
    best_params = None
    
    # Try multiple random initializations with gradient descent
    for trial in range(50):
        if time.time() > deadline:
            break
        V_init = np.random.randn(rank_sub, 4) * 0.3
        W_init = np.random.randn(rank_sub, 6) * 0.3
        Z_init = np.random.randn(rank_sub, 6) * 0.3
        params = np.concatenate([V_init.flatten(), W_init.flatten(), Z_init.flatten()])
        
        lr = 0.005
        for step in range(500):
            if time.time() > deadline:
                break
            V_ = params[0:rank_sub*4].reshape(rank_sub, 4)
            W_ = params[rank_sub*4:rank_sub*4+rank_sub*6].reshape(rank_sub, 6)
            Z_ = params[rank_sub*4+rank_sub*6:].reshape(rank_sub, 6)
            
            T_comp = np.zeros(4*6*6)
            for r in range(rank_sub):
                T_comp += np.einsum('a,b,c->abc', V_[r], W_[r], Z_[r]).flatten()
            
            residual = T_comp - T_flat
            loss = float(np.sum(residual**2))
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
            
            if loss < 1e-8:
                break
            
            # Numerical gradient (expensive but simple)
            grad = np.zeros_like(params)
            eps = 1e-6
            for idx in range(len(params)):
                if time.time() > deadline:
                    break
                old = params[idx]
                params[idx] = old + eps
                V_2 = params[0:rank_sub*4].reshape(rank_sub, 4)
                W_2 = params[rank_sub*4:rank_sub*4+rank_sub*6].reshape(rank_sub, 6)
                Z_2 = params[rank_sub*4+rank_sub*6:].reshape(rank_sub, 6)
                T_comp_2 = np.zeros(4*6*6)
                for r in range(rank_sub):
                    T_comp_2 += np.einsum('a,b,c->abc', V_2[r], W_2[r], Z_2[r]).flatten()
                loss_2 = float(np.sum((T_comp_2 - T_flat)**2))
                grad[idx] = (loss_2 - loss) / eps
                params[idx] = old
            
            params -= lr * grad
        
        if best_loss < 1e-6:
            break
    
    if best_params is not None and best_loss < 1e-6:
        V_ = best_params[0:rank_sub*4].reshape(rank_sub, 4)
        W_ = best_params[rank_sub*4:rank_sub*4+rank_sub*6].reshape(rank_sub, 6)
        Z_ = best_params[rank_sub*4+rank_sub*6:].reshape(rank_sub, 6)
        
        # Round to nearest half-integers
        V_r = np.round(V_ * 2) / 2
        W_r = np.round(W_ * 2) / 2
        Z_r = np.round(Z_ * 2) / 2
        
        # Verify exactness
        T_comp = np.zeros(4*6*6)
        for r in range(rank_sub):
            T_comp += np.einsum('a,b,c->abc', V_r[r], W_r[r], Z_r[r]).flatten()
        
        if np.allclose(T_comp, T_flat, atol=1e-9):
            # Build 2x4x6 decomposition using two 2x2x6 subproblems
            # Each 2x2x6 uses two copies of the 2x2x3 decomposition
            total_rank = 2 * rank_sub  # 14
            
            # U: 2x4 = 8 rows, total_rank cols
            # V: 4x6 = 24 rows, total_rank cols  
            # W: 2x6 = 12 rows, total_rank cols (using W[k*n+i, r] convention)
            
            U_full = [[0.0] * total_rank for _ in range(8)]
            V_full = [[0.0] * total_rank for _ in range(24)]
            W_full = [[0.0] * total_rank for _ in range(12)]
            
            # Left subproblem: A[:,0:2] (2x2), B[0:2,:] (2x6)
            # Right subproblem: A[:,2:4] (2x2), B[2:4,:] (2x6)
            
            for sub in range(2):  # 0=left, 1=right
                j_offset = sub * 2
                # For k in 0..2: use first copy of 2x2x3
                # For k in 3..5: use second copy of 2x2x3
                for k_block in range(2):  # 0: k=0,1,2; 1: k=3,4,5
                    k_offset = k_block * 3
                    for r in range(rank_sub):
                        r_idx = sub * rank_sub + k_block * rank_sub + r
                        # U: rows i*4+j for i in 0,1; j in 0,1
                        for i in range(2):
                            for j in range(2):
                                U_full[i*4 + j_offset + j][r_idx] = float(V_r[r][i*2 + j])
                        # V: rows j*6+k for j in 0,1; k in 0..5
                        for j in range(2):
                            for k in range(3):
                                V_full[j_offset + j]*6 + (k_offset + k) if False else 0
                                V_full[(j_offset + j)*6 + k_offset + k][r_idx] = float(W_r[r][j*3 + k])
                        # W: rows k*n+i for k in 0..5; i in 0,1
                        for i in range(2):
                            for k in range(3):
                                W_full[(k_offset + k)*2 + i][r_idx] = float(Z_r[r][i*3 + k])
            
            # Convert to lists of lists with proper types
            U_out = [[float(x) for x in row] for row in U_full]
            V_out = [[float(x) for x in row] for row in V_full]
            W_out = [[float(x) for x in row] for row in W_full]
            
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: schoolbook decomposition
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
