"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # For (2,3,9), decompose into three (2,3,3) subproblems
    # Try to find a low-rank decomposition for (2,3,3)
    
    def verify_233(U, V, W, rank):
        """Verify that U,V,W form a valid decomposition for (2,3,3)."""
        n2, m2, p2 = 2, 3, 3
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    val = np.sum(U[i*m2+j, :] * V[j*p2+k, :] * W[k*n2+i, :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    def try_rank_r(r):
        """Try to find a rank-r decomposition for (2,3,3) using random initialization and refinement."""
        n2, m2, p2 = 2, 3, 3
        np.random.seed(0)
        
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n2*m2, r) * 0.5
            V = np.random.randn(m2*p2, r) * 0.5
            W = np.random.randn(p2*n2, r) * 0.5
            
            # Gradient descent
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                # Compute gradient of error
                err = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            # Target: sum_r U[i*m2+j,r]*V[j*p2+k,r]*W[k*n2+i,r] = 1
                            prod = U[i*m2+j, :] * V[j*p2+k, :] * W[k*n2+i, :]
                            val = np.sum(prod)
                            err += (val - 1.0)**2
                            # Gradient w.r.t. U[i*m2+j, r]
                            grad_U[i*m2+j, :] += 2*(val-1.0) * V[j*p2+k, :] * W[k*n2+i, :]
                            grad_V[j*p2+k, :] += 2*(val-1.0) * U[i*m2+j, :] * W[k*n2+i, :]
                            grad_W[k*n2+i, :] += 2*(val-1.0) * U[i*m2+j, :] * V[j*p2+k, :]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if err < 1e-10:
                    break
            
            if verify_233(U, V, W, r):
                return U, V, W
        
        return None
    
    # Try ranks from 9 upward
    best_decomp = None
    for r in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
        if time.time() > deadline:
            break
        result = try_rank_r(r)
        if result is not None:
            best_decomp = result
            best_rank = r
            break
    
    if best_decomp is not None:
        U233, V233, W233 = best_decomp
        # Replicate for (2,3,9): three independent (2,3,3) blocks
        total_rank = best_rank * 3
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        for block in range(3):
            # Block b uses columns [b*best_rank, (b+1)*best_rank)
            # U: rows i*m+j for i in 0..1, j in 0..2 -> same for all blocks
            # V: rows j*p+k for j in 0..2, k in 0..8
            #   Block 0: k=0,1,2 -> V indices j*3+0, j*3+1, j*3+2
            #   Block 1: k=3,4,5 -> V indices j*3+3, j*3+4, j*3+5
            #   Block 2: k=6,7,8 -> V indices j*3+6, j*3+7, j*3+8
            # W: rows k*n+i for k in 0..8, i in 0..1
            #   Block 0: k=0,1,2 -> W indices 0*2+i, 1*2+i, 2*2+i = i, 2+i, 4+i
            #   Block 1: k=3,4,5 -> W indices 3*2+i, 4*2+i, 5*2+i = 6+i, 8+i, 10+i
            #   Block 2: k=6,7,8 -> W indices 6*2+i, 7*2+i, 8*2+i = 12+i, 14+i, 16+i
            
            for r_idx in range(best_rank):
                col = block * best_rank + r_idx
                # U: same for all blocks
                U[:, col] = U233[:, r_idx]
                # V: map j*3+k_block to j*p2+k for p2=3
                for j in range(3):
                    for k_sub in range(3):
                        k_global = block * 3 + k_sub
                        V[j * p + k_global, col] = V233[j * 3 + k_sub, r_idx]
                # W: map k_sub*n2+i to k_global*n+i
                for k_sub in range(3):
                    for i in range(2):
                        k_global = block * 3 + k_sub
                        W[k_global * n + i, col] = W233[k_sub * 2 + i, r_idx]
        
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(total_rank):
                        val += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if abs(val - 1.0) > 1e-6:
                        # Fallback to schoolbook
                        break
                else:
                    continue
                break
            else:
                continue
            break
        else:
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
