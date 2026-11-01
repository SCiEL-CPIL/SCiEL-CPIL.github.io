"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 110.0  # Safety margin
    
    # Schoolbook decomposition as fallback (rank 36)
    def schoolbook():
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

    def verify(U, V, W):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Check if sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    # Try to find a lower rank decomposition (Rank 35)
    # Strategy: Start with schoolbook (36 cols), drop the last col, and optimize the remaining 35 to compensate.
    target_rank = 35
    if time.time() < deadline - 5:
        # Get schoolbook factors
        sb = schoolbook()
        U_sb = np.array(sb['U'], dtype=float)
        V_sb = np.array(sb['V'], dtype=float)
        W_sb = np.array(sb['W'], dtype=float)
        
        # Initialize with first 35 columns
        U0 = U_sb[:, :target_rank].flatten()
        V0 = V_sb[:, :target_rank].flatten()
        W0 = W_sb[:, :target_rank].flatten()
        
        x0 = np.concatenate([U0, V0, W0])
        
        def objective(x):
            U_flat = x[:n*m*target_rank]
            V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
            W_flat = x[n*m*target_rank + m*p*target_rank:]
            
            U = U_flat.reshape((n*m, target_rank))
            V = V_flat.reshape((m*p, target_rank))
            W = W_flat.reshape((p*n, target_rank))
            
            residuals = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        residuals.append(val - 1.0)
                        
            return residuals

        try:
            # Use least_squares to find a solution
            res = least_squares(objective, x0, method='lm', max_nfev=1000)
            
            if res.cost < 1e-8:
                U_flat = res.x[:n*m*target_rank]
                V_flat = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                W_flat = res.x[n*m*target_rank + m*p*target_rank:]
                
                U = U_flat.reshape((n*m, target_rank))
                V = V_flat.reshape((m*p, target_rank))
                W = W_flat.reshape((p*n, target_rank))
                
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                W_rounded = np.round(W * 2) / 2.0
                
                if verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        except Exception:
            pass

    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
