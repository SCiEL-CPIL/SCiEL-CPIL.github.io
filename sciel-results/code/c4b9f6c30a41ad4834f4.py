"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import linalg

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        rank = n * m * p  # 72
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
    
    best_result = schoolbook()
    best_rank = n * m * p  # 72

    # Verify a candidate decomposition using matrix multiplication
    def verify(U, V, W, rank):
        """
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        Check if sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k
        """
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Compute T[i, j, k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # P[i, j, k, c] = U[i, j, c] * V[j, k, c]
        P = np.einsum('ijc,jkc->ijkc', U_3d, V_3d)
        
        # T[i, j, k] = sum_c P[i, j, k, c] * W[k, i, c]
        T = np.einsum('ijkc,kic->ijk', P, W_3d)
        
        return np.allclose(T, 1.0, atol=1e-9)

    def try_rank(r, num_attempts=100):
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            
            # Randomize U and V
            # Use sparse random matrices with integer entries
            # Sparsity 0.5-0.7
            sparsity = np.random.uniform(0.5, 0.7)
            
            U = np.zeros((n * m, r), dtype=float)
            V = np.zeros((m * p, r), dtype=float)
            
            # Fill U
            mask_U = np.random.rand(n * m, r) < sparsity
            U[mask_U] = np.random.choice([-1, 1], size=mask_U.sum())
            
            # Fill V
            mask_V = np.random.rand(m * p, r) < sparsity
            V[mask_V] = np.random.choice([-1, 1], size=mask_V.sum())
            
            # Build A matrix
            num_eqs = n * m * p
            num_vars = p * n * r
            A = np.zeros((num_eqs, num_vars), dtype=float)
            
            U_3d = U.reshape(n, m, r)
            V_3d = V.reshape(m, p, r)
            P = np.einsum('ijc,jkc->ijkc', U_3d, V_3d)
            
            ii, jj, kk = np.indices((n, m, p))
            rows = ii * m * p + jj * p + kk
            cols = kk * n + ii # Note: col index in W is k*n + i
            
            for c in range(r):
                Pc = P[:, :, :, c]
                vals = Pc.flatten()
                mask = vals != 0
                if np.any(mask):
                    A_c = np.zeros((num_eqs, p * n), dtype=float)
                    A_c[rows[mask], cols[mask]] = vals[mask]
                    A[:, c*p*n : (c+1)*p*n] += A_c
            
            b = np.ones(num_eqs, dtype=float)
            
            try:
                w_flat, residuals, rank_A, s = linalg.lstsq(A, b, cond=None)
            except Exception:
                continue
            
            w_int = np.round(w_flat * 2)
            error = np.linalg.norm(w_flat * 2 - w_int)
            
            if error < 1e-5:
                W_flat = w_int / 2.0
                W = W_flat.reshape((p * n, r))
                
                if verify(U.tolist(), V.tolist(), W.tolist(), r):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Search for lower ranks
    for r in range(71, 3, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, num_attempts=100)
        
        if result is not None:
            best_rank = r
            best_result = result
            
    return best_result
# EVOLVE-BLOCK-END
