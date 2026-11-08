import numpy as np
import random
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb, "rank": rank_sb}
    
    # Try to find lower rank
    # R must be at least max(n*m, m*p, p*n) / something?
    # Lower bound on rank is max(n*m, m*p, p*n) = 20.
    # But for matrix mult, it's higher.
    # Let's try from 12 to 55
    
    for R in range(12, 56):
        if time.time() > deadline:
            break
            
        # Try a few random configurations
        trials = 10
        for t in range(trials):
            if time.time() > deadline:
                break
                
            # Generate random U, W with small integer entries
            # U shape (n*m, R), W shape (p*n, R)
            U = np.zeros((n * m, R), dtype=int)
            W = np.zeros((p * n, R), dtype=int)
            
            # Fill with random -1, 0, 1
            for i in range(n * m):
                for r in range(R):
                    U[i, r] = random.randint(-1, 1)
            for i in range(p * n):
                for r in range(R):
                    W[i, r] = random.randint(-1, 1)
            
            # Solve for V
            # V shape (m*p, R)
            # Equations: for each k, j2, we have a system for V[j2*p + k, :]
            # System: sum_r U[i, j1, r] * W[k, i, r] * V[j2, k, r] = delta(j1, j2)
            
            V = np.zeros((m * p, R), dtype=float)
            valid = True
            
            for k in range(p):
                for j2 in range(m):
                    # Unknowns: V[j2*p + k, 0..R-1]
                    # Equations: for i in 0..n-1, j1 in 0..m-1
                    # A * x = b
                    A = np.zeros((n * m, R))
                    b = np.zeros(n * m)
                    
                    idx = 0
                    for i in range(n):
                        for j1 in range(m):
                            # Coefficients for V[j2, k, r]
                            for r in range(R):
                                # U index: i*m + j1
                                # W index: k*n + i
                                A[idx, r] = U[i * m + j1, r] * W[k * n + i, r]
                            # RHS
                            b[idx] = 1 if j1 == j2 else 0
                            idx += 1
                    
                    # Solve A x = b
                    # Use least squares if overdetermined
                    try:
                        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                        # Check if residual is small
                        if residuals and residuals[0] > 1e-6:
                            valid = False
                            break
                        V[j2 * p + k, :] = x
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Check if V can be scaled to integers
                # Find LCM of denominators
                # If V is rational, we can scale
                # For simplicity, check if V * 2 is close to integer
                # Or just scale by a large number and round
                # But we need EXACT decomposition.
                # If lstsq gives a solution, it's exact for the linear system.
                # We need to verify the full tensor condition.
                
                # Verify
                U_int = U
                W_int = W
                # Scale V to integers
                # Try scaling factors
                found_int = False
                for scale in [1, 2, 10, 100]:
                    V_scaled = V * scale
                    if np.allclose(V_scaled, np.round(V_scaled), atol=1e-9):
                        V_int = np.round(V_scaled).astype(int)
                        # Verify full decomposition
                        ok = True
                        for i in range(n):
                            for j1 in range(m):
                                for k in range(p):
                                    for j2 in range(m):
                                        s = 0
                                        for r in range(R):
                                            s += U_int[i * m + j1, r] * V_int[j2 * p + k, r] * W_int[k * n + i, r]
                                        expected = 1 if j1 == j2 else 0
                                        if s != expected:
                                            ok = False
                                            break
                                    if not ok: break
                                if not ok: break
                            if not ok: break
                        if ok:
                            # Convert to lists
                            U_list = U_int.tolist()
                            V_list = V_int.tolist()
                            W_list = W_int.tolist()
                            return {"U": U_list, "V": V_list, "W": W_list}
                        found_int = True
                        break
                
    return best_result
# EVOLVE-BLOCK-END
