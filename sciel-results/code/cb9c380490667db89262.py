"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit
    
    def schoolbook():
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}, rank
    
    def verify(U, V, W):
        """Check if the decomposition is valid."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def to_half_int(x):
        """Round to nearest half-integer."""
        return int(2 * x + 0.5) / 2.0 if x >= 0 else -int(2 * (-x) + 0.5) / 2.0
    
    def try_rank_search(max_rank, num_trials=50):
        """Try to find a valid decomposition of given rank."""
        n, m, p = DIMENSIONS
        nm, mp, pn = n*m, m*p, p*n
        
        for trial in range(num_trials):
            if time.time() > deadline - 20:
                break
            
            # Initialize with random values
            np.random.seed(trial * 12345 + 678)
            U = np.random.randn(nm, max_rank) * 0.5
            V = np.random.randn(mp, max_rank) * 0.5
            W = np.random.randn(pn, max_rank) * 0.5
            
            # Iteratively refine to satisfy constraints
            # We need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all (i,j,k)
            # Use gradient-like updates
            
            for iteration in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Compute current values
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(max_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            errors.append(s - 1)
                
                max_err = max(abs(e) for e in errors)
                if max_err < 1e-10:
                    # Round to half-integers and verify
                    U_h = np.round(U * 2) / 2
                    V_h = np.round(V * 2) / 2
                    W_h = np.round(W * 2) / 2
                    
                    # Verify exact
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for c in range(max_rank):
                                    s += U_h[i*m+j, c] * V_h[j*p+k, c] * W_h[k*n+i, c]
                                if abs(s - 1) > 1e-9:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid:
                        # Check entries are half-integers
                        if np.allclose(U_h * 2, np.round(U_h * 2)) and \
                           np.allclose(V_h * 2, np.round(V_h * 2)) and \
                           np.allclose(W_h * 2, np.round(W_h * 2)):
                            U_list = [[float(U_h[i*m+j, c]) for c in range(max_rank)] for i in range(n) for j in range(m)]
                            V_list = [[float(V_h[j*p+k, c]) for c in range(max_rank)] for j in range(m) for k in range(p)]
                            W_list = [[float(W_h[k*n+i, c]) for c in range(max_rank)] for k in range(p) for i in range(n)]
                            return {"U": U_list, "V": V_list, "W": W_list}, max_rank
                
                # Update U, V, W using simple gradient step
                # For each (i,j,k), the error is e = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1
                # Gradient w.r.t U[i,j,c] is V[j,k,c]*W[k,i,c]*e
                # We want to reduce e, so subtract gradient * lr
                
                lr = 0.01
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(max_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            e = s - 1
                            for c in range(max_rank):
                                grad_u = V[j*p+k, c] * W[k*n+i, c] * e
                                grad_v = U[i*m+j, c] * W[k*n+i, c] * e
                                grad_w = U[i*m+j, c] * V[j*p+k, c] * e
                                U[i*m+j, c] -= lr * grad_u
                                V[j*p+k, c] -= lr * grad_v
                                W[k*n+i, c] -= lr * grad_w
        
        return None, max_rank
    
    # Start with schoolbook
    best_result, best_rank = schoolbook()
    
    # Try to find lower rank decompositions
    for target_rank in range(44, 20, -1):
        if time.time() > deadline - 25:
            break
        
        result, rank = try_rank_search(target_rank, num_trials=30)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
