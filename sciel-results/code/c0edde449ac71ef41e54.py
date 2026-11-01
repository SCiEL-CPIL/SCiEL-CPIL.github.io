import time
import numpy as np
import itertools

DIMENSIONS = [2, 3, 3]

def verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Check if U, V, W define the tensor
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # T[i, j, k] should be 1 for all i in [0, n), j in [0, m), k in [0, p).
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def tensor_residual(U, V, W):
    """Compute the maximum absolute residual of the tensor decomposition."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                res = abs(s - 1.0)
                if res > max_res:
                    max_res = res
    return max_res

def round_to_half(x):
    """Round to nearest half-integer."""
    return round(2 * x) / 2.0

def polish(U, V, W, max_iters=100, tol=1e-12):
    """Try to refine the decomposition to exact values using Newton's method."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Convert to numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for _ in range(max_iters):
        max_res = tensor_residual(U_np.tolist(), V_np.tolist(), W_np.tolist())
        if max_res < tol:
            break
        
        # Compute Jacobian and use Gauss-Newton to solve for corrections
        # This is a complex system; instead, let's try a simpler approach:
        # For each term, compute the residual and adjust one factor at a time
        
        # Actually, let's just try rounding to half-integers and checking
        U_rounded = [[round_to_half(x) for x in row] for row in U_np.tolist()]
        V_rounded = [[round_to_half(x) for x in row] for row in V_np.tolist()]
        W_rounded = [[round_to_half(x) for x in row] for row in W_np.tolist()]
        
        if verify(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        
        # If rounding doesn't work, try small perturbations
        improved = False
        for factor_name in ['U', 'V', 'W']:
            if factor_name == 'U':
                for idx in range(len(U_np)):
                    for c in range(rank):
                        for delta in [-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5]:
                            old = U_np[idx][c]
                            U_np[idx][c] = old + delta
                            if tensor_residual(U_np.tolist(), V_np.tolist(), W_np.tolist()) < max_res:
                                max_res = tensor_residual(U_np.tolist(), V_np.tolist(), W_np.tolist())
                                improved = True
                            else:
                                U_np[idx][c] = old
                        # Try rounding this element
                        U_np[idx][c] = round_to_half(old)
                        if tensor_residual(U_np.tolist(), V_np.tolist(), W_np.tolist()) < max_res:
                            max_res = tensor_residual(U_np.tolist(), V_np.tolist(), W_np.tolist())
                            improved = True
                        else:
                            U_np[idx][c] = old
            elif factor_name == 'V':
                for idx in range(len(V_np)):
                    for c in range(rank):
                        for delta in [-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5]:
                            old = V_np[idx][c]
                            V_np[idx][c] = old + delta
                            if tensor_residual(U_np.tolist(), V_np.tolist(), W_np.tolist()) < max_res:
                                max_res = tensor_residual(U_np.tolist(), V_np.tolist(), W_np.tolist())
                                improved = True
                            else:
                                V_np[idx][c] = old
                        V_np[idx][c] = round_to_half(old)
                        if tensor_residual(U_np.tolist(), V_np.tolist(), W_np.tolist()) < max_res:
                            max_res = tensor_residual(U_np.tolist(), V_np.tolist(), W_np.tolist())
                            improved = True
                        else:
                            V_np[idx][c] = old
            else:  # W
                for idx in range(len(W_np)):
                    for c in range(rank):
                        for delta in [-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5]:
                            old = W_np[idx][c]
                            W_np[idx][c] = old + delta
                            if tensor_residual(U_np.tolist(), V_np.tolist(), W_np.tolist()) < max_res:
                                max_res = tensor_residual(U_np.tolist(), V_np.tolist(), W_np.tolist())
                                improved = True
                            else:
                                W_np[idx][c] = old
                        W_np[idx][c] = round_to_half(old)
                        if tensor_residual(U_np.tolist(), V_np.tolist(), W_np.tolist()) < max_res:
                            max_res = tensor_residual(U_np.tolist(), V_np.tolist(), W_np.tolist())
                            improved = True
                        else:
                            W_np[idx][c] = old
        
        if not improved:
            break
    
    U_final = U_np.tolist()
    V_final = V_np.tolist()
    W_final = W_np.tolist()
    
    # Final rounding attempt
    U_rounded = [[round_to_half(x) for x in row] for row in U_final]
    V_rounded = [[round_to_half(x) for x in row] for row in V_final]
    W_rounded = [[round_to_half(x) for x in row] for row in W_final]
    
    if verify(U_rounded, V_rounded, W_rounded):
        return U_rounded, V_rounded, W_rounded
    
    return U_final, V_final, W_final

def als_step(U, V, W, fix_factor, lr=0.1, max_iters=50):
    """Perform ALS updates on one factor while keeping others fixed."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for _ in range(max_iters):
        if fix_factor == 'U':
            for i in range(n):
                for j in range(m):
                    # Target: T[i,j,k] = 1 for all k
                    # U[i*m+j, c] * sum_k V[j*p+k, c] * W[k*n+i, c] = 1
                    # Let's compute the contribution from each c
                    for c in range(rank):
                        # Compute the "response" from V and W for this (i,j,c)
                        resp = 0.0
                        for k in range(p):
                            resp += V_np[j * p + k][c] * W_np[k * n + i][c]
                        if abs(resp) > 1e-10:
                            U_np[i * m + j][c] = 1.0 / resp
                        else:
                            U_np[i * m + j][c] = 0.0
        elif fix_factor == 'V':
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        resp = 0.0
                        for i in range(n):
                            resp += U_np[i * m + j][c] * W_np[k * n + i][c]
                        if abs(resp) > 1e-10:
                            V_np[j * p + k][c] = 1.0 / resp
                        else:
                            V_np[j * p + k][c] = 0.0
        else:  # W
            for k in range(p):
                for i in range(n):
                    for c in range(rank):
                        resp = 0.0
                        for j in range(m):
                            resp += U_np[i * m + j][c] * V_np[j * p + k][c]
                        if abs(resp) > 1e-10:
                            W_np[k * n + i][c] = 1.0 / resp
                        else:
                            W_np[k * n + i][c] = 0.0
    
    return U_np.tolist(), V_np.tolist(), W_np.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Start with schoolbook rank 18
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: ceil(n*m*p / max(n,m,p)) = ceil(18/3) = 6
    # But for matrix multiplication, the rank is at least max(n*m, m*p, p*n) = max(6, 9, 6) = 9
    # Actually, the rank is at least the max of the dimensions of the factor spaces
    # For 2x3 * 3x3, the rank should be at least 9 (m*p)
    
    # Try ranks from 9 up to 17
    for target_rank in range(9, 18):
        if time.time() > deadline - 30:  # Leave 30s for polish
            break
        
        # Multiple random restarts
        n_restarts = 50
        for restart in range(n_restarts):
            if time.time() > deadline - 30:
                break
            
            # Random initialization with small values
            U_init = np.random.uniform(-1, 1, (n * m, target_rank))
            V_init = np.random.uniform(-1, 1, (m * p, target_rank))
            W_init = np.random.uniform(-1, 1, (p * n, target_rank))
            
            # Normalize to avoid numerical issues
            U_init /= (np.linalg.norm(U_init) + 1e-10)
            V_init /= (np.linalg.norm(V_init) + 1e-10)
            W_init /= (np.linalg.norm(W_init) + 1e-10)
            
            # Run ALS for a few iterations
            U_cur, V_cur, W_cur = U_init.tolist(), V_init.tolist(), W_init.tolist()
            
            for als_iter in range(20):
                if time.time() > deadline - 30:
                    break
                U_cur, V_cur, W_cur = als_step(U_cur, V_cur, W_cur, 'U', max_iters=5)
                U_cur, V_cur, W_cur = als_step(U_cur, V_cur, W_cur, 'V', max_iters=5)
                U_cur, V_cur, W_cur = als_step(U_cur, V_cur, W_cur, 'W', max_iters=5)
                
                # Check residual
                res = tensor_residual(U_cur, V_cur, W_cur)
                if res < 1e-6:
                    # Try to polish to exact
                    U_polished, V_polished, W_polished = polish(U_cur, V_cur, W_cur)
                    if verify(U_polished, V_polished, W_polished):
                        best_result = {"U": U_polished, "V": V_polished, "W": W_polished}
                        best_rank = target_rank
                        break
                if time.time() > deadline - 30:
                    break
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
    
    # Return the best result found
    return best_result
