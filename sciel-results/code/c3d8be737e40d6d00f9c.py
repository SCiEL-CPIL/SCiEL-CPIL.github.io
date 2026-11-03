"""
Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses a local search approach to find a low-rank decomposition.
"""

import time
import random
import copy

DIMENSIONS = [4, 4, 5]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Compute U * V^T * W^T and check against the tensor
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # Tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # For exact verification, check that the bilinear form matches
    # The matrix multiplication tensor: (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So T[i,j,k] = 1 if this term contributes, but actually the tensor is defined as
    # T[i,j,k] such that C_{i,k} = sum_{j} T[i,j,k] * A_{i,j} * B_{j,k}
    # Wait, the standard tensor decomposition for matrix multiplication:
    # C = A*B, C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T has entries T[i,j,k] = 1 for all i,j,k (since every term contributes)
    # No wait. The matrix multiplication tensor is defined such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # because C_{i,k} = sum_j (A_{i,j} * B_{j,k}) and each (i,j,k) triple has coefficient 1.
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-6:
                    return False, s
    return True, 1.0

def _count_errors(U, V, W, n, m, p, tol=1e-6):
    """Count the number of (i,j,k) triples that don't satisfy the equation."""
    rank = len(U[0])
    errors = 0
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                err = abs(s - 1.0)
                if err > tol:
                    errors += 1
                if err > max_err:
                    max_err = err
    return errors, max_err

def _random_decomposition(n, m, p, rank, seed=None):
    """Generate a random initial decomposition."""
    if seed is not None:
        random.seed(seed)
    U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n*m)]
    V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m*p)]
    W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p*n)]
    return U, V, W

def _local_search(U, V, W, n, m, p, max_iters=5000, tol=1e-9, time_limit=30):
    """Local search to refine a decomposition."""
    rank = len(U[0])
    start_time = time.time()
    
    U = copy.deepcopy(U)
    V = copy.deepcopy(V)
    W = copy.deepcopy(W)
    
    best_errors, _ = _count_errors(U, V, W, n, m, p)
    best_U, best_V, best_W = copy.deepcopy(U), copy.deepcopy(V), copy.deepcopy(W)
    
    step_size = 0.1
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if best_errors == 0:
            break
        
        # Try a random perturbation
        matrix_choice = random.choice(['U', 'V', 'W'])
        if matrix_choice == 'U':
            row = random.randint(0, n*m-1)
            col = random.randint(0, rank-1)
            delta = random.uniform(-step_size, step_size)
            U[row][col] += delta
            new_errors, _ = _count_errors(U, V, W, n, m, p)
            if new_errors < best_errors:
                best_errors = new_errors
                best_U, best_V, best_W = copy.deepcopy(U), copy.deepcopy(V), copy.deepcopy(W)
            else:
                U[row][col] -= delta
        elif matrix_choice == 'V':
            row = random.randint(0, m*p-1)
            col = random.randint(0, rank-1)
            delta = random.uniform(-step_size, step_size)
            V[row][col] += delta
            new_errors, _ = _count_errors(U, V, W, n, m, p)
            if new_errors < best_errors:
                best_errors = new_errors
                best_U, best_V, best_W = copy.deepcopy(U), copy.deepcopy(V), copy.deepcopy(W)
            else:
                V[row][col] -= delta
        else:
            row = random.randint(0, p*n-1)
            col = random.randint(0, rank-1)
            delta = random.uniform(-step_size, step_size)
            W[row][col] += delta
            new_errors, _ = _count_errors(U, V, W, n, m, p)
            if new_errors < best_errors:
                best_errors = new_errors
                best_U, best_V, best_W = copy.deepcopy(U), copy.deepcopy(V), copy.deepcopy(W)
            else:
                W[row][col] -= delta
        
        # Gradually reduce step size
        if it % 500 == 0 and it > 0:
            step_size *= 0.9
    
    return best_U, best_V, best_W, best_errors

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # The schoolbook rank is n*m*p = 80
    # We want to find a lower rank.
    # Known bounds: for 4x4*4x5, the minimum rank is not known exactly,
    # but we can try ranks below 80.
    
    # Start with schoolbook as a fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Use local search from random initializations for various ranks
    
    for target_rank in range(60, 80):  # Try ranks from 60 to 79
        if time.time() - start_time > deadline - 30:  # Leave 30s margin
            break
        
        found = False
        max_attempts = 10
        
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline - 15:
                break
            
            U, V, W = _random_decomposition(n, m, p, target_rank, seed=42 + attempt * 1000 + target_rank)
            
            # Run local search
            U, V, W, errors = _local_search(U, V, W, n, m, p, max_iters=3000, time_limit=10)
            
            if errors == 0:
                # Verify exactly
                valid, _ = _verify(U, V, W, n, m, p)
                if valid:
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Try even lower ranks
            continue
        else:
            # If we can't find rank target_rank, try lower? No, break.
            # Actually, let's try a few more ranks
            pass
    
    return best_result
