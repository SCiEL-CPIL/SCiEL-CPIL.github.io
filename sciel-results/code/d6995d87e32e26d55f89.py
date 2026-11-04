import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

def _check_decomposition(U, V, W, n, m, p, rank, tol=1e-8):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    return np.allclose(T_decomp, T, atol=tol)

def _solve_subproblem(n, m, p, target_rank, max_time=20.0):
    """
    Try to find a decomposition of rank target_rank for n x m x p multiplication.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    values = [-1, -0.5, 0, 0.5, 1]
    
    def compute_error(U, V, W):
        T_decomp = np.zeros((n*m, m*p, p*n))
        for r in range(target_rank):
            T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.linalg.norm(T_decomp - T)
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        U = np.random.choice(values, size=(n*m, target_rank))
        V = np.random.choice(values, size=(m*p, target_rank))
        W = np.random.choice(values, size=(p*n, target_rank))
        
        err = compute_error(U, V, W)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-8:
            return U, V, W
            
        # Local search
        for it in range(500):
            if time.time() > deadline:
                break
                
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                arr = U
                rows, cols = n*m, target_rank
            elif mat_choice == 'V':
                arr = V
                rows, cols = m*p, target_rank
            else:
                arr = W
                rows, cols = p*n, target_rank
                
            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)
            
            old_val = arr[row, col]
            new_val = random.choice(values)
            
            if new_val == old_val:
                continue
                
            arr[row, col] = new_val
            new_err = compute_error(U, V, W)
            
            if new_err < err:
                err = new_err
                if err < 1e-8:
                    return U, V, W
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                arr[row, col] = old_val
                
    # If we didn't find exact, check if best is close enough to snap?
    # For now, only return exact solutions.
    if best_err < 1e-8:
        return best_U, best_V, best_W
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # p=5, split into p1=3 and p2=2
    
    p1 = 3
    p2 = 2
    
    # Solve 3x3x3
    res_333 = None
    rank_333 = 27 # Schoolbook
    # Try to find lower rank for 3x3x3
    # Known best is 23. Try 23, 24, 25, 26, 27
    for r in range(23, 28):
        if time.time() > deadline - 20: # Leave time for 332 and combination
            break
        res_333 = _solve_subproblem(3, 3, p1, r, max_time=10.0)
        if res_333:
            rank_333 = r
            break
            
    if not res_333:
        # Fall back to schoolbook
        U_333 = [[0]*rank_333 for _ in range(9)]
        V_333 = [[0]*rank_333 for _ in range(9)]
        W_333 = [[0]*rank_333 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(p1):
                    U_333[i*3+j][col] = 1
                    V_333[j*p1+k][col] = 1
                    W_333[k*3+i][col] = 1
                    col += 1
        res_333 = (np.array(U_333), np.array(V_333), np.array(W_333))
        rank_333 = 27

    # Solve 3x3x2
    res_332 = None
    rank_332 = 18 # Schoolbook
    # Try to find lower rank for 3x3x2
    # Lower bound 9. Try 9..18
    for r in range(9, 19):
        if time.time() > deadline - 10:
            break
        res_332 = _solve_subproblem(3, 3, p2, r, max_time=5.0)
        if res_332:
            rank_332 = r
            break
            
    if not res_332:
        # Fall back to schoolbook
        U_332 = [[0]*rank_332 for _ in range(9)]
        V_332 = [[0]*rank_332 for _ in range(6)]
        W_332 = [[0]*rank_332 for _ in range(6)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(p2):
                    U_332[i*3+j][col] = 1
                    V_332[j*p2+k][col] = 1
                    W_332[k*3+i][col] = 1
                    col += 1
        res_332 = (np.array(U_332), np.array(V_332), np.array(W_332))
        rank_332 = 18

    U_333, V_333, W_333 = res_333
    U_332, V_332, W_332 = res_332
    
    total_rank = rank_333 + rank_332
    
    # Construct combined matrices
    U_total = np.zeros((9, total_rank))
    V_total = np.zeros((15, total_rank))
    W_total = np.zeros((15, total_rank))
    
    # Part 1: 3x3x3 (k=0,1,2)
    U_total[:, :rank_333] = U_333
    # V_333 has rows for j=0,1,2 and k=0,1,2. Index j*3+k.
    # In V_total, rows are j*5+k.
    # For k=0,1,2, V_total[j*5+k] = V_333[j*3+k]
    for j in range(3):
        for k in range(p1):
            V_total[j*5+k, :rank_333] = V_333[j*3+k, :rank_333]
    # W_333 has rows for k=0,1,2 and i=0,1,2. Index k*3+i.
    # In W_total, rows are k*3+i.
    # For k=0,1,2, W_total[k*3+i] = W_333[k*3+i]
    W_total[:9, :rank_333] = W_333
    
    # Part 2: 3x3x2 (k=3,4)
    U_total[:, rank_333:] = U_332
    # V_332 has rows for j=0,1,2 and k'=0,1. Index j*2+k'.
    # In V_total, rows are j*5+k where k=3,4.
    # k=3 corresponds to k'=0, k=4 corresponds to k'=1.
    for j in range(3):
        for kp in range(p2):
            k = p1 + kp
            V_total[j*5+k, rank_333:] = V_332[j*2+kp, :]
    # W_332 has rows for k'=0,1 and i=0,1,2. Index k'*3+i.
    # In W_total, rows are k*3+i where k=3,4.
    # k=3 corresponds to k'=0, k=4 corresponds to k'=1.
    for kp in range(p2):
        k = p1 + kp
        # W_332 rows for this kp are W_332[kp*3 + i]
        # W_total rows for this k are W_total[k*3 + i]
        W_total[k*3:(k+1)*3, rank_333:] = W_332[kp*3:(kp+1)*3, :]
        
    # Convert to lists of lists
    U_best = [[float(x) for x in row] for row in U_total]
    V_best = [[float(x) for x in row] for row in V_total]
    W_best = [[float(x) for x in row] for row in W_total]
    
    # Final verification
    if not _check_decomposition(U_best, V_best, W_best, n, m, p, total_rank):
        # Fall back to full schoolbook
        schoolbook_rank = n * m * p
        U_fallback = [[0] * schoolbook_rank for _ in range(n * m)]
        V_fallback = [[0] * schoolbook_rank for _ in range(m * p)]
        W_fallback = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j][column] = 1
                    V_fallback[j * p + k][column] = 1
                    W_fallback[k * n + i][column] = 1
                    column += 1
        return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
        
    return {"U": U_best, "V": V_best, "W": W_best}
