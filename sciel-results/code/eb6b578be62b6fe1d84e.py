"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check diagonal: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal
    checked = set()
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k2 in range(p):
                    for i2 in range(n):
                        a = i * m + j
                        b = j2 * p + k2
                        c = k2 * n + i2
                        if (a, b, c) in checked:
                            continue
                        checked.add((a, b, c))
                        
                        expected = 1.0 if (i == i2 and j == j2) else 0.0
                        
                        val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                        if abs(val - expected) > 1e-9:
                            return False
    
    return True

def _search_rank(R, time_limit=100):
    """
    Attempt to find a rank R decomposition using local search.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # We will try to find U, V, W with entries in {-1, -0.5, 0, 0.5, 1}
    # Initialize with random sparse matrices
    for attempt in range(100): # Max attempts
        if time.time() - start_time > time_limit:
            break
            
        U = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=(n*m, R))
        V = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=(m*p, R))
        W = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=(p*n, R))
        
        # Local search: iterate through entries, change them to minimize error
        # Error = || T - sum_r U[:,r] V[:,r] W[:,r] ||^2
        # This is expensive to compute fully. We can use a greedy approach.
        
        # Instead of full local search, we can try to solve a system of linear equations
        # for one matrix at a time, holding others fixed.
        # But the constraint is integer/half-integer.
        
        # Simple heuristic: 
        # 1. Compute current error.
        # 2. If error is 0, return.
        # 3. Pick a random column r.
        # 4. Try to adjust U[:,r], V[:,r], W[:,r] to reduce error.
        
        # This is very slow. 
        # Alternative: Use the fact that we can split into blocks.
        # If R is even, R = 2 * R_block.
        # We can search for a rank R_block decomposition for 2x3x3.
        
        if R % 2 == 0:
            R_block = R // 2
            # Search for rank R_block for 2x3x3
            # T_block is 2x3x3
            T_block = np.zeros((n*m, m*3, 3*n))
            for i in range(n):
                for j in range(m):
                    for k in range(3):
                        T_block[i*m+j, j*3+k, k*n+i] = 1.0
            
            # Try to find U_b, V_b, W_b for T_block
            found_block = False
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                U_b = np.random.choice([-1, -0.5, 0, 0, 0.5, 1], size=(n*m, R_block))
                V_b = np.random.choice([-1, -0.5, 0, 0, 0.5, 1], size=(m*3, R_block))
                W_b = np.random.choice([-1, -0.5, 0, 0, 0.5, 1], size=(3*n, R_block))
                
                # Check if U_b, V_b, W_b decompose T_block
                # We can use a simple verification
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(3):
                            a = i*m+j
                            b = j*3+k
                            c = k*n+i
                            val = np.sum(U_b[a, :] * V_b[b, :] * W_b[c, :])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    # Check off-diagonals for block
                    for i in range(n):
                        for j in range(m):
                            for j2 in range(m):
                                for k2 in range(3):
                                    for i2 in range(n):
                                        a = i*m+j
                                        b = j2*3+k2
                                        c = k2*n+i2
                                        expected = 1.0 if (i == i2 and j == j2) else 0.0
                                        val = np.sum(U_b[a, :] * V_b[b, :] * W_b[c, :])
                                        if abs(val - expected) > 1e-9:
                                            valid = False
                                            break
                                    if not valid: break
                                if not valid: break
                            if not valid: break
                        if not valid: break
                    if valid:
                        # We found a block decomposition!
                        # Construct full U, V, W
                        # U is same for both blocks? No, U is 2x3, same for both.
                        # V is 3x6 = [V1, V2]. We use V_b for V1 and V_b for V2?
                        # Wait, the tensor is T = T1 + T2 where T1 uses B1 and T2 uses B2.
                        # U is the same for both.
                        # V1 = V_b, V2 = V_b.
                        # W1 = W_b, W2 = W_b.
                        # But we need to map indices.
                        # V_b is 3x3. V is 3x6.
                        # V[0:3, 0:3] = V_b
                        # V[0:3, 3:6] = V_b
                        # W_b is 3x2 (p*n = 3*2=6? No, p=3, n=2 for block. p*n=6).
                        # W is 6x2 (p=6, n=2).
                        # W[0:3, :] = W_b
                        # W[3:6, :] = W_b
                        # U is 6xR_block (n*m=6).
                        # Full U is 6x(2*R_block).
                        # U[:, 0:R_block] = U_b
                        # U[:, R_block:2*R_block] = U_b
                        
                        U_full = np.zeros((n*m, 2*R_block))
                        V_full = np.zeros((m*6, 2*R_block))
                        W_full = np.zeros((6*n, 2*R_block))
                        
                        U_full[:, :R_block] = U_b
                        U_full[:, R_block:] = U_b
                        
                        V_full[:, :R_block] = V_b
                        V_full[:, R_block:] = V_b
                        
                        W_full[:3*n, :R_block] = W_b
                        W_full[3*n:, R_block:] = W_b
                        
                        # Verify full
                        if _verify(U_full.tolist(), V_full.tolist(), W_full.tolist()):
                            return U_full.tolist(), V_full.tolist(), W_full.tolist()
                        
                        found_block = True
                        break
        # If R is odd or block search fails, continue to next attempt
    return None

def construct():
    start = time.time()
    deadline = start + 115  # 115 second limit to be safe
    
    # Start with schoolbook
    result = _schoolbook()
    best_rank = 36
    
    # Try to find lower ranks
    # We know 36 is schoolbook.
    # Try 30, 28, 26, 24, 22, 20, 18
    for R in [30, 28, 26, 24, 22, 20, 18]:
        if time.time() > deadline - 10:
            break
        sol = _search_rank(R, time_limit=min(10, deadline - time.time() - 5))
        if sol:
            U, V, W = sol
            result = {"U": U, "V": V, "W": W}
            best_rank = R
            break # Found a valid lower rank, return it
    
    return result
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
