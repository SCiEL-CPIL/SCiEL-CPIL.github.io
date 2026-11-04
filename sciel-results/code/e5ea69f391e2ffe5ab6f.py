"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify_fast(U, V, W, n, m, p):
    """Vectorized verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the matmul tensor T[i,j,k] = 1
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # For each (i,j,k), compute sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # We can do this via einsum or manual loops
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U3[i,j,:], V3[j,k,:] * W3[k,i,:])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _is_valid_entry(x):
    """Check if entry is integer or half-integer."""
    try:
        x = float(x)
    except:
        return False
    if abs(x - round(x)) < 1e-9:
        return True
    if abs(2*x - round(2*x)) < 1e-9:
        return True
    return False

def _check_all_entries(U, V, W):
    for row in U:
        for x in row:
            if not _is_valid_entry(x):
                return False
    for row in V:
        for x in row:
            if not _is_valid_entry(x):
                return False
    for row in W:
        for x in row:
            if not _is_valid_entry(x):
                return False
    return True

def _schoolbook(n, m, p):
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

def _try_kronecker_based(n, m, p, max_attempts=200, seed=42):
    """
    Try to exploit Kronecker structure.
    For 3x4x4: decompose 4x4 matmul as rank-9 (Strassen-like or direct),
    then combine with 3x4 part.
    
    Strategy: Use the fact that 4x4 matmul can be done in rank 9 (optimal for 4x4 is 9? No, it's 9 for 2x2x2 Strassen... actually 4x4x4 has rank 16 schoolbook, and lower bounds exist).
    
    Actually, let's try a different approach: structured random search with constraints.
    """
    rng = random.Random(seed)
    
    # Try various rank targets
    for target_rank in range(30, 48):  # Try ranks from 30 to 47
        for attempt in range(max_attempts):
            if time.time() > 130:
                return None
            
            U = [[0]*target_rank for _ in range(n*m)]
            V = [[0]*target_rank for _ in range(m*p)]
            W = [[0]*target_rank for _ in range(p*n)]
            
            # Generate structured random columns
            for c in range(target_rank):
                # Each column should be sparse with values in {-1, 0, 1, -0.5, 0.5}
                for idx in range(n*m):
                    r = rng.random()
                    if r < 0.3:
                        U[idx][c] = rng.choice([-1, 1])
                    elif r < 0.4:
                        U[idx][c] = rng.choice([-0.5, 0.5])
                for idx in range(m*p):
                    r = rng.random()
                    if r < 0.3:
                        V[idx][c] = rng.choice([-1, 1])
                    elif r < 0.4:
                        V[idx][c] = rng.choice([-0.5, 0.5])
                for idx in range(p*n):
                    r = rng.random()
                    if r < 0.3:
                        W[idx][c] = rng.choice([-1, 1])
                    elif r < 0.4:
                        W[idx][c] = rng.choice([-0.5, 0.5])
            
            if _verify_fast(U, V, W, n, m, p):
                if _check_all_entries(U, V, W):
                    return {"U": U, "V": V, "W": W}
    return None

def _try_block_structure(n, m, p):
    """
    Try to use block structure: decompose 4x4 matmul separately, then combine.
    4x4 matmul: A(4x4) * B(4x4) = C(4x4)
    We can use rank-16 schoolbook for 4x4, but maybe lower.
    
    Actually, for the full 3x4x4:
    A is 3x4, B is 4x4, C is 3x4.
    
    We can view this as: for each row i of A (i=0,1,2), compute row_i(A) * B.
    row_i(A) is 1x4, B is 4x4, result is 1x4.
    
    So we need to compute 3 separate 1x4 * 4x4 multiplications.
    Each 1x4 * 4x4 can be done in rank 16 (schoolbook) or lower.
    
    Total rank would be 3 * rank(1x4*4x4).
    If rank(1x4*4x4) = 16, total = 48 (schoolbook).
    If we can do 1x4*4x4 in rank 12, total = 36.
    
    Let's try to find a rank-12 decomposition for 1x4 * 4x4.
    """
    # This is getting complex. Let's stick with random search for now.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try structured search for lower rank
    result = _try_kronecker_based(n, m, p, max_attempts=100, seed=42)
    if result is not None:
        best_result = result
        best_rank = len(result['U'][0])
    
    return best_result
# EVOLVE-BLOCK-END
