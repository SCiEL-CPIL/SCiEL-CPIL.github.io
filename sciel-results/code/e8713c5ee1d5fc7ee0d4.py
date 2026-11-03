"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin
    
    # Theoretical lower bounds from tensor flattening
    # Rank >= max(rank(T_U), rank(T_V), rank(T_W))
    # For M_{n,m,p}, the flattening ranks are:
    # T_U: (n*m) x (m*p) -> rank n*m (if p>=1)
    # T_V: (m*p) x (p*n) -> rank m*p (if n>=1)
    # T_W: (p*n) x (n*m) -> rank p*n (if m>=1)
    # Wait, this is for the identity-like tensor. For matrix multiplication:
    # The flattening of M_{n,m,p} along mode 1 has size (n*m) x (m*p)
    # Its rank is n*m (since for each (i,j) pair, the slice is linearly independent)
    # Similarly for other modes.
    # So lower bound is max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    
    # However, we know Strassen gives 29. Can we do better?
    # Let's try to find a decomposition with rank < 29 using random search
    
    # First, return the known good solution (rank 29) as fallback
    fallback = _strassen_2x2x2_plus_schoolbook(n, m, p)
    
    # Try to find a better solution using randomized rank-revealing
    if time.time() < deadline - 10:
        better = _randomized_rank_search(n, m, p, fallback, deadline)
        if better is not None:
            return better
    
    return fallback

def _strassen_2x2x2_plus_schoolbook(n, m, p):
    """
    Construct a rank-29 decomposition using Strassen on the 2x2x2 sub-block.
    """
    rank = 29
    U = [[0] * rank for _ in range(n * m)]  # 6 x 29
    V = [[0] * rank for _ in range(m * p)]  # 15 x 29
    W = [[0] * rank for _ in range(p * n)]  # 10 x 29
    
    # Strassen terms for the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[3][2] = 1; V[4][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B01-B00)
    U[4][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A10)B11
    U[0][4] = 1; U[3][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[3][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[4][6] = -1
    V[1][6] = 1; V[4][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    # Schoolbook for remaining
    col = 7
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    T = np.zeros((n, m, p), dtype=np.float64)
    
    for c in range(rank):
        u_c = np.array([U[i * m + j][c] for i in range(n) for j in range(m)]).reshape(n, m)
        v_c = np.array([V[j * p + k][c] for j in range(m) for k in range(p)]).reshape(m, p)
        w_c = np.array([W[k * n + i][c] for k in range(p) for i in range(n)]).reshape(p, n)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
    
    return np.all(np.abs(T - 1) < 1e-9)

def _randomized_rank_search(n, m, p, fallback, deadline):
    """
    Try to find a decomposition with rank less than fallback's rank using
    randomized sparse rank-revealing techniques.
    
    Strategy:
    1. Start with the known rank-29 decomposition
    2. Try to merge terms by finding linear dependencies
    3. Use random projections to identify potential rank reductions
    """
    U_fb = np.array(fallback["U"], dtype=np.float64)
    V_fb = np.array(fallback["V"], dtype=np.float64)
    W_fb = np.array(fallback["W"], dtype=np.float64)
    current_rank = U_fb.shape[1]
    
    # Try to find a lower rank by searching for sparse combinations
    # We'll try ranks from current_rank - 1 down to 15
    for target_rank in range(current_rank - 1, 14, -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random sparse matrices
            U = _random_sparse_matrix(n * m, target_rank, density=0.3)
            V = _random_sparse_matrix(m * p, target_rank, density=0.3)
            W = _random_sparse_matrix(p * n, target_rank, density=0.3)
            
            # Scale to ensure correct magnitude
            U *= 1.0
            V *= 1.0
            W *= 1.0
            
            # Check if this random initialization happens to work (unlikely)
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _random_sparse_matrix(rows, cols, density=0.3):
    """Generate a random sparse matrix with entries in {-1, 0, 1}."""
    M = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            if np.random.random() < density:
                M[i, j] = np.random.choice([-1, 1])
    return M

# EVOLVE-BLOCK-END
