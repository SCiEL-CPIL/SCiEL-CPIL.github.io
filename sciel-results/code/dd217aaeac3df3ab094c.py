import time
import random
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    n, m, p = DIMENSIONS
    n_m = n * m  # 8
    m_p = m * p  # 24
    p_n = p * n  # 12
    
    # First, build the current rank-42 solution
    rank42_U, rank42_V, rank42_W = _build_rank42()
    
    # Verify it's correct
    if time.time() < deadline - 10.0:
        if _verify_decomposition(rank42_U, rank42_V, rank42_W, n, m, p):
            # Try to find a lower rank solution
            best_U, best_V, best_W = rank42_U, rank42_V, rank42_W
            best_rank = 42
            
            # Try ranks from 41 down to 30
            for target_rank in range(41, 29, -1):
                if time.time() > deadline - 5.0:
                    break
                found = _search_rank(target_rank, n, m, p, start_time, deadline)
                if found is not None:
                    best_U, best_V, best_W = found
                    best_rank = target_rank
                    break  # Found the lowest rank, stop
            
            return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": rank42_U, "V": rank42_V, "W": rank42_W}


def _build_rank42():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]

    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    
    # Build the multiplication tensor T[i, j, k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition should satisfy: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0  # T[i,j,k] = 1 for all i,j,k in standard matrix mult
                actual = 0.0
                for r in range(rank):
                    actual += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True


def _search_rank(target_rank, n, m, p, start_time, deadline):
    """Try to find a rank-target_rank decomposition using random search."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Use random initialization and check if we can find a valid solution
    # This is a hard problem, so we use a heuristic approach
    
    max_iterations = 50
    for iter in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization with small integers/half-integers
        U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(n_m)]
        V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(m_p)]
        W = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(p_n)]
        
        if _verify_decomposition(U, V, W, n, m, p):
            return (U, V, W)
    
    return None
# EVOLVE-BLOCK-END
