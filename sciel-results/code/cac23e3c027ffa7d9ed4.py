"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 4

    # Standard Strassen-based decomposition: rank 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)

    # Strassen coefficients for 2x2 * 2x2
    u_coeffs = [
        [[1, 0], [0, 1]],   # M1
        [[0, 0], [1, 1]],   # M2
        [[1, 0], [0, 0]],   # M3
        [[0, 0], [0, 1]],   # M4
        [[1, 1], [0, 0]],   # M5
        [[-1, 0], [1, 0]],  # M6
        [[0, 1], [0, -1]],  # M7
    ]
    v_coeffs = [
        [[1, 0], [0, 1]],   # M1
        [[1, 0], [0, 0]],   # M2
        [[0, 1], [0, -1]],  # M3
        [[-1, 0], [1, 0]],  # M4
        [[0, 0], [0, 1]],   # M5
        [[1, 1], [0, 0]],   # M6
        [[0, 0], [1, 1]],   # M7
    ]
    w_coeffs = [
        [[1, 0], [0, 1]],   # M1: C[0,0]+=1, C[1,1]+=1
        [[0, 1], [0, -1]],  # M2: C[1,0]+=1, C[1,1]+=-1
        [[0, 0], [1, 1]],   # M3: C[0,1]+=1, C[1,1]+=1
        [[1, 0], [0, 0]],   # M4: C[0,0]+=1
        [[-1, 0], [1, 0]],  # M5: C[0,0]+=-1, C[0,1]+=1
        [[0, 1], [0, 0]],   # M6: C[1,0]+=1
        [[-1, 0], [0, 1]],  # M7: C[0,0]+=-1, C[1,1]+=1
    ]

    # 4 subproblems
    for s in range(4):
        j_block = s // 2
        k_block = s % 2
        for f in range(7):
            col = s * 7 + f
            for pi in range(2):
                for pj in range(2):
                    val = u_coeffs[f][pi][pj]
                    if val != 0:
                        j = 2 * j_block + pj
                        U[pi * m + j][col] += val
            for pj in range(2):
                for qk in range(2):
                    val = v_coeffs[f][pj][qk]
                    if val != 0:
                        j = 2 * j_block + pj
                        k = 2 * k_block + qk
                        V[j * p + k][col] += val
            for qk in range(2):
                for pi in range(2):
                    val = w_coeffs[f][qk][pi]
                    if val != 0:
                        k = 2 * k_block + qk
                        W[k * n + pi][col] += val

    # Convert to numpy for easier manipulation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Verify the decomposition
    def is_valid(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    # Check if the current solution is valid
    if not is_valid(U, V, W, rank):
        # Fallback to schoolbook
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

    # Try to reduce rank by checking for linear dependencies in the tensor components
    # This is a heuristic search for a lower rank decomposition
    # We attempt to see if we can remove a column (rank 1 component) while maintaining validity
    # This is not a full rank minimization, but a simple structural improvement attempt
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # We will try to see if rank 27 or 26 is possible by random perturbation and re-optimization
    # Given the time limit, we will only do a few iterations
    
    for target_rank in [27, 26]:
        if time.time() > deadline - 20:
            break
            
        # This is a placeholder for a more advanced search
        # In a real scenario, we would use a solver like SMT or Gröbner bases
        # Here we just return the best found so far
        
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
