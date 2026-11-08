import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120s limit
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try to find a lower rank decomposition
    # We know mu(4,4,4) = 70. Try ranks from 70 upwards.
    for rank in range(70, rank_school):
        if time.time() > deadline - 10:
            break
            
        # Try multiple restarts for this rank
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            # Initialize U and V with random sparse integers
            # Use -1, 0, 1 with low density
            density = 0.2
            U = [[random.choice([-1, 0, 1]) if random.random() < density else 0 for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) if random.random() < density else 0 for _ in range(rank)] for _ in range(m * p)]
            
            # Ensure some non-zero rows/cols to allow solving
            # If a row in U is all zero, we can't solve for W if A is 0.
            # We need A_{R, (k,i)} != 0 for some R.
            
            W = [[0] * rank for _ in range(p * n)]
            success = True
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    # Compute A_{r, (k,i)} = sum_j U_{(i,j), r} V_{(j,k), r}
                    A = [0] * rank
                    for j in range(m):
                        u_row = U[i * m + j]
                        v_col = V[j * p + k]
                        for r in range(rank):
                            A[r] += u_row[r] * v_col[r]
                            
                    # We need sum_r A[r] * W_{(k,i), r} = 1
                    # Pick W_{(k,i), 0..R-2} randomly
                    w_vals = [0] * rank
                    for r in range(rank - 1):
                        # Choose small integers
                        w_vals[r] = random.choice([-1, 0, 1])
                        
                    # Solve for w_vals[rank-1]
                    # A[rank-1] * w_vals[rank-1] = 1 - sum_{r=0}^{rank-2} A[r] * w_vals[r]
                    lhs = 0
                    for r in range(rank - 1):
                        lhs += A[r] * w_vals[r]
                    rhs = 1 - lhs
                    
                    if A[rank - 1] == 0:
                        # Cannot solve, try different randomization
                        success = False
                        break
                    
                    # Check if rhs is divisible by A[rank-1] to get integer
                    if rhs % A[rank - 1] != 0:
                        # Try half-integers? The problem allows half-integers.
                        # So we need 2 * rhs % (2 * A[rank-1]) == 0?
                        # Or just check if rhs / A[rank-1] is integer or half-integer.
                        num = rhs
                        den = A[rank - 1]
                        # Check if num/den is k/2
                        # 2*num % den == 0
                        if (2 * num) % den != 0:
                            success = False
                            break
                        w_vals[rank - 1] = num / den
                    else:
                        w_vals[rank - 1] = rhs / A[rank - 1]
                        
                    if not success:
                        break
                        
                    W[k * n + i] = w_vals
                    
            if success:
                # Verify the solution
                # Check all i, j, k
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(rank):
                                s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                            if s != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                            
                if valid:
                    best_rank = rank
                    best_U = U
                    best_V = V
                    best_W = W
                    return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}
