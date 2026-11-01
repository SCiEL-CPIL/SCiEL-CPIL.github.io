"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Uses a structured search over low-rank tensor decompositions with rational entries.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    
    # Dimensions
    dim_A = n * m  # 6
    dim_B = m * p  # 12
    dim_C = p * n  # 8
    
    # The tensor T[a, b, c] is 1 if a=(i,j), b=(j,k), c=(i,k) and 0 otherwise.
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    def get_T_entry(a, b, c):
        i, j = divmod(a, m)
        j2, k = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j == j2 and i == i2 and k == k2:
            return 1
        return 0

    def validate(U, V, W, rank):
        """Check if U, V, W with given rank exactly decompose the multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check all (a, b, c)
        # T_recon[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We check if T_recon matches T
        for a in range(dim_A):
            for b in range(dim_B):
                for c in range(dim_C):
                    target = get_T_entry(a, b, c)
                    val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                    if abs(val - target) > 1e-6:
                        return False
        return True

    def schoolbook_solution():
        rank = dim_A * dim_B // dim_C # No, rank is n*m*p
        rank = n * m * p
        U = [[0] * rank for _ in range(dim_A)]
        V = [[0] * rank for _ in range(dim_B)]
        W = [[0] * rank for _ in range(dim_C)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_solution = schoolbook_solution()
    best_rank = n * m * p
    
    # Try to find lower rank
    # Lower bound is max(dim_A, dim_B, dim_C) = 12
    # We try ranks from 12 to best_rank - 1
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Strategy: Fix V and W, solve for U.
    # For each a, we have a linear system A_a U_a = t_a
    # A_a is (dim_B * dim_C) x rank
    # t_a is (dim_B * dim_C) x 1
    
    def try_solve_for_U(V, W, rank):
        """
        Given V, W, check if we can find U such that the decomposition holds.
        Returns U if successful, None otherwise.
        """
        U = []
        num_eq = dim_B * dim_C
        for a in range(dim_A):
            # Build A_a and t_a
            A = np.zeros((num_eq, rank))
            t = np.zeros(num_eq)
            idx = 0
            for b in range(dim_B):
                for c in range(dim_C):
                    # A[idx, r] = V[b, r] * W[c, r]
                    A[idx, :] = V[b] * W[c]
                    t[idx] = get_T_entry(a, b, c)
                    idx += 1
            
            # Solve A U_a = t
            # We need an exact solution.
            # Use least squares and check residual
            try:
                U_a, residuals, rank_A, s = np.linalg.lstsq(A, t, rcond=None)
                # Check if the solution is exact
                residual = np.linalg.norm(A.dot(U_a) - t)
                if residual < 1e-6:
                    # Also check if U_a entries are "nice" (optional, but good for integer/half-integer)
                    U.append(U_a)
                else:
                    return None
            except:
                return None
        
        # If we reached here, we have U, V, W
        # Convert to lists
        U_list = [list(map(float, u)) for u in U]
        V_list = [list(map(float, v)) for v in V]
        W_list = [list(map(float, w)) for w in W]
        
        # Final validation
        if validate(U_list, V_list, W_list, rank):
            return U_list
        return None

    # Search for solutions
    random.seed(42)
    
    for target_rank in range(12, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations of V and W
        # Use small integers for V and W to increase chance of exact solutions
        max_attempts = 50 if target_rank < 16 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize V and W with random small integers
            # Scale is important. If too large, numerical issues. If too small, rank deficiency.
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(dim_B)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(dim_C)]
            
            # Ensure some non-zero rows
            if all(all(x == 0 for x in row) for row in V):
                continue
            if all(all(x == 0 for x in row) for row in W):
                continue
            
            U = try_solve_for_U(V, W, target_rank)
            if U is not None:
                best_solution = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break

    return best_solution
# EVOLVE-BLOCK-END
