"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    deadline = time.time() + 140.0
    
    # Best result found so far (initialized to schoolbook)
    best_U = [[0] * schoolbook_rank for _ in range(n * m)]
    best_V = [[0] * schoolbook_rank for _ in range(m * p)]
    best_W = [[0] * schoolbook_rank for _ in range(p * n)]
    best_rank = schoolbook_rank
    
    # Fill schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def _to_fraction_list(matrix):
        """Convert numpy float matrix to list of lists of Fractions (half-integers allowed)."""
        if matrix is None:
            return None
        result = []
        for row in matrix:
            new_row = []
            for val in row:
                # Round to nearest 0.5
                half_int = round(val * 2) / 2.0
                new_row.append(Fraction(half_int))
            result.append(new_row)
        return result
    
    def _verify(U_list, V_list, W_list, rank):
        """Verify exact decomposition."""
        if len(U_list) != n * m or len(V_list) != m * p or len(W_list) != p * n:
            return False
        if len(U_list[0]) != rank or len(V_list[0]) != rank or len(W_list[0]) != rank:
            return False
        
        # Check tensor decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # T[i,j,k] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        total += U_list[i * m + j][c] * V_list[j * p + k][c] * W_list[k * n + i][c]
                    if total != 1:
                        return False
        return True
    
    # Try to find lower rank decompositions using randomized search + local optimization
    # Strategy: For each target rank r from min_rank to best_rank-1:
    #   1. Try random initialization with structured patterns
    #   2. Use local search to refine
    
    # Known lower bound: max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    # But for matrix multiplication, the lower bound is often higher.
    # For 4x4 * 4x8, let's try ranks from 32 upward.
    
    min_possible_rank = max(n * m, m * p, p * n)
    
    # Try a few target ranks
    target_ranks = list(range(min_possible_rank, best_rank))
    
    # Shuffle target ranks to try different orders
    np.random.seed(42)
    np.random.shuffle(target_ranks)
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10.0:
            break
        if target_rank >= best_rank:
            continue
            
        # Multiple restarts for this rank
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize with random binary-like patterns
            # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Fill with random values in {-1, 0, 1} or {-0.5, 0, 0.5, 1}
            for c in range(target_rank):
                # Randomly assign non-zero entries
                for idx in range(n * m):
                    if np.random.rand() < 0.3:
                        U[idx, c] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
                for idx in range(m * p):
                    if np.random.rand() < 0.3:
                        V[idx, c] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
                for idx in range(p * n):
                    if np.random.rand() < 0.3:
                        W[idx, c] = np.random.choice([-1, 1]) * np.random.choice([0.5, 1.0])
            
            # Local search: try to improve the decomposition
            # Objective: minimize sum of squared errors over all (i,j,k)
            def _compute_error(U_mat, V_mat, W_mat):
                errors = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                            errors += (val - 1.0) ** 2
                return errors
            
            current_error = _compute_error(U, V, W)
            
            # Simple local search: perturb entries
            for step in range(50):
                if time.time() > deadline - 3.0:
                    break
                    
                # Pick a random entry to perturb
                mat_choice = np.random.choice([0, 1, 2])
                if mat_choice == 0:
                    idx = np.random.randint(n * m)
                    col_idx = np.random.randint(target_rank)
                    old_val = U[idx, col_idx]
                    new_val = old_val + np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    U[idx, col_idx] = new_val
                elif mat_choice == 1:
                    idx = np.random.randint(m * p)
                    col_idx = np.random.randint(target_rank)
                    old_val = V[idx, col_idx]
                    new_val = old_val + np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    V[idx, col_idx] = new_val
                else:
                    idx = np.random.randint(p * n)
                    col_idx = np.random.randint(target_rank)
                    old_val = W[idx, col_idx]
                    new_val = old_val + np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    W[idx, col_idx] = new_val
                
                new_error = _compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                else:
                    # Revert
                    if mat_choice == 0:
                        U[idx, col_idx] = old_val
                    elif mat_choice == 1:
                        V[idx, col_idx] = old_val
                    else:
                        W[idx, col_idx] = old_val
            
            # Check if we found an exact solution
            if current_error < 1e-10:
                # Convert to exact fractions
                U_list = _to_fraction_list(U)
                V_list = _to_fraction_list(V)
                W_list = _to_fraction_list(W)
                
                if U_list and V_list and W_list:
                    if _verify(U_list, V_list, W_list, target_rank):
                        best_U = U_list
                        best_V = V_list
                        best_W = W_list
                        best_rank = target_rank
                        break
    
    # Return best result found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
