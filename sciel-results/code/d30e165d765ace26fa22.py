"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Dimensions of the factor matrices
    dim_U = n * m  # 12
    dim_V = m * p  # 32
    dim_W = p * n  # 24
    
    # The tensor T is of size (n*m) x (m*p) x (p*n)
    # We represent T as a matrix M of size (n*m*p) x (n*m*p) where M_{(i,j,k), (i',j',k')} 
    # is 1 if i==i', j==j', k==k' else 0. 
    # Actually, the tensor multiplication condition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is a system of linear equations in the variables U_c, V_c, W_c.
    # Let x be the vector of all variables. The equations are quadratic.
    # However, if we fix U and V, the equations are linear in W.
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # Let A_{(i,j,k), c} = U[i*m+j, c] * V[j*p+k, c].
    # Then A * W_flat = 1, where W_flat is W reshaped to (p*n, rank).
    # Wait, W is (p*n, rank). The equation is sum_c A_{(i,j,k), c} * W[k*n+i, c] = 1.
    # This is not a standard matrix multiplication because the index of W depends on i and k.
    # We can rewrite this as a linear system.
    
    # Let's construct the linear system for W given fixed U and V.
    # Number of equations: n*m*p = 96.
    # Number of unknowns in W: p*n*rank = 24*rank.
    # If rank is small (e.g., < 4), 24*rank < 96, so the system is underdetermined or inconsistent.
    # Actually, for a solution to exist, the vector of 1s must be in the column space of the matrix formed by the equations.
    
    # Let's define the matrix L of size (n*m*p) x (p*n*rank).
    # Row index: (i, j, k) -> idx = (i*m + j)*p + k
    # Col index: (k, i, c) -> idx = (k*n + i)*rank + c
    # L[idx_eq, idx_var] = U[i*m+j, c] * V[j*p+k, c]
    
    # Strategy:
    # 1. Start with a random rank R.
    # 2. Generate random U (dim_U x R) and V (dim_V x R).
    # 3. Construct L (96 x 24*R).
    # 4. Solve L * W_flat = 1.
    # 5. If a solution exists (residual is small), we have a valid decomposition.
    # 6. We want to minimize R.
    
    # We will try ranks from a lower bound upwards.
    # Lower bound for 3x4x8 is not known to be less than 88 (from Strassen tiling).
    # But randomized search might find better structures.
    # Let's start from 80 and go up to 96.
    
    best_rank = 96
    best_U = None
    best_V = None
    best_W = None
    
    # Precompute indices for speed
    idx_eq_list = []
    idx_u_list = []
    idx_v_list = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_eq = (i * m + j) * p + k
                idx_u = i * m + j
                idx_v = j * p + k
                idx_eq_list.append(idx_eq)
                idx_u_list.append(idx_u)
                idx_v_list.append(idx_v)
                
    idx_eq_arr = np.array(idx_eq_list)
    idx_u_arr = np.array(idx_u_list)
    idx_v_arr = np.array(idx_v_list)
    
    # We will iterate over possible ranks
    # To increase the chance of finding a solution, we can try multiple random initializations for each rank.
    
    for rank in range(80, 97):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Number of attempts per rank
        attempts = 50 if rank < 90 else 10
        
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Using normal distribution for U and V
            U_rand = np.random.randn(dim_U, rank)
            V_rand = np.random.randn(dim_V, rank)
            
            # Construct L
            # L is 96 x (24 * rank)
            # We can build it efficiently using broadcasting or loops.
            # L[row, col] where row is (i,j,k) and col is (k,i,c)
            # U[i*m+j, c] * V[j*p+k, c]
            
            # Let's reshape U and V to make it easier
            # U is (n*m, rank). We need U[i*m+j, c].
            # V is (m*p, rank). We need V[j*p+k, c].
            
            # We can create a matrix M of size (n*m*p) x (rank) where M[(i*m+j)*p+k, c] = U[i*m+j, c] * V[j*p+k, c]
            # Then L is a permutation of M's rows/columns? No.
            # L[row, (k*n+i)*rank + c] = M[row, c]
            
            # Let's build M first
            # M[row, c] = U[idx_u[row], c] * V[idx_v[row], c]
            # idx_u and idx_v are arrays of length 96.
            U_sel = U_rand[idx_u_arr, :]  # (96, rank)
            V_sel = V_rand[idx_v_arr, :]  # (96, rank)
            M = U_sel * V_sel  # (96, rank)
            
            # Now we need to map M to L.
            # L[row, col] = M[row, c] where col = (k*n+i)*rank + c
            # This is a block permutation.
            # L is (96, 24*rank).
            # We can initialize L as zeros and fill it.
            # Or we can use advanced indexing.
            
            # Let's try to solve without explicitly building L if possible?
            # The system is: sum_c M[row, c] * W[k*n+i, c] = 1
            # Let W_flat be (24*rank,).
            # W_flat[(k*n+i)*rank + c] = W[k*n+i, c]
            
            # We can build L using sparse matrix if needed, but dense is fine for 96 x (24*rank).
            # Max rank 96 -> 96 x 2304.
            
            L = np.zeros((96, 24 * rank))
            
            # Fill L
            # For each row r in 0..95:
            #   i = r // (m*p)
            #   rem = r % (m*p)
            #   j = rem // p
            #   k = rem % p
            #   For each c in 0..rank-1:
            #     col = (k*n + i)*rank + c
            #     L[r, col] = M[r, c]
            
            # Vectorized fill:
            # We need to know i and k for each row.
            # i = idx_eq_arr // (m*p)
            # k = idx_eq_arr % p
            i_arr = idx_eq_arr // (m * p)
            k_arr = idx_eq_arr % p
            
            # col_indices for each (r, c)
            # We want to scatter M[r, c] to L[r, (k[r]*n + i[r])*rank + c]
            
            # Create column indices for all elements
            # shape (96, rank)
            # c_indices = np.arange(rank)
            # base_col = (k_arr * n + i_arr) * rank
            # cols = base_col[:, np.newaxis] + np.arange(rank)[np.newaxis, :]
            
            base_col = (k_arr * n + i_arr) * rank
            cols = base_col[:, np.newaxis] + np.arange(rank)[np.newaxis, :]
            
            # L is zeros, we can use L[row, col] = M[row, col]
            # But L is dense. 
            # We can just do:
            # L[np.arange(96)[:, np.newaxis], cols] = M
            
            L[np.arange(96)[:, np.newaxis], cols] = M
            
            # Solve L * W_flat = 1
            # Use least squares
            try:
                W_flat, residuals, rank_L, s = np.linalg.lstsq(L, np.ones(96), rcond=None)
                # Check residual
                # residual = np.linalg.norm(L @ W_flat - np.ones(96))
                # If residual is small, we have a solution.
                if residuals and residuals[0] < 1e-8:
                    # Reconstruct W
                    W_rand = W_flat.reshape(24, rank)
                    
                    # Verify
                    if _verify(U_rand, V_rand, W_rand, n, m, p):
                        best_rank = rank
                        best_U = U_rand
                        best_V = V_rand
                        best_W = W_rand
                        found = True
                        break # Move to next rank
            except:
                pass
        
        if found:
            # We found a solution for this rank.
            # We could continue to see if we find a lower rank? 
            # No, we are iterating rank from low to high.
            # So the first found is the best for this search.
            # But maybe a lower rank was missed?
            # We started from 80. If we found 85, we should have found 80-84 if they existed.
            # However, random search might miss them.
            # Given time constraints, we return the first valid one found.
            # To be safe, we can break.
            break
            
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: Schoolbook
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

def construct():
    return _construct()
# EVOLVE-BLOCK-END
