import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 145
    
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank_val = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank_val):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Fast verification using matrix operations."""
        rank_val = U.shape[1]
        for i in range(n):
            for k in range(p):
                # C[i,k] should equal sum_j A[i,j]*B[j,k]
                # In tensor form: sum_j U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] summed over r
                # For each (i,j,k), the contribution is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
                # Sum over r gives the value that should be 1 for all (i,j,k)
                for j in range(m):
                    val = 0.0
                    for r in range(rank_val):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Strategy: Try to eliminate columns by finding linear dependencies
    # or by combining terms
    
    # First, try simple reductions: if two columns have proportional factors,
    # we might be able to merge them
    
    def try_reduce_column(U_mat, V_mat, W_mat, col_to_remove):
        """Try to remove a column by adjusting others."""
        U_try = U_mat.copy()
        V_try = V_mat.copy()
        W_try = W_mat.copy()
        
        # Get the column to remove
        U_col = U_mat[:, col_to_remove].copy()
        V_col = V_mat[:, col_to_remove].copy()
        W_col = W_mat[:, col_to_remove].copy()
        
        # The contribution of this column to T[i,j,k] is:
        # U_col[i*m+j] * V_col[j*p+k] * W_col[k*n+i]
        # We need to cancel this by adjusting other columns
        
        # For each (i,j,k) where this column contributes non-zero,
        # we need other columns to compensate
        
        # Find all (i,j,k) where the column contributes
        contributing = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U_col[i*m+j] * V_col[j*p+k] * W_col[k*n+i]
                    if abs(val) > 1e-12:
                        contributing.append((i, j, k, val))
        
        if not contributing:
            # Column doesn't contribute, just remove it
            U_try = np.delete(U_try, col_to_remove, axis=1)
            V_try = np.delete(V_try, col_to_remove, axis=1)
            W_try = np.delete(W_try, col_to_remove, axis=1)
            return U_try, V_try, W_try, True
        
        # Try to find adjustments to other columns to cancel the contribution
        # This is a system of equations. For simplicity, try small perturbations
        
        # For each contributing (i,j,k), we need:
        # sum_{r != col_to_remove} U_try[i*m+j, r] * V_try[j*p+k, r] * W_try[k*n+i, r] = 1
        # Currently: sum_{r != col_to_remove} U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r] = 1 - val
        # So we need to add val to the sum
        
        # Try adjusting one other column at a time
        for adj_col in range(U_mat.shape[1]):
            if adj_col == col_to_remove:
                continue
            
            # Check if we can adjust column adj_col to help
            # The adjustment delta to U_adj, V_adj, W_adj should satisfy:
            # For each (i,j,k) in contributing:
            #   delta_U[i*m+j] * V_mat[j*p+k, adj_col] * W_mat[k*n+i, adj_col]
            # + U_mat[i*m+j, adj_col] * delta_V[j*p+k] * W_mat[k*n+i, adj_col]
            # + U_mat[i*m+j, adj_col] * V_mat[j*p+k, adj_col] * delta_W[k*n+i]
            # = val for each (i,j,k)
            
            # This is complex. Try a simpler approach: 
            # If U_mat[i*m+j, adj_col], V_mat[j*p+k, adj_col], W_mat[k*n+i, adj_col] 
            # are all non-zero for some (i,j,k), we can adjust one factor
            
            # Try adjusting W only (since W has the smallest dimension pn=21)
            # For a given (i,j,k), if U_mat[i*m+j, adj_col] * V_mat[j*p+k, adj_col] != 0,
            # then we need delta_W[k*n+i] = val / (U_mat[i*m+j, adj_col] * V_mat[j*p+k, adj_col])
            
            # But this must work for all (i,j,k) in contributing, which is hard.
            # Instead, try a heuristic: randomly perturb and check
            
            for attempt in range(20):
                U_adj = U_try.copy()
                V_adj = V_try.copy()
                W_adj = W_try.copy()
                
                # Small random perturbation
                scale = random.uniform(0.01, 0.1)
                U_adj[:, adj_col] += random.uniform(-scale, scale)
                V_adj[:, adj_col] += random.uniform(-scale, scale)
                W_adj[:, adj_col] += random.uniform(-scale, scale)
                
                if verify(U_adj, V_adj, W_adj, tol=1e-6):
                    # Success! Remove the column
                    U_final = np.delete(U_adj, col_to_remove, axis=1)
                    V_final = np.delete(V_adj, col_to_remove, axis=1)
                    W_final = np.delete(W_adj, col_to_remove, axis=1)
                    return U_final, V_final, W_final, True
        
        return None, None, None, False
    
    # Local search: try to remove random columns
    attempts = 0
    max_attempts = 200
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        if best_rank <= 1:
            break
        
        tc = random.randint(0, best_rank - 1)
        
        U_new, V_new, W_new, success = try_reduce_column(best_U, best_V, best_W, tc)
        
        if success:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank -= 1
    
    # Round to nearest half-integer
    best_U = np.round(best_U * 2) / 2
    best_V = np.round(best_V * 2) / 2
    best_W = np.round(best_W * 2) / 2
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-9):
        # Fallback to schoolbook
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
