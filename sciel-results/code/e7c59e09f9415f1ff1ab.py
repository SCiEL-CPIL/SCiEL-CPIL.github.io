import time
import sys
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Schoolbook decomposition as guaranteed fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def verify_decomp_fast(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Fast vectorized verification using einsum."""
        # Compute the reconstructed tensor
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T[a, b, c] - 1.0) > tol:
                        return False
        
        # Check zero entries - all other combinations
        # We need T[a,b,c] = 0 for all (a,b,c) not corresponding to valid (i,j,k)
        # Valid: a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
        # This means: i = a//m, j = a%m, k = b%p, j2 = b//p, i2 = c//n, k2 = c%n
        # Valid iff j == j2 and i == i2 and k == k2
        
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    i2 = c // n
                    k2 = c % n
                    
                    if not (j == j2 and i == i2 and k == k2):
                        if abs(T[a, b, c]) > tol:
                            return False
        
        return True
    
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        """Check if decomposition is exact for all entries."""
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        return verify_decomp_fast(U_mat, V_mat, W_mat, n, m, p, tol)
    
    # Try to find lower rank decomposition using structured approach
    # Strategy: Use the fact that matrix multiplication can sometimes be 
    # decomposed using algebraic identities
    
    # For now, let's try a few specific structured decompositions
    # that might achieve rank < 60
    
    # Approach: Try to find a decomposition by exploiting linearity
    # The tensor T_{a,b,c} = 1 if (a,b,c) corresponds to valid (i,j,k) else 0
    
    # Let's try a randomized greedy approach with small rational coefficients
    # Coefficient set: {-1, 0, 1, -1/2, 1/2}
    
    np.random.seed(42)
    coeff_set = [Fraction(-1), Fraction(0), Fraction(1), Fraction(-1,2), Fraction(1,2)]
    
    best_found = False
    
    # Try ranks from 55 down to 50 (conservative range)
    for target_rank in range(55, 49, -1):
        if time.time() > deadline - 15:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random small rational coefficients
            U_mat = np.zeros((n*m, target_rank))
            V_mat = np.zeros((m*p, target_rank))
            W_mat = np.zeros((p*n, target_rank))
            
            # Random sparse initialization
            for r_idx in range(target_rank):
                # Each rank-1 tensor has a few non-zero entries
                num_nonzero_u = np.random.randint(1, 4)
                num_nonzero_v = np.random.randint(1, 4)
                num_nonzero_w = np.random.randint(1, 4)
                
                u_indices = np.random.choice(n*m, size=num_nonzero_u, replace=False)
                v_indices = np.random.choice(m*p, size=num_nonzero_v, replace=False)
                w_indices = np.random.choice(p*n, size=num_nonzero_w, replace=False)
                
                for idx in u_indices:
                    U_mat[idx, r_idx] = float(np.random.choice([-1, 1, 0.5, -0.5]))
                for idx in v_indices:
                    V_mat[idx, r_idx] = float(np.random.choice([-1, 1, 0.5, -0.5]))
                for idx in w_indices:
                    W_mat[idx, r_idx] = float(np.random.choice([-1, 1, 0.5, -0.5]))
            
            # Verify
            if verify_decomp_fast(U_mat, V_mat, W_mat, n, m, p, tol=1e-6):
                # Found a valid decomposition!
                # Try to reduce rank further via null-space projection
                best_rank = target_rank
                best_U = U_mat.tolist()
                best_V = V_mat.tolist()
                best_W = W_mat.tolist()
                best_found = True
                break
        
        if best_found:
            break
    
    # If we found a better decomposition, try to minimize rank further
    if best_found and best_rank < rank_schoolbook:
        # Try null-space projection to eliminate redundant columns
        # This is a refinement pass
        if time.time() < deadline - 20:
            U_arr = np.array(best_U, dtype=float)
            V_arr = np.array(best_V, dtype=float)
            W_arr = np.array(best_W, dtype=float)
            
            r = U_arr.shape[1]
            
            # Try to find if any column is a linear combination of others
            # For simplicity, check if removing any single column still gives valid decomp
            for col_to_remove in range(r - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                
                # Remove this column
                U_test = np.delete(U_arr, col_to_remove, axis=1)
                V_test = np.delete(V_arr, col_to_remove, axis=1)
                W_test = np.delete(W_arr, col_to_remove, axis=1)
                
                if verify_decomp_fast(U_test, V_test, W_test, n, m, p, tol=1e-6):
                    # Success! Reduce rank by 1
                    U_arr = U_test
                    V_arr = V_test
                    W_arr = W_test
                    best_U = U_arr.tolist()
                    best_V = V_arr.tolist()
                    best_W = W_arr.tolist()
                    best_rank -= 1
                    # Continue trying to remove more columns
                    r = U_arr.shape[1]
    
    # Return the best decomposition found
    return {"U": best_U, "V": best_V, "W": best_W}
