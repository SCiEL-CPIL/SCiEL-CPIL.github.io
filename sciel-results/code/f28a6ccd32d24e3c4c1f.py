import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    dim_A = n * m  # 8
    dim_B = m * p  # 24
    dim_C = p * n  # 12
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Build the rank-42 Strassen decomposition
    rank42 = 42
    U42 = np.zeros((dim_A, rank42))
    V42 = np.zeros((dim_B, rank42))
    W42 = np.zeros((dim_C, rank42))
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        """Add a 2x2x2 Strassen block at position (i_start, j_start, k_start)"""
        # Strassen's 7 products for 2x2 matrix multiplication
        # U indices: (i_start+di)*m + (j_start+dj)
        # V indices: (j_start+dj)*p + (k_start+dk)
        # W indices: (k_start+dk)*n + (i_start+di)
        
        # Product 0: (A00+A11)*(B00+B11)
        U42[(i_start+0)*m + (j_start+0), offset+0] += 1
        U42[(i_start+1)*m + (j_start+1), offset+0] += 1
        V42[(j_start+0)*p + (k_start+0), offset+0] += 1
        V42[(j_start+1)*p + (k_start+1), offset+0] += 1
        W42[(k_start+0)*n + (i_start+0), offset+0] += 1
        W42[(k_start+1)*n + (i_start+1), offset+0] += 1
        
        # Product 1: (A10+A11)*B00
        U42[(i_start+1)*m + (j_start+0), offset+1] += 1
        U42[(i_start+1)*m + (j_start+1), offset+1] += 1
        V42[(j_start+0)*p + (k_start+0), offset+1] += 1
        W42[(k_start+0)*n + (i_start+1), offset+1] += 1
        W42[(k_start+1)*n + (i_start+1), offset+1] -= 1
        
        # Product 2: A00*(B01-B11)
        U42[(i_start+0)*m + (j_start+0), offset+2] += 1
        V42[(j_start+0)*p + (k_start+1), offset+2] += 1
        V42[(j_start+1)*p + (k_start+1), offset+2] -= 1
        W42[(k_start+1)*n + (i_start+0), offset+2] += 1
        W42[(k_start+1)*n + (i_start+1), offset+2] += 1
        
        # Product 3: A11*(B10-B00)
        U42[(i_start+1)*m + (j_start+1), offset+3] += 1
        V42[(j_start+1)*p + (k_start+0), offset+3] += 1
        V42[(j_start+0)*p + (k_start+0), offset+3] -= 1
        W42[(k_start+0)*n + (i_start+0), offset+3] += 1
        W42[(k_start+0)*n + (i_start+1), offset+3] += 1
        
        # Product 4: (A00+A01)*B11
        U42[(i_start+0)*m + (j_start+0), offset+4] += 1
        U42[(i_start+0)*m + (j_start+1), offset+4] += 1
        V42[(j_start+1)*p + (k_start+1), offset+4] += 1
        W42[(k_start+0)*n + (i_start+0), offset+4] -= 1
        W42[(k_start+1)*n + (i_start+0), offset+4] += 1
        
        # Product 5: (A10-A00)*(B00+B01)
        U42[(i_start+1)*m + (j_start+0), offset+5] += 1
        U42[(i_start+0)*m + (j_start+0), offset+5] -= 1
        V42[(j_start+0)*p + (k_start+0), offset+5] += 1
        V42[(j_start+0)*p + (k_start+1), offset+5] += 1
        W42[(k_start+1)*n + (i_start+1), offset+5] += 1
        
        # Product 6: (A01-A11)*(B10+B11)
        U42[(i_start+0)*m + (j_start+1), offset+6] += 1
        U42[(i_start+1)*m + (j_start+1), offset+6] -= 1
        V42[(j_start+1)*p + (k_start+0), offset+6] += 1
        V42[(j_start+1)*p + (k_start+1), offset+6] += 1
        W42[(k_start+0)*n + (i_start+0), offset+6] += 1
    
    # Six 2x2x2 blocks covering the 2x4x6 multiplication
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # Block (i, j_pair, k_pair) where j_pair is {0,1}, {2,3}, {4,5} and k_pair is {0,1}, {2,3}, {4,5}
    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)
    
    # Precompute the tensor entries for fast verification
    # tensor[(i,j,k)] = 1 if (i,j,k) is a valid entry, else 0
    # We only need to check the n*m*p = 48 non-zero entries
    
    def verify_solution_fast(U, V, W, rank, atol=1e-9):
        """Fast verification using precomputed indices"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    w = k * n + i
                    total = np.dot(U[a, :rank], V[b, :rank] * W[w, :rank])
                    if abs(total - 1.0) > atol:
                        return False
        return True
    
    # Verify the rank-42 solution
    if not verify_solution_fast(U42, V42, W42, rank42):
        # Fallback to schoolbook rank 48
        rank_school = n * m * p
        U_school = np.zeros((dim_A, rank_school))
        V_school = np.zeros((dim_B, rank_school))
        W_school = np.zeros((dim_C, rank_school))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Strategy: Try to find linear combinations of columns that allow elimination
    # For each column r, check if there exist coefficients c_s (s != r) such that
    # U[:,r] = sum_s c_s * U[:,s], V[:,r] = sum_s c_s * V[:,s], W[:,r] = sum_s c_s * W[:,s]
    # This is a strong condition. Instead, we use a weaker but more practical approach:
    # Try to find a lower-rank decomposition by solving a system of equations
    
    # Build the constraint matrix for the tensor equations
    # Each of the 48 (i,j,k) entries gives one equation
    # Unknowns: U[a,r], V[b,r], W[w,r] for r in range(target_rank)
    # This is a nonlinear system, so we use a different approach
    
    # Alternative: Use the fact that the Strassen blocks are independent
    # The 6 blocks each have 7 products. Check if any block can be compressed.
    
    # For now, try a simpler approach: check if any single column is redundant
    # by seeing if zeroing it still gives a valid decomposition
    
    def try_eliminate_column(U, V, W, rank, col_idx, atol=1e-6):
        """Check if column col_idx can be eliminated"""
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        U_test[:, col_idx] = 0
        V_test[:, col_idx] = 0
        W_test[:, col_idx] = 0
        return verify_solution_fast(U_test, V_test, W_test, rank, atol)
    
    # Try to eliminate columns one by one
    U = U42.copy()
    V = V42.copy()
    W = W42.copy()
    current_rank = rank42
    
    # Try all columns in order
    for col in range(42):
        if time.time() > deadline:
            break
        
        # Map original column index to current position
        # Since we're removing columns, we need to track which original columns remain
        # For simplicity, just try removing from the current set
        
        if current_rank == 0:
            break
            
        # Try removing the last column first (easier to handle)
        test_col = current_rank - 1
        if try_eliminate_column(U, V, W, current_rank, test_col):
            # Remove the last column
            U = U[:, :current_rank-1]
            V = V[:, :current_rank-1]
            W = W[:, :current_rank-1]
            current_rank -= 1
            # Restart the loop since indices changed
            col = -1  # Reset to try again from the beginning
    
    # Round to half-integers
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Final verification
    if not verify_solution_fast(U, V, W, current_rank, atol=1e-9):
        # Return the known rank-42 solution
        U, V, W = U42, V42, W42
        current_rank = rank42
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
