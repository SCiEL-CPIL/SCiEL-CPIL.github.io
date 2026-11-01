"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Start with schoolbook decomposition
    U = [[Fraction(0)] * schoolbook_rank for _ in range(n * m)]
    V = [[Fraction(0)] * schoolbook_rank for _ in range(m * p)]
    W = [[Fraction(0)] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_rank = schoolbook_rank
    
    # Helper: verify decomposition
    def verify(U_mat, V_mat, W_mat, rank):
        """Check if decomposition exactly matches matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Check: sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] == 1
                    # But wait, the tensor is T[i,j,k] = 1 if we're computing (A@B)[i,k]
                    # Actually the tensor for matrix mult: T[i,k,j] = delta[j,j]... 
                    # Let me re-read: W[k*n+i, r] - so W index is (k,i)
                    # U index is (i,j), V index is (j,k)
                    # The tensor T[i,j,k] should equal 1 for the multiplication
                    # Actually for matrix multiplication C = A*B where A is n x m, B is m x p:
                    # C[i,k] = sum_j A[i,j] * B[j,k]
                    # So the tensor has T[i,j,k] = 1 (it's the identity in the middle dimension)
                    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    # Wait, W is indexed as W[k*n+i, r], so W row index is k*n+i
                    # Let me check: for each (i,j,k), we need sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
                    total = Fraction(0)
                    for r in range(rank):
                        total += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if total != 1:
                        return False
        return True
    
    # Verify initial
    if not verify(U, V, W, schoolbook_rank):
        raise ValueError("Initial schoolbook decomposition failed verification")
    
    # Local search: try to merge pairs of terms
    # If we have two terms c1, c2, can we find a new term c_new such that
    # U[i,j,c_new] = U[i,j,c1] + U[i,j,c2] (or other combination)
    # and similarly for V, W, and then remove c1 and c2?
    
    # More systematically: try all pairs of columns to see if they can be merged
    # A merge is possible if there exist coefficients a, b (half-integers) such that
    # the new column is a*col1 + b*col2 and this satisfies the tensor equation
    
    # Alternative approach: try to find linear dependencies
    # For each pair of columns (c1, c2), check if we can write the tensor contribution
    # of these two columns as a single rank-1 term
    
    # Let's try a different approach: random local search with exact arithmetic
    # Try to find a lower rank by attempting to add random combinations and checking
    
    # Actually, let's try a structured approach:
    # For 2x3x4, known lower bounds suggest rank might be around 20-22
    # Let's try to find a decomposition by solving a system of equations
    
    # Method: Try to find a rank-r decomposition by random initialization and local search
    for target_rank in range(20, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        max_attempts = 50 if target_rank <= 22 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization with small half-integers
            U_try = [[random.choice([Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]) 
                     for _ in range(target_rank)] for _ in range(n * m)]
            V_try = [[random.choice([Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]) 
                     for _ in range(target_rank)] for _ in range(m * p)]
            W_try = [[random.choice([Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2)]) 
                     for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U_try, V_try, W_try, target_rank):
                best_U = U_try
                best_V = V_try
                best_W = W_try
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    # Try to improve further with local search from best found
    if best_rank < schoolbook_rank:
        # Try to merge columns
        for merge_attempt in range(200):
            if time.time() > deadline - 5:
                break
            if best_rank <= 20:  # Stop if we're already at a good rank
                break
                
            c1 = random.randint(0, best_rank - 1)
            c2 = random.randint(0, best_rank - 1)
            if c1 == c2:
                continue
            
            # Try to merge c1 and c2 into a new column
            # New column should satisfy: for all (i,j,k), 
            # U[i,j,new]*V[j,k,new]*W[k,i,new] = 
            # U[i,j,c1]*V[j,k,c1]*W[k,i,c1] + U[i,j,c2]*V[j,k,c2]*W[k,i,c2]
            
            # This is hard to solve directly. Instead, try simple combinations
            # Try: new_col = c1 + c2 (element-wise)
            # Check if removing c1 and c2 and adding new_col still works
            
            U_new = [row[:] for row in best_U]
            V_new = [row[:] for row in best_V]
            W_new = [row[:] for row in best_W]
            
            # Create new column as sum
            new_col_idx = best_rank - 1  # Use last column as new
            for i in range(n * m):
                U_new[i][new_col_idx] = best_U[i][c1] + best_U[i][c2]
            for j in range(m * p):
                V_new[j][new_col_idx] = best_V[j][c1] + best_V[j][c2]
            for k in range(p * n):
                W_new[k][new_col_idx] = best_W[k][c1] + best_W[k][c2]
            
            # Zero out columns c1 and c2
            for i in range(n * m):
                U_new[i][c1] = Fraction(0)
                U_new[i][c2] = Fraction(0)
            for j in range(m * p):
                V_new[j][c1] = Fraction(0)
                V_new[j][c2] = Fraction(0)
            for k in range(p * n):
                W_new[k][c1] = Fraction(0)
                W_new[k][c2] = Fraction(0)
            
            if verify(U_new, V_new, W_new, best_rank):
                # Check if we can actually reduce rank by removing zero columns
                # Find which columns are all zero
                active_cols = []
                for c in range(best_rank):
                    if any(U_new[i][c] != 0 for i in range(n * m)) or \
                       any(V_new[j][c] != 0 for j in range(m * p)) or \
                       any(W_new[k][c] != 0 for k in range(p * n)):
                        active_cols.append(c)
                
                if len(active_cols) < best_rank:
                    # Compress
                    new_rank = len(active_cols)
                    U_comp = [[U_new[i][c] for c in active_cols] for i in range(n * m)]
                    V_comp = [[V_new[j][c] for c in active_cols] for j in range(m * p)]
                    W_comp = [[W_new[k][c] for c in active_cols] for k in range(p * n)]
                    
                    if verify(U_comp, V_comp, W_comp, new_rank):
                        best_U = U_comp
                        best_V = V_comp
                        best_W = W_comp
                        best_rank = new_rank
    
    # Convert to list of lists of floats (or keep as Fraction if needed)
    # The problem says integer/half-integer entries, so convert to float
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
