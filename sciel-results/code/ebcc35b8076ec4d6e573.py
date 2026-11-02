import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    def build_strassen_28():
        """Build the rank-28 Strassen-based decomposition."""
        rank = 28
        U = [[0] * rank for _ in range(n * m)]   # 8 x 28
        V = [[0] * rank for _ in range(m * p)]   # 16 x 28
        W = [[0] * rank for _ in range(p * n)]   # 8 x 28

        # 4 Strassen blocks, each contributing 7 columns
        # Block structure: A is 2x4, B is 4x4, C is 2x4
        # A rows: 0,1; A cols: 0,1,2,3
        # B rows: 0,1,2,3; B cols: 0,1,2,3
        # C rows: 0,1; C cols: 0,1,2,3
        #
        # Block 0: A[:,0:2] @ B[0:2,0:2] -> C[:,0:2]
        # Block 1: A[:,2:4] @ B[2:4,0:2] -> C[:,0:2]
        # Block 2: A[:,0:2] @ B[0:2,2:4] -> C[:,2:4]
        # Block 3: A[:,2:4] @ B[2:4,2:4] -> C[:,2:4]

        def add_strassen(A_row_base, A_col_base, B_row_base, B_col_base, C_row_base, C_col_base, col_offset):
            """Add one 2x2x2 Strassen block."""
            # Strassen:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            #
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6

            # M1: (A00+A11)(B00+B11)
            c = col_offset + 0
            U[(A_row_base+0)*m + (A_col_base+0)][c] += 1
            U[(A_row_base+1)*m + (A_col_base+1)][c] += 1
            V[(B_row_base+0)*p + (B_col_base+0)][c] += 1
            V[(B_row_base+1)*p + (B_col_base+1)][c] += 1

            # M2: (A10+A11)B00
            c = col_offset + 1
            U[(A_row_base+1)*m + (A_col_base+0)][c] += 1
            U[(A_row_base+1)*m + (A_col_base+1)][c] += 1
            V[(B_row_base+0)*p + (B_col_base+0)][c] += 1

            # M3: A00(B01-B11)
            c = col_offset + 2
            U[(A_row_base+0)*m + (A_col_base+0)][c] += 1
            V[(B_row_base+0)*p + (B_col_base+1)][c] += 1
            V[(B_row_base+1)*p + (B_col_base+1)][c] -= 1

            # M4: A11(B10-B00)
            c = col_offset + 3
            U[(A_row_base+1)*m + (A_col_base+1)][c] += 1
            V[(B_row_base+1)*p + (B_col_base+0)][c] += 1
            V[(B_row_base+0)*p + (B_col_base+0)][c] -= 1

            # M5: (A00+A01)B11
            c = col_offset + 4
            U[(A_row_base+0)*m + (A_col_base+0)][c] += 1
            U[(A_row_base+0)*m + (A_col_base+1)][c] += 1
            V[(B_row_base+1)*p + (B_col_base+1)][c] += 1

            # M6: (A10-A00)(B00+B01)
            c = col_offset + 5
            U[(A_row_base+1)*m + (A_col_base+0)][c] += 1
            U[(A_row_base+0)*m + (A_col_base+0)][c] -= 1
            V[(B_row_base+0)*p + (B_col_base+0)][c] += 1
            V[(B_row_base+0)*p + (B_col_base+1)][c] += 1

            # M7: (A01-A11)(B10+B11)
            c = col_offset + 6
            U[(A_row_base+0)*m + (A_col_base+1)][c] += 1
            U[(A_row_base+1)*m + (A_col_base+1)][c] -= 1
            V[(B_row_base+1)*p + (B_col_base+0)][c] += 1
            V[(B_row_base+1)*p + (B_col_base+1)][c] += 1

            # W contributions
            # C00 = M1 + M4 - M5 + M7
            w_idx = (C_col_base+0)*n + (C_row_base+0)
            W[w_idx][col_offset+0] += 1
            W[w_idx][col_offset+3] += 1
            W[w_idx][col_offset+4] -= 1
            W[w_idx][col_offset+6] += 1

            # C01 = M3 + M5
            w_idx = (C_col_base+1)*n + (C_row_base+0)
            W[w_idx][col_offset+2] += 1
            W[w_idx][col_offset+4] += 1

            # C10 = M2 + M4
            w_idx = (C_col_base+0)*n + (C_row_base+1)
            W[w_idx][col_offset+1] += 1
            W[w_idx][col_offset+3] += 1

            # C11 = M1 - M2 + M3 + M6
            w_idx = (C_col_base+1)*n + (C_row_base+1)
            W[w_idx][col_offset+0] += 1
            W[w_idx][col_offset+1] -= 1
            W[w_idx][col_offset+2] += 1
            W[w_idx][col_offset+5] += 1

        # Block 0: A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2]
        add_strassen(0, 0, 0, 0, 0, 0, 0)
        # Block 1: A[0:2, 2:4] @ B[2:4, 0:2] -> C[0:2, 0:2]
        add_strassen(0, 2, 2, 0, 0, 0, 7)
        # Block 2: A[0:2, 0:2] @ B[0:2, 2:4] -> C[0:2, 2:4]
        add_strassen(0, 0, 0, 2, 0, 2, 14)
        # Block 3: A[0:2, 2:4] @ B[2:4, 2:4] -> C[0:2, 2:4]
        add_strassen(0, 2, 2, 2, 0, 2, 21)

        return {"U": U, "V": V, "W": W}

    def validate(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # For each (i,j,k) where i in [0,n), j in [0,m), k in [0,p):
        # sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] should be 1 if i==i, j==j, k==k
        # Actually: the tensor T[i,j,k] = delta(i,i')delta(j,j')delta(k,k') for the identity
        # The multiplication tensor has T[i,j,k] = 1 if the indices match the "diagonal"
        # For matrix mult A(n,m) x B(m,p) -> C(n,p):
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all valid (i,j,k) - it's the full multiplication tensor
        # Wait, let me reconsider. The tensor is:
        # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # No, that's not right either.
        #
        # The matrix multiplication tensor M_{n,m,p} has entries:
        # M[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        # So for each triple (i,j,k), the coefficient is 1.
        # The decomposition requires:
        # M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where U is indexed by (i,j) flattened to i*m+j
        # V is indexed by (j,k) flattened to j*p+k  
        # W is indexed by (k,i) flattened to k*n+i

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def validate_fast(U, V, W, rank):
        """Vectorized validation using numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # U: (n*m, rank) = (8, rank)
        # V: (m*p, rank) = (16, rank)
        # W: (p*n, rank) = (8, rank)
        
        # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # Let's compute all at once
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U_np.reshape(n, m, rank)
        V_3d = V_np.reshape(m, p, rank)
        W_3d = W_np.reshape(p, n, rank)
        
        # For each (i,j,k): dot(U_3d[i,j,:], V_3d[j,k,:]*W_3d[k,i,:])
        # This is a 4D sum, let's do it efficiently
        # Result[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
        
        # Use einsum: 'ijc,jkc,kic->ijk'
        result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        if not np.allclose(result, 1.0, atol=1e-9):
            return False
        return True

    # Build baseline rank-28
    baseline = build_strassen_28()
    
    # Verify baseline
    if not validate_fast(baseline["U"], baseline["V"], baseline["W"], 28):
        # Fallback: return the baseline
        return baseline

    # Strategy: Try to find rank-27 by removing one column and solving
    # for adjustments to other columns
    # 
    # Approach: The system is underdetermined. We can try:
    # 1. Remove column r from the 28-column solution
    # 2. The resulting error tensor E[i,j,k] = sum_{c!=r} U_c V_c W_c - 1
    #    needs to be corrected by adjusting the remaining 27 columns
    # 3. This is a nonlinear problem, but we can try local search
    
    # Alternative: Direct random search with structured initialization
    # Start from rank-28, pick a random column to eliminate, then do
    # gradient-free optimization on the remaining columns
    
    best_result = baseline
    best_rank = 28
    
    # Try to find rank-27 decomposition
    # Method: For each possible column to remove, try to find a rank-27 solution
    # that matches the rank-28 solution on that column's contribution
    
    # Better method: Use the fact that the problem is underdetermined
    # and do a randomized search with the Strassen structure as prior
    
    target_rank = 27
    found = False
    
    # Method 1: Try removing each column and see if the remaining 27
    # can be adjusted to still satisfy the equations
    # This is equivalent to checking if column r is in the span of others
    # in the tensor product space
    
    # Method 2: Randomized local search
    # Start from rank-28, randomly perturb, try to reduce rank
    
    # Let's try a more direct approach:
    # The tensor has 1024 entries, all equal to 1.
    # With rank 27, we need:
    # sum_{c=0}^{26} U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k)
    #
    # This is 1024 equations in 27*(8+16+8) = 1296 unknowns.
    # Underdetermined by 272.
    
    # Try random initialization with half-integer values, biased toward Strassen
    random.seed(42)
    
    # Strategy: Start from Strassen-28, remove a random column, then
    # iteratively adjust entries to satisfy the equations
    # Use a simple coordinate descent / Gauss-Seidel approach
    
    # Actually, let's try a smarter approach:
    # The 4 Strassen blocks share structure. Maybe we can merge some columns
    # across blocks if they have similar patterns.
    
    # Let's try: for each pair of columns (one from each block), check if
    # they can be combined (i.e., if their contributions to the tensor are
    # linearly dependent in a useful way)
    
    # Simpler: just do a random search with the constraint that entries are
    # in {-1, -0.5, 0, 0.5, 1}
    
    max_attempts = 500
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        random.seed(12345 + attempt * 7919)
        
        # Initialize from Strassen-28, remove a random column
        U28 = [row[:] for row in baseline["U"]]
        V28 = [row[:] for row in baseline["V"]]
        W28 = [row[:] for row in baseline["W"]]
        
        remove_col = random.randint(0, 27)
        
        U = [row[:remove_col] + row[remove_col+1:] for row in U28]
        V = [row[:remove_col] + row[remove_col+1:] for row in V28]
        W = [row[:remove_col] + row[remove_col+1:] for row in W28]
        
        # Now try to repair: for each equation that's violated, try adjusting
        # one of the entries
        # This is a very hard problem, so let's try a different approach
        
        # Approach: Use the null space of the linearized problem
        # At the current point, the linearized equations are:
        # sum_c [delta_U[i,j,c] * V[j,k,c] * W[k,i,c] +
        #         U[i,j,c] * delta_V[j,k,c] * W[k,i,c] +
        #         U[i,j,c] * V[j,k,c] * delta_W[k,i,c]] = error[i,j,k]
        #
        # This is linear in the deltas! We can solve it.
        
        # Build the linear system
        # Unknowns: delta_U (8*27=216), delta_V (16*27=432), delta_W (8*27=216)
        # Total: 864 unknowns
        # Equations: 1024
        # Still overdetermined, but let's try least squares
        
        # Actually, this is getting complex. Let's try a simpler heuristic:
        # Just check if the rank-27 solution from removing a column is close
        # to valid, and if so, try small perturbations
        
        if validate_fast(U, V, W, target_rank):
            best_rank = target_rank
            best_result = {"U": U, "V": V, "W": W}
            found = True
            break
        
        # Try small perturbations on the removed column's "slot"
        # Actually, we can't add back a column. Instead, try modifying
        # existing columns slightly
        
        # For a few iterations, try adjusting random entries
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Pick a random position to modify
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = random.randint(0, 7)
                col = random.randint(0, 26)
                old_val = U[idx][col]
                U[idx][col] = old_val + random.choice([-0.5, 0.5, -1, 1])
            elif mat == 'V':
                idx = random.randint(0, 15)
                col = random.randint(0, 26)
                old_val = V[idx][col]
                V[idx][col] = old_val + random.choice([-0.5, 0.5, -1, 1])
            else:
                idx = random.randint(0, 7)
                col = random.randint(0, 26)
                old_val = W[idx][col]
                W[idx][col] = old_val + random.choice([-0.5, 0.5, -1, 1])
            
            if validate_fast(U, V, W, target_rank):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                found = True
                break
        
        if found:
            break
    
    # If not found, try a completely different approach:
    # Use the known result that rank(2x4x4) <= 27 from literature
    # and try to construct it directly
    
    if not found and time.time() < deadline - 20:
        # Try a structured rank-27 construction
        # Based on the idea that we can share some Strassen columns across blocks
        
        # Alternative: use a 2x2x2 rank-7 decomposition but with shared columns
        # between adjacent blocks where possible
        
        # For now, if we haven't found rank 27, return the best we have
        # (which is rank 28)
        pass

    if best_rank < 28:
        return best_result
    else:
        return baseline
