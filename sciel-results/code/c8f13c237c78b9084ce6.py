"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s margin before 150s kill
    
    # Default rank-11 solution (Strassen for 2x2 x 2x2 + schoolbook for col 2)
    default_rank = 11
    U_def = [[0] * default_rank for _ in range(n * m)]
    V_def = [[0] * default_rank for _ in range(m * p)]
    W_def = [[0] * default_rank for _ in range(p * n)]
    
    U_def[0][0] = 1; U_def[3][0] = 1
    V_def[0][0] = 1; V_def[4][0] = 1
    U_def[2][1] = 1; U_def[3][1] = 1
    V_def[0][1] = 1
    U_def[0][2] = 1
    V_def[1][2] = 1; V_def[4][2] = -1
    U_def[3][3] = 1
    V_def[3][3] = 1; V_def[0][3] = -1
    U_def[0][4] = 1; U_def[1][4] = 1
    V_def[4][4] = 1
    U_def[2][5] = 1; U_def[0][5] = -1
    V_def[0][5] = 1; V_def[1][5] = 1
    U_def[1][6] = 1; U_def[3][6] = -1
    V_def[3][6] = 1; V_def[4][6] = 1
    W_def[0][0] = 1; W_def[0][3] = 1; W_def[0][4] = -1; W_def[0][6] = 1
    W_def[2][2] = 1; W_def[2][4] = 1
    W_def[1][1] = 1; W_def[1][3] = 1
    W_def[3][0] = 1; W_def[3][1] = -1; W_def[3][2] = 1; W_def[3][5] = 1
    U_def[0][7] = 1
    V_def[2][7] = 1
    U_def[1][8] = 1
    V_def[5][8] = 1
    U_def[2][9] = 1
    V_def[2][9] = 1
    U_def[3][10] = 1
    V_def[5][10] = 1
    W_def[4][7] = 1; W_def[4][8] = 1
    W_def[5][9] = 1; W_def[5][10] = 1
    
    def verify(U, V, W, rank):
        """Check if the decomposition exactly reproduces matrix multiplication."""
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    # C[i][j] should get contribution U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                    # Actually: C[i][j] = sum_k A[i][k]*B[k][j]
                    # Tensor: T[i,j,k] = sum_r U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                    # T[i,j,k] should be 1 if i==i, j==j, k==k (identity for matmul)
                    # Actually T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k}... no
                    # The matrix multiplication tensor: T[i,j,k] = 1 iff we're computing C[i][j] from A[i][k]*B[k][j]
                    # So T[i,j,k] = 1 for all valid (i,j,k) - it's the identity tensor
                    expected = 1
                    actual = 0
                    for r in range(rank):
                        actual += U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                    if actual != expected:
                        return False
        return True
    
    # Verify default
    if verify(U_def, V_def, W_def, default_rank):
        # Try to find rank-10 by removing one column and adjusting
        # Strategy: for each column c to remove, solve for adjusted coefficients
        # in remaining columns using the sparse support pattern
        
        # Extract support pattern from default solution
        # For each position (row, col), note which columns are non-zero
        U_support = set()
        for i in range(n*m):
            for r in range(default_rank):
                if U_def[i][r] != 0:
                    U_support.add((i, r))
        V_support = set()
        for i in range(m*p):
            for r in range(default_rank):
                if V_def[i][r] != 0:
                    V_support.add((i, r))
        W_support = set()
        for i in range(p*n):
            for r in range(default_rank):
                if W_def[i][r] != 0:
                    W_support.add((i, r))
        
        # Try each column removal
        for col_to_remove in range(default_rank):
            if time.time() > deadline:
                break
            
            remaining_cols = [c for c in range(default_rank) if c != col_to_remove]
            new_rank = len(remaining_cols)
            
            # Build variable index mapping
            # Variables: U[i][r] for i in range(n*m), r in remaining_cols (only where support exists)
            # V[i][r] for i in range(m*p), r in remaining_cols (only where support exists)  
            # W[i][r] for i in range(p*n), r in remaining_cols (only where support exists)
            
            U_vars = {}
            var_idx = 0
            for i in range(n*m):
                for r in remaining_cols:
                    if (i, r) in U_support:
                        U_vars[(i, r)] = var_idx
                        var_idx += 1
            
            V_vars = {}
            for i in range(m*p):
                for r in remaining_cols:
                    if (i, r) in V_support:
                        V_vars[(i, r)] = var_idx
                        var_idx += 1
            
            W_vars = {}
            for i in range(p*n):
                for r in remaining_cols:
                    if (i, r) in W_support:
                        W_vars[(i, r)] = var_idx
                        var_idx += 1
            
            total_vars = var_idx
            
            # Build linear system: for each (i,j,k), sum_r U[i*m+k][r]*V[k*p+j][r]*W[j*n+i][r] = 1
            # This is nonlinear (trilinear). Use the structure: fix two, solve for third.
            # Alternative: use the fact that we can scale columns.
            # If we remove column c, we need to redistribute its contribution.
            # 
            # Simpler approach: try all combinations of which column to merge into which other
            # by checking if the tensor equations can be satisfied.
            
            # Try: for each pair (c_remove, c_target), see if we can absorb column c_remove
            # into column c_target by adjusting W entries.
            # The key insight: if columns c1 and c2 have the same U and V patterns
            # (up to scaling), they can be merged by adjusting W.
            
            # Check if removing column c is possible by seeing if its U,V pattern
            # is a linear combination of remaining columns' patterns.
            
            # Build U column vectors for remaining columns
            U_cols = {}
            for r in remaining_cols:
                U_cols[r] = [U_def[i][r] for i in range(n*m)]
            
            V_cols = {}
            for r in remaining_cols:
                V_cols[r] = [V_def[i][r] for i in range(m*p)]
            
            # For the removed column, check if its U vector is in span of remaining U vectors
            # AND its V vector is in span of remaining V vectors
            # AND the W adjustment works out.
            
            # This is getting complex. Try a more direct search:
            # Randomly perturb and project to integer solutions.
            
            if time.time() > deadline:
                break
        
        # Try randomized search for rank-10
        if time.time() < deadline - 30:
            best_found = None
            np.random.seed(42)
            
            for attempt in range(500):
                if time.time() > deadline - 10:
                    break
                
                # Start from default, zero out one column, try to fix
                col_to_zero = np.random.randint(default_rank)
                U_try = [row[:] for row in U_def]
                V_try = [row[:] for row in V_def]
                W_try = [row[:] for row in W_def]
                
                # Zero out column col_to_zero
                for i in range(n*m):
                    U_try[i][col_to_zero] = 0
                for i in range(m*p):
                    V_try[i][col_to_zero] = 0
                for i in range(p*n):
                    W_try[i][col_to_zero] = 0
                
                # Now try to fix by adjusting other columns
                # Use iterative refinement: for each (i,j,k) equation, adjust
                # This is a trilinear system - hard to solve directly.
                
                # Alternative: try small integer perturbations to W
                # to compensate for the missing column's contribution.
                
                # The missing column contributed:
                # For each (i,j,k): delta = U_def[i*m+k][col_to_zero] * V_def[k*p+j][col_to_zero] * W_def[j*n+i][col_to_zero]
                # We need to cancel this delta using other columns.
                
                # Check if delta is zero for all (i,j,k) (i.e., column was redundant)
                redundant = True
                for i in range(n):
                    for j in range(p):
                        for k in range(m):
                            delta = U_def[i*m+k][col_to_zero] * V_def[k*p+j][col_to_zero] * W_def[j*n+i][col_to_zero]
                            if delta != 0:
                                redundant = False
                                break
                        if not redundant:
                            break
                    if not redundant:
                        break
                
                if redundant:
                    # Great! Column was redundant, we have a rank-10 solution
                    rank_10 = default_rank - 1
                    U_10 = [[U_try[i][r] for r in range(rank_10)] for i in range(n*m)]
                    V_10 = [[V_try[i][r] for r in range(rank_10)] for i in range(m*p)]
                    W_10 = [[W_try[i][r] for r in range(rank_10)] for i in range(p*n)]
                    
                    # Verify
                    if verify(U_10, V_10, W_10, rank_10):
                        best_found = (U_10, V_10, W_10, rank_10)
                        break
                
                if time.time() > deadline - 10:
                    break
            
            if best_found:
                U, V, W, rank = best_found
                return {"U": U, "V": V, "W": W}
        
        # Return default rank-11
        return {"U": U_def, "V": V_def, "W": W_def}
    else:
        # Should not happen, but return default anyway
        return {"U": U_def, "V": V_def, "W": W_def}
# EVOLVE-BLOCK-END
