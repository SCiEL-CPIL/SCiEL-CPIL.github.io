"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Precompute tensor indices
    tensor_indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def verify(U_list, V_list, W_list, rank):
        """Verify exact decomposition with integer/half-integer entries."""
        for (i, j, k) in tensor_indices:
            val = 0.0
            for c in range(rank):
                val += U_list[i * m + j][c] * V_list[j * p + k][c] * W_list[k * n + i][c]
            if abs(val - 1.0) > 1e-9:
                return False
        return True
    
    def verify_fast(U, V, W, rank):
        """Fast verification using numpy broadcasting."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # For each (i,j,k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        for (i, j, k) in tensor_indices:
            u_row = U[i * m + j, :]
            v_row = V[j * p + k, :]
            w_row = W[k * n + i, :]
            val = np.dot(u_row, v_row * w_row)
            if abs(val - 1.0) > 1e-9:
                return False
        return True
    
    # Strategy: Tensor train / MPS-inspired structured search
    # The multiplication tensor T[i,j,k] = delta_{i,j,k} has structure:
    # For 3x3x8, we can exploit that the last dimension (k) has 8 components
    # which can be decomposed into blocks.
    
    # Approach: Fix U and V with structured patterns, solve for W via linear system
    # Then round to half-integers and verify
    
    def try_structured_decomposition(rank, seed_offset):
        """Try to find a decomposition of given rank using structured U, V."""
        if time.time() > deadline:
            return None
        
        np.random.seed(seed_offset)
        
        # Generate structured U and V with some zeros for sparsity
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        
        # Pattern 1: Block-diagonal structure in U and V
        # Each (i,j) pair gets a subset of columns
        cols_per_ij = rank // (n * m)
        remainder = rank % (n * m)
        
        col = 0
        for i in range(n):
            for j in range(m):
                count = cols_per_ij + (1 if (i * m + j) < remainder else 0)
                for c in range(col, col + count):
                    U[i * m + j, c] = 1.0
                    V[j * p + (c % p), c] = 1.0
                col += count
        
        # Now solve for W: For each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is linear in W entries
        # Build linear system: A @ w_flat = b
        # where w_flat has p*n*rank entries
        
        # Actually, let's use a different approach:
        # For each column c, define the contribution as outer product
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # Let's try random sparse U, V and solve for W
        U = np.random.randint(-2, 3, (n * m, rank)) * 0.5
        V = np.random.randint(-2, 3, (m * p, rank)) * 0.5
        
        # Build linear system for W
        # Unknowns: W[k*n+i, c] for k in range(p), i in range(n), c in range(rank)
        # Total unknowns: p * n * rank
        num_unknowns = p * n * rank
        
        # Equations: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # num_equations = n * m * p
        num_equations = n * m * p
        
        # Build A matrix (num_equations x num_unknowns)
        A = np.zeros((num_equations, num_unknowns))
        b = np.ones(num_equations)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        # W index: k*n + i, column c
                        w_idx = (k * n + i) * rank + c
                        A[eq_idx, w_idx] = U[i * m + j, c] * V[j * p + k, c]
                    eq_idx += 1
        
        # Solve least squares
        try:
            W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
            W = W_flat.reshape((p * n, rank))
        except:
            return None
        
        # Round to half-integers
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        if verify_fast(U, V, W_rounded, rank):
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W_rounded]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try tensor-train-inspired structured decompositions
    # Key insight: The 3x3x8 tensor can be viewed as 3 copies of 3x3x~2.67
    # or we can exploit the block structure
    
    # Strategy: Use the fact that for matrix multiplication, there are known
    # submultiplicative bounds and structured decompositions
    
    # Try ranks from 71 down to 50
    for rank in range(best_rank - 1, 49, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random seeds
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            result = try_structured_decomposition(rank, attempt * 1000 + rank)
            if result is not None:
                best_rank = rank
                best_result = result
                found = True
                break
        
        if found:
            # Found a valid decomposition, try to go lower
            continue
    
    # Additional strategy: Direct search with better initialization
    # Use the structure that U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # suggests that for each c, we have a "rank-1" contribution
    
    # Try to find decompositions where each column c corresponds to a
    # specific pattern of (i,j,k) indices
    
    if time.time() < deadline - 10:
        # Try combinatorial structured patterns
        for rank in range(min(best_rank - 1, 60), 49, -1):
            if time.time() > deadline:
                break
            
            # Pattern: Each column c selects a subset of (i,j,k) indices
            # where the product U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # and is 0 elsewhere
            
            # This is equivalent to finding a set of "rectangles" that cover
            # the tensor exactly once
            
            # For 3x3x8, try patterns based on modular arithmetic
            for pattern_seed in range(5):
                if time.time() > deadline:
                    break
                
                np.random.seed(pattern_seed * 7 + rank)
                
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                # Assign each of the 72 tensor entries to one of the rank columns
                # such that each column gets a "consistent" pattern
                tensor_to_col = {}
                col_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Use a structured assignment
                            # Group by (i, j) and spread across k
                            col = (i * m + j + k * 3) % rank
                            tensor_to_col[(i, j, k)] = col
                
                # For each column, determine which (i,j,k) it covers
                col_covers = {c: [] for c in range(rank)}
                for (i, j, k), c in tensor_to_col.items():
                    col_covers[c].append((i, j, k))
                
                # For each column c, we need to assign U, V, W values
                # such that the product is 1 for covered indices and 0 for others
                # This is hard in general, but let's try simple values
                
                valid = True
                for c in range(rank):
                    if not col_covers[c]:
                        continue
                    
                    # Get all (i,j,k) covered by this column
                    covered = col_covers[c]
                    
                    # For each i in {0,1,2}, j in {0,1,2}, k in {0,...,7}
                    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be:
                    # 1 if (i,j,k) in covered, 0 otherwise
                    
                    # Verify the current assignment works
                    for (i, j, k) in tensor_indices:
                        expected = 1.0 if (i, j, k) in covered else 0.0
                        actual = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if abs(actual - expected) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    if verify(U_list, V_list, W_list, rank):
                        best_rank = rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        break
    
    return best_result
# EVOLVE-BLOCK-END
