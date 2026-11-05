import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with rank 56 (8 independent Strassen blocks)
    rank = 56
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen 2x2x2 decomposition
    # U[i,j,r] * V[j,k,r] * W[k,i,r] = C[i,k]
    strassen_U = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(1,0): 1, (1,1): 1},   # r=1
        {(0,0): 1},              # r=2
        {(1,1): 1},              # r=3
        {(0,0): 1, (0,1): 1},   # r=4
        {(1,0): 1, (0,0): -1},  # r=5
        {(0,1): 1, (1,1): -1},  # r=6
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,0): 1},              # r=1
        {(0,1): 1, (1,1): -1},  # r=2
        {(1,0): 1, (0,0): -1},  # r=3
        {(1,1): 1},              # r=4
        {(0,0): 1, (0,1): 1},   # r=5
        {(1,0): 1, (1,1): 1},   # r=6
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,1): 1, (1,1): -1},  # r=1
        {(1,0): 1, (1,1): 1},   # r=2
        {(0,0): 1, (0,1): 1},   # r=3
        {(0,0): -1, (1,0): 1},  # r=4
        {(1,1): 1},              # r=5
        {(0,0): 1},              # r=6
    ]
    
    def add_block(base_rank, i_base, j_base, k_base):
        """Add a Strassen decomposition for A[i_base:i_base+2, j_base:j_base+2] * 
           B[j_base:j_base+2, k_base:k_base+2] starting at rank index base_rank."""
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * 4 + gj, cr] += val
            
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * 8 + gk, cr] += val
            
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * 2 + gi, cr] += val
    
    # A is 2x4, B is 4x8
    # Partition: A = [A0 | A1] where A0 is cols 0,1 and A1 is cols 2,3
    # B = [B_top; B_bottom] where B_top is rows 0,1 and B_bottom is rows 2,3
    # C = A0 * B_top + A1 * B_bottom
    
    # B_top is 2x8, partition into 4 2x2 blocks: [Bt0 | Bt1 | Bt2 | Bt3]
    # B_bottom is 2x8, partition into 4 2x2 blocks: [Bb0 | Bb1 | Bb2 | Bb3]
    
    # A0 * Bt0: i_base=0, j_base=0, k_base=0
    add_block(0, 0, 0, 0)
    
    # A0 * Bt1: i_base=0, j_base=0, k_base=2
    add_block(7, 0, 0, 2)
    
    # A0 * Bt2: i_base=0, j_base=0, k_base=4
    add_block(14, 0, 0, 4)
    
    # A0 * Bt3: i_base=0, j_base=0, k_base=6
    add_block(21, 0, 0, 6)
    
    # A1 * Bb0: i_base=0, j_base=2, k_base=0
    add_block(28, 0, 2, 0)
    
    # A1 * Bb1: i_base=0, j_base=2, k_base=2
    add_block(35, 0, 2, 2)
    
    # A1 * Bb2: i_base=0, j_base=2, k_base=4
    add_block(42, 0, 2, 4)
    
    # A1 * Bb3: i_base=0, j_base=2, k_base=6
    add_block(49, 0, 2, 6)
    
    # Try to reduce rank by finding redundant columns
    # The idea: if some column of the tensor can be expressed as a linear combination
    # of other columns, we can eliminate it
    
    # Convert to tensor form: T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to find if we can reduce rank by finding linear dependencies
    
    deadline = time.time() + 120
    
    def verify_rank(U, V, W, rank):
        """Verify that the decomposition is valid for the given rank."""
        # Build the tensor
        T = np.zeros((n, m, p), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
        
        # Check against direct matrix multiplication
        # We need to check that T[i,j,k] = (A @ B)[i,k] for all i,j,k
        # But A and B are variable, so we check the structure
        # The matrix multiplication tensor satisfies: T[i,j,k] = delta[j,j] (i.e., it's the identity in j)
        # More precisely, the tensor for matrix multiplication is T[i,j,k] = 1 if j==j else 0
        # Wait, that's not right. The tensor is T[i,j,k] such that C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k]
        # So T[i,j,k] = 1 for all i,j,k? No, that's not right either.
        # 
        # The matrix multiplication tensor is defined as:
        # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k ⊗ (A[i,:] · B[:,k])
        # Actually, the standard definition is:
        # The bilinear map (A,B) -> A*B can be written as sum_r U_r(A) * V_r(B) * W_r
        # where W_r is a matrix.
        # 
        # The tensor is T[i,j,k] = 1 for all i,j,k? No.
        # Let me think again. We have C[i,k] = sum_j A[i,j] B[j,k].
        # This can be written as C = sum_r (U_r(A) * V_r(B)) * W_r where W_r is a p x n matrix.
        # The tensor T is defined such that C[i,k] = sum_{j,r} A[i,j] B[j,k] U[i,j,r] V[j,k,r] W[k,i,r]
        # For this to equal sum_j A[i,j] B[j,k], we need:
        # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{i,i'} ... no.
        # 
        # Actually, the condition is:
        # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
        # because C[i,k] = sum_j A[i,j] B[j,k] * 1
        # 
        # Wait, that's only if the tensor is the identity in j. Let me reconsider.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_j sum_r A[i,j] U[i,j,r] * B[j,k] V[j,k,r] * W[k,i,r]
        # For this to hold for all A, B, we need:
        # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k
        # 
        # Yes, that's the condition.
        
        # Check the condition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try greedy elimination: for each column r, check if removing it still gives a valid decomposition
    # (i.e., can we express column r as a linear combination of other columns?)
    
    # Actually, a better approach: try to find linear dependencies in the rank-56 decomposition
    # by looking at the structure of the 8 independent Strassen blocks.
    
    # Each Strassen block uses 7 ranks. If we can share some ranks between blocks, we can reduce.
    
    # Let's try a different approach: use the known result that the rank of 2x4x8 matrix multiplication
    # is at most 56, and try to find a lower bound by searching for a valid decomposition with lower rank.
    
    # For now, let's try to eliminate columns that are zero or redundant.
    
    # First, check which columns are actually used (non-zero)
    used_cols = set()
    for r in range(rank):
        if np.any(U[:, r] != 0) or np.any(V[:, r] != 0) or np.any(W[:, r] != 0):
            used_cols.add(r)
    
    # All 56 columns should be used in the Strassen decomposition.
    
    # Try to find if any column can be expressed as a linear combination of others
    # by solving a linear system.
    
    # For a given column r0, we want to find coefficients c_r such that:
    # U[:, r0] = sum_{r != r0} c_r * U[:, r]
    # V[:, r0] = sum_{r != r0} c_r * V[:, r]
    # W[:, r0] = sum_{r != r0} c_r * W[:, r]
    
    # This is a linear system. Let's set up the augmented matrix.
    
    def try_eliminate_column(U, V, W, rank, r0):
        """Try to eliminate column r0 by expressing it as a linear combination of other columns."""
        # Build the linear system
        # We have 3*(nm + mp + pn) equations
        # U part: nm equations
        # V part: mp equations
        # W part: pn equations
        
        # Set up A_mat * c = b
        # c has rank-1 unknowns (all r != r0)
        
        other_cols = [r for r in range(rank) if r != r0]
        num_unknowns = len(other_cols)
        
        if num_unknowns == 0:
            return None
        
        # Build the system
        A_mat = np.zeros((n*m + m*p + p*n, num_unknowns), dtype=np.float64)
        b = np.zeros(n*m + m*p + p*n, dtype=np.float64)
        
        # U part
        for i in range(n*m):
            for idx, r in enumerate(other_cols):
                A_mat[i, idx] = U[i, r]
            b[i] = U[i, r0]
        
        # V part
        for i in range(m*p):
            for idx, r in enumerate(other_cols):
                A_mat[n*m + i, idx] = V[i, r]
            b[n*m + i] = V[i, r0]
        
        # W part
        for i in range(p*n):
            for idx, r in enumerate(other_cols):
                A_mat[n*m + m*p + i, idx] = W[i, r]
            b[n*m + m*p + i] = W[i, r0]
        
        # Solve using least squares
        try:
            c, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b, rcond=None)
        except:
            return None
        
        # Check if the solution is good
        # If the residual is small, then column r0 can be eliminated
        if len(residuals) > 0 and residuals[0] < 1e-6:
            return c
        return None
    
    # Try to eliminate columns
    new_rank = rank
    for r0 in range(rank - 1, -1, -1):
        if time.time() > deadline - 5:
            break
        
        c = try_eliminate_column(U, V, W, new_rank, r0)
        if c is not None:
            # Eliminate column r0
            other_cols = [r for r in range(new_rank) if r != r0]
            for idx, r in enumerate(other_cols):
                U[:, r] += c[idx] * U[:, r0]
                V[:, r] += c[idx] * V[:, r0]
                W[:, r] += c[idx] * W[:, r0]
            
            # Remove column r0
            keep = [r for r in range(new_rank) if r != r0]
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
            new_rank -= 1
    
    # Verify the final decomposition
    if not verify_rank(U, V, W, new_rank):
        # Fallback to the original rank-56 decomposition
        U = np.zeros((n * m, 56), dtype=np.float64)
        V = np.zeros((m * p, 56), dtype=np.float64)
        W = np.zeros((p * n, 56), dtype=np.float64)
        
        add_block(0, 0, 0, 0)
        add_block(7, 0, 0, 2)
        add_block(14, 0, 0, 4)
        add_block(21, 0, 0, 6)
        add_block(28, 0, 2, 0)
        add_block(35, 0, 2, 2)
        add_block(42, 0, 2, 4)
        add_block(49, 0, 2, 6)
        
        new_rank = 56
    
    # Convert to lists for the return interface
    U_list = [[U[i, j] for j in range(new_rank)] for i in range(n * m)]
    V_list = [[V[i, j] for j in range(new_rank)] for i in range(m * p)]
    W_list = [[W[i, j] for j in range(new_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
