"""Matrix multiplication 2x3 x 3x9 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    Uses a block structure: split 9 into three 3s, find low-rank 2x3x3 decomp, tile it.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    def verify_full(U, V, W, n, m, p):
        """Verify the decomposition is exact."""
        rank = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(total - 1.0) > 1e-6:
                        return False
        return True
    
    def verify_block(U_b, V_b, W_b, n, m, p_block):
        """Verify a block decomposition."""
        rank = len(U_b[0])
        U_b = np.array(U_b, dtype=float)
        V_b = np.array(V_b, dtype=float)
        W_b = np.array(W_b, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    total = np.sum(U_b[i*m+j] * V_b[j*p_block+k] * W_b[k*n+i])
                    if abs(total - 1.0) > 1e-6:
                        return False
        return True
    
    def schoolbook_full(n, m, p):
        """Return schoolbook rank-nmp decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a rank-5 decomposition for 2x3x3
    # Using random initialization + gradient-free search
    best_block_rank = None
    best_block = None
    
    # First, try a structured rank-5 decomposition
    # The 2x3x3 tensor: all 18 entries are 1
    # Let's try a specific structured form
    
    # Approach: Use the fact that for 2x3x3, we can potentially share structure
    # between the two rows of the output.
    
    # Let me try to solve this using numpy's least squares with a structured ansatz.
    # For rank r, we have r terms. Each term c has:
    #   U[:,c] in R^6, V[:,c] in R^9, W[:,c] in R^6
    # Constraint: for all (i,j,k), sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*2+i,c] = 1
    
    # This is a system of 18 equations in 6r + 9r + 6r = 21r unknowns.
    # For r=5: 105 unknowns, 18 equations. Very underdetermined.
    # For r=4: 84 unknowns, 18 equations.
    # For r=3: 63 unknowns, 18 equations.
    
    # The question is whether a solution exists with the multiplicative constraint.
    
    # Let me try a direct numerical approach for small ranks.
    
    def try_rank_r(n, m, p_block, r, seed=42, max_iter=5000):
        """Try to find rank-r decomposition using iterative refinement."""
        np.random.seed(seed)
        
        # Initialize randomly
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p_block, r)
        W = np.random.randn(p_block*n, r)
        
        # Scale to reasonable magnitude
        U /= np.sqrt(r)
        V /= np.sqrt(r)
        W /= np.sqrt(r)
        
        # Iterative: fix two factors, solve for the third
        best_obj = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            # Fix V, W, solve for U
            # For each (i,j): sum_c U[i*m+j,c] * (V[j*p_block+k,c] * W[k*n+i,c] summed over k) = 1
            # Actually, for each (i,j,k): sum_c U[i*m+j,c] * V[j*p_block+k,c] * W[k*n+i,c] = 1
            # Fix V and W, for each (i,j): we have a linear system in U[i*m+j,:]
            # sum_c U[i*m+j,c] * V[j*p_block+k,c] * W[k*n+i,c] = 1 for each k
            # This is: U[i*m+j,:] @ diag(V[j*p_block+k,:] * W[k*n+i,:]) = 1 for each k
            # i.e., U[i*m+j,:] @ M_k = 1 where M_k[c] = V[j*p_block+k,c] * W[k*n+i,c]
            # So U[i*m+j,:] is a vector u such that u @ M_k = 1 for all k
            # This is a linear system: M^T @ u = 1 (where M has rows M_k)
            
            # Build the system for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # For this (i,j), we need: for each k, sum_c U[i*m+j,c]*V[j*p_block+k,c]*W[k*n+i,c] = 1
                    # Let A_k[c] = V[j*p_block+k,c] * W[k*n+i,c]
                    # We need U[i*m+j,:] @ A_k = 1 for all k
                    # This is a system: (A^T) @ U[i*m+j,:]^T = ones(p_block)
                    A = np.zeros((p_block, r))
                    for k in range(p_block):
                        A[k] = V[j*p_block+k] * W[k*n+i]
                    # Solve A^T @ u = ones(p_block)
                    # u is r-dimensional
                    try:
                        u, residuals, rank_s, sv = np.linalg.lstsq(A.T, np.ones(p_block), rcond=None)
                        U_new[i*m+j] = u
                    except:
                        U_new[i*m+j] = U[i*m+j]
            U = U_new
            
            # Fix U, W, solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p_block):
                    # For this (j,k), we need: for each i, sum_c U[i*m+j,c]*V[j*p_block+k,c]*W[k*n+i,c] = 1
                    # Let B_i[c] = U[i*m+j,c] * W[k*n+i,c]
                    # We need V[j*p_block+k,:] @ B_i = 1 for all i
                    B = np.zeros((n, r))
                    for i in range(n):
                        B[i] = U[i*m+j] * W[k*n+i]
                    try:
                        v, residuals, rank_s, sv = np.linalg.lstsq(B.T, np.ones(n), rcond=None)
                        V_new[j*p_block+k] = v
                    except:
                        V_new[j*p_block+k] = V[j*p_block+k]
            V = V_new
            
            # Fix U, V, solve for W
            W_new = np.zeros_like(W)
            for k in range(p_block):
                for i in range(n):
                    # For this (k,i), we need: for each j, sum_c U[i*m+j,c]*V[j*p_block+k,c]*W[k*n+i,c] = 1
                    # Let C_j[c] = U[i*m+j,c] * V[j*p_block+k,c]
                    # We need W[k*n+i,:] @ C_j = 1 for all j
                    C = np.zeros((m, r))
                    for j in range(m):
                        C[j] = U[i*m+j] * V[j*p_block+k]
                    try:
                        w, residuals, rank_s, sv = np.linalg.lstsq(C.T, np.ones(m), rcond=None)
                        W_new[k*n+i] = w
                    except:
                        W_new[k*n+i] = W[k*n+i]
            W = W_new
            
            # Check objective
            obj = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p_block):
                        val = np.sum(U[i*m+j] * V[j*p_block+k] * W[k*n+i])
                        obj += (val - 1.0)**2
            if obj < best_obj:
                best_obj = obj
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if obj < 1e-10:
                break
        
        # Check if solution is valid
        if best_obj < 1e-8:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from low to high for the 2x3x3 block
    block_p = 3
    found_block = None
    for r in [5, 6, 7, 8]:
        if time.time() > deadline - 30:
            break
        result = try_rank_r(n, m, block_p, r, seed=42+r, max_iter=3000)
        if result is not None:
            U_b, V_b, W_b = result
            if verify_block(U_b.tolist(), V_b.tolist(), W_b.tolist(), n, m, block_p):
                found_block = (r, U_b, V_b, W_b)
                break
    
    if found_block is not None:
        r_block, U_b, V_b, W_b = found_block
        num_blocks = p // block_p  # 3
        total_rank = num_blocks * r_block
        
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        for q in range(num_blocks):
            base_col = q * r_block
            for c in range(r_block):
                col = base_col + c
                for i in range(n):
                    for j in range(m):
                        U[i*m+j][col] = U_b[i*m+j][c]
                for j in range(m):
                    for k in range(block_p):
                        V[j*p + q*block_p + k][col] = V_b[j*block_p + k][c]
                for k in range(block_p):
                    for i in range(n):
                        W[(q*block_p + k)*n + i][col] = W_b[k*n + i][c]
        
        if verify_full(U, V, W, n, m, p):
            # Convert to exact rationals if possible
            return {"U": U, "V": V, "W": W}
    
    # Fallback: try a different block structure or schoolbook
    # Try 2x3x9 directly with moderate rank
    if time.time() < deadline - 20:
        for r in [15, 18, 20]:
            result = try_rank_r(n, m, p, r, seed=100+r, max_iter=2000)
            if result is not None:
                U_d, V_d, W_d = result
                if verify_full(U_d.tolist(), V_d.tolist(), W_d.tolist(), n, m, p):
                    return {"U": U_d.tolist(), "V": V_d.tolist(), "W": W_d.tolist()}
    
    # Final fallback: schoolbook
    return schoolbook_full(n, m, p)
# EVOLVE-BLOCK-END
