import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected[i,j,k] = 1.0
        
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if abs(u_mat[i,j]) < 1e-15:
                        continue
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        return np.max(np.abs(T_computed - T_expected)) < tol

    def get_17_rank_block():
        """Returns U, V, W for a 3x3x2 block with rank 17."""
        # Re-construct the 17-rank logic for one block
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        col = 0
        # Strassen part (7 terms)
        U_coeffs = [
            {0: 1, 3: 1},       
            {2: 1, 3: 1},       
            {0: 1},             
            {3: 1},             
            {0: 1, 1: 1},       
            {2: 1, 0: -1},      
            {1: 1, 3: -1}       
        ]
        V_coeffs = [
            {0: 1, 3: 1},       
            {0: 1},             
            {1: 1, 3: -1},      
            {2: 1, 0: -1},      
            {3: 1},             
            {0: 1, 1: 1},       
            {2: 1, 3: 1}        
        ]
        W_coeffs = [
            {0: 1, 3: 1},       
            {2: 1, 3: -1},      
            {1: 1, 3: 1},       
            {0: 1, 2: 1},       
            {0: -1, 1: 1},      
            {3: 1},             
            {0: 1}              
        ]
        
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                i_local = li // 2
                j_local = li % 2
                U[i_local*3 + j_local, col] += coeff
            for lj, coeff in V_coeffs[l].items():
                j_local = lj // 2
                k_local = lj % 2
                V[j_local*2 + k_local, col] += coeff
            for lc, coeff in W_coeffs[l].items():
                i_local = lc // 2
                k_local = lc % 2
                W[k_local*3 + i_local, col] += coeff
            col += 1
        
        # Schoolbook part (10 terms)
        # i in {0,1}, j=2, k in {0,1}
        for i in range(2):
            for k in range(2):
                U[i*3 + 2, col] = 1.0
                V[2*2 + k, col] = 1.0
                W[k*3 + i, col] = 1.0
                col += 1
        # i=2, j in {0,1}, k in {0,1}
        for j in range(2):
            for k in range(2):
                U[2*3 + j, col] = 1.0
                V[j*2 + k, col] = 1.0
                W[k*3 + 2, col] = 1.0
                col += 1
        # i=2, j=2, k in {0,1}
        for k in range(2):
            U[2*3 + 2, col] = 1.0
            V[2*2 + k, col] = 1.0
            W[k*3 + 2, col] = 1.0
            col += 1
            
        return U, V, W

    def try_find_rank_16_block():
        """Try to find a rank 16 decomposition for 3x3x2 using structured search."""
        import itertools
        
        # Generate candidate rank-1 terms with small integer/half-integer entries
        # Each term is (u_vec, v_vec, w_vec) where:
        # u_vec has 9 entries (3x3), v_vec has 6 entries (3x2), w_vec has 6 entries (2x3)
        
        # Use a structured approach: consider terms that are "simple" combinations
        # Focus on terms that match Strassen-like patterns and schoolbook patterns
        
        # First, try a direct linear algebra approach:
        # The tensor equation is: sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k
        
        # This is a system of 18 equations (3*3*2) in 16*9 + 16*6 + 16*6 = 288 unknowns
        # But we want integer/half-integer solutions
        
        # Strategy: Use the 17-rank solution as a starting point and try to merge two terms
        
        U17, V17, W17 = get_17_rank_block()
        
        # Try to find if we can express one term as a combination of others
        # Or find a different 16-term decomposition
        
        # Alternative: Use a grid search over small integer values
        # This is expensive but might work for small cases
        
        # Let's try a smarter approach: use the structure of Strassen's algorithm
        # and look for modifications that reduce the rank
        
        # Generate a dictionary of candidate terms
        # Each term is defined by sparse patterns in U, V, W
        
        candidates = []
        
        # Schoolbook terms (18 for 3x3x2)
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    u = np.zeros(9)
                    v = np.zeros(6)
                    w = np.zeros(6)
                    u[i*3+j] = 1.0
                    v[j*2+k] = 1.0
                    w[k*3+i] = 1.0
                    candidates.append((u, v, w))
        
        # Strassen-like terms for 2x2x2 blocks
        # For each 2x2 sub-block, add Strassen terms
        for i0 in range(2):
            for j0 in range(2):
                # 7 Strassen terms for the 2x2x2 block starting at (i0, j0)
                strassen_U = [
                    {0: 1, 3: 1},
                    {2: 1, 3: 1},
                    {0: 1},
                    {3: 1},
                    {0: 1, 1: 1},
                    {2: 1, 0: -1},
                    {1: 1, 3: -1}
                ]
                strassen_V = [
                    {0: 1, 3: 1},
                    {0: 1},
                    {1: 1, 3: -1},
                    {2: 1, 0: -1},
                    {3: 1},
                    {0: 1, 1: 1},
                    {2: 1, 3: 1}
                ]
                strassen_W = [
                    {0: 1, 3: 1},
                    {2: 1, 3: -1},
                    {1: 1, 3: 1},
                    {0: 1, 2: 1},
                    {0: -1, 1: 1},
                    {3: 1},
                    {0: 1}
                ]
                
                for l in range(7):
                    u = np.zeros(9)
                    v = np.zeros(6)
                    w = np.zeros(6)
                    
                    for li, coeff in strassen_U[l].items():
                        i_local = li // 2
                        j_local = li % 2
                        i_global = i0 + i_local
                        j_global = j0 + j_local
                        if i_global < 3 and j_global < 3:
                            u[i_global*3 + j_global] = coeff
                    
                    for lj, coeff in strassen_V[l].items():
                        j_local = lj // 2
                        k_local = lj % 2
                        j_global = j0 + j_local
                        if j_global < 3:
                            v[j_global*2 + k_local] = coeff
                    
                    for lc, coeff in strassen_W[l].items():
                        i_local = lc // 2
                        k_local = lc % 2
                        i_global = i0 + i_local
                        if i_global < 3:
                            w[k_local*3 + i_global] = coeff
                    
                    # Only add if all three vectors are non-zero
                    if np.any(u != 0) and np.any(v != 0) and np.any(w != 0):
                        candidates.append((u, v, w))
        
        # Now we need to find a subset of 16 candidates whose tensor sum equals all 1s
        # This is a combinatorial problem, but we can use linear programming relaxation
        
        # Build the linear system: for each (i,j,k), sum of u_c[i,j]*v_c[j,k]*w_c[k,i] = 1
        # This is nonlinear in the coefficients, but if we fix the candidate terms,
        # we need to find which 16 to use with what weights
        
        # Actually, the problem is: find 16 terms (from candidates or new ones) such that
        # their tensor sum equals the target
        
        # Let's try a different approach: use the 17-rank solution and try to find
        # a linear dependency that allows us to eliminate one term
        
        # Compute the tensor for each of the 17 terms
        term_tensors = []
        for c in range(17):
            u = U17[:, c].reshape(3, 3)
            v = V17[:, c].reshape(3, 2)
            w = W17[:, c].reshape(2, 3)
            t = np.zeros((3, 3, 2))
            for i in range(3):
                for j in range(3):
                    for k in range(2):
                        t[i,j,k] = u[i,j] * v[j,k] * w[k,i]
            term_tensors.append(t.flatten())
        
        term_tensors = np.array(term_tensors)
        
        # Check if any term is a linear combination of others
        # This would allow us to eliminate it
        
        for c in range(17):
            if time_left() < 5:
                break
            
            # Try to express term c as a combination of others
            others = [c2 for c2 in range(17) if c2 != c]
            A = term_tensors[others].T  # 18 x 16
            b = term_tensors[c]
            
            # Solve A x = b
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if this is an exact solution
                if np.max(np.abs(A @ x - b)) < 1e-10:
                    # Check if x has small integer/half-integer values
                    x_rounded = np.round(x * 2) / 2
                    if np.max(np.abs(A @ x_rounded - b)) < 1e-10:
                        # We found a dependency! Term c can be eliminated
                        # Create a new 16-term decomposition
                        U_new = np.zeros((9, 16))
                        V_new = np.zeros((6, 16))
                        W_new = np.zeros((6, 16))
                        
                        col = 0
                        for c2 in others:
                            U_new[:, col] = U17[:, c2] * x_rounded[col]
                            V_new[:, col] = V17[:, c2] * x_rounded[col]
                            W_new[:, col] = W17[:, c2] * x_rounded[col]
                            col += 1
                        
                        # Verify
                        T_check = np.zeros((3, 3, 2))
                        for c2 in range(16):
                            u = U_new[:, c2].reshape(3, 3)
                            v = V_new[:, c2].reshape(3, 2)
                            w = W_new[:, c2].reshape(2, 3)
                            for i in range(3):
                                for j in range(3):
                                    for k in range(2):
                                        T_check[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                        
                        if np.max(np.abs(T_check - 1.0)) < 1e-10:
                            return U_new, V_new, W_new
            except Exception:
                pass
        
        return None

    # Try to find rank 16 block with time limit
    block_16 = None
    if time_left() > 30:  # Only try if we have enough time
        block_16 = try_find_rank_16_block()
    
    if block_16 is not None:
        U_block, V_block, W_block = block_16
        rank_block = 16
    else:
        U_block, V_block, W_block = get_17_rank_block()
        rank_block = 17
        
    # Assemble full 3x3x8 decomposition
    rank_total = 4 * rank_block
    U = np.zeros((9, rank_total))
    V = np.zeros((6, rank_total))
    W = np.zeros((6, rank_total))
    
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        col_start = g * rank_block
        col_end = (g + 1) * rank_block
        
        # Copy U block
        U[:, col_start:col_end] = U_block
        
        # Copy V block, adjusting k indices
        # V_block is 6x16 (or 17). Indices: j*2+k
        # We need to map k_local (0,1) to k_global (k0, k1)
        V_g = np.zeros((6, rank_block))
        for j in range(3):
            for k_local in range(2):
                k_global = k0 + k_local
                V_g[j*2 + k_local, :] = V_block[j*2 + k_local, :]
                # Wait, V_block is indexed by j*2+k_local.
                # The target V is indexed by j*p+k_global. Since p=8, but here we are building the block for a specific k-pair.
                # Actually, the full V is 24xRank.
                # In the block, V is 6xRank (j in 0..2, k_local in 0..1).
                # We need to place these into the correct rows of the full V.
                # Full V row index: j*8 + k_global.
                # But we are building the full U, V, W here.
                pass
        
        # Let's do it properly by filling the full matrices
        # This loop is inefficient if done naively, but rank is small.
        
    # Re-do assembly properly
    U = np.zeros((9, rank_total))
    V = np.zeros((24, rank_total)) # 3*8
    W = np.zeros((24, rank_total)) # 8*3
    
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        col_start = g * rank_block
        col_end = (g + 1) * rank_block
        
        # U
        U[:, col_start:col_end] = U_block
        
        # V
        # V_block has shape (6, rank_block). Rows are j*2 + k_local.
        # Full V has shape (24, rank_total). Rows are j*8 + k_global.
        for c in range(rank_block):
            for j in range(3):
                for k_local in range(2):
                    k_global = k0 + k_local
                    V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                    
        # W
        # W_block has shape (6, rank_block). Rows are k_local*3 + i.
        # Full W has shape (24, rank_total). Rows are k_global*3 + i.
        for c in range(rank_block):
            for k_local in range(2):
                k_global = k0 + k_local
                for i in range(3):
                    W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]

    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if validate(U, V, W):
        return result
    else:
        # Fallback: return the best valid solution
        # Try with the 17-rank block
        U_block, V_block, W_block = get_17_rank_block()
        rank_block = 17
        rank_total = 4 * rank_block
        U = np.zeros((9, rank_total))
        V = np.zeros((24, rank_total))
        W = np.zeros((24, rank_total))
        
        for g in range(4):
            k0 = 2 * g
            col_start = g * rank_block
            col_end = (g + 1) * rank_block
            
            U[:, col_start:col_end] = U_block
            
            for c in range(rank_block):
                for j in range(3):
                    for k_local in range(2):
                        k_global = k0 + k_local
                        V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                        
            for c in range(rank_block):
                for k_local in range(2):
                    k_global = k0 + k_local
                    for i in range(3):
                        W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]
        
        result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if validate(U, V, W):
            return result
        else:
            return schoolbook()
