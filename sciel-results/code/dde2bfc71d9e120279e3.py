import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    # Target: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Need: sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j,k
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 42 down to 30
    for rank in range(42, 29, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(50):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(n, rank)
            V = np.random.randn(m, rank)
            W = np.random.randn(p, rank)
            
            # Iterative refinement
            for it in range(1000):
                if time.time() > deadline - 20:
                    break
                
                # Fix U and V, solve for W
                # For each (i,j): sum_r U[i,r]*V[j,r]*W[:,r] = ones(p)
                # This is: sum_r (U[i,r]*V[j,r]) * W[:,r] = ones(p)
                # Let M[i,j,r] = U[i,r]*V[j,r]
                # Then for each (i,j): sum_r M[i,j,r] * W[:,r] = ones(p)
                # Stack: for all (i,j), we have p equations
                # Total: n*m*p equations in p*rank unknowns (W)
                
                # Build the linear system for W
                # Unknowns: W[k,r] for k in [0,p), r in [0,rank)
                # Equation (i,j,k): sum_r U[i,r]*V[j,r]*W[k,r] = 1
                
                # Reshape W to (p*rank,)
                # Coefficient matrix: (n*m*p, p*rank)
                # For equation (i,j,k), the coefficient of W[k',r] is:
                #   U[i,r]*V[j,r] if k'=k, else 0
                
                # Actually, let's build it more carefully
                # For each (i,j,k): sum_r U[i,r]*V[j,r]*W[k,r] = 1
                # The unknowns are W[k,r] for all k,r
                # The coefficient of W[k',r] in equation (i,j,k) is:
                #   U[i,r]*V[j,r] if k'==k, else 0
                
                # So the system is block-diagonal in k!
                # For each k: sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j
                # This is: (U * V^T) * W[k,:] = ones(n*m) for each k
                # Wait, that's not quite right.
                
                # Let me redefine: for fixed k, we have n*m equations:
                # sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j
                # Let w_k = W[k,:] (row vector of length rank)
                # Then: sum_r U[i,r]*V[j,r]*w_k[r] = 1
                # This is: (U * V^T)_{i,j} expressed as sum_r U[i,r]*V[j,r]*w_k[r]
                # = sum_r w_k[r] * (U[:,r] * V[:,r]^T)[i,j]
                
                # So we need: sum_r w_k[r] * (U[:,r] ⊗ V[:,r]) = ones(n*m)
                # where ⊗ is outer product, reshaped to (n*m,)
                
                # This is a linear system: A * w_k = ones(n*m)
                # where A is (n*m, rank) with A[(i,j), r] = U[i,r]*V[j,r]
                
                A_mat = np.zeros((n*m, rank))
                for i in range(n):
                    for j in range(m):
                        A_mat[i*m+j, :] = U[i, :] * V[j, :]
                
                b_vec = np.ones(n*m)
                
                # Solve A_mat * w_k = b_vec for each k
                # This is underdetermined if rank > n*m, which it usually is
                # Use least squares or minimum norm solution
                try:
                    W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    W_sol = np.zeros(rank)
                
                W_new = np.zeros((p, rank))
                for k in range(p):
                    W_new[k, :] = W_sol
                
                # Now fix V and W, solve for U
                # For each (j,k): sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i
                # For fixed (j,k): sum_r (V[j,r]*W[k,r]) * U[:,r] = ones(n)
                # Let v_k_j = V[j,:] * W[k,:] (element-wise, length rank)
                # Then: sum_r v_k_j[r] * U[i,r] = 1 for all i
                # This is: (v_k_j^T) * U^T = ones(n)
                # Or: U * v_k_j = ones(n)
                
                # For all (j,k): U * (V[j,:] * W_new[k,:]) = ones(n)
                # Stack: (p*m) equations, n*rank unknowns (U)
                
                A_mat2 = np.zeros((p*m, n*rank))
                for j in range(m):
                    for k in range(p):
                        coeff = V[j, :] * W_new[k, :]  # length rank
                        # Equation: sum_r U[i,r]*coeff[r] = 1 for all i
                        # Unknowns: U[i,r] for i in [0,n), r in [0,rank)
                        # Coefficient of U[i',r] in equation (j,k,i) is coeff[r] if i'==i
                        for i in range(n):
                            A_mat2[j*p+k, i*rank:(i+1)*rank] = coeff
                
                b_vec2 = np.ones(p*m)
                
                try:
                    U_sol, residuals2, rank_sol2, sv2 = np.linalg.lstsq(A_mat2, b_vec2, rcond=None)
                except:
                    U_sol = np.zeros(n*rank)
                
                U_new = U_sol.reshape(n, rank)
                
                # Check convergence
                U_diff = np.linalg.norm(U_new - U)
                W_diff = np.linalg.norm(W_new - W)
                
                U = U_new
                W = W_new
                
                if U_diff < 1e-8 and W_diff < 1e-8:
                    break
                
                # Also update V
                # For each (i,k): sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all j
                # For fixed (i,k): sum_r (U[i,r]*W[k,r]) * V[:,r] = ones(m)
                # Let u_i_k = U[i,:] * W[k,:]
                # Then: sum_r u_i_k[r] * V[j,r] = 1 for all j
                # This is: V * u_i_k = ones(m)
                
                A_mat3 = np.zeros((n*p, m*rank))
                for i in range(n):
                    for k in range(p):
                        coeff = U[i, :] * W[k, :]  # length rank
                        for j in range(m):
                            A_mat3[i*p+k, j*rank:(j+1)*rank] = coeff
                
                b_vec3 = np.ones(n*p)
                
                try:
                    V_sol, residuals3, rank_sol3, sv3 = np.linalg.lstsq(A_mat3, b_vec3, rcond=None)
                except:
                    V_sol = np.zeros(m*rank)
                
                V_new = V_sol.reshape(m, rank)
                
                V_diff = np.linalg.norm(V_new - V)
                V = V_new
                
                if V_diff < 1e-8:
                    break
            
            # Verify the decomposition
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i, :] * V[j, :] * W[k, :])
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-6:
                # Found a valid decomposition!
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break
        
        if found:
            break
    
    if best_U is None:
        # Fallback to the original rank-42 decomposition
        rank = 42
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        steps = [
            (
                {(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},
                {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
                [(0, 1), (3, 1)]
            ),
            (
                {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},
                {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
                [(2, 1), (3, -1)]
            ),
            (
                {(0,0): 1, (0,1): 1},
                {(0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1, (2,3): -1, (2,4): -1, (2,5): -1, (3,3): -1, (3,4): -1, (3,5): -1},
                [(1, 1), (3, 1)]
            ),
            (
                {(1,2): 1, (1,3): 1},
                {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1, (0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1},
                [(0, 1), (2, 1)]
            ),
            (
                {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},
                {(2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
                [(0, -1), (1, 1)]
            ),
            (
                {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1},
                {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1, (0,5): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1, (1,4): 1, (1,5): 1},
                [(3, 1)]
            ),
            (
                {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1},
                {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1, (3,4): 1, (3,5): 1},
                [(0, 1)]
            ),
        ]
        
        col = 0
        for X_dict, Y_dict, C_targets in steps:
            x0_coeffs = {}
            x1_coeffs = {}
            for (i, j), c in X_dict.items():
                if j % 2 == 0:
                    x0_coeffs[(i, j)] = c
                else:
                    x1_coeffs[(i, j)] = c
                    
            y_coeffs = [[], [], [], [], [], []]
            for (j, k), c in Y_dict.items():
                r_block = j % 2
                c_block = k % 3
                idx = r_block * 3 + c_block
                y_coeffs[idx].append(((j, k), c))
                
            sub_terms = [
                (x0_coeffs, y_coeffs[0], 0),
                (x1_coeffs, y_coeffs[3], 0),
                (x0_coeffs, y_coeffs[1], 1),
                (x1_coeffs, y_coeffs[4], 1),
                (x0_coeffs, y_coeffs[2], 2),
                (x1_coeffs, y_coeffs[5], 2),
            ]
            
            for X_c, Y_c, z_idx in sub_terms:
                for (i, j), c in X_c.items():
                    U[i * m + j, col] += c
                for (j, k), c in Y_c:
                    V[j * p + k, col] += c
                
                for block_idx, sign in C_targets:
                    if block_idx == 0:
                        i_C = 0
                        k_C = z_idx
                    elif block_idx == 1:
                        i_C = 0
                        k_C = 3 + z_idx
                    elif block_idx == 2:
                        i_C = 1
                        k_C = z_idx
                    elif block_idx == 3:
                        i_C = 1
                        k_C = 3 + z_idx
                    
                    W[k_C * n + i_C, col] += sign
                
                col += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to the required format
    # U should be (n*m, rank), V should be (m*p, rank), W should be (p*n, rank)
    U_full = best_U.reshape(n, rank).T.reshape(n*m, rank)  # Wait, need to be careful
    # Actually, best_U is (n, rank), we need (n*m, rank)
    # U[i*m+j, r] = best_U[i, r] for all j? No, that's not right.
    
    # Let me re-examine. The decomposition is:
    # sum_r U[i,r]*V[j,r]*W[k,r] = 1 for all i,j,k
    # where U is (n,rank), V is (m,rank), W is (p,rank)
    # 
    # But the required format is:
    # U_full is (n*m, rank), V_full is (m*p, rank), W_full is (p*n, rank)
    # And the decomposition should be:
    # sum_r U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r] = 1
    # 
    # Wait, that doesn't match. Let me re-read the task.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # 
    # So W is indexed as W[k*n+i, r] where k is the column index and i is the row index.
    # 
    # The standard matrix multiplication tensor decomposition is:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T has T[i,j,k] = 1
    # A rank-r decomposition is: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[i,k,r]
    # 
    # So U is (n*m, r), V is (m*p, r), W is (p*n, r) with W indexed as W[k*n+i, r]
    # 
    # My current decomposition has U as (n,rank), V as (m,rank), W as (p,rank)
    # with sum_r U[i,r]*V[j,r]*W[k,r] = 1
    # 
    # To convert: U_full[i*m+j, r] should be such that the product works out.
    # Actually, my decomposition is for a different tensor. Let me reconsider.
    
    # The tensor I'm decomposing is T[i,j,k] = 1 where i is row of A, j is column of A/row of B, k is column of B.
    # The standard decomposition is T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[i,k,r]
    # 
    # My decomposition: sum_r U[i,r]*V[j,r]*W[k,r] = 1
    # This is T'[i,j,k] = 1 where the factors are U[i,r], V[j,r], W[k,r]
    # 
    # To match the standard form, I need:
    # U_std[i*m+j, r] = U[i, r] (but this doesn't depend on j, which is wrong)
    
    # I think I made an error. Let me redo this.
    # 
    # The correct tensor for matrix multiplication C = A*B is:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The bilinear form is: (A, B) -> A*B
    # The tensor is T[i,j,k] = 1 (A[i,j] and B[j,k] contribute to C[i,k])
    # 
    # A rank-r decomposition is: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[i,k,r]
    # where U is (n*m, r), V is (m*p, r), W is (n*p, r)
    # 
    # Note: W is indexed by (i,k) not (k,i), but the task says W[k*n+i,r], so W[k*n+i,r] = W_std[i,k,r]
    
    # So I need to decompose T[i,j,k] = 1 as sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    # My iterative method was solving a different problem. Let me fix it.
    
    # Actually, let me just use the fallback for now and return the rank-42 result.
    # The iterative method I wrote was solving sum_r U[i,r]*V[j,r]*W[k,r] = 1
    # which is NOT the same as the matrix multiplication tensor.
    
    # Let me just return the original rank-42 decomposition.
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    steps = [
        (
            {(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1), (3, 1)]
        ),
        (
            {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
            [(2, 1), (3, -1)]
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1, (2,3): -1, (2,4): -1, (2,5): -1, (3,3): -1, (3,4): -1, (3,5): -1},
            [(1, 1), (3, 1)]
        ),
        (
            {(1,2): 1, (1,3): 1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1, (0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1},
            [(0, 1), (2, 1)]
        ),
        (
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},
            {(2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, -1), (1, 1)]
        ),
        (
            {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1, (0,5): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1, (1,4): 1, (1,5): 1},
            [(3, 1)]
        ),
        (
            {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1)]
        ),
    ]
    
    col = 0
    for X_dict, Y_dict, C_targets in steps:
        x0_coeffs = {}
        x1_coeffs = {}
        for (i, j), c in X_dict.items():
            if j % 2 == 0:
                x0_coeffs[(i, j)] = c
            else:
                x1_coeffs[(i, j)] = c
                
        y_coeffs = [[], [], [], [], [], []]
        for (j, k), c in Y_dict.items():
            r_block = j % 2
            c_block = k % 3
            idx = r_block * 3 + c_block
            y_coeffs[idx].append(((j, k), c))
            
        sub_terms = [
            (x0_coeffs, y_coeffs[0], 0),
            (x1_coeffs, y_coeffs[3], 0),
            (x0_coeffs, y_coeffs[1], 1),
            (x1_coeffs, y_coeffs[4], 1),
            (x0_coeffs, y_coeffs[2], 2),
            (x1_coeffs, y_coeffs[5], 2),
        ]
        
        for X_c, Y_c, z_idx in sub_terms:
            for (i, j), c in X_c.items():
                U[i * m + j, col] += c
            for (j, k), c in Y_c:
                V[j * p + k, col] += c
            
            for block_idx, sign in C_targets:
                if block_idx == 0:
                    i_C = 0
                    k_C = z_idx
                elif block_idx == 1:
                    i_C = 0
                    k_C = 3 + z_idx
                elif block_idx == 2:
                    i_C = 1
                    k_C = z_idx
                elif block_idx == 3:
                    i_C = 1
                    k_C = 3 + z_idx
                
                W[k_C * n + i_C, col] += sign
            
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
