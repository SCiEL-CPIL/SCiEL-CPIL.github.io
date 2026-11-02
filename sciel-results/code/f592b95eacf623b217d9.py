import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def find_2x2x5_rank10():
    # n=2, m=2, p=5
    # U: 4x10, V: 10x10, W: 10x10
    # Equations: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    # i,j in 0,1; k in 0..4
    
    # We can set U to be a random matrix with small integer entries and solve for V, W?
    # Or set V to be identity-like?
    # Let's try to set U and V to be sparse and solve for W.
    # If U and V are fixed, W is determined by:
    # sum_r V[j,k,r] W[k,i,r] = (sum_r U[i,j,r] V[j,k,r] W[k,i,r] ... wait)
    # The equation is linear in W if U and V are fixed.
    # Let A_{i,j,k} = U[i,j,r] V[j,k,r]. Then sum_r A_{i,j,k,r} W[k,i,r] = 1.
    # This is a linear system for W.
    
    # We will try random sparse U and V matrices with integer entries.
    # Then solve for W. If W has integer or half-integer entries, we have a solution.
    
    n, m, p = 2, 2, 5
    R = 10
    
    for _ in range(1000):
        # Random sparse U, V
        U = np.zeros((n*m, R), dtype=int)
        V = np.zeros((m*p, R), dtype=int)
        
        # Each column of U and V should have a few 1s and -1s
        for r in range(R):
            # Pick 2 random indices for U
            idx_u = np.random.choice(n*m, 2, replace=False)
            U[idx_u, r] = np.random.choice([-1, 1], 2)
            
            # Pick 2 random indices for V
            idx_v = np.random.choice(m*p, 2, replace=False)
            V[idx_v, r] = np.random.choice([-1, 1], 2)
        
        # Build linear system for W
        # Variables: W[k, i, r] for k in 0..4, i in 0..1, r in 0..9
        # Total variables: 5 * 2 * 10 = 100
        # Equations: i in 0..1, j in 0..1, k in 0..4 -> 20 equations
        
        # Equation for (i, j, k):
        # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        
        # We can group by (k, i).
        # For a fixed (k, i), the variable is W[k, i, r].
        # The equation involves W[k, i, r] for all r.
        # But the coefficient depends on j.
        # So for each (k, i), we have 2 equations (for j=0, j=1) involving the same 10 variables W[k, i, 0..9].
        
        W_solution = np.zeros((p, n, R))
        
        for k in range(p):
            for i in range(n):
                # Variables: W[k, i, 0..R-1]
                # Equations for j=0 and j=1
                # sum_r U[i,0,r] V[0,k,r] W[k,i,r] = 1
                # sum_r U[i,1,r] V[1,k,r] W[k,i,r] = 1
                
                A_mat = np.zeros((2, R))
                b_vec = np.array([1.0, 1.0])
                
                for r in range(R):
                    A_mat[0, r] = U[i*m + 0, r] * V[0*p + k, r]
                    A_mat[1, r] = U[i*m + 1, r] * V[1*p + k, r]
                
                # Solve A_mat @ W_vec = b_vec
                # This is under-determined (2 equations, 10 unknowns).
                # We want an integer solution.
                # We can set 8 variables to 0 and solve for 2?
                # Or use a simple heuristic.
                
                # Try to find a sparse solution.
                # Pick 2 columns of A_mat that are linearly independent.
                # If found, set other W to 0.
                
                # Find 2 independent columns
                found = False
                for r1 in range(R):
                    for r2 in range(r1+1, R):
                        col1 = A_mat[:, r1]
                        col2 = A_mat[:, r2]
                        det = np.linalg.det(np.column_stack((col1, col2)))
                        if abs(det) > 1e-9:
                            # Solve
                            # A_mat[:, [r1, r2]] @ [w1, w2] = b
                            try:
                                w_vals = np.linalg.solve(np.column_stack((col1, col2)), b_vec)
                                if np.allclose(w_vals, np.round(w_vals), atol=1e-5):
                                    W_vec = np.zeros(R)
                                    W_vec[r1] = int(np.round(w_vals[0]))
                                    W_vec[r2] = int(np.round(w_vals[1]))
                                    W_solution[k, i, :] = W_vec
                                    found = True
                                    break
                            except np.linalg.LinAlgError:
                                pass
                    if found: break
                if not found:
                    # Try half-integers?
                    # Or just fail this iteration
                    break
            
            if not found:
                break
        
        if found:
            # Verify solution
            # Check all 20 equations
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U[i*m+j, r] * V[j*p+k, r] * W_solution[k, i, r]
                        if abs(val - 1) > 1e-5:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = [[W_solution[k, i, r] for r in range(R)] for k in range(p) for i in range(n)]
                # W shape is p*n = 10.
                # W_list is list of 10 lists of 10 ints.
                return U_list, V_list, W_list
                
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find 2x2x5 rank 10
    # If found, use it 4 times
    # Time limit for search: 50s
    while time.time() - start_time < 50:
        res = find_2x2x5_rank10()
        if res:
            U_225, V_225, W_225 = res
            # Construct 4x4x5
            # 4 blocks of 2x2x5
            # U: 16 x 40
            # V: 20 x 40
            # W: 20 x 40
            
            R_total = 40
            U = [[0] * R_total for _ in range(n * m)]
            V = [[0] * R_total for _ in range(m * p)]
            W = [[0] * R_total for _ in range(p * n)]
            
            # Block 0: A00, B0 -> C00
            # Block 1: A01, B1 -> C01
            # Block 2: A10, B0 -> C10
            # Block 3: A11, B1 -> C11
            
            # Indices:
            # A: i in 0..3, j in 0..3
            # B: j in 0..3, k in 0..4
            # C: i in 0..3, k in 0..4
            
            # Block 0 (i=0,1, j=0,1):
            # U[i, j, r] = U_225[i, j, r]
            # V[j, k, r] = V_225[j, k, r]
            # W[k, i, r] = W_225[k, i, r]
            
            for block in range(4):
                r_offset = block * 10
                i_block = block // 2
                j_block = block % 2
                
                # i in 0,1 -> i_abs = i_block*2 + i
                # j in 0,1 -> j_abs = j_block*2 + j
                # k in 0..4 -> k_abs = k
                
                for i in range(2):
                    for j in range(2):
                        i_abs = i_block * 2 + i
                        j_abs = j_block * 2 + j
                        for r in range(10):
                            U[i_abs * m + j_abs][r_offset + r] = U_225[i * 2 + j][r] # U_225 index i*2+j? No, U_225 is 4x10. i,j in 0,1.
                            # U_225[i, j] is at index i*2+j? No, n=2, m=2. Index i*m+j = i*2+j.
                            
                for j in range(2):
                    for k in range(5):
                        j_abs = j_block * 2 + j
                        for r in range(10):
                            V[j_abs * p + k][r_offset + r] = V_225[j * 5 + k][r] # V_225 index j*5+k
                
                for k in range(5):
                    for i in range(2):
                        i_abs = i_block * 2 + i
                        for r in range(10):
                            W[k * n + i_abs][r_offset + r] = W_225[k * 2 + i][r] # W_225 index k*2+i? No, W_225 is 10x10. Index k*n+i = k*2+i.
            
            return {"U": U, "V": V, "W": W}
            
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
