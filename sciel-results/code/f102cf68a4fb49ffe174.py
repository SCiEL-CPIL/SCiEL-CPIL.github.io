import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    def time_left():
        return deadline - time.time()
    
    # Build the rank-42 Strassen solution
    # A is 2x4, B is 4x6
    # Split A into [A1 | A2] (each 2x2)
    # Split B into [[B11, B12, B13], [B21, B22, B23]] (each 2x2)
    # C = A*B, C is 2x6
    # C[:, 0:2] = A1*B11 + A2*B21
    # C[:, 2:4] = A1*B12 + A2*B22
    # C[:, 4:6] = A1*B13 + A2*B23
    
    # For each output block column, we compute sum of two 2x2 matrix products
    # Using Strassen for each 2x2 product: rank 7 each, so rank 14 per output block
    # Total: 3 * 14 = 42
    
    rank_42 = 42
    U = np.zeros((n*m, rank_42))  # 8 x 42
    V = np.zeros((m*p, rank_42))  # 24 x 42
    W = np.zeros((p*n, rank_42))  # 12 x 42
    
    # Strassen terms for 2x2 matrix multiplication
    # For X*Y where X = [[x1,x2],[x3,x4]], Y = [[y1,y2],[y3,y4]]
    # m1 = (x1+x4)(y1+y4)
    # m2 = (x3+x4)y1
    # m3 = x1(y2-y4)
    # m4 = x4(y3-y1)
    # m5 = (x1+x2)y4
    # m6 = (x3-x1)(y1+y2)
    # m7 = (x2-x4)(y3+y4)
    # Result: [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    strassen_u = [
        [1, 0, 0, 1],  # m1: x1+x4
        [0, 0, 1, 1],  # m2: x3+x4
        [1, 0, 0, 0],  # m3: x1
        [0, 0, 0, 1],  # m4: x4
        [1, 1, 0, 0],  # m5: x1+x2
        [-1, 0, 1, 0], # m6: x3-x1
        [0, 1, 0, -1], # m7: x2-x4
    ]
    strassen_v = [
        [1, 0, 1, 0],  # m1: y1+y4
        [1, 0, 0, 0],  # m2: y1
        [0, 1, 0, -1], # m3: y2-y4
        [0, -1, 1, 0], # m4: y3-y1
        [0, 0, 0, 1],  # m5: y4
        [1, 1, 0, 0],  # m6: y1+y2
        [0, 0, 1, 1],  # m7: y3+y4
    ]
    strassen_w = [
        [1, 0, -1, 1],  # m1-m2+m3+m6 for (0,0)... wait let me redo
    ]
    
    # Let me be more careful. For 2x2 matrix mult X*Y:
    # X = [[x0, x1], [x2, x3]], Y = [[y0, y1], [y2, y3]]
    # (X*Y)[0,0] = x0*y0 + x1*y2
    # (X*Y)[0,1] = x0*y1 + x1*y3
    # (X*Y)[1,0] = x2*y0 + x3*y2
    # (X*Y)[1,1] = x2*y1 + x3*y3
    #
    # Strassen:
    # m1 = (x0+x3)(y0+y3)
    # m2 = (x2+x3)y0
    # m3 = x0(y1-y3)
    # m4 = x3(y2-y0)
    # m5 = (x0+x1)y3
    # m6 = (x2-x0)(y0+y1)
    # m7 = (x1-x3)(y2+y3)
    #
    # (X*Y)[0,0] = m1 + m4 - m5 + m7
    # (X*Y)[0,1] = m3 + m5
    # (X*Y)[1,0] = m2 + m4
    # (X*Y)[1,1] = m1 - m2 + m3 + m6
    
    strassen_u_coeffs = [
        [1, 0, 0, 1],   # m1: x0+x3
        [0, 0, 1, 1],   # m2: x2+x3
        [1, 0, 0, 0],   # m3: x0
        [0, 0, 0, 1],   # m4: x3
        [1, 1, 0, 0],   # m5: x0+x1
        [-1, 0, 1, 0],  # m6: x2-x0
        [0, 1, 0, -1],  # m7: x1-x3
    ]
    strassen_v_coeffs = [
        [1, 0, 1, 0],   # m1: y0+y3
        [1, 0, 0, 0],   # m2: y0
        [0, 1, 0, -1],  # m3: y1-y3
        [0, -1, 1, 0],  # m4: y2-y0
        [0, 0, 0, 1],   # m5: y3
        [1, 1, 0, 0],   # m6: y0+y1
        [0, 0, 1, 1],   # m7: y2+y3
    ]
    strassen_w_coeffs = [
        [1, 0, -1, 1],  # m1+m4-m5+m7 for (0,0)
        [0, 1, 0, 0],   # m3+m5 for (0,1)
        [0, 0, 1, 0],   # m2+m4 for (1,0)
        [1, -1, 1, 1],  # m1-m2+m3+m6 for (1,1)
    ]
    # Wait, strassen_w should be per-output-element, not per-term.
    # Let me restructure: for each Strassen term t (0..6), and each output element (0..3),
    # the coefficient in W.
    
    # Actually, let me think of it as: for each term t, U has a row pattern, V has a row pattern,
    # and W has a column pattern (which output elements get this term and with what sign).
    
    # For term t, the contribution to output element e is:
    # U_pattern[t] dot X * V_pattern[t] dot Y * W_coeff[t][e]
    
    # So W is a 4x7 matrix where W[e][t] is the coefficient.
    
    strassen_w_matrix = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # output (0,0): m1+m4-m5+m7
        [0, 0, 1, 0, 1, 0, 0],   # output (0,1): m3+m5
        [0, 1, 0, 1, 0, 0, 0],   # output (1,0): m2+m4
        [1, -1, 1, 0, 0, 1, 0],  # output (1,1): m1-m2+m3+m6
    ])
    
    # Now build the full 42-rank solution
    # A is 2x4: A[i,j] where i in {0,1}, j in {0,1,2,3}
    # U rows: index = i*m + j = i*4 + j, so rows 0-7
    #   Row 0: A[0,0], Row 1: A[0,1], Row 2: A[0,2], Row 3: A[0,3]
    #   Row 4: A[1,0], Row 5: A[1,1], Row 6: A[1,2], Row 7: A[1,3]
    #
    # B is 4x6: B[i,j] where i in {0,1,2,3}, j in {0,1,2,3,4,5}
    # V rows: index = i*p + j = i*6 + j, so rows 0-23
    #
    # C is 2x6: C[i,j] where i in {0,1}, j in {0,1,2,3,4,5}
    # W rows: index = k*n + i = j*2 + i (using k=j for column index in B, i for row in C)
    # Actually W index: the output is C[i,j], and W is indexed by (k*n+i) where k is the column of B (=column of C) and i is the row of A (=row of C).
    # So W index = j*2 + i where j is column of C (0..5), i is row of C (0..1)
    # So W has 12 rows: 
    #   j=0: W[0]=C[0,0], W[1]=C[1,0]
    #   j=1: W[2]=C[0,1], W[3]=C[1,1]
    #   j=2: W[4]=C[0,2], W[5]=C[1,2]
    #   j=3: W[6]=C[0,3], W[7]=C[1,3]
    #   j=4: W[8]=C[0,4], W[9]=C[1,4]
    #   j=5: W[10]=C[0,5], W[11]=C[1,5]
    
    # For output block column bc (0, 1, 2), corresponding to C columns 2*bc and 2*bc+1:
    #   C[:, 2*bc:2*bc+2] = A1 * B1_bc + A2 * B2_bc
    #   where A1 = A[:, 0:2] (rows 0-1 of U, cols 0-1), A2 = A[:, 2:4] (rows 0-1 of U, cols 2-3)
    #   B1_bc = B[0:2, 2*bc:2*bc+2], B2_bc = B[2:4, 2*bc:2*bc+2]
    
    col_offset = 0
    for bc in range(3):  # output block column
        for a_block in range(2):  # which A block (0 or 1)
            # A_block is A[:, a_block*2:(a_block+1)*2]
            # B_block is B[a_block*2:(a_block+1)*2, bc*2:(bc+1)*2]
            for t in range(7):  # Strassen term
                col = col_offset + t
                
                # U: for A_block, extract the 2x2 submatrix
                # A_block[i,j] corresponds to U row (i*4 + a_block*2 + j)
                for i in range(2):
                    for j in range(2):
                        u_row = i * 4 + a_block * 2 + j
                        U[u_row, col] += strassen_u_coeffs[t][i * 2 + j]
                
                # V: for B_block, extract the 2x2 submatrix
                # B_block[i,j] corresponds to V row ((a_block*2+i)*6 + bc*2 + j)
                for i in range(2):
                    for j in range(2):
                        v_row = (a_block * 2 + i) * 6 + bc * 2 + j
                        V[v_row, col] += strassen_v_coeffs[t][i * 2 + j]
                
                # W: for output block, the 2x2 result
                # Output element (i,j) in the block corresponds to C row i, column bc*2+j
                # W index = (bc*2+j)*2 + i
                for i in range(2):
                    for j in range(2):
                        w_row = (bc * 2 + j) * 2 + i
                        W[w_row, col] += strassen_w_matrix[i * 2 + j][t]
        
            col_offset += 7
    
    # Verify the rank-42 solution
    def verify(U_mat, V_mat, W_mat):
        r = U_mat.shape[1]
        # Compute the tensor: T[a,b,d] = sum_c U[a,c]*V[b,c]*W[d,c]
        # Expected: T[a,b,d] = 1 if (a//m, a%m) == (d%n, d//n) and (a%m, b//p) == (b//p, b%p)... 
        # Let me re-derive:
        # a indexes A: a = i*m + j, so i = a//m, j = a%m
        # b indexes B: b = j'*p + k, so j' = b//p, k = b%p
        # d indexes C: d = k'*n + i', so k' = d//n, i' = d%n
        # C[i',k'] = sum_j A[i',j] * B[j,k']
        # So T[a,b,d] = 1 if i == i' and j == j' and k == k', else 0
        
        errors = 0
        max_err = 0
        for a in range(n*m):
            for b in range(m*p):
                for d in range(p*n):
                    val = np.sum(U_mat[a, :] * V_mat[b, :] * W_mat[d, :])
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_d = d // n
                    i_d = d % n
                    if i_a == i_d and j_a == j_b and k_b == k_d:
                        expected = 1.0
                    else:
                        expected = 0.0
                    err = abs(val - expected)
                    if err > max_err:
                        max_err = err
                    if err > 1e-9:
                        errors += 1
        return errors == 0, max_err
    
    valid, max_err = verify(U, V, W)
    best_rank = 42
    best_U, best_V, best_W = U, V, W
    
    if not valid:
        print(f"Rank-42 solution invalid, max_err={max_err}")
        # Fall back to trivial rank-48
        rank_48 = n * m * p
        U48 = np.zeros((n*m, rank_48))
        V48 = np.zeros((m*p, rank_48))
        W48 = np.zeros((p*n, rank_48))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U48[i*m+j, col] = 1
                    V48[j*p+k, col] = 1
                    W48[k*n+i, col] = 1
                    col += 1
        best_rank = 48
        best_U, best_V, best_W = U48, V48, W48
    else:
        print(f"Rank-42 solution valid, max_err={max_err}")
    
    # Try to find lower rank solutions using optimization
    # Strategy: for target rank r, try to find U(r), V(r), W(r) that satisfy the constraints
    # Use alternating minimization: fix two matrices, optimize the third
    
    def solve_for_W(U_mat, V_mat):
        # Solve for W given U and V
        # T[a,b,d] = sum_c U[a,c]*V[b,c]*W[d,c]
        # For each d, we have: sum_c (U*V)[a,b,c] * W[d,c] = expected[a,b,d]
        # This is a linear system in W[d,:]
        # Let M[a,b,c] = U[a,c]*V[b,c]
        # Then for each d: sum_c M[a,b,c]*W[d,c] = expected[a,b,d]
        # This is a system of (n*m)*(m*p) equations in (p*n) unknowns per d... 
        # Actually, for fixed d, W[d,:] is a vector of length r.
        # The equations are: for all (a,b), sum_c U[a,c]*V[b,c]*W[d,c] = expected[a,b,d]
        # This is (n*m)*(m*p) equations in r unknowns. Usually overdetermined.
        # Use least squares.
        
        r = U_mat.shape[1]
        W_sol = np.zeros((p*n, r))
        
        # Build the system: for each d, A_d * W[d,:] = b_d
        # where A_d is a matrix of size (n*m*m*p) x r
        # A_d[a*m*p + b, c] = U[a,c] * V[b,c]
        # b_d[a*m*p + b] = expected[a,b,d]
        
        A_full = np.zeros((n*m, m*p, r))
        for a in range(n*m):
            for b in range(m*p):
                for c in range(r):
                    A_full[a, b, c] = U_mat[a, c] * V_mat[b, c]
        
        for d in range(p*n):
            b_d = np.zeros(n*m * m*p)
            for a in range(n*m):
                for b in range(m*p):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_d = d // n
                    i_d = d % n
                    if i_a == i_d and j_a == j_b and k_b == k_d:
                        b_d[a * m*p + b] = 1.0
            
            A_d = A_full.reshape(n*m * m*p, r)
            # Solve A_d * x = b_d in least squares sense
            try:
                W_sol[d, :], residuals, rank, sv = np.linalg.lstsq(A_d, b_d, rcond=None)
            except:
                W_sol[d, :] = 0
            # Check if the solution is exact
            if np.max(np.abs(A_d @ W_sol[d, :] - b_d)) > 1e-9:
                return None  # No exact solution
        
        return W_sol
    
    def solve_for_U(V_mat, W_mat):
        r = V_mat.shape[1]
        U_sol = np.zeros((n*m, r))
        
        A_full = np.zeros((m*p, p*n, r))
        for b in range(m*p):
            for d in range(p*n):
                for c in range(r):
                    A_full[b, d, c] = V_mat[b, c] * W_mat[d, c]
        
        for a in range(n*m):
            b_a = np.zeros(m*p * p*n)
            for b in range(m*p):
                for d in range(p*n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_d = d // n
                    i_d = d % n
                    if i_a == i_d and j_a == j_b and k_b == k_d:
                        b_a[b * p*n + d] = 1.0
            
            A_a = A_full.reshape(m*p * p*n, r)
            try:
                U_sol[a, :], residuals, rank, sv = np.linalg.lstsq(A_a, b_a, rcond=None)
            except:
                U_sol[a, :] = 0
            if np.max(np.abs(A_a @ U_sol[a, :] - b_a)) > 1e-9:
                return None
        
        return U_sol
    
    def solve_for_V(U_mat, W_mat):
        r = U_mat.shape[1]
        V_sol = np.zeros((m*p, r))
        
        A_full = np.zeros((n*m, p*n, r))
        for a in range(n*m):
            for d in range(p*n):
                for c in range(r):
                    A_full[a, d, c] = U_mat[a, c] * W_mat[d, c]
        
        for b in range(m*p):
            b_b = np.zeros(n*m * p*n)
            for a in range(n*m):
                for d in range(p*n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_d = d // n
                    i_d = d % n
                    if i_a == i_d and j_a == j_b and k_b == k_d:
                        b_b[a * p*n + d] = 1.0
            
            A_b = A_full.reshape(n*m * p*n, r)
            try:
                V_sol[b, :], residuals, rank, sv = np.linalg.lstsq(A_b, b_b, rcond=None)
            except:
                V_sol[b, :] = 0
            if np.max(np.abs(A_b @ V_sol[b, :] - b_b)) > 1e-9:
                return None
        
        return V_sol
    
    # Try alternating minimization for lower ranks
    np.random.seed(42)
    
    for target_rank in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if time_left() < 10:
            break
        
        found = False
        for restart in range(20):
            if time_left() < 5:
                break
            
            # Initialize randomly
            U_r = np.random.randn(n*m, target_rank)
            V_r = np.random.randn(m*p, target_rank)
            W_r = np.random.randn(p*n, target_rank)
            
            # Alternating minimization
            success = False
            for it in range(50):
                if time_left() < 3:
                    break
                
                # Solve for W
                W_new = solve_for_W(U_r, V_r)
                if W_new is not None:
                    W_r = W_new
                    # Verify
                    valid_w, _ = verify(U_r, V_r, W_r)
                    if valid_w:
                        success = True
                        break
                
                # Solve for U
                U_new = solve_for_U(V_r, W_r)
                if U_new is not None:
                    U_r = U_new
                    valid_u, _ = verify(U_r, V_r, W_r)
                    if valid_u:
                        success = True
                        break
                
                # Solve for V
                V_new = solve_for_V(U_r, W_r)
                if V_new is not None:
                    V_r = V_new
                    valid_v, _ = verify(U_r, V_r, W_r)
                    if valid_v:
                        success = True
                        break
            
            if success:
                print(f"Found rank-{target_rank} solution!")
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                found = True
                break
        
        if found:
            # Try even lower
            continue
        else:
            # If we failed at this rank, lower ranks are unlikely to work easily
            # but let's try one more level down
            continue
    
    # Convert to list format if needed, but numpy arrays should be fine
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
