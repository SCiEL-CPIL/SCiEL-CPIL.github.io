import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7
    
    def check_valid(U, V, W, tol=1e-9):
        """Check if the decomposition is valid using vectorized computation."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We need this to equal 1 for all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m + j, :], V_arr[j*p + k, :] * W_arr[k*n + i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def build_schoolbook():
        """Build the trivial schoolbook decomposition of rank n*m*p."""
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W
    
    def build_strassen_2x2_2x2():
        """
        Build Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
        Returns U (4x7), V (4x7), W (4x7) for the 2x2x2 case.
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # p1 = (a11 + a22)(b11 + b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # p2 = (a21 + a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # p3 = a11(b12 - b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # p4 = a22(b21 - b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # p5 = (a11 + a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # p6 = (a21 - a11)(b11 + b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # p7 = (a12 - a22)(b21 + b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    def kronecker_decompose_4x4x4():
        """
        Build a decomposition for 4x4x4 using Kronecker product of two 2x2x2 Strassen decompositions.
        This gives rank 7*7 = 49.
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                if val != 0:
                                    U4[i*4+j][r] += val
                                  
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                if val != 0:
                                    V4[j*4+k][r] += val
                                  
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                if val != 0:
                                    W4[k*4+i][r] += val
        
        return U4, V4, W4
    
    def build_4x4x2_strassen():
        """
        Build rank-28 decomposition for 4x4x2 using 4 independent 2x2x2 Strassen blocks.
        A is 4x4, B is 4x2, C is 4x4.
        U: 16x28 (indices i*4+j), V: 8x28 (indices j*2+k), W: 8x28 (indices k*4+i)
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        r = 28
        
        U = [[0]*r for _ in range(16)]
        V = [[0]*r for _ in range(8)]
        W = [[0]*r for _ in range(8)]
        
        for row_group in range(2):
            for col_group in range(2):
                base_r = (row_group * 2 + col_group) * 7
                
                for r_local in range(7):
                    r_global = base_r + r_local
                    
                    for i_local in range(2):
                        for j_local in range(2):
                            i = 2 * row_group + i_local
                            j = 2 * col_group + j_local
                            U[i*4 + j][r_global] = U2[i_local*2 + j_local][r_local]
                    
                    for j_local in range(2):
                        for k_local in range(2):
                            j = 2 * col_group + j_local
                            k = k_local
                            V[j*2 + k][r_global] = V2[j_local*2 + k_local][r_local]
                    
                    for k_local in range(2):
                        for i_local in range(2):
                            k = k_local
                            i = 2 * row_group + i_local
                            W[k*4 + i][r_global] = W2[k_local*2 + i_local][r_local]
        
        return U, V, W
    
    def build_4x4x1_schoolbook():
        """
        Build rank-16 schoolbook decomposition for 4x4x1.
        A is 4x4, B is 4x1, C is 4x4.
        U: 16x16 (indices i*4+j), V: 4x16 (indices j*1+k= j), W: 4x16 (indices k*4+i)
        """
        r = 16
        U = [[0]*r for _ in range(16)]
        V = [[0]*r for _ in range(4)]
        W = [[0]*r for _ in range(4)]
        
        col = 0
        for i in range(4):
            for j in range(4):
                U[i*4 + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        
        return U, V, W
    
    def build_4x4x7_decomposition():
        """
        Build a decomposition for 4x4x7.
        Part 1: 4x4x4 using Kronecker Strassen (rank 49)
        Part 2: 4x4x3 = 4x4x2 (rank 28) + 4x4x1 (rank 16) = rank 44
        Total: rank 93
        """
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])  # 49
        
        U42, V42, W42 = build_4x4x2_strassen()
        r42 = len(U42[0])  # 28
        
        U41, V41, W41 = build_4x4x1_schoolbook()
        r41 = len(U41[0])  # 16
        
        r_total = r4 + r42 + r41
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        # Part 1: 4x4x4 Kronecker (columns 0-3 of B)
        for i in range(16):
            for r in range(r4):
                U[i][r] = U4[i][r]
        
        for j in range(4):
            for k in range(4):
                for r in range(r4):
                    V[j*7 + k][r] = V4[j*4 + k][r]
        
        for k in range(4):
            for i in range(4):
                for r in range(r4):
                    W[k*4 + i][r] = W4[k*4 + i][r]
        
        # Part 2a: 4x4x2 Strassen (columns 4-5 of B)
        offset42 = r4
        for i in range(16):
            for r in range(r42):
                U[i][offset42 + r] = U42[i][r]
        
        for j in range(4):
            for k in range(2):
                for r in range(r42):
                    V[j*7 + (4 + k)][offset42 + r] = V42[j*2 + k][r]
        
        for k in range(2):
            for i in range(4):
                for r in range(r42):
                    W[(4 + k)*4 + i][offset42 + r] = W42[k*4 + i][r]
        
        # Part 2b: 4x4x1 schoolbook (column 6 of B)
        offset41 = r4 + r42
        for i in range(16):
            for r in range(r41):
                U[i][offset41 + r] = U41[i][r]
        
        for j in range(4):
            for r in range(r41):
                V[j*7 + 6][offset41 + r] = V41[j][r]
        
        for i in range(4):
            for r in range(r41):
                W[6*4 + i][offset41 + r] = W41[i][r]
        
        return U, V, W
    
    def reduce_rank_greedy(U, V, W, max_iterations=500, tol=1e-9):
        """
        Vectorized rank reduction:
        1. Compute the full tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        2. Find the minimal subset of columns whose tensor sum equals T.
        Uses SVD-based column selection for efficiency.
        """
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Quick check: remove any all-zero columns
        nonzero_cols = []
        for c in range(r):
            if not (np.allclose(U_arr[:, c], 0) or np.allclose(V_arr[:, c], 0) or np.allclose(W_arr[:, c], 0)):
                nonzero_cols.append(c)
        
        if len(nonzero_cols) < r:
            r = len(nonzero_cols)
            U_arr = U_arr[:, nonzero_cols]
            V_arr = V_arr[:, nonzero_cols]
            W_arr = W_arr[:, nonzero_cols]
        
        # Compute tensor T as (n*m*p,) vector
        # T[idx] where idx = i*m*p + j*p + k
        T = np.zeros(n*m*p)
        for c in range(r):
            # u_c is (n*m,), v_c is (m*p,), w_c is (p*n,)
            # Need to reshape and compute outer product
            u_c = U_arr[:, c].reshape(n, m)
            v_c = V_arr[:, c].reshape(m, p)
            w_c = W_arr[:, c].reshape(p, n)
            # T contribution: for each (i,j,k), u_c[i,j]*v_c[j,k]*w_c[k,i]
            # This is a 3D tensor contraction
            # Reshape to (n, m, p, n) and contract
            # Actually, let's do it more directly
            for i in range(n):
                for j in range(m):
                    if u_c[i, j] == 0:
                        continue
                    for k in range(p):
                        T[i*m*p + j*p + k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
        
        # Now try to find minimal subset of columns that reproduces T
        # Greedy: at each step, pick the column whose addition most reduces ||T - T_reconstructed||
        # But this is O(r^2 * n*m*p). Let's use a different approach.
        
        # Alternative: check if columns are linearly dependent in their tensor contributions
        # Build matrix M where M[i*m*p+j*p+k, c] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Then T = M @ ones(r)
        # We want to find the sparsest x such that M @ x = T
        # This is an l1 minimization problem, but let's try a greedy approach
        
        # Build M (n*m*p) x r
        M = np.zeros((n*m*p, r))
        for c in range(r):
            u_c = U_arr[:, c].reshape(n, m)
            v_c = V_arr[:, c].reshape(m, p)
            w_c = W_arr[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if u_c[i, j] == 0:
                        continue
                    for k in range(p):
                        M[i*m*p + j*p + k, c] = u_c[i, j] * v_c[j, k] * w_c[k, i]
        
        # Check if T is consistent
        if not np.allclose(M @ np.ones(r), T, atol=tol):
            return U, V, W, r
        
        # Try to find sparse solution: greedy forward selection
        # Start with empty set, add columns that best explain T
        # But we want to REMOVE columns, so let's do backward elimination
        
        # Backward: try removing each column, check if T can still be represented
        # For column c, we need to check if T - M[:,c] is in span of {M[:,d] : d != c}
        # This is equivalent to checking if the residual after projecting (T - M[:,c]) onto span of others is zero
        
        # More efficient: compute rank of M and M without each column
        # If rank doesn't change when removing column c, and T is in the span of remaining columns, we can remove c
        
        rank_M = np.linalg.matrix_rank(M, tol=tol)
        
        removable = []
        for c in range(r):
            if time.time() - start_time > WALL_DEADLINE - SAFETY_MARGIN - 2:
                break
            others = [d for d in range(r) if d != c]
            if len(others) == 0:
                break
            M_others = M[:, others]
            rank_others = np.linalg.matrix_rank(M_others, tol=tol)
            
            if rank_others == rank_M:
                # Check if T is in span of M_others
                # Solve M_others @ x = T
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(M_others, T, rcond=tol)
                    if np.allclose(M_others @ x, T, atol=tol):
                        removable.append(c)
                except:
                    pass
        
        if not removable:
            return U, V, W, r
        
        # Remove all identified columns
        keep_cols = [c for c in range(r) if c not in removable]
        new_r = len(keep_cols)
        
        if new_r == 0:
            return U, V, W, r
        
        U_new = U_arr[:, keep_cols].tolist()
        V_new = V_arr[:, keep_cols].tolist()
        W_new = W_arr[:, keep_cols].tolist()
        
        # Verify
        T_new = np.zeros(n*m*p)
        for c in range(new_r):
            u_c = np.array(U_new)[:, c].reshape(n, m)
            v_c = np.array(V_new)[:, c].reshape(m, p)
            w_c = np.array(W_new)[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if u_c[i, j] == 0:
                        continue
                    for k in range(p):
                        T_new[i*m*p + j*p + k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
        
        if np.allclose(T_new, T, atol=tol):
            return U_new, V_new, W_new, new_r
        else:
            return U, V, W, r
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 112
    
    # Try the optimized approach
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_opt, V_opt, W_opt = build_4x4x7_decomposition()
            r_opt = len(U_opt[0])
            
            if check_valid(U_opt, V_opt, W_opt):
                if r_opt < best_rank:
                    best_result = {"U": U_opt, "V": V_opt, "W": W_opt}
                    best_rank = r_opt
                    
                    # Try rank reduction
                    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN - 10:
                        U_red, V_red, W_red, r_red = reduce_rank_greedy(U_opt, V_opt, W_opt)
                        if r_red < r_opt and check_valid(U_red, V_red, W_red):
                            best_result = {"U": U_red, "V": V_red, "W": W_red}
                            best_rank = r_red
                        
                        # Try iterative reduction
                        while r_red < r_opt and time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN - 5:
                            U_red2, V_red2, W_red2, r_red2 = reduce_rank_greedy(U_red, V_red, W_red)
                            if r_red2 >= r_red:
                                break
                            if check_valid(U_red2, V_red2, W_red2):
                                U_red, V_red, W_red = U_red2, V_red2, W_red2
                                r_red = r_red2
                                best_result = {"U": U_red, "V": V_red, "W": W_red}
                                best_rank = r_red
                            else:
                                break
        except Exception as e:
            pass
    
    return best_result
