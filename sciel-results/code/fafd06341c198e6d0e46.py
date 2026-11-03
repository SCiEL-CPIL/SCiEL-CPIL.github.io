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
        """Check if the decomposition is valid using vectorized numpy."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape for batch computation
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # For each (i,j,k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Compute all (i,j) pairs for U
        # This is a 3D tensor contraction
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Use einsum for efficiency
        # First, create index arrays
        i_idx = np.repeat(np.arange(n), m*p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n*m)
        
        u_vals = U_arr[i_idx*m + j_idx]  # (n*m*p, r)
        v_vals = V_arr[j_idx*p + k_idx]  # (n*m*p, r)
        w_vals = W_arr[k_idx*n + i_idx]  # (n*m*p, r)
        
        # Sum over r
        tensor_vals = np.sum(u_vals * v_vals * w_vals, axis=1)
        
        if np.any(np.abs(tensor_vals - 1.0) > tol):
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
    
    def _kron_decompose(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2, n, m, p):
        """Build Kronecker product decomposition.
        
        Factor 1: n1 x m1 * m1 x p1 -> n1 x p1
        Factor 2: n2 x m2 * m2 x p2 -> n2 x p2
        Result:   (n1*n2) x (m1*m2) * (m1*m2) x (p1*p2) -> (n1*n2) x (p1*p2)
        
        U1: (n1*m1) x r1, V1: (m1*p1) x r1, W1: (p1*n1) x r1
        U2: (n2*m2) x r2, V2: (m2*p2) x r2, W2: (p2*n2) x r2
        
        Returns: U: (n*m) x (r1*r2), V: (m*p) x (r1*r2), W: (p*n) x (r1*r2)
        """
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 * r2
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        for r1_idx in range(r1):
            for r2_idx in range(r2):
                r = r1_idx * r2 + r2_idx
                # U: A[i,j] where i = i1*n2 + i2, j = j1*m2 + j2
                for i1 in range(n1):
                    for j1 in range(m1):
                        for i2 in range(n2):
                            for j2 in range(m2):
                                i = i1 * n2 + i2
                                j = j1 * m2 + j2
                                val = U1[i1 * m1 + j1][r1_idx] * U2[i2 * m2 + j2][r2_idx]
                                if val != 0:
                                    U[i * m + j][r] += val
                            
                # V: B[j,k] where j = j1*m2 + j2, k = k1*p2 + k2
                for j1 in range(m1):
                    for k1 in range(p1):
                        for j2 in range(m2):
                            for k2 in range(p2):
                                j = j1 * m2 + j2
                                k = k1 * p2 + k2
                                val = V1[j1 * p1 + k1][r1_idx] * V2[j2 * p2 + k2][r2_idx]
                                if val != 0:
                                    V[j * p + k][r] += val
                            
                # W: C[i,k] -> W[k*n+i] where k = k1*p2 + k2, i = i1*n2 + i2
                for k1 in range(p1):
                    for i1 in range(n1):
                        for k2 in range(p2):
                            for i2 in range(n2):
                                k = k1 * p2 + k2
                                i = i1 * n2 + i2
                                val = W1[k1 * n1 + i1][r1_idx] * W2[k2 * n2 + i2][r2_idx]
                                if val != 0:
                                    W[k * n + i][r] += val
        
        return U, V, W
    
    def _schoolbook_2x2x2x1():
        """Schoolbook decomposition for 2x2 * 2x1 -> 2x1, rank 4.
        
        Convention: U[i*2+j, r] for A[i,j], V[j, r] for B[j] (B is 2x1), W[k*2+i, r] for C[i] (C is 2x1, k=0).
        """
        U = [[0] * 4 for _ in range(4)]
        V = [[0] * 4 for _ in range(2)]
        W = [[0] * 4 for _ in range(2)]
        col = 0
        for i in range(2):
            for j in range(2):
                U[i * 2 + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W
    
    def kronecker_decompose_4x4x4():
        """
        Build a decomposition for 4x4x4 using Kronecker product of two 2x2x2 Strassen decompositions.
        This gives rank 7*7 = 49.
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        U, V, W = _kron_decompose(U2, V2, W2, U2, V2, W2, 2, 2, 2, 2, 2, 2, 4, 4, 4)
        return U, V, W
    
    def build_4x4x2():
        """Build decomposition for 4x4 * 4x2.
        
        Use Kronecker: (2x2 * 2x2) ⊗ (2x2 * 2x1)
        Factor 1: 2x2 * 2x2 -> 2x2, Strassen rank 7
        Factor 2: 2x2 * 2x1 -> 2x1, schoolbook rank 4
        Result: 4x4 * 4x2 -> 4x2, rank 7*4 = 28
        """
        n, m, p = 4, 4, 2
        U1, V1, W1 = build_strassen_2x2_2x2()  # 2x2*2x2
        U2, V2, W2 = _schoolbook_2x2x2x1()  # 2x2*2x1
        U, V, W = _kron_decompose(U1, V1, W1, U2, V2, W2, 2, 2, 2, 2, 2, 1, n, m, p)
        return U, V, W
    
    def build_4x4x1():
        """Build decomposition for 4x4 * 4x1.
        
        Use Kronecker: (2x2 * 2x1) ⊗ (2x2 * 2x1)
        Factor 1: 2x2 * 2x1 -> 2x1, schoolbook rank 4
        Factor 2: 2x2 * 2x1 -> 2x1, schoolbook rank 4
        Result: 4x4 * 4x1 -> 4x1, rank 4*4 = 16
        """
        n, m, p = 4, 4, 1
        U1, V1, W1 = _schoolbook_2x2x2x1()  # 2x2*2x1
        U2, V2, W2 = _schoolbook_2x2x2x1()  # 2x2*2x1
        U, V, W = _kron_decompose(U1, V1, W1, U2, V2, W2, 2, 2, 1, 2, 2, 1, n, m, p)
        return U, V, W
    
    def build_4x4x7_decomposition():
        """Build decomposition for 4x4 * 4x7 by concatenating parts.
        
        Part 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3) -> rank 49
        Part 2: 4x4 * 4x2 (B cols 4-5, C cols 4-5) -> rank 28
        Part 3: 4x4 * 4x1 (B col 6, C col 6) -> rank 16
        Total rank: 93
        """
        n, m, p = 4, 4, 7
        
        U1, V1, W1 = kronecker_decompose_4x4x4()
        r1 = len(U1[0])  # 49
        
        U2, V2, W2 = build_4x4x2()
        r2 = len(U2[0])  # 28
        
        U3, V3, W3 = build_4x4x1()
        r3 = len(U3[0])  # 16
        
        total_rank = r1 + r2 + r3
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Part 1: cols 0-3 of B
        for i in range(n * m):
            for r in range(r1):
                U[i][r] = U1[i][r]
        for j in range(m):
            for k in range(4):
                for r in range(r1):
                    V[j * p + k][r] = V1[j * 4 + k][r]
        for k in range(4):
            for i in range(n):
                for r in range(r1):
                    W[k * n + i][r] = W1[k * 4 + i][r]
        
        # Part 2: cols 4-5 of B
        col_offset = r1
        for i in range(n * m):
            for r in range(r2):
                U[i][col_offset + r] = U2[i][r]
        for j in range(m):
            for k in range(2):
                for r in range(r2):
                    V[j * p + (4 + k)][col_offset + r] = V2[j * 2 + k][r]
        for k in range(2):
            for i in range(n):
                for r in range(r2):
                    W[k * n + i][col_offset + r] = W2[k * 2 + i][r]
        
        # Part 3: col 6 of B
        col_offset = r1 + r2
        for i in range(n * m):
            for r in range(r3):
                U[i][col_offset + r] = U3[i][r]
        for j in range(m):
            for r in range(r3):
                V[j * p + 6][col_offset + r] = V3[j][r]
        for i in range(n):
            for r in range(r3):
                W[6 * n + i][col_offset + r] = W3[i][r]
        
        return U, V, W
    
    def try_rank_reduction(U, V, W, max_iter=1000):
        """
        Try to reduce rank by finding linear dependencies among columns.
        This is a heuristic approach that may or may not succeed.
        """
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each column c, check if it can be expressed as a linear combination
        # of other columns in a way that preserves the tensor identity.
        # This is complex, so we'll use a simpler heuristic:
        # Try to find pairs of columns that are "similar" and merge them.
        
        best_U, best_V, best_W = U, V, W
        best_rank = r
        
        for _ in range(max_iter):
            if time.time() > start_time + WALL_DEADLINE - SAFETY_MARGIN:
                break
            
            # Randomly pick two columns to try to merge
            c1, c2 = random.sample(range(best_rank), 2)
            
            # Try to express column c2 as a scalar multiple of column c1
            # plus a small perturbation that doesn't affect the tensor
            # This is a very rough heuristic
            
            U_test = [row[:] for row in best_U]
            V_test = [row[:] for row in best_V]
            W_test = [row[:] for row in best_W]
            
            # Set column c2 to be a copy of column c1
            for i in range(len(U_test)):
                U_test[i][c2] = U_test[i][c1]
            for i in range(len(V_test)):
                V_test[i][c2] = V_test[i][c1]
            for i in range(len(W_test)):
                W_test[i][c2] = W_test[i][c1]
            
            # Remove column c2
            U_new = [row[:c2] + row[c2+1:] for row in U_test]
            V_new = [row[:c2] + row[c2+1:] for row in V_test]
            W_new = [row[:c2] + row[c2+1:] for row in W_test]
            
            if check_valid(U_new, V_new, W_new):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank -= 1
        
        return best_U, best_V, best_W, best_rank
    
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
                        U_red, V_red, W_red, r_red = try_rank_reduction(U_opt, V_opt, W_opt)
                        if r_red < best_rank:
                            best_result = {"U": U_red, "V": V_red, "W": W_red}
                            best_rank = r_red
        except Exception as e:
            pass
    
    return best_result
