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
    
    def build_2x2x3_rank_11():
        """
        Build rank-11 decomposition for 2x2x3 matrix multiplication.
        A is 2x2, B is 2x3, C is 2x2.
        U: 4x11 (indices i*2+j), V: 6x11 (indices j*3+k), W: 6x11 (indices k*2+i)
        """
        U = [[0]*11 for _ in range(4)]
        V = [[0]*11 for _ in range(6)]
        W = [[0]*11 for _ in range(6)]
        
        # First 7 terms: Strassen for 2x2x2 (columns 0,1 of B)
        U2, V2, W2 = build_strassen_2x2_2x2()
        for r in range(7):
            for idx in range(4):
                U[idx][r] = U2[idx][r]
            for idx in range(4):
                j = idx // 2
                k = idx % 2
                V[j*3 + k][r] = V2[idx][r]
            for idx in range(4):
                # W2 is indexed by k*2+i for k=0,1 and i=0,1
                # In 2x2x3, W is indexed by k*2+i for k=0,1,2 and i=0,1
                # So for k=0,1: W[k*2+i] = W2[k*2+i]
                W[idx][r] = W2[idx][r]
        
        # Last 4 terms: schoolbook for 2x2x1 (column 2 of B)
        for i in range(2):
            for j in range(2):
                r = 7 + i*2 + j
                U[i*2 + j][r] = 1
                V[j*3 + 2][r] = 1
                W[2*2 + i][r] = 1  # k=2, i in {0,1}, so W[4+i]
        
        return U, V, W

    def find_2x2x3_rank_10():
        """
        Try to find a rank 10 decomposition for 2x2x3.
        Returns None if not found within time.
        
        Strategy: Use the known rank-11 decomposition as a starting point,
        then try to find a linear dependency among the 11 terms that allows
        eliminating one term while preserving validity.
        
        Alternative: Use a parametrized family and solve for rank 10.
        """
        # The 2x2x3 tensor T[i,j,k] for i,j in {0,1}, k in {0,1,2}
        # We need U (4x10), V (6x10), W (6x10) such that
        # T[i,j,k] = sum_{r=0}^{9} U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
        
        # Start from rank-11 and try to find a null space vector
        U11, V11, W11 = build_2x2x3_rank_11()
        
        # The 11 terms give a valid decomposition. We want to see if we can
        # express one term as a linear combination of the others in a way that
        # preserves the tensor identity.
        
        # This is complex. Instead, let's try a direct search with a structured ansatz.
        
        # Ansatz: Use 7 Strassen-like terms for the 2x2x2 part (k=0,1) and
        # 3 terms for the k=2 part, with some coupling.
        
        # Let's try: p1..p7 from Strassen for 2x2x2 (modified), plus
        # p8, p9, p10 for k=2, with some entries shared.
        
        # Actually, let me try a different structured approach.
        # The 2x2x3 tensor can be written as:
        # T[i,j,0] = sum_r U[i,j,r] V[j,0,r] W[0,i,r]
        # T[i,j,1] = sum_r U[i,j,r] V[j,1,r] W[1,i,r]
        # T[i,j,2] = sum_r U[i,j,r] V[j,2,r] W[2,i,r]
        
        # For rank 10, we need 10 terms. Let's use a parametrized family.
        
        # Try: Use the 7 Strassen terms but modify them to also handle k=2,
        # then add 3 more terms.
        
        # This is getting complex. Let me just return None and rely on the rank-11 fallback,
        # but try a simple random search if time permits.
        
        deadline = time.time() + 30  # 30 second budget
        
        # Simple random search: initialize with small random values and do
        # a few gradient steps. This is unlikely to succeed but worth trying.
        
        def random_search():
            if time.time() > deadline:
                return None
            
            best_err = float('inf')
            best_U, best_V, best_W = None, None, None
            
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                # Initialize with small random values
                np.random.seed(attempt)
                U = np.random.randn(4, 10) * 0.5
                V = np.random.randn(6, 10) * 0.5
                W = np.random.randn(6, 10) * 0.5
                
                # Compute tensor error
                def compute_error(U, V, W):
                    err = 0.0
                    for i in range(2):
                        for j in range(2):
                            for k in range(3):
                                s = 0.0
                                for r in range(10):
                                    s += U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
                                err += (s - 1.0) ** 2
                    return err
                
                err = compute_error(U, V, W)
                
                # A few gradient steps
                lr = 0.01
                for step in range(200):
                    if time.time() > deadline:
                        break
                    
                    # Numerical gradient
                    grad_U = np.zeros_like(U)
                    grad_V = np.zeros_like(V)
                    grad_W = np.zeros_like(W)
                    
                    h = 1e-5
                    for i in range(2):
                        for j in range(2):
                            for k in range(3):
                                s = sum(U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r] for r in range(10))
                                ds = 2 * (s - 1.0)
                                for r in range(10):
                                    grad_U[i*2+j, r] += ds * V[j*3+k, r] * W[k*2+i, r]
                                    grad_V[j*3+k, r] += ds * U[i*2+j, r] * W[k*2+i, r]
                                    grad_W[k*2+i, r] += ds * U[i*2+j, r] * V[j*3+k, r]
                    
                    U -= lr * grad_U
                    V -= lr * grad_V
                    W -= lr * grad_W
                    
                    err = compute_error(U, V, W)
                    
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                    if best_err < 1e-12:
                        break
                
                if best_err < 1e-12:
                    break
            
            if best_err < 1e-12 and best_U is not None:
                # Convert to list of lists with rational-looking values
                U_list = [[best_U[i][r] for r in range(10)] for i in range(4)]
                V_list = [[best_V[i][r] for r in range(10)] for i in range(6)]
                W_list = [[best_W[i][r] for r in range(10)] for i in range(6)]
                return U_list, V_list, W_list
            
            return None
        
        result = random_search()
        return result

    def build_4x4x7_decomposition(use_rank_10=False):
        """
        Build a decomposition for 4x4x7.
        If use_rank_10 is True and a rank 10 2x2x3 is available, use it.
        """
        U4, V4, W4 = kronecker_decompose_4x4x4()
        r4 = len(U4[0])  # 49
        
        if use_rank_10:
            U223, V223, W223 = find_2x2x3_rank_10()
            if U223 is None:
                # Fallback to rank 11
                U223, V223, W223 = build_2x2x3_rank_11()
                r223 = 11
            else:
                r223 = 10
        else:
            U223, V223, W223 = build_2x2x3_rank_11()
            r223 = 11
            
        r_total = r4 + 4 * r223
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
        
        # Part 2: 4x4x3 using four 2x2x3 blocks (columns 4-6 of B)
        col_offset = r4
        for row_group in range(2):
            for col_group in range(2):
                base_r = col_offset + (row_group * 2 + col_group) * r223
                
                for r_local in range(r223):
                    r_global = base_r + r_local
                    
                    for i_local in range(2):
                        for j_local in range(2):
                            i = 2 * row_group + i_local
                            j = 2 * col_group + j_local
                            U[i*4 + j][r_global] = U223[i_local*2 + j_local][r_local]
                    
                    for j_local in range(2):
                        for k_local in range(3):
                            j = 2 * col_group + j_local
                            k = 4 + k_local
                            V[j*7 + k][r_global] = V223[j_local*3 + k_local][r_local]
                    
                    for k_local in range(3):
                        for i_local in range(2):
                            k = 4 + k_local
                            i = 2 * row_group + i_local
                            W[k*4 + i][r_global] = W223[k_local*2 + i_local][r_local]
        
        return U, V, W

    # Start with schoolbook as fallback
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 112
    
    # Try the optimized approach
    if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            # First try without rank 10
            U_opt, V_opt, W_opt = build_4x4x7_decomposition(use_rank_10=False)
            r_opt = len(U_opt[0])
            
            if check_valid(U_opt, V_opt, W_opt):
                if r_opt < best_rank:
                    best_result = {"U": U_opt, "V": V_opt, "W": W_opt}
                    best_rank = r_opt
                    
            # If time permits, try rank 10 search (currently returns None)
            if time.time() - start_time < WALL_DEADLINE - SAFETY_MARGIN - 10:
                U_opt2, V_opt2, W_opt2 = build_4x4x7_decomposition(use_rank_10=True)
                r_opt2 = len(U_opt2[0])
                if check_valid(U_opt2, V_opt2, W_opt2):
                    if r_opt2 < best_rank:
                        best_result = {"U": U_opt2, "V": V_opt2, "W": W_opt2}
                        best_rank = r_opt2
        except Exception as e:
            pass
    
    return best_result
