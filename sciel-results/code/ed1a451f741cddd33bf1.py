"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=5
    # U: (n*m) x rank = 16 x rank
    # V: (m*p) x rank = 20 x rank  
    # W: (p*n) x rank = 20 x rank
    # Convention: W[k*n+i, r]
    
    start_time = time.time()
    deadline = 145  # 5s safety margin before 150s kill
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        if not U or not V or not W:
            return False
        rank = len(U[0]) if U else 0
        if rank == 0:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Schoolbook decomposition as baseline
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
    
    best = schoolbook()
    best_rank = n * m * p  # 80
    
    def strassen_2x2():
        """Return U, V, W for 2x2*2x2 Strassen decomposition.
        U: 4x7, V: 4x7, W: 4x7
        Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # r=0: M1 = (A11+A22)*(B11+B22)
        U[0][0] = 1  # A11
        U[3][0] = 1  # A22
        V[0][0] = 1  # B11
        V[3][0] = 1  # B22
        W[0][0] = 1  # C11 (k=0,i=0 -> k*2+i=0)
        W[3][0] = 1  # C22 (k=1,i=1 -> k*2+i=3)
        
        # r=1: M2 = (A21+A22)*B11
        U[2][1] = 1  # A21
        U[3][1] = 1  # A22
        V[0][1] = 1  # B11
        W[2][1] = 1  # C21 (k=1,i=0 -> k*2+i=2)
        
        # r=2: M3 = A11*(B12-B22)
        U[0][2] = 1  # A11
        V[1][2] = 1  # B12
        V[3][2] = -1  # B22
        W[1][2] = 1  # C12 (k=0,i=1 -> k*2+i=1)
        
        # r=3: M4 = A22*(B21-B11)
        U[3][3] = 1  # A22
        V[2][3] = 1  # B21
        V[0][3] = -1  # B11
        W[0][3] = 1  # C11
        W[2][3] = 1  # C21
        
        # r=4: M5 = (A11+A12)*B22
        U[0][4] = 1  # A11
        U[1][4] = 1  # A12
        V[3][4] = 1  # B22
        W[0][4] = -1  # C11
        W[1][4] = 1  # C12
        
        # r=5: M6 = (A21-A11)*(B11+B12)
        U[2][5] = 1  # A21
        U[0][5] = -1  # A11
        V[0][5] = 1  # B11
        V[1][5] = 1  # B12
        W[3][5] = 1  # C22
        
        # r=6: M7 = (A12-A22)*(B21+B22)
        U[1][6] = 1  # A12
        U[3][6] = -1  # A22
        V[2][6] = 1  # B21
        V[3][6] = 1  # B22
        W[0][6] = 1  # C11
        W[3][6] = 1  # C22
        
        return U, V, W
    
    def build_strassen_block():
        """Build 4x4*4x5 using Strassen for 2x2*2x2 and schoolbook for 2x2*2x3."""
        U_list = []
        V_list = []
        W_list = []
        
        sU, sV, sW = strassen_2x2()
        
        def add_strassen(a_map, b_map, c_map):
            """Add a Strassen-based 2x2*2x2 block multiplication."""
            for r in range(7):
                U_new = [0]*16
                V_new = [0]*20
                W_new = [0]*20
                for si, val in enumerate(sU):
                    if val != 0:
                        U_new[a_map[si]] = val
                for sj, val in enumerate(sV):
                    if val != 0:
                        V_new[b_map[sj]] = val
                for sk, val in enumerate(sW):
                    if val != 0:
                        W_new[c_map[sk]] = val
                U_list.append(U_new)
                V_list.append(V_new)
                W_list.append(W_new)
        
        def add_schoolbook_2x2x2x3(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            """Add schoolbook terms for 2x2 * 2x3 block multiplication."""
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        U_new = [0]*16
                        V_new = [0]*20
                        W_new = [0]*20
                        U_new[(a_row_start+i)*4 + (a_col_start+j)] = 1
                        V_new[(b_row_start+j)*5 + (b_col_start+k)] = 1
                        W_new[(c_col_start+k)*4 + (c_row_start+i)] = 1
                        U_list.append(U_new)
                        V_list.append(V_new)
                        W_list.append(W_new)
        
        # Block (1,1): C11 += A11*B11
        a_map1 = [0, 1, 4, 5]      
        b_map1 = [0, 1, 5, 6]      
        c_map1 = [0, 1, 4, 5]
        add_strassen(a_map1, b_map1, c_map1)
        
        # Block (1,2): C12 += A11*B12
        add_schoolbook_2x2x2x3(0, 0, 0, 2, 0, 2)
        
        # Block (2,1): C21 += A21*B11
        a_map2 = [8, 9, 12, 13]    
        b_map2 = [0, 1, 5, 6]
        c_map2 = [2, 3, 6, 7]
        add_strassen(a_map2, b_map2, c_map2)
        
        # Block (2,2): C22 += A21*B12
        add_schoolbook_2x2x2x3(2, 0, 0, 2, 2, 2)
        
        # Block (3,1): C11 += A12*B21
        a_map3 = [2, 3, 6, 7]      
        b_map3 = [10, 11, 15, 16]  
        c_map3 = [0, 1, 4, 5]      
        add_strassen(a_map3, b_map3, c_map3)
        
        # Block (3,2): C12 += A12*B22
        add_schoolbook_2x2x2x3(0, 2, 2, 2, 0, 2)
        
        # Block (4,1): C21 += A22*B21
        a_map4 = [10, 11, 14, 15]  
        b_map4 = [10, 11, 15, 16]
        c_map4 = [2, 3, 6, 7]      
        add_strassen(a_map4, b_map4, c_map4)
        
        # Block (4,2): C22 += A22*B22
        add_schoolbook_2x2x2x3(2, 2, 2, 2, 2, 2)
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Initialize with Strassen block decomposition
    strassen_result = build_strassen_block()
    if verify(strassen_result["U"], strassen_result["V"], strassen_result["W"]):
        if len(strassen_result["U"]) < best_rank:
            best = strassen_result
            best_rank = len(strassen_result["U"])
    
    # Attempt to reduce rank using ALS and rounding
    def attempt_rank_reduction(U_init, V_init, W_init, target_rank, max_iters=50):
        """
        Try to find a decomposition of rank 'target_rank' by optimizing
        U, V, W initialized from the first 'target_rank' columns of the init.
        """
        if target_rank <= 0:
            return None
        
        U0 = np.array(U_init[:target_rank], dtype=float)
        V0 = np.array(V_init[:target_rank], dtype=float)
        W0 = np.array(W_init[:target_rank], dtype=float)
        
        # Add some random noise to break symmetry and escape local minima
        # But keep it small to stay close to the known solution
        noise_scale = 0.01
        U = U0 + noise_scale * np.random.randn(*U0.shape)
        V = V0 + noise_scale * np.random.randn(*V0.shape)
        W = W0 + noise_scale * np.random.randn(*W0.shape)
        
        # Precompute indices for efficiency
        # We want to minimize sum_{i,j,k} (sum_r U[i,j,r] V[j,k,r] W[k,i,r] - delta_{i,j,k})^2
        # This is equivalent to minimizing the Frobenius norm of the tensor difference.
        
        def compute_error(U, V, W):
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i, r] * V[j, r] * W[k, r]
                        err += (s - 1.0)**2
            return err
        
        # ALS Steps
        # We update one matrix at a time.
        # For fixed V, W, the problem for U is linear:
        # sum_{i,j,k} (sum_r U[i,r] V[j,r] W[k,r] - 1)^2
        # This can be viewed as a linear regression for each U[i, :].
        
        # Let's implement a simple gradient descent or normal equations solver.
        # Given the small size, normal equations are feasible.
        
        # Actually, let's use a simpler heuristic: 
        # The tensor T_{ijk} = delta_{ijk}.
        # We want T = sum_r U_r \otimes V_r \otimes W_r.
        
        # Update U:
        # For each i, we want to minimize sum_{j,k} (sum_r U[i,r] V[j,r] W[k,r] - 1)^2
        # Let A[i, r, j, k] = V[j,r] * W[k,r]
        # This is a linear system in U[i, :].
        
        # To make it faster, we'll just do a few iterations of coordinate descent or simple GD.
        # Given time constraints, let's try a few random restarts with simple GD.
        
        lr = 0.01
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for i in range(n):
                grad = np.zeros(target_rank)
                for j in range(m):
                    for k in range(p):
                        # Derivative w.r.t U[i,r]
                        # 2 * (sum_r' U[i,r'] V[j,r'] W[k,r'] - 1) * V[j,r] * W[k,r]
                        s = np.dot(U[i], V[j] * W[k])
                        grad += 2 * (s - 1) * (V[j] * W[k])
                U[i] -= lr * grad
            
            # Update V
            for j in range(m):
                grad = np.zeros(target_rank)
                for i in range(n):
                    for k in range(p):
                        s = np.dot(U[i], V[j] * W[k])
                        grad += 2 * (s - 1) * (U[i] * W[k])
                V[j] -= lr * grad
            
            # Update W
            for k in range(p):
                grad = np.zeros(target_rank)
                for i in range(n):
                    for j in range(m):
                        s = np.dot(U[i], V[j] * W[k])
                        grad += 2 * (s - 1) * (U[i] * V[j])
                W[k] -= lr * grad
            
            err = compute_error(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_err < 1e-6:
                break
        
        # Round to half-integers
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Convert to list of lists
        U_list = U_rounded.T.tolist()
        V_list = V_rounded.T.tolist()
        W_list = W_rounded.T.tolist()
        
        # Verify
        if verify(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # If exact verification fails, try a tighter rounding or different init
        # For now, return None if not exact
        return None

    # Try to reduce rank from current best
    current_rank = best_rank
    # We know Strassen gives 64. Let's try 63, 62, etc.
    # But we need a valid init. We can use the first R columns of the current best.
    
    # Let's try a few random restarts for rank = current_rank - 1, current_rank - 2, etc.
    # But we must be careful with time.
    
    for target_rank in range(current_rank - 1, 0, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            
            result = attempt_rank_reduction(best["U"], best["V"], best["W"], target_rank)
            if result:
                if len(result["U"]) < best_rank:
                    best = result
                    best_rank = len(result["U"])
                    found = True
                    break
        
        if found:
            # If we found a lower rank, we can try to go even lower
            continue
        else:
            # If we failed to find rank R, it's likely we won't find R-1 easily with this init
            # But let's just break to save time for now, or continue cautiously
            # Given the difficulty, let's stop after first failure for now
            break

    return best

# EVOLVE-BLOCK-END
