import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    # Fallback: Rank 50 decomposition from current program
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Part 1: A_L (2x2) * B_T (2x9)
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            # Strassen
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        # Last col of B_T
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        # Part 2: A_R (2x1) * B_B (1x9)
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        U = np.array(U, dtype=np.int64)
        V = np.array(V, dtype=np.int64)
        W = np.array(W, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if val != 1:
                        return False
        return True

    def try_structured_rank(r):
        """
        Try to find a rank-r decomposition by:
        1. Fixing W to have a specific sparse structure (2 non-zeros per column, one for each i in {0,1})
        2. Solving for U and V using integer linear algebra
        """
        if time.time() > deadline - 10:
            return None
        
        # Strategy: For each column c, W[:,c] has non-zeros at exactly 2 positions (one for i=0, one for i=1)
        # W[k*n+i, c] is non-zero for exactly one k per i
        # This means for each c, we pick k0(c) for i=0 and k1(c) for i=1
        
        # We'll try different patterns for W
        # Pattern: W[k*n+i, c] = val where k is determined by a function of c
        
        for trial in range(50):
            if time.time() > deadline - 5:
                return None
            
            # Randomly assign which k each column uses for each i
            W = np.zeros((p_rows, r), dtype=np.int64)
            for c in range(r):
                # For i=0: pick a random k in [0, p)
                k0 = random.randint(0, p-1)
                W[k0*n + 0, c] = random.choice([-1, 1, 2, -2])
                # For i=1: pick a random k in [0, p)
                k1 = random.randint(0, p-1)
                W[k1*n + 1, c] = random.choice([-1, 1, 2, -2])
            
            # Now solve for V and U
            # For each (i, j, k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is a system of equations. Let's try to solve it.
            
            # Build the system: For each (i,j,k), we have:
            # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
            # Let X[i*m+j, j*p+k, c] = U[i*m+j,c] * V[j*p+k,c]
            # Then sum_c X[i*m+j, j*p+k, c] * W[k*n+i,c] = 1
            
            # This is a rank-constrained system. Let's try a different approach:
            # Fix V sparsely, then solve for U.
            
            # Try: V has 1 non-zero per column
            V = np.zeros((m_rows, r), dtype=np.int64)
            for c in range(r):
                # Pick a random (j, k) for the non-zero
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                V[j*p+k, c] = random.choice([-1, 1, 2, -2])
            
            # Now for each (i, j), solve for U[i*m+j, :]:
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
            # This is: M[i,j] @ U[i*m+j,:] = ones(p)
            # where M[i,j][k,c] = V[j*p+k,c] * W[k*n+i,c]
            
            U = np.zeros((n_rows, r), dtype=np.int64)
            valid = True
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r), dtype=np.int64)
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    
                    # Solve M @ u = ones(p)
                    # Use least squares, check if integer solution exists
                    try:
                        u_float, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        u_int = np.round(u_float).astype(np.int64)
                        # Check exactness
                        if not np.allclose(M @ u_int, np.ones(p), atol=1e-10):
                            valid = False
                            break
                        U[i*m+j, :] = u_int
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Full verification
            if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), r):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        return None

    def try_systematic_rank(r):
        """
        More systematic approach: Use the structure of the problem.
        For n=2, m=3, p=9, the tensor has 2*3*9 = 54 entries, all = 1.
        
        Key insight: We can try to find a decomposition where each column of W
        corresponds to a specific (k0, k1) pair, and the U and V matrices are sparse.
        """
        if time.time() > deadline - 10:
            return None
        
        # Try using a more structured approach based on the Strassen-like decomposition
        # but with optimization
        
        # Approach: Start from the rank-50 solution and try to merge columns
        base = current_solution()
        U0 = np.array(base["U"], dtype=np.int64)
        V0 = np.array(base["V"], dtype=np.int64)
        W0 = np.array(base["W"], dtype=np.int64)
        
        # Try to find linear dependencies among the columns
        # If two columns are linearly dependent in a certain way, we might merge them
        
        # For now, try a random perturbation approach
        for trial in range(30):
            if time.time() > deadline - 5:
                return None
            
            # Start from base and try to reduce rank by finding redundant columns
            # This is complex; instead, try a fresh random search with better initialization
            
            # Initialize W with the pattern from the base solution
            W = np.zeros((p_rows, r), dtype=np.int64)
            V = np.zeros((m_rows, r), dtype=np.int64)
            U = np.zeros((n_rows, r), dtype=np.int64)
            
            # Copy first min(r, 50) columns from base
            copy_r = min(r, 50)
            W[:, :copy_r] = W0[:, :copy_r]
            V[:, :copy_r] = V0[:, :copy_r]
            U[:, :copy_r] = U0[:, :copy_r]
            
            # If r < 50, we need to drop some columns and adjust
            # This is hard; skip for now and just verify if r == 50
            if r == 50:
                if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), r):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
            
            # For r < 50, try a different approach
            # Use a greedy column selection
            if r < 50:
                # Select r columns from the 50 that best cover the tensor
                # This is a set cover problem; use greedy
                selected = []
                remaining = set(range(54))  # 54 = n*m*p entries
                
                for _ in range(r):
                    best_col = -1
                    best_count = -1
                    for c in range(50):
                        if c in selected:
                            continue
                        # Count how many new entries this column covers
                        count = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    if (i,j,k) in remaining:
                                        if U0[i*m+j, c] != 0 and V0[j*p+k, c] != 0 and W0[k*n+i, c] != 0:
                                            count += 1
                        if count > best_count:
                            best_count = count
                            best_col = c
                    if best_col >= 0:
                        selected.append(best_col)
                        # Update remaining (this is approximate)
                
                if len(selected) == r:
                    U_sel = U0[:, selected]
                    V_sel = V0[:, selected]
                    W_sel = W0[:, selected]
                    if verify_decomposition(U_sel.tolist(), V_sel.tolist(), W_sel.tolist(), r):
                        return {
                            "U": U_sel.tolist(),
                            "V": V_sel.tolist(),
                            "W": W_sel.tolist()
                        }
        
        return None

    # Try to find lower rank solutions
    best_result = None
    
    # First try structured random search for ranks 49 down to 40
    for r in range(49, 39, -1):
        if time.time() > deadline - 15:
            break
        res = try_structured_rank(r)
        if res:
            best_result = res
            break
    
    # If not found, try systematic approach
    if best_result is None and time.time() < deadline - 10:
        for r in range(49, 44, -1):
            if time.time() > deadline - 5:
                break
            res = try_systematic_rank(r)
            if res:
                best_result = res
                break
    
    # Fallback to rank 50
    if best_result is None:
        best_result = current_solution()
    
    return best_result
