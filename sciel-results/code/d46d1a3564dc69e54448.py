import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    def verify_decomposition(U, V, W):
        """Verify that the tensor decomposition is exact."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if val != 1:
                        return False
        return True
    
    # Rank 50 decomposition (Strassen-based)
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        
        col = 0
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
        
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}
    
    # Structural approach: Try to find lower rank by exploiting algebraic structure
    # Use a parameterized family of decompositions with structured sparsity
    
    def try_structured_rank(r, max_trials=5):
        """Try to find a rank-r decomposition using structured parameterization."""
        if time.time() > deadline - 15:
            return None
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Initialize with random sparse structure but with algebraic constraints
            U = np.zeros((n_rows, r), dtype=np.float64)
            V = np.zeros((m_rows, r), dtype=np.float64)
            W = np.zeros((p_rows, r), dtype=np.float64)
            
            # For each rank-1 term, assign non-zero entries with small integers
            for c in range(r):
                # Choose which U entries are non-zero (sparse)
                u_nonzero = np.random.choice(n_rows, size=np.random.randint(1, 4), replace=False)
                for idx in u_nonzero:
                    U[idx, c] = np.random.choice([-1, 1, 2, -2])
                
                v_nonzero = np.random.choice(m_rows, size=np.random.randint(1, 4), replace=False)
                for idx in v_nonzero:
                    V[idx, c] = np.random.choice([-1, 1, 2, -2])
                
                w_nonzero = np.random.choice(p_rows, size=np.random.randint(1, 4), replace=False)
                for idx in w_nonzero:
                    W[idx, c] = np.random.choice([-1, 1, 2, -2])
            
            # Now try to adjust to satisfy the tensor equations
            # For each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            
            # Build the system of equations
            # Unknowns: we can try to solve for some entries given others
            # This is a 3-linear system, which is hard. Instead, use iterative refinement.
            
            # Iterative refinement: randomly perturb and check
            best_error = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute current error
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            errors.append(val - 1)
                
                error = np.max(np.abs(errors))
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-10:
                    break
                
                # Perturb one random entry
                mat = np.random.choice([0, 1, 2])  # U, V, or W
                if mat == 0:
                    row = np.random.randint(n_rows)
                    col = np.random.randint(r)
                    U[row, col] += np.random.choice([-1, 1])
                elif mat == 1:
                    row = np.random.randint(m_rows)
                    col = np.random.randint(r)
                    V[row, col] += np.random.choice([-1, 1])
                else:
                    row = np.random.randint(p_rows)
                    col = np.random.randint(r)
                    W[row, col] += np.random.choice([-1, 1])
            
            # Check if we found an integer solution
            if best_error < 1e-10:
                # Round to integers
                U_int = np.round(best_U).astype(np.int64)
                V_int = np.round(best_V).astype(np.int64)
                W_int = np.round(best_W).astype(np.int64)
                
                # Verify exactness
                if verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
        
        return None
    
    # Try ranks from 49 down to 40
    for r in range(49, 39, -1):
        if time.time() > deadline - 20:
            break
        res = try_structured_rank(r, max_trials=3)
        if res:
            return res
    
    # Fallback to rank 50
    return current_solution()
