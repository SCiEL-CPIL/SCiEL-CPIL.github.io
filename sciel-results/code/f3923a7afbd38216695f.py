import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def to_list(mat):
        return mat.tolist()

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        rank = U.shape[1]
        
        # Check shapes
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        
        # For each (i,j,k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # And that for all other w, the contribution is 0
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_target = k * n + i
                    
                    # Compute contributions to all w
                    contrib = np.zeros(p*n)
                    for r in range(rank):
                        uv = U[u_idx, r] * V[v_idx, r]
                        if uv != 0:
                            contrib += uv * W[:, r]
                    
                    # Check target
                    if abs(contrib[w_target] - 1.0) > 1e-9:
                        return False
                    # Check all others are zero
                    for ww in range(p*n):
                        if ww != w_target and abs(contrib[ww]) > 1e-9:
                            return False
        return True

    # Known rank 17 solution as fallback
    rank_17 = 17
    U17 = np.zeros((n * m, rank_17))
    V17 = np.zeros((m * p, rank_17))
    W17 = np.zeros((p * n, rank_17))
    
    # Term 1
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    # Term 2
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    # Term 3
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    # Term 4
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    # Term 5
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    # Term 6
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    # Term 7
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    # Term 8
    U17[0, 7] = 1
    V17[2, 7] = 1
    W17[4, 7] = 1
    
    # Term 9
    U17[1, 8] = 1
    V17[5, 8] = 1
    W17[4, 8] = 1
    
    # Term 10
    U17[3, 9] = 1
    V17[2, 9] = 1
    W17[5, 9] = 1
    
    # Term 11
    U17[4, 10] = 1
    V17[5, 10] = 1
    W17[5, 10] = 1
    
    # Term 12
    U17[2, 11] = 1
    V17[6, 11] = 1
    W17[0, 11] = 1
    
    # Term 13
    U17[2, 12] = 1
    V17[7, 12] = 1
    W17[2, 12] = 1
    
    # Term 14
    U17[2, 13] = 1
    V17[8, 13] = 1
    W17[4, 13] = 1
    
    # Term 15
    U17[5, 14] = 1
    V17[6, 14] = 1
    W17[1, 14] = 1
    
    # Term 16
    U17[5, 15] = 1
    V17[7, 15] = 1
    W17[3, 15] = 1
    
    # Term 17
    U17[5, 16] = 1
    V17[8, 16] = 1
    W17[5, 16] = 1

    # Structured block decomposition approach:
    # Decompose 2x3 * 3x3 into blocks:
    # A = [[A11, A12, A13], [A21, A22, A23]] (2x3)
    # B = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]] (3x3)
    # 
    # C = A*B (2x3)
    # C11 = A11*B11 + A12*B21 + A13*B31
    # C12 = A11*B12 + A12*B22 + A13*B32
    # C13 = A11*B13 + A12*B23 + A13*B33
    # C21 = A21*B11 + A22*B21 + A23*B31
    # C22 = A21*B12 + A22*B22 + A23*B32
    # C23 = A21*B13 + A22*B23 + A23*B33
    #
    # Strategy: Use Strassen (rank 7) for the 2x2 block (rows 1-2, cols 1-2 of B),
    # then handle the third column of B and cross terms.
    #
    # Let's try to find a rank-16 solution by systematic search.
    # We'll use a more structured random search with constraints.

    # Attempt 1: Structured search with Strassen-like patterns
    # The 2x2x2 tensor has rank 7 (Strassen). We'll embed this and search for the rest.
    
    # First, let's try a direct rank-16 search with better heuristics
    best_rank = 17
    best_solution = (U17, V17, W17)
    
    # Try rank 16 with structured initialization
    for trial in range(5000):
        if time.time() > deadline - 5.0:
            break
        
        R = 16
        np.random.seed(trial)
        
        # Use a mixture of sparse and slightly denser patterns
        density = 0.15 + 0.1 * (trial % 5)  # Vary density
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        
        for r in range(R):
            # For each rank term, choose a few non-zero entries
            n_u_nonzero = np.random.choice([1, 2, 2, 3], p=[0.3, 0.3, 0.2, 0.2])
            n_v_nonzero = np.random.choice([1, 2, 2, 3], p=[0.3, 0.3, 0.2, 0.2])
            
            u_indices = np.random.choice(n*m, size=n_u_nonzero, replace=False)
            v_indices = np.random.choice(m*p, size=n_v_nonzero, replace=False)
            
            for idx in u_indices:
                U[idx, r] = np.random.choice([-1, 1])
            for idx in v_indices:
                V[idx, r] = np.random.choice([-1, 1])
        
        # Build linear system for W
        A_mat = np.zeros((18, p*n*R))
        b_vec = np.ones(18)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    for r in range(R):
                        coeff = U[u, r] * V[v, r]
                        A_mat[eq_idx, w * R + r] = coeff
                    eq_idx += 1
        
        try:
            W_sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            W_sol = W_sol.reshape((p*n, R))
            
            # Check if close to integer or half-integer
            W_int = np.round(W_sol * 2) / 2
            if np.allclose(W_sol, W_int, atol=1e-6):
                if verify_decomposition(U, V, W_int, n, m, p):
                    return {"U": to_list(U), "V": to_list(V), "W": to_list(W_int)}
        except:
            pass

    # Attempt 2: Use the rank-17 solution as a starting point and try to merge terms
    # Look for pairs of terms that can be combined
    U_cur = U17.copy()
    V_cur = V17.copy()
    W_cur = W17.copy()
    
    for merge_trial in range(1000):
        if time.time() > deadline - 5.0:
            break
        
        R = 17
        U_try = U_cur.copy()
        V_try = V_cur.copy()
        W_try = W_cur.copy()
        
        # Try to zero out one column by adjusting others
        # Pick a random column to eliminate
        col_to_elim = np.random.randint(R)
        
        # For each row in U, V, W, try to find a linear combination
        # that zeros out the selected column
        success = True
        for r in range(R):
            if r == col_to_elim:
                continue
            
            # Try to express column r as a combination of other columns
            # This is complex; instead, try random perturbations
            pass
        
        # Simpler: try random rank-16 subspaces
        np.random.seed(merge_trial + 10000)
        U_rand = np.random.choice([-1, 0, 1], size=(n*m, 16))
        V_rand = np.random.choice([-1, 0, 1], size=(m*p, 16))
        
        A_mat = np.zeros((18, 96))
        b_vec = np.ones(18)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    for r in range(16):
                        coeff = U_rand[u, r] * V_rand[v, r]
                        A_mat[eq_idx, w * 16 + r] = coeff
                    eq_idx += 1
        
        try:
            W_sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
            W_sol = W_sol.reshape((p*n, 16))
            
            W_int = np.round(W_sol * 2) / 2
            if np.allclose(W_sol, W_int, atol=1e-6):
                if verify_decomposition(U_rand, V_rand, W_int, n, m, p):
                    return {"U": to_list(U_rand), "V": to_list(V_rand), "W": to_list(W_int)}
        except:
            pass

    # Return the best known solution (rank 17)
    return {"U": to_list(U17), "V": to_list(V17), "W": to_list(W17)}
