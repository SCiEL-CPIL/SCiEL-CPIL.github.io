import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def to_list(mat):
        return mat.tolist()

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        rank = U.shape[1]
        
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_target = k * n + i
                    
                    contrib = np.zeros(p*n)
                    for r in range(rank):
                        uv = U[u_idx, r] * V[v_idx, r]
                        if uv != 0:
                            contrib += uv * W[:, r]
                    
                    if abs(contrib[w_target] - 1.0) > 1e-8:
                        return False
                    for ww in range(p*n):
                        if ww != w_target and abs(contrib[ww]) > 1e-8:
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

    # Verify the rank-17 solution
    if verify_decomposition(U17, V17, W17, n, m, p):
        best_solution = (U17, V17, W17)
        best_rank = 17
    else:
        best_solution = None
        best_rank = 18

    # Search for rank 16 solution
    # Strategy: random sparse U,V with entries in {-1, 0, 1}, solve for W, check half-integer
    R = 16
    num_trials = 0
    max_trials = 50000
    
    # Precompute equation structure
    # For each (i,j,k): u_idx = i*m+j, v_idx = j*p+k, w_target = k*n+i
    eq_u = []
    eq_v = []
    eq_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_u.append(i * m + j)
                eq_v.append(j * p + k)
                eq_w.append(k * n + i)
    eq_u = np.array(eq_u)
    eq_v = np.array(eq_v)
    eq_w = np.array(eq_w)
    
    # Build the coefficient matrix structure for W
    # A[eq, w*R + r] = U[eq_u[eq], r] * V[eq_v[eq], r]
    
    found = False
    trial = 0
    while not found and trial < max_trials:
        if time.time() > deadline - 3.0:
            break
        
        trial += 1
        num_trials += 1
        
        # Vary the sparsity pattern
        # Use different densities and value sets
        if trial % 3 == 0:
            # Dense random
            U = np.random.choice([-1, 0, 1], size=(n*m, R), p=[0.3, 0.4, 0.3])
            V = np.random.choice([-1, 0, 1], size=(m*p, R), p=[0.3, 0.4, 0.3])
        elif trial % 3 == 1:
            # Sparse
            U = np.random.choice([-1, 0, 1], size=(n*m, R), p=[0.2, 0.6, 0.2])
            V = np.random.choice([-1, 0, 1], size=(m*p, R), p=[0.2, 0.6, 0.2])
        else:
            # Very sparse
            U = np.random.choice([-1, 0, 1], size=(n*m, R), p=[0.15, 0.7, 0.15])
            V = np.random.choice([-1, 0, 1], size=(m*p, R), p=[0.15, 0.7, 0.15])
        
        # Build A matrix: 18 x (9*16) = 18 x 144
        A = np.zeros((18, p*n*R))
        for eq in range(18):
            u_idx = eq_u[eq]
            v_idx = eq_v[eq]
            w_idx = eq_w[eq]
            for r in range(R):
                coeff = U[u_idx, r] * V[v_idx, r]
                if coeff != 0:
                    A[eq, w_idx * R + r] = coeff
        
        # Solve for W
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(18), rcond=None)
        except:
            continue
        
        W_sol = W_flat.reshape((p*n, R))
        
        # Check if close to half-integer
        W_half = np.round(W_sol * 2) / 2.0
        if np.max(np.abs(W_sol - W_half)) < 1e-7:
            if verify_decomposition(U, V, W_half, n, m, p):
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W_half)}
    
    # If we didn't find rank 16 with random search, try a more structured approach
    # Use the rank-17 solution and try to merge two terms into one
    # This is equivalent to finding a rank-16 decomposition
    
    # Alternative: try rank 16 with structured initialization based on Strassen
    # For the 2x2x2 subproblem (rows 1-2, cols 1-2 of B), use Strassen's 7 terms
    # Then add 9 more terms for the remaining parts
    
    # Let's try a different approach: use integer linear programming via scipy
    # Actually, let's just do more random trials with different seeds
    
    trial2 = 0
    max_trials2 = 30000
    while not found and trial2 < max_trials2:
        if time.time() > deadline - 3.0:
            break
        
        trial2 += 1
        
        # Use a different random strategy: fix some structure
        # Each rank term should have a "meaningful" pattern
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        
        for r in range(R):
            # Choose 1-3 non-zero entries for U
            n_u = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
            u_idx = np.random.choice(n*m, size=n_u, replace=False)
            U[u_idx, r] = np.random.choice([-1, 1], size=n_u)
            
            # Choose 1-3 non-zero entries for V
            n_v = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
            v_idx = np.random.choice(m*p, size=n_v, replace=False)
            V[v_idx, r] = np.random.choice([-1, 1], size=n_v)
        
        # Build A matrix
        A = np.zeros((18, p*n*R))
        for eq in range(18):
            u_idx = eq_u[eq]
            v_idx = eq_v[eq]
            w_idx = eq_w[eq]
            for r in range(R):
                coeff = U[u_idx, r] * V[v_idx, r]
                if coeff != 0:
                    A[eq, w_idx * R + r] = coeff
        
        # Solve for W
        try:
            W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(18), rcond=None)
        except:
            continue
        
        W_sol = W_flat.reshape((p*n, R))
        
        # Check if close to half-integer
        W_half = np.round(W_sol * 2) / 2.0
        if np.max(np.abs(W_sol - W_half)) < 1e-7:
            if verify_decomposition(U, V, W_half, n, m, p):
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W_half)}
    
    # If no rank-16 found, return the best we have (rank 17)
    if best_solution is not None:
        U, V, W = best_solution
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Fallback: schoolbook rank 18
    U18 = np.zeros((n*m, 18))
    V18 = np.zeros((m*p, 18))
    W18 = np.zeros((p*n, 18))
    for r, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
        U18[i*m+j, r] = 1
        V18[j*p+k, r] = 1
        W18[k*n+i, r] = 1
    
    return {"U": to_list(U18), "V": to_list(V18), "W": to_list(W18)}
